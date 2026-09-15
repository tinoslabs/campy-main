"""A minimal ONVIF Profile S client: ask a camera where its video is.

Most CCTV cameras speak RTSP, but every vendor invents its own URL for it —
Hikvision wants ``/Streaming/Channels/101``, Dahua wants
``/cam/realmonitor?channel=1&subtype=0``, and a dozen others differ again.
ONVIF exists so you do not have to know: you give the camera its IP address
and credentials, and it tells you its own stream URL.

This module implements just enough of ONVIF to do that — ``GetCapabilities``
to find the media service, ``GetProfiles`` to list the streams a camera
publishes, and ``GetStreamUri`` to turn one into an RTSP URL — with the
WS-Security UsernameToken digest that ONVIF devices authenticate with.

It is deliberately dependency-free beyond ``requests``: the full ONVIF WSDL
stack is large, and the four calls below are all that connecting a camera
needs.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree

import requests

logger = logging.getLogger("campy.onvif")

NS = {
    "s": "http://www.w3.org/2003/05/soap-envelope",
    "tds": "http://www.onvif.org/ver10/device/wsdl",
    "trt": "http://www.onvif.org/ver10/media/wsdl",
    "tt": "http://www.onvif.org/ver10/schema",
}
_WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
_WSU = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
_DIGEST = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
)
_B64 = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-soap-message-security-1.0#Base64Binary"
)

DEFAULT_TIMEOUT = 8.0


class OnvifError(RuntimeError):
    """The camera could not be reached, or refused to answer."""


@dataclass(frozen=True)
class StreamProfile:
    """One stream a camera publishes — typically a main and a sub stream."""

    token: str
    name: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    encoding: str = ""

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def label(self) -> str:
        if self.width and self.height:
            return f"{self.name} ({self.width}x{self.height} {self.encoding or ''})".strip()
        return self.name or self.token


def _security_header(username: str, password: str) -> str:
    """WS-Security UsernameToken with a password digest.

    ONVIF never sends the password itself: the digest is
    ``base64(sha1(nonce + created + password))``, which is why the nonce and
    the timestamp have to travel with it.
    """
    if not username:
        return ""
    nonce = os.urandom(16)
    created = datetime.now(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = hashlib.sha1(nonce + created.encode() + password.encode()).digest()
    return (
        f'<s:Header><wsse:Security xmlns:wsse="{_WSSE}" xmlns:wsu="{_WSU}">'
        "<wsse:UsernameToken>"
        f"<wsse:Username>{_escape(username)}</wsse:Username>"
        f'<wsse:Password Type="{_DIGEST}">{base64.b64encode(digest).decode()}</wsse:Password>'
        f'<wsse:Nonce EncodingType="{_B64}">{base64.b64encode(nonce).decode()}</wsse:Nonce>'
        f"<wsu:Created>{created}</wsu:Created>"
        "</wsse:UsernameToken></wsse:Security></s:Header>"
    )


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _call(url: str, body: str, username: str, password: str, timeout: float) -> ElementTree.Element:
    """Send one SOAP 1.2 request and return the parsed body."""
    envelope = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
        'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
        'xmlns:tt="http://www.onvif.org/ver10/schema">'
        f"{_security_header(username, password)}"
        f"<s:Body>{body}</s:Body></s:Envelope>"
    )
    try:
        response = requests.post(
            url,
            data=envelope.encode("utf-8"),
            headers={"Content-Type": "application/soap+xml; charset=utf-8"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise OnvifError(f"Could not reach {url}: {exc}") from exc

    if response.status_code == 401:
        raise OnvifError("The camera rejected these credentials.")
    if response.status_code >= 400 and not response.content:
        raise OnvifError(f"The camera answered {response.status_code}.")

    try:
        root = ElementTree.fromstring(response.content)
    except ElementTree.ParseError as exc:
        raise OnvifError(f"The camera did not return valid ONVIF XML: {exc}") from exc

    fault = root.find(".//s:Fault", NS)
    if fault is not None:
        reason = fault.find(".//s:Text", NS)
        detail = (reason.text or "").strip() if reason is not None else "unknown SOAP fault"
        raise OnvifError(f"The camera refused the request: {detail}")

    body_element = root.find("s:Body", NS)
    if body_element is None:
        raise OnvifError("The camera returned a SOAP envelope with no body.")
    return body_element


def device_service_url(host: str, port: int = 80) -> str:
    host = (host or "").strip()
    if "://" in host:
        parts = urlsplit(host)
        host = parts.hostname or host
    port = int(port or 80)
    scheme = "https" if port == 443 else "http"
    authority = host if port in (80, 443) else f"{host}:{port}"
    return f"{scheme}://{authority}/onvif/device_service"


def _rehost(url: str, host: str) -> str:
    """Point a URL the camera gave us back at the address we can actually reach.

    Cameras routinely report the address they believe they have, which behind
    NAT or on a second interface is not the one we connected to. The path and
    port it returns are still right, so only the host is replaced.
    """
    if not url or not host:
        return url
    parts = urlsplit(url)
    if not parts.hostname or parts.hostname == host:
        return url
    authority = host if parts.port is None else f"{host}:{parts.port}"
    if parts.username:
        credentials = parts.username + (f":{parts.password}" if parts.password else "")
        authority = f"{credentials}@{authority}"
    return urlunsplit((parts.scheme, authority, parts.path, parts.query, parts.fragment))


def get_media_service(host, port=80, username="", password="", timeout=DEFAULT_TIMEOUT) -> str:
    """Ask the device where its media service lives."""
    endpoint = device_service_url(host, port)
    body = "<tds:GetCapabilities><tds:Category>Media</tds:Category></tds:GetCapabilities>"
    try:
        element = _call(endpoint, body, username, password, timeout)
    except OnvifError:
        # Some older firmware only answers GetCapabilities without a category,
        # and a few serve media on the device endpoint itself.
        element = _call(endpoint, "<tds:GetCapabilities/>", username, password, timeout)

    address = element.find(".//tt:Media/tt:XAddr", NS)
    if address is None or not (address.text or "").strip():
        return endpoint
    return _rehost(address.text.strip(), _hostname(host))


def _hostname(host: str) -> str:
    host = (host or "").strip()
    if "://" in host:
        return urlsplit(host).hostname or host
    return host.split(":")[0]


def get_profiles(media_url, username="", password="", timeout=DEFAULT_TIMEOUT) -> list[StreamProfile]:
    """List the streams this camera publishes."""
    element = _call(media_url, "<trt:GetProfiles/>", username, password, timeout)
    profiles: list[StreamProfile] = []
    for node in element.findall(".//trt:Profiles", NS):
        token = node.get("token") or ""
        if not token:
            continue
        name = node.findtext("tt:Name", default="", namespaces=NS) or token
        config = node.find(".//tt:VideoEncoderConfiguration", NS)
        width = height = 0
        fps = 0.0
        encoding = ""
        if config is not None:
            encoding = config.findtext("tt:Encoding", default="", namespaces=NS) or ""
            width = _int(config.findtext(".//tt:Width", default="", namespaces=NS))
            height = _int(config.findtext(".//tt:Height", default="", namespaces=NS))
            fps = float(_int(config.findtext(".//tt:FrameRateLimit", default="", namespaces=NS)))
        profiles.append(
            StreamProfile(token=token, name=name, width=width, height=height,
                          fps=fps, encoding=encoding)
        )
    if not profiles:
        raise OnvifError("The camera reported no video profiles.")
    return profiles


def _int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def get_stream_uri(media_url, token, username="", password="", timeout=DEFAULT_TIMEOUT) -> str:
    """Turn a profile token into an RTSP URL."""
    body = (
        "<trt:GetStreamUri><trt:StreamSetup>"
        "<tt:Stream>RTP-Unicast</tt:Stream>"
        "<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>"
        f"</trt:StreamSetup><trt:ProfileToken>{_escape(token)}</trt:ProfileToken>"
        "</trt:GetStreamUri>"
    )
    element = _call(media_url, body, username, password, timeout)
    uri = element.findtext(".//tt:Uri", default="", namespaces=NS).strip()
    if not uri:
        raise OnvifError("The camera did not return a stream URL for that profile.")
    return uri


def choose_profile(profiles: list[StreamProfile], min_width: int = 640) -> StreamProfile:
    """Pick the cheapest stream still wide enough to analyse.

    Analysis runs at a few hundred pixels across, so decoding a 4K main stream
    burns CPU for detail that is thrown away in the first step. The smallest
    stream at or above ``min_width`` is the right trade; if every stream is
    smaller, take the largest one available.
    """
    sized = [p for p in profiles if p.pixels]
    if not sized:
        return profiles[0]
    big_enough = [p for p in sized if p.width >= min_width]
    if big_enough:
        return min(big_enough, key=lambda p: p.pixels)
    return max(sized, key=lambda p: p.pixels)


def discover_stream_url(
    host, port=80, username="", password="", *, min_width=640, timeout=DEFAULT_TIMEOUT
) -> tuple[str, StreamProfile]:
    """Full discovery: address and credentials in, RTSP URL out."""
    if not (host or "").strip():
        raise OnvifError("No ONVIF host was configured for this camera.")
    media_url = get_media_service(host, port, username, password, timeout)
    profiles = get_profiles(media_url, username, password, timeout)
    profile = choose_profile(profiles, min_width=min_width)
    uri = get_stream_uri(media_url, profile.token, username, password, timeout)
    return _rehost(uri, _hostname(host)), profile


def with_credentials(url: str, username: str, password: str) -> str:
    """RTSP needs the credentials again; ONVIF rarely embeds them in the URL."""
    if not url or not username or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" in rest.split("/", 1)[0]:
        return url
    return f"{scheme}://{username}:{password}@{rest}"


def redact(url: str) -> str:
    """Never let a password reach a log line or a status message."""
    return re.sub(r"://([^:/@]+):([^@/]*)@", r"://\1:•••@", url or "")
