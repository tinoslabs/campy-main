"""Ask a camera one question, so a failure can be named.

OpenCV reports a stream it could not open as a plain False — the 401, the 404
and the unplugged cable all arrive identically, and the page can only say "no
frame available". FFmpeg prints the real reason to its own stderr, where it is
no use to anyone.

So when a stream will not open, we ask the camera ourselves: a single RTSP
OPTIONS request, and the status line it answers with. That is the difference
between telling somebody their password is wrong and telling them to go and
check a switch.
"""
from __future__ import annotations

import base64
import hashlib
import re
import socket
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

DEFAULT_PORT = 554
TIMEOUT = 5.0


@dataclass(frozen=True)
class ProbeResult:
    reachable: bool
    status: int | None
    detail: str

    @property
    def unauthorized(self) -> bool:
        return self.status == 401


def _auth_header(challenge: str, username: str, password: str, url: str) -> str | None:
    """Answer a WWW-Authenticate challenge for OPTIONS.

    Basic is a base64 pair. Digest is MD5(MD5(user:realm:pass):nonce:MD5(OPTIONS:uri)),
    which is what most cameras — Hikvision and Dahua among them — actually use.
    """
    if challenge.lower().startswith("basic"):
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        return f"Basic {token}"
    if not challenge.lower().startswith("digest"):
        return None

    fields = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    realm, nonce = fields.get("realm", ""), fields.get("nonce", "")
    if not nonce:
        return None
    ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode()).hexdigest()  # noqa: S324
    ha2 = hashlib.md5(f"OPTIONS:{url}".encode()).hexdigest()                  # noqa: S324
    response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()       # noqa: S324
    return (f'Digest username="{username}", realm="{realm}", nonce="{nonce}", '
            f'uri="{url}", response="{response}"')


def _request(sock, url, cseq, auth=None) -> str:
    lines = [f"OPTIONS {url} RTSP/1.0", f"CSeq: {cseq}", "User-Agent: CampyAI"]
    if auth:
        lines.append(f"Authorization: {auth}")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
    return sock.recv(2048).decode("latin-1", "replace")


def _status_of(response: str) -> int | None:
    match = re.match(r"RTSP/\d\.\d\s+(\d{3})", response.strip())
    return int(match.group(1)) if match else None


def probe(url: str, username: str = "", password: str = "", timeout: float = TIMEOUT) -> ProbeResult:
    """Find out what the camera says about this address."""
    parts = urlsplit(url)
    host = parts.hostname
    if not host:
        return ProbeResult(False, None, "That stream URL has no host in it.")
    port = parts.port or DEFAULT_PORT
    username = username or unquote(parts.username or "")
    password = password or unquote(parts.password or "")
    # The request line must not carry the credentials.
    bare = f"{parts.scheme}://{host}:{port}{parts.path or '/'}"

    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            response = _request(sock, bare, 1)
            status = _status_of(response)

            if status == 401:
                challenge = ""
                for line in response.splitlines():
                    if line.lower().startswith("www-authenticate:"):
                        challenge = line.split(":", 1)[1].strip()
                        break
                if not username:
                    return ProbeResult(True, 401,
                                       "The camera requires a username and password.")
                auth = _auth_header(challenge, username, password, bare)
                if auth is None:
                    return ProbeResult(True, 401,
                                       "The camera rejected the login. Check the username "
                                       "and password on the camera itself.")
                status = _status_of(_request(sock, bare, 2, auth))
                if status == 401:
                    return ProbeResult(True, 401,
                                       "The camera rejected this username and password.")

            if status is None:
                return ProbeResult(True, None,
                                   "The camera answered, but not in RTSP. Check the port.")
            if status == 404:
                return ProbeResult(True, 404,
                                   "The camera is reachable but has no stream at that path. "
                                   "Check the channel number in the URL.")
            if status >= 400:
                return ProbeResult(True, status, f"The camera answered RTSP {status}.")
            return ProbeResult(True, status, "The camera accepted the connection.")

    except socket.timeout:
        return ProbeResult(False, None,
                           f"No answer from {host}:{port}. Check the address and that this "
                           "server can reach the camera's network.")
    except ConnectionRefusedError:
        return ProbeResult(False, None,
                           f"{host}:{port} refused the connection. Check the port — RTSP is "
                           "usually 554.")
    except OSError as exc:
        return ProbeResult(False, None, f"Could not reach {host}:{port} — {exc}")


def probe_http(url: str, username: str = "", password: str = "",
               timeout: float = TIMEOUT) -> ProbeResult:
    """The same question for an MJPEG or snapshot camera on HTTP."""
    import requests
    from requests.auth import HTTPBasicAuth, HTTPDigestAuth

    parts = urlsplit(url)
    if not parts.hostname:
        return ProbeResult(False, None, "That stream URL has no host in it.")
    username = username or unquote(parts.username or "")
    password = password or unquote(parts.password or "")
    # Credentials travel in the header, never in the requested URL.
    bare = parts._replace(netloc=parts.netloc.rpartition("@")[2]).geturl()

    for auth in (HTTPDigestAuth(username, password) if username else None,
                 HTTPBasicAuth(username, password) if username else None):
        try:
            response = requests.get(bare, auth=auth, timeout=timeout, stream=True)
            response.close()
        except requests.Timeout:
            return ProbeResult(False, None,
                               f"No answer from {parts.hostname}. Check the address and "
                               "that this server can reach the camera's network.")
        except requests.ConnectionError:
            return ProbeResult(False, None,
                               f"Nothing answered at {parts.netloc.rpartition('@')[2]}. "
                               "Check the address and port.")
        except requests.RequestException as exc:
            return ProbeResult(False, None, f"Could not reach the camera — {exc}")

        if response.status_code != 401:
            break

    if response.status_code == 401:
        if not username:
            return ProbeResult(True, 401, "The camera requires a username and password.")
        return ProbeResult(True, 401, "The camera rejected this username and password.")
    if response.status_code == 404:
        return ProbeResult(True, 404,
                           "The camera is reachable but has nothing at that path. Check "
                           "the URL — an MJPEG path is often /video.mjpg or /videostream.cgi.")
    if response.status_code >= 400:
        return ProbeResult(True, response.status_code,
                           f"The camera answered HTTP {response.status_code}.")
    return ProbeResult(True, response.status_code, "The camera accepted the connection.")
