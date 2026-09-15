"""The ONVIF path: address and credentials in, a working RTSP URL out.

These exercise the SOAP client against recorded-shape responses rather than a
real camera, because the interesting failures are all in how a camera answers:
a NAT'd address, a sub-stream worth preferring, a rejected password.
"""
from __future__ import annotations

import base64
import hashlib
from unittest import mock

from django.test import TestCase

from apps.cameras import onvif

CAPABILITIES = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl"
            xmlns:tt="http://www.onvif.org/ver10/schema">
 <s:Body><tds:GetCapabilitiesResponse><tds:Capabilities>
   <tt:Media><tt:XAddr>http://192.168.9.9/onvif/media_service</tt:XAddr></tt:Media>
 </tds:Capabilities></tds:GetCapabilitiesResponse></s:Body></s:Envelope>"""

PROFILES = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
            xmlns:tt="http://www.onvif.org/ver10/schema">
 <s:Body><trt:GetProfilesResponse>
  <trt:Profiles token="main"><tt:Name>MainStream</tt:Name>
    <tt:VideoEncoderConfiguration><tt:Encoding>H264</tt:Encoding>
      <tt:Resolution><tt:Width>3840</tt:Width><tt:Height>2160</tt:Height></tt:Resolution>
      <tt:RateControl><tt:FrameRateLimit>25</tt:FrameRateLimit></tt:RateControl>
    </tt:VideoEncoderConfiguration></trt:Profiles>
  <trt:Profiles token="sub"><tt:Name>SubStream</tt:Name>
    <tt:VideoEncoderConfiguration><tt:Encoding>H264</tt:Encoding>
      <tt:Resolution><tt:Width>704</tt:Width><tt:Height>576</tt:Height></tt:Resolution>
      <tt:RateControl><tt:FrameRateLimit>12</tt:FrameRateLimit></tt:RateControl>
    </tt:VideoEncoderConfiguration></trt:Profiles>
  <trt:Profiles token="tiny"><tt:Name>Mobile</tt:Name>
    <tt:VideoEncoderConfiguration><tt:Encoding>H264</tt:Encoding>
      <tt:Resolution><tt:Width>352</tt:Width><tt:Height>288</tt:Height></tt:Resolution>
    </tt:VideoEncoderConfiguration></trt:Profiles>
 </trt:GetProfilesResponse></s:Body></s:Envelope>"""

STREAM_URI = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
            xmlns:tt="http://www.onvif.org/ver10/schema">
 <s:Body><trt:GetStreamUriResponse><trt:MediaUri>
   <tt:Uri>rtsp://192.168.9.9:554/Streaming/Channels/102</tt:Uri>
 </trt:MediaUri></trt:GetStreamUriResponse></s:Body></s:Envelope>"""

FAULT = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
 <s:Body><s:Fault><s:Reason><s:Text>Sender not authorized</s:Text></s:Reason>
 </s:Fault></s:Body></s:Envelope>"""


class FakeResponse:
    def __init__(self, body: str, status: int = 200):
        self.content = body.encode()
        self.status_code = status


def replies(*bodies):
    """Answer each POST in order, recording what was sent."""
    sent = []
    remaining = list(bodies)

    def post(url, data=None, headers=None, timeout=None):
        sent.append({"url": url, "body": data.decode(), "timeout": timeout})
        return remaining.pop(0) if remaining else FakeResponse(FAULT)

    post.sent = sent
    return post


class DiscoveryTests(TestCase):
    def test_it_finds_the_rtsp_url_from_an_address_and_a_password(self):
        post = replies(FakeResponse(CAPABILITIES), FakeResponse(PROFILES), FakeResponse(STREAM_URI))
        with mock.patch("apps.cameras.onvif.requests.post", post):
            url, profile = onvif.discover_stream_url("192.168.1.50", 80, "admin", "secret")

        self.assertEqual(url, "rtsp://192.168.1.50:554/Streaming/Channels/102")
        self.assertEqual(profile.token, "sub")
        self.assertEqual(post.sent[0]["url"], "http://192.168.1.50/onvif/device_service")

    def test_it_prefers_the_cheapest_stream_still_worth_analysing(self):
        """A 4K main stream costs CPU to decode for detail analysis discards."""
        post = replies(FakeResponse(PROFILES))
        with mock.patch("apps.cameras.onvif.requests.post", post):
            found = onvif.get_profiles("http://cam/onvif/media_service")

        self.assertEqual(onvif.choose_profile(found, min_width=640).token, "sub")
        # When nothing meets the floor, the largest available wins instead.
        self.assertEqual(onvif.choose_profile(found, min_width=5000).token, "main")

    def test_it_points_the_stream_back_at_the_address_we_can_reach(self):
        """Cameras report the IP they think they have, which NAT makes wrong."""
        self.assertEqual(
            onvif._rehost("rtsp://10.0.0.5:554/live", "203.0.113.7"),
            "rtsp://203.0.113.7:554/live",
        )
        self.assertEqual(
            onvif._rehost("rtsp://10.0.0.5/live", "203.0.113.7"),
            "rtsp://203.0.113.7/live",
        )

    def test_it_authenticates_with_a_password_digest_not_the_password(self):
        post = replies(FakeResponse(CAPABILITIES), FakeResponse(PROFILES), FakeResponse(STREAM_URI))
        with mock.patch("apps.cameras.onvif.requests.post", post):
            onvif.discover_stream_url("192.168.1.50", 80, "admin", "hunter2")

        body = post.sent[0]["body"]
        self.assertNotIn("hunter2", body)
        self.assertIn("PasswordDigest", body)

        nonce = base64.b64decode(_between(body, "<wsse:Nonce", "</wsse:Nonce>").split(">", 1)[1])
        created = _between(body, "<wsu:Created>", "</wsu:Created>")
        expected = base64.b64encode(
            hashlib.sha1(nonce + created.encode() + b"hunter2").digest()
        ).decode()
        self.assertIn(expected, body)

    def test_a_rejected_password_says_so_in_words_an_installer_can_act_on(self):
        post = replies(FakeResponse(FAULT), FakeResponse(FAULT))
        with mock.patch("apps.cameras.onvif.requests.post", post):
            with self.assertRaises(onvif.OnvifError) as caught:
                onvif.discover_stream_url("192.168.1.50", 80, "admin", "wrong")
        self.assertIn("Sender not authorized", str(caught.exception))

    def test_an_unreachable_camera_is_reported_not_raised_as_a_transport_error(self):
        import requests as requests_module

        def boom(*args, **kwargs):
            raise requests_module.ConnectTimeout("timed out")

        with mock.patch("apps.cameras.onvif.requests.post", boom):
            with self.assertRaises(onvif.OnvifError):
                onvif.discover_stream_url("192.168.1.50")

    def test_credentials_are_added_for_rtsp_but_never_duplicated(self):
        self.assertEqual(
            onvif.with_credentials("rtsp://cam/live", "admin", "pw"),
            "rtsp://admin:pw@cam/live",
        )
        self.assertEqual(
            onvif.with_credentials("rtsp://someone:else@cam/live", "admin", "pw"),
            "rtsp://someone:else@cam/live",
        )

    def test_passwords_never_survive_into_a_status_message(self):
        self.assertEqual(
            onvif.redact("rtsp://admin:hunter2@cam/live"), "rtsp://admin:•••@cam/live"
        )

    def test_the_device_endpoint_follows_the_configured_port(self):
        self.assertEqual(onvif.device_service_url("10.0.0.4"), "http://10.0.0.4/onvif/device_service")
        self.assertEqual(
            onvif.device_service_url("10.0.0.4", 8000), "http://10.0.0.4:8000/onvif/device_service"
        )
        self.assertEqual(
            onvif.device_service_url("10.0.0.4", 443), "https://10.0.0.4/onvif/device_service"
        )


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


class CameraIntegrationTests(TestCase):
    """The camera row, not just the protocol: discovery, caching and fallback."""

    def setUp(self):
        from apps.cameras.models import Camera, Site
        from tests.test_platform import full_workspace

        self.organization, _ = full_workspace("Onvif Co")
        site = Site.objects.create(organization=self.organization, name="Plant")
        self.camera = Camera.objects.create(
            organization=self.organization, site=site, name="Gate",
            protocol=Camera.Protocol.ONVIF, onvif_host="192.168.1.50", onvif_port=80,
            username="admin", password="secret",
        )

    def test_discovery_caches_the_url_so_a_restart_need_not_ask_again(self):
        from apps.cameras.services import resolve_onvif_url

        post = replies(FakeResponse(CAPABILITIES), FakeResponse(PROFILES), FakeResponse(STREAM_URI))
        with mock.patch("apps.cameras.onvif.requests.post", post):
            url = resolve_onvif_url(self.camera)

        self.assertEqual(url, "rtsp://admin:secret@192.168.1.50:554/Streaming/Channels/102")
        self.camera.refresh_from_db()
        self.assertEqual(self.camera.stream_url, "rtsp://192.168.1.50:554/Streaming/Channels/102")
        self.assertIn("SubStream", self.camera.status_detail)

    def test_a_camera_that_stops_answering_onvif_falls_back_to_its_known_url(self):
        from apps.cameras.services import resolve_onvif_url

        self.camera.stream_url = "rtsp://192.168.1.50:554/Streaming/Channels/102"
        self.camera.save(update_fields=["stream_url"])

        def boom(*args, **kwargs):
            import requests as requests_module
            raise requests_module.ConnectTimeout("no route")

        with mock.patch("apps.cameras.onvif.requests.post", boom):
            url = resolve_onvif_url(self.camera)
        self.assertEqual(url, "rtsp://admin:secret@192.168.1.50:554/Streaming/Channels/102")

    def test_a_camera_that_never_worked_reports_why_instead_of_crashing_the_worker(self):
        from apps.cameras.services import UnavailableSource, build_source

        def boom(*args, **kwargs):
            import requests as requests_module
            raise requests_module.ConnectTimeout("no route")

        with mock.patch("apps.cameras.onvif.requests.post", boom):
            source = build_source(self.camera)

        self.assertIsInstance(source, UnavailableSource)
        self.assertFalse(source.open())
        self.assertIn("ONVIF discovery failed", source.description)

    def test_the_connection_test_hands_that_reason_to_the_operator(self):
        from apps.cameras.services import check_connection

        def boom(*args, **kwargs):
            import requests as requests_module
            raise requests_module.ConnectTimeout("no route")

        with mock.patch("apps.cameras.onvif.requests.post", boom):
            ok, message = check_connection(self.camera)

        self.assertFalse(ok)
        self.assertIn("ONVIF discovery failed", message)
