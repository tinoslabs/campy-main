"""Naming the failure a decoder reports only as False.

The log that prompted this read "method OPTIONS failed: 401 Unauthorized" in
FFmpeg's own output while the page said "No frame available". The probe asks
the camera directly so the answer reaches the person who can act on it.
"""
from __future__ import annotations

import base64
import hashlib
import socket
import threading

from django.test import TestCase

from apps.cameras.rtsp_probe import probe, probe_http

REALM, NONCE = "IP Camera(C2201)", "4e4f4e43453a31323334"


class FakeCamera:
    """A camera that challenges the way Hikvision and Dahua do."""

    def __init__(self, scheme="digest", username="admin", password="Tinos@1122",
                 final_status=200):
        self.scheme, self.username, self.password = scheme, username, password
        self.final_status = final_status
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.sock.settimeout(10)
        self.port = self.sock.getsockname()[1]
        self.received = []
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def expected_digest(self, uri):
        ha1 = hashlib.md5(f"{self.username}:{REALM}:{self.password}".encode()).hexdigest()
        ha2 = hashlib.md5(f"OPTIONS:{uri}".encode()).hexdigest()
        return hashlib.md5(f"{ha1}:{NONCE}:{ha2}".encode()).hexdigest()

    def serve(self):
        try:
            conn, _ = self.sock.accept()
            conn.settimeout(6)
            for _ in range(3):
                data = conn.recv(2048).decode("latin-1", "replace")
                if not data:
                    break
                self.received.append(data)
                seq = data.split("CSeq:")[1].split("\r\n")[0].strip()
                uri = data.split(" ", 2)[1]
                if "Authorization:" in data and self.accepts(data, uri):
                    conn.sendall(f"RTSP/1.0 {self.final_status} OK\r\nCSeq: {seq}\r\n\r\n".encode())
                    break
                header = (f'Digest realm="{REALM}", nonce="{NONCE}"'
                          if self.scheme == "digest" else f'Basic realm="{REALM}"')
                conn.sendall(f"RTSP/1.0 401 Unauthorized\r\nCSeq: {seq}\r\n"
                             f"WWW-Authenticate: {header}\r\n\r\n".encode())
            conn.close()
        except Exception:                                  # noqa: BLE001 - test double
            pass
        finally:
            self.sock.close()

    def accepts(self, data, uri):
        header = data.split("Authorization:")[1].split("\r\n")[0].strip()
        if self.scheme == "digest":
            return self.expected_digest(uri) in header
        expected = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        return expected in header

    def url(self, path="/Streaming/Channels/501"):
        return f"rtsp://127.0.0.1:{self.port}{path}"


class ProbeTests(TestCase):
    def test_the_right_password_authenticates_over_digest(self):
        camera = FakeCamera(scheme="digest")
        result = probe(camera.url(), "admin", "Tinos@1122")
        self.assertTrue(result.reachable)
        self.assertFalse(result.unauthorized)
        self.assertIn("accepted", result.detail)

    def test_the_right_password_authenticates_over_basic(self):
        camera = FakeCamera(scheme="basic")
        result = probe(camera.url(), "admin", "Tinos@1122")
        self.assertFalse(result.unauthorized, result.detail)

    def test_a_wrong_password_is_reported_as_a_rejected_login(self):
        camera = FakeCamera(scheme="digest")
        result = probe(camera.url(), "admin", "not-the-password")
        self.assertTrue(result.reachable)
        self.assertTrue(result.unauthorized)
        self.assertIn("rejected this username and password", result.detail)

    def test_no_credentials_at_all_says_the_camera_wants_a_login(self):
        camera = FakeCamera(scheme="digest")
        result = probe(camera.url(), "", "")
        self.assertTrue(result.unauthorized)
        self.assertIn("requires a username and password", result.detail)

    def test_credentials_inside_the_url_are_used(self):
        camera = FakeCamera(scheme="digest")
        url = camera.url().replace("rtsp://", "rtsp://admin:Tinos%401122@")
        self.assertFalse(probe(url).unauthorized)

    def test_the_request_line_never_carries_the_credentials(self):
        camera = FakeCamera(scheme="digest")
        probe(camera.url().replace("rtsp://", "rtsp://admin:Tinos%401122@"))
        self.assertTrue(camera.received)
        self.assertNotIn("Tinos", camera.received[0].split("\r\n")[0])

    def test_a_missing_stream_points_at_the_channel(self):
        camera = FakeCamera(scheme="digest", final_status=404)
        result = probe(camera.url(), "admin", "Tinos@1122")
        self.assertEqual(result.status, 404)
        self.assertIn("channel", result.detail)

    def test_nothing_listening_is_a_refused_connection(self):
        spare = socket.socket()
        spare.bind(("127.0.0.1", 0))
        port = spare.getsockname()[1]
        spare.close()

        result = probe(f"rtsp://127.0.0.1:{port}/live", "admin", "x")

        self.assertFalse(result.reachable)
        self.assertIn("554", result.detail)

    def test_an_unroutable_address_times_out_with_advice(self):
        result = probe("rtsp://192.0.2.1:554/live", "admin", "x", timeout=1.0)
        self.assertFalse(result.reachable)
        self.assertIn("192.0.2.1", result.detail)

    def test_a_url_with_no_host_says_so(self):
        self.assertIn("no host", probe("rtsp://").detail)


class FailureReportingTests(TestCase):
    """What the camera page says after a run that could not read anything."""

    def setUp(self):
        from apps.cameras.models import Camera, Site
        from tests.test_platform import full_workspace

        self.organization, self.owner = full_workspace("Reporting Co")
        self.site = Site.objects.create(organization=self.organization, name="Plant")
        self.Camera = Camera

    def camera_for(self, fake, username="admin", password="Tinos@1122"):
        return self.Camera.objects.create(
            organization=self.organization, site=self.site, name="Gate",
            protocol=self.Camera.Protocol.RTSP, stream_url=fake.url(),
            username=username, password=password, enabled_analytics=["gesture"],
        )

    def test_a_rejected_login_is_reported_as_such(self):
        from apps.cameras.services import explain_stream_failure

        fake = FakeCamera(scheme="digest", password="the-real-one")
        camera = self.camera_for(fake, password="the-wrong-one")

        self.assertIn("rejected this username and password",
                      explain_stream_failure(camera))

    def test_an_unreachable_camera_names_the_address(self):
        from apps.cameras.services import explain_stream_failure

        spare = socket.socket()
        spare.bind(("127.0.0.1", 0))
        port = spare.getsockname()[1]
        spare.close()

        camera = self.Camera.objects.create(
            organization=self.organization, site=self.site, name="Gone",
            protocol=self.Camera.Protocol.RTSP,
            stream_url=f"rtsp://127.0.0.1:{port}/live", username="admin", password="x")

        self.assertIn("refused the connection", explain_stream_failure(camera))

    def test_the_reason_reaches_the_camera_page(self):
        from apps.cameras.services import explain_stream_failure

        fake = FakeCamera(scheme="digest", password="the-real-one")
        camera = self.camera_for(fake, password="the-wrong-one")
        camera.mark_offline(explain_stream_failure(camera))

        self.client.force_login(self.owner)
        html = self.client.get(camera.get_absolute_url()).content.decode()

        self.assertIn("rejected this username and password", html)
        self.assertNotIn("the-wrong-one", html, "the password must never render")

    def test_a_run_that_reads_nothing_is_reported_as_an_error(self):
        fake = FakeCamera(scheme="digest", password="the-real-one")
        camera = self.camera_for(fake, password="the-wrong-one")

        self.client.force_login(self.owner)
        response = self.client.post(
            f"/app/cameras/{camera.uid}/run/", {"frames": 2}, follow=True)

        levels = [m.level_tag for m in response.context["messages"]]
        self.assertIn("error", levels, f"expected an error, got {levels}")
        self.assertNotIn("success", levels)


class HttpProbeTests(TestCase):
    """MJPEG and snapshot cameras deserve the same answer as RTSP ones."""

    def setUp(self):
        import base64
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        expected = base64.b64encode(b"admin:Tinos@1122").decode()
        status_for_authorised = self.authorised_status

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                auth = self.headers.get("Authorization", "")
                if auth == f"Basic {expected}":
                    self.send_response(status_for_authorised)
                    self.end_headers()
                    return
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="IP Camera"')
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)

    authorised_status = 200

    def url(self, path="/video.mjpg"):
        return f"http://127.0.0.1:{self.port}{path}"

    def test_the_right_password_is_accepted(self):
        result = probe_http(self.url(), "admin", "Tinos@1122")
        self.assertFalse(result.unauthorized, result.detail)
        self.assertIn("accepted", result.detail)

    def test_a_wrong_password_is_named_as_a_rejected_login(self):
        result = probe_http(self.url(), "admin", "wrong")
        self.assertTrue(result.unauthorized)
        self.assertIn("rejected this username and password", result.detail)

    def test_no_credentials_says_the_camera_wants_a_login(self):
        result = probe_http(self.url(), "", "")
        self.assertTrue(result.unauthorized)
        self.assertIn("requires a username and password", result.detail)

    def test_credentials_inside_the_url_are_used(self):
        url = self.url().replace("http://", "http://admin:Tinos%401122@")
        self.assertFalse(probe_http(url).unauthorized)

    def test_nothing_listening_is_reported_as_unreachable(self):
        import socket

        spare = socket.socket()
        spare.bind(("127.0.0.1", 0))
        port = spare.getsockname()[1]
        spare.close()

        result = probe_http(f"http://127.0.0.1:{port}/video.mjpg", "admin", "x")

        self.assertFalse(result.reachable)
        self.assertIn("Check the address and port", result.detail)

    def test_a_url_with_no_host_says_so(self):
        self.assertIn("no host", probe_http("http://").detail)


class HttpNotFoundTests(HttpProbeTests):
    """The same camera, but nothing at that path."""

    authorised_status = 404

    def test_the_right_password_is_accepted(self):
        result = probe_http(self.url(), "admin", "Tinos@1122")
        self.assertEqual(result.status, 404)
        self.assertIn("nothing at that path", result.detail)

    def test_an_mjpeg_path_is_suggested(self):
        self.assertIn("video.mjpg", probe_http(self.url(), "admin", "Tinos@1122").detail)
