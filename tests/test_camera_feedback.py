"""When a camera shows no picture, the page must say why.

"No frame available — check the connection" sent operators to inspect
firewalls when the real cause was that the video decoder was not installed at
all: OpenCV sat in the production requirements while the documented quick start
installs the core file, so every camera except the built-in simulator failed
silently, and Test Connection blamed the network.
"""
from __future__ import annotations

import builtins
import sys
from unittest import mock

from django.test import TestCase
from django.urls import reverse

from apps.cameras.models import Camera, Site
from apps.cameras.services import (
    UnavailableSource, build_source, capture_preview, check_connection, decoder_error,
)
from tests.test_platform import full_workspace


def without_opencv():
    """Pretend this deployment followed the documented install."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    modules = {k: v for k, v in sys.modules.items() if k != "cv2"}
    return mock.patch.dict(sys.modules, modules, clear=True), mock.patch(
        "builtins.__import__", fake_import
    )


class DecoderTests(TestCase):
    def setUp(self):
        self.organization, self.owner = full_workspace("Feedback Co")
        self.site = Site.objects.create(organization=self.organization, name="Site")
        self.rtsp = Camera.objects.create(
            organization=self.organization, site=self.site, name="Gate",
            protocol=Camera.Protocol.RTSP, stream_url="rtsp://10.0.0.9:554/live",
        )
        self.demo = Camera.objects.create(
            organization=self.organization, site=self.site, name="Demo",
            protocol=Camera.Protocol.DEMO,
        )

    def test_the_decoder_is_installed(self):
        """It is a hard requirement: without it only the simulator works."""
        self.assertIsNone(decoder_error())

    def test_a_missing_decoder_is_named_rather_than_blamed_on_the_network(self):
        sys_patch, import_patch = without_opencv()
        with sys_patch, import_patch:
            source = build_source(self.rtsp)
            self.assertIsInstance(source, UnavailableSource)
            self.assertFalse(source.open())
            self.assertIn("opencv-python-headless", source.reason)

            ok, message = check_connection(self.rtsp)
            self.assertFalse(ok)
            self.assertIn("opencv-python-headless", message)
            self.assertNotIn("network route", message)

    def test_the_simulator_still_runs_without_a_decoder(self):
        """The demo path must not depend on it — that is the point of it."""
        sys_patch, import_patch = without_opencv()
        with sys_patch, import_patch:
            self.assertIsNotNone(capture_preview(self.demo))

    def test_a_camera_with_no_url_says_so_instead_of_failing_to_connect(self):
        blank = Camera.objects.create(organization=self.organization, site=self.site,
                                      name="Unconfigured", protocol=Camera.Protocol.RTSP)
        source = build_source(blank)
        self.assertIsInstance(source, UnavailableSource)
        self.assertIn("No stream URL", source.reason)


class ConnectionHintTests(TestCase):
    def setUp(self):
        self.organization, self.owner = full_workspace("Hint Co")
        self.site = Site.objects.create(organization=self.organization, name="Site")

    def camera(self, **kwargs):
        return Camera.objects.create(organization=self.organization, site=self.site,
                                     name=kwargs.pop("name", "Cam"), **kwargs)

    def test_a_demo_camera_needs_no_explanation(self):
        self.assertEqual(self.camera(protocol=Camera.Protocol.DEMO).connection_hint, "")

    def test_a_missing_url_is_the_explanation(self):
        hint = self.camera(protocol=Camera.Protocol.RTSP).connection_hint
        self.assertIn("No stream URL", hint)

    def test_an_onvif_camera_with_no_address_is_the_explanation(self):
        hint = self.camera(protocol=Camera.Protocol.ONVIF).connection_hint
        self.assertIn("ONVIF address", hint)

    def test_whatever_the_worker_last_reported_is_used(self):
        camera = self.camera(protocol=Camera.Protocol.RTSP, stream_url="rtsp://x/y",
                             status=Camera.Status.OFFLINE,
                             status_detail="Stream stopped delivering frames")
        self.assertEqual(camera.connection_hint, "Stream stopped delivering frames")

    def test_a_camera_nobody_has_connected_yet_says_that(self):
        camera = self.camera(protocol=Camera.Protocol.RTSP, stream_url="rtsp://x/y",
                             status=Camera.Status.PENDING)
        self.assertIn("Test connection", camera.connection_hint)

    def test_the_decoder_hint_outranks_everything_else(self):
        camera = self.camera(protocol=Camera.Protocol.RTSP, stream_url="rtsp://x/y",
                             status_detail="something else")
        sys_patch, import_patch = without_opencv()
        with sys_patch, import_patch:
            self.assertIn("opencv-python-headless", camera.connection_hint)


class LiveViewHeaderTests(TestCase):
    """The header claimed a green "live" dot and an fps for every camera."""

    def setUp(self):
        self.organization, self.owner = full_workspace("Header Co")
        self.site = Site.objects.create(organization=self.organization, name="Site")
        self.client.force_login(self.owner)

    def page(self, camera):
        return self.client.get(reverse("dashboard:camera_detail",
                                       args=[camera.uid])).content.decode()

    def test_a_camera_that_has_never_connected_is_not_shown_as_live(self):
        camera = Camera.objects.create(
            organization=self.organization, site=self.site, name="Never",
            protocol=Camera.Protocol.RTSP, status=Camera.Status.PENDING, target_fps=6.0)

        html = self.page(camera)

        self.assertNotIn("dot-live", html)
        self.assertIn("Pending setup", html)
        self.assertIn("fps when running", html)

    def test_an_offline_camera_explains_itself_in_the_frame_area(self):
        camera = Camera.objects.create(
            organization=self.organization, site=self.site, name="Down",
            protocol=Camera.Protocol.RTSP, stream_url="rtsp://x/y",
            status=Camera.Status.OFFLINE, status_detail="Stream stopped delivering frames")

        html = self.page(camera)

        self.assertIn("Stream stopped delivering frames", html)
        self.assertNotIn("check the connection", html)

    def test_a_healthy_camera_reports_the_rate_it_actually_achieved(self):
        from django.utils import timezone

        camera = Camera.objects.create(
            organization=self.organization, site=self.site, name="Working",
            protocol=Camera.Protocol.DEMO, status=Camera.Status.ONLINE,
            last_frame_at=timezone.now(), target_fps=6.0, health={"fps": 5.4})

        html = self.page(camera)

        self.assertIn("dot-live", html)
        self.assertIn("5.4 fps analysed", html)

    def test_a_healthy_camera_that_has_not_reported_a_rate_yet(self):
        from django.utils import timezone

        camera = Camera.objects.create(
            organization=self.organization, site=self.site, name="Fresh",
            protocol=Camera.Protocol.DEMO, status=Camera.Status.ONLINE,
            last_frame_at=timezone.now(), health={})

        html = self.page(camera)

        self.assertIn("dot-live", html)
        self.assertNotIn("fps when running", html)

    def test_the_observed_rate_survives_junk_in_the_health_column(self):
        camera = Camera.objects.create(
            organization=self.organization, site=self.site, name="Odd",
            protocol=Camera.Protocol.DEMO, health={"fps": "not a number"})
        self.assertIsNone(camera.observed_fps)
        camera.health = {"fps": 0}
        self.assertIsNone(camera.observed_fps)


class PreviewEndpointTests(TestCase):
    def setUp(self):
        self.organization, self.owner = full_workspace("Preview Co")
        self.site = Site.objects.create(organization=self.organization, name="Site")
        self.client.force_login(self.owner)

    def test_a_demo_camera_serves_a_real_jpeg(self):
        camera = Camera.objects.create(organization=self.organization, site=self.site,
                                       name="Demo", protocol=Camera.Protocol.DEMO)

        response = self.client.get(reverse("dashboard:camera_preview", args=[camera.uid]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "image/jpeg")
        self.assertTrue(response.content.startswith(b"\xff\xd8"), "not a JPEG")

    def test_an_unreachable_camera_returns_no_content_rather_than_an_error(self):
        camera = Camera.objects.create(organization=self.organization, site=self.site,
                                       name="Broken", protocol=Camera.Protocol.RTSP)

        response = self.client.get(reverse("dashboard:camera_preview", args=[camera.uid]))

        self.assertEqual(response.status_code, 204)

    def test_one_workspace_cannot_read_anothers_camera(self):
        other, _ = full_workspace("Somebody Else")
        their_site = Site.objects.create(organization=other, name="Theirs")
        theirs = Camera.objects.create(organization=other, site=their_site,
                                       name="Theirs", protocol=Camera.Protocol.DEMO)

        response = self.client.get(reverse("dashboard:camera_preview", args=[theirs.uid]))

        self.assertEqual(response.status_code, 404)
