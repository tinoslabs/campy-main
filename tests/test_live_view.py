"""The live view, and the cost of watching a camera.

Every preview image used to build a source, open the stream, read a frame and
tear it down — then the browser asked again four seconds later. On RTSP that
handshake is seconds and on ONVIF it re-ran SOAP discovery first, so the "live"
view was a slideshow of stale pictures and the camera paid for a session per
image. One reader now holds the stream open and every viewer reads its newest
frame.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from django.test import TestCase
from django.urls import reverse

from apps.cameras import live
from apps.cameras.models import Camera, Site
from apps.cameras.services import FrameSource
from tests.test_platform import full_workspace


class CountingSource(FrameSource):
    """A source that records how often it is opened."""

    opens = 0
    closes = 0
    lock = threading.Lock()

    def __init__(self, fail=False):
        self.fail = fail
        self.index = 0

    @classmethod
    def reset(cls):
        with cls.lock:
            cls.opens = cls.closes = 0

    def open(self):
        with CountingSource.lock:
            CountingSource.opens += 1
        return not self.fail

    def read(self):
        self.index += 1
        time.sleep(0.01)
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = self.index % 255      # a different picture each time
        return frame

    def close(self):
        with CountingSource.lock:
            CountingSource.closes += 1

    @property
    def description(self):
        return "CountingSource"


class LiveViewTestCase(TestCase):
    def setUp(self):
        self.organization, self.owner = full_workspace("Live Co")
        self.site = Site.objects.create(organization=self.organization, name="Site")
        self.camera = Camera.objects.create(
            organization=self.organization, site=self.site, name="Cam",
            protocol=Camera.Protocol.DEMO, status=Camera.Status.ONLINE, target_fps=30)
        CountingSource.reset()
        live.broker.stop_all()
        self.addCleanup(live.broker.stop_all)

    def use_counting_source(self, fail=False):
        patcher = self.settings()          # placeholder to keep addCleanup tidy
        del patcher
        from unittest import mock

        patch = mock.patch("apps.cameras.services.build_source",
                           side_effect=lambda camera: CountingSource(fail=fail))
        patch.start()
        self.addCleanup(patch.stop)


class ReaderTests(LiveViewTestCase):
    def test_one_reader_serves_many_viewers(self):
        """Twenty people watching must not be twenty sessions on the camera."""
        self.use_counting_source()

        frames = [live.first_frame(self.camera, timeout=5) for _ in range(20)]

        self.assertTrue(all(f is not None for f in frames))
        self.assertEqual(CountingSource.opens, 1, "the camera was opened more than once")

    def test_a_second_request_is_served_from_memory(self):
        self.use_counting_source()
        live.first_frame(self.camera, timeout=5)

        started = time.perf_counter()
        frame = live.first_frame(self.camera, timeout=5)
        elapsed = (time.perf_counter() - started) * 1000

        self.assertIsNotNone(frame)
        self.assertLess(elapsed, 50, "a cached frame should not cost milliseconds")

    def test_the_picture_keeps_moving_without_being_asked(self):
        self.use_counting_source()
        first = live.first_frame(self.camera, timeout=5)
        time.sleep(0.3)
        later = live.first_frame(self.camera, timeout=5)

        self.assertGreater(later.index, first.index,
                           "the reader should keep producing between requests")

    def test_a_reader_stops_when_nobody_is_watching(self):
        self.use_counting_source()
        original = live.IDLE_SHUTDOWN_SECONDS
        live.IDLE_SHUTDOWN_SECONDS = 0.2
        self.addCleanup(setattr, live, "IDLE_SHUTDOWN_SECONDS", original)

        live.first_frame(self.camera, timeout=5)
        deadline = time.time() + 5
        while live.broker.open_readers and time.time() < deadline:
            time.sleep(0.1)

        self.assertEqual(live.broker.open_readers, 0, "an unwatched camera was left open")
        self.assertEqual(CountingSource.closes, 1, "the stream was not closed")

    def test_a_stream_that_will_not_open_yields_no_frame(self):
        self.use_counting_source(fail=True)
        self.assertIsNone(live.first_frame(self.camera, timeout=2))

    def test_the_number_of_open_readers_is_capped(self):
        self.use_counting_source()
        original = live.MAX_READERS
        live.MAX_READERS = 2
        self.addCleanup(setattr, live, "MAX_READERS", original)

        cameras = [Camera.objects.create(
            organization=self.organization, site=self.site, name=f"Cam {i}",
            protocol=Camera.Protocol.DEMO, status=Camera.Status.ONLINE) for i in range(4)]
        served = [live.first_frame(c, timeout=3) is not None for c in cameras]

        self.assertLessEqual(live.broker.open_readers, 2)
        self.assertTrue(any(served), "the cap should not stop every camera")


class WorkerSharingTests(LiveViewTestCase):
    """A camera being analysed is already open; watching it must be free."""

    def test_the_worker_feeds_viewers_without_a_second_session(self):
        self.use_counting_source()
        live.broker.watch(self.camera)
        self.assertTrue(live.broker.wants_frames(self.camera))

        frame = np.full((36, 64, 3), 120, dtype=np.uint8)
        live.broker.publish(self.camera, frame)

        served = live.first_frame(self.camera, timeout=3)
        self.assertIsNotNone(served)
        self.assertEqual(CountingSource.opens, 0,
                         "the camera was opened even though the worker had it")

    def test_nobody_watching_means_nothing_to_encode(self):
        """The worker asks before paying for a JPEG."""
        self.assertFalse(live.broker.wants_frames(self.camera))

    def test_stale_worker_frames_fall_back_to_opening_the_stream(self):
        """If the worker stops, the live view must recover on its own."""
        self.use_counting_source()
        live.broker.publish(self.camera, np.zeros((36, 64, 3), dtype=np.uint8))

        original = live.STALL_SECONDS
        live.STALL_SECONDS = 0.05
        self.addCleanup(setattr, live, "STALL_SECONDS", original)
        time.sleep(0.1)

        self.assertIsNotNone(live.first_frame(self.camera, timeout=5))
        self.assertEqual(CountingSource.opens, 1)


class IdentityTests(LiveViewTestCase):
    """Which camera a frame belongs to must not depend on a reusable integer."""

    def test_a_reused_primary_key_cannot_serve_the_wrong_camera(self):
        """SQLite hands the next row the highest deleted id, so this is reachable.

        Serving a cached frame for whoever holds that id now is one workspace
        seeing another's camera, not merely a stale picture.
        """
        self.use_counting_source()
        live.first_frame(self.camera, timeout=5)
        self.assertEqual(live.broker.open_readers, 1)

        stale_pk = self.camera.pk
        self.camera.delete()

        other, _ = full_workspace("Another Tenant")
        their_site = Site.objects.create(organization=other, name="Theirs")
        replacement = Camera.objects.create(
            organization=other, site=their_site, name="Theirs",
            protocol=Camera.Protocol.DEMO, status=Camera.Status.ONLINE)
        replacement.pk = stale_pk          # the id the deleted camera had

        self.assertNotEqual(live.Broker.key(replacement), str(self.camera.uid),
                            "the broker must key on something that is never reused")


class EndpointTests(LiveViewTestCase):
    def setUp(self):
        super().setUp()
        self.client.force_login(self.owner)

    def test_the_preview_returns_a_jpeg_and_says_how_old_it_is(self):
        self.use_counting_source()
        response = self.client.get(
            reverse("dashboard:camera_preview", args=[self.camera.uid]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "image/jpeg")
        self.assertTrue(response.content.startswith(b"\xff\xd8"))
        self.assertLess(float(response["X-Frame-Age"]), 5.0)

    def test_the_stream_delivers_several_frames_in_one_response(self):
        self.use_counting_source()
        original = live.settings.CAMPY["LIVE_STREAM_SECONDS"]
        live.settings.CAMPY["LIVE_STREAM_SECONDS"] = 1.0
        self.addCleanup(live.settings.CAMPY.__setitem__, "LIVE_STREAM_SECONDS", original)

        response = self.client.get(
            reverse("dashboard:camera_stream", args=[self.camera.uid]))

        self.assertEqual(response.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", response["Content-Type"])
        body = b"".join(response.streaming_content)
        self.assertGreater(body.count(b"--campyframe"), 2,
                           "a stream should carry more than one frame")
        self.assertIn(b"\xff\xd8", body)

    def test_a_camera_that_cannot_be_read_returns_no_content(self):
        self.use_counting_source(fail=True)
        response = self.client.get(
            reverse("dashboard:camera_preview", args=[self.camera.uid]))
        self.assertEqual(response.status_code, 204)

    def test_one_workspace_cannot_stream_anothers_camera(self):
        other, _ = full_workspace("Somebody Else")
        their_site = Site.objects.create(organization=other, name="Theirs")
        theirs = Camera.objects.create(organization=other, site=their_site, name="Theirs",
                                       protocol=Camera.Protocol.DEMO)

        for route in ("dashboard:camera_preview", "dashboard:camera_stream"):
            with self.subTest(route=route):
                response = self.client.get(reverse(route, args=[theirs.uid]))
                self.assertEqual(response.status_code, 404)
