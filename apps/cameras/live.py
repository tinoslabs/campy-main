"""Keep camera streams open, so the live view is live.

The preview endpoint used to build a source, open the stream, read a frame and
tear the whole thing down — for every single image. On a LAN MJPEG camera that
is wasteful; on RTSP it is seconds, because opening a session means OPTIONS,
DESCRIBE, SETUP, PLAY and then waiting for a keyframe, and on ONVIF it re-ran
SOAP discovery first. The browser then asked again four seconds later.

So one reader thread per camera holds the stream open and keeps the newest
frame, already JPEG-encoded, in a slot that any number of viewers can read
without touching the camera. Readers start when somebody watches and stop
themselves when nobody has for a while, so an unwatched camera holds no
session.
"""
from __future__ import annotations

import io
import logging
import threading
import time

import numpy as np
from django.conf import settings

logger = logging.getLogger("campy.live")

#: Stop reading a camera nobody has looked at for this long.
IDLE_SHUTDOWN_SECONDS = 20.0
#: Give up on a stream that stops producing, and let the next viewer retry.
STALL_SECONDS = 15.0
#: Most cameras any one process will hold open at once.
MAX_READERS = 24
#: Encoded once per frame, however many people are watching.
JPEG_QUALITY = 78


class Frame:
    __slots__ = ("jpeg", "width", "height", "captured_at", "index")

    def __init__(self, jpeg: bytes, width: int, height: int, index: int):
        self.jpeg = jpeg
        self.width = width
        self.height = height
        self.captured_at = time.time()
        self.index = index

    @property
    def age(self) -> float:
        return time.time() - self.captured_at


def encode(frame: np.ndarray) -> bytes | None:
    from PIL import Image

    try:
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
            buffer, format="JPEG", quality=JPEG_QUALITY
        )
        return buffer.getvalue()
    except Exception:  # noqa: BLE001 - a bad frame must not kill the reader
        logger.exception("Could not encode a frame")
        return None


class FrameSlot:
    """One camera's newest frame, and a way to wait for the next."""

    def __init__(self):
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._closed = threading.Event()
        self._index = 0
        self._last_wanted = time.time()

    def touch(self) -> None:
        """Somebody is watching; keep producing."""
        self._last_wanted = time.time()

    @property
    def wanted_recently(self) -> bool:
        return time.time() - self._last_wanted <= IDLE_SHUTDOWN_SECONDS

    def publish(self, jpeg: bytes, width: int, height: int) -> None:
        with self._new_frame:
            self._index += 1
            self._frame = Frame(jpeg, width, height, self._index)
            self._new_frame.notify_all()

    def close(self) -> None:
        self._closed.set()
        with self._new_frame:
            self._new_frame.notify_all()

    def latest(self) -> Frame | None:
        self.touch()
        with self._lock:
            return self._frame

    def wait_for_frame(self, after: int, timeout: float) -> Frame | None:
        """Block until a frame newer than *after* arrives, or the timeout ends."""
        self.touch()
        deadline = time.time() + timeout
        with self._new_frame:
            while not self._closed.is_set():
                if self._frame is not None and self._frame.index > after:
                    return self._frame
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._new_frame.wait(remaining)
        return None


class PushedReader(FrameSlot):
    """Frames handed over by the analytics worker, which already decoded them.

    A camera that is being analysed is already open somewhere; opening it a
    second time to look at it wastes a session the camera may not have — plenty
    of them allow only one or two — and shows a different moment than the one
    the detectors are working on.
    """

    def __init__(self, camera):
        super().__init__()
        self.name = camera.name
        self.error = ""

    @property
    def alive(self) -> bool:
        frame = self._frame
        return frame is not None and frame.age <= STALL_SECONDS

    def start(self) -> None:          # nothing to start: the worker feeds it
        return None

    def stop(self) -> None:
        self.close()


class CameraReader:
    """Holds one camera's stream open and publishes its newest frame."""

    def __init__(self, camera):
        self.camera_id = camera.pk
        self.camera_uid = str(camera.uid)
        self.name = camera.name
        self.target_fps = max(float(camera.target_fps or 6), 1.0)
        self._camera = camera
        self._slot = FrameSlot()
        self._error: str = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"live-{camera.pk}", daemon=True
        )

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._slot.close()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive() and not self._stop.is_set()

    def touch(self) -> None:
        self._slot.touch()

    # -- reading -----------------------------------------------------------
    def _run(self) -> None:
        from apps.cameras.services import UnavailableSource, build_source

        source = build_source(self._camera)
        if not source.open():
            reason = getattr(source, "reason", "") if isinstance(source, UnavailableSource) else ""
            self._fail(reason or "Could not open the stream.")
            return

        interval = 1.0 / self.target_fps
        empty_since: float | None = None
        try:
            while not self._stop.is_set():
                if not self._slot.wanted_recently:
                    logger.debug("Closing %s — nobody watching", self.name)
                    break

                started = time.perf_counter()
                frame = source.read()
                if frame is None:
                    empty_since = empty_since or time.time()
                    if time.time() - empty_since > STALL_SECONDS:
                        self._fail("The stream stopped sending frames.")
                        return
                    time.sleep(0.2)
                    continue

                empty_since = None
                jpeg = encode(frame)
                if jpeg is not None:
                    self._slot.publish(jpeg, frame.shape[1], frame.shape[0])

                # Pace to the camera's analysis rate rather than spinning.
                remaining = interval - (time.perf_counter() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
        except Exception:  # noqa: BLE001 - a reader must never take the server with it
            logger.exception("Live reader for %s stopped", self.name)
            self._fail("The stream failed while running.")
        finally:
            source.close()
            self._stop.set()
            self._slot.close()

    def _fail(self, reason: str) -> None:
        self._error = reason
        self._stop.set()
        self._slot.close()

    # -- reading out -------------------------------------------------------
    @property
    def error(self) -> str:
        return self._error

    def latest(self) -> Frame | None:
        return self._slot.latest()

    def wait_for_frame(self, after: int, timeout: float) -> Frame | None:
        return self._slot.wait_for_frame(after, timeout)


class Broker:
    """Who is producing frames for which camera, and who wants them.

    Two kinds of producer, and they are not interchangeable. A worker
    analysing a camera already has it open and hands its frames over; that is
    always preferred, because opening the camera twice wastes a session it may
    not have. Otherwise a reader opens the stream itself. Interest is tracked
    separately from either, so a worker can know whether encoding is worth the
    cost before anything is producing.
    """

    def __init__(self):
        # Keyed by UUID, not primary key: an integer can be reused by a later
        # row, and serving one camera's frames for another is a leak across
        # workspaces, not merely a stale picture.
        self._readers: dict[str, CameraReader] = {}
        self._pushed: dict[str, PushedReader] = {}
        self._interest: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key(camera) -> str:
        return str(camera.uid)

    # -- interest ----------------------------------------------------------
    def watch(self, camera) -> None:
        """Record that somebody is looking, without demanding a frame yet."""
        with self._lock:
            self._interest[self.key(camera)] = time.time()

    def wants_frames(self, camera) -> bool:
        """Is anybody watching? The worker asks before paying to encode."""
        with self._lock:
            last = self._interest.get(self.key(camera), 0.0)
        return time.time() - last <= IDLE_SHUTDOWN_SECONDS

    # -- producers ---------------------------------------------------------
    def publish(self, camera, frame: np.ndarray) -> None:
        """Take a frame the worker has already decoded."""
        jpeg = encode(frame)
        if jpeg is None:
            return
        with self._lock:
            key = self.key(camera)
            slot = self._pushed.get(key)
            if slot is None:
                slot = PushedReader(camera)
                self._pushed[key] = slot
            # A worker feeding this camera makes our own reader redundant.
            reader = self._readers.pop(key, None)
            if reader is not None:
                reader.stop()
        slot.publish(jpeg, frame.shape[1], frame.shape[0])

    def reader_for(self, camera):
        """Whatever can supply this camera's frames, starting one if needed."""
        with self._lock:
            self._interest[self.key(camera)] = time.time()

            key = self.key(camera)
            pushed = self._pushed.get(key)
            if pushed is not None:
                if pushed.alive:
                    pushed.touch()
                    return pushed
                self._pushed.pop(key, None)         # the worker has stopped

            reader = self._readers.get(key)
            if reader is not None and reader.alive:
                reader.touch()
                return reader
            if reader is not None:
                self._readers.pop(key, None)
                if reader.error:
                    logger.debug("Reader for %s ended: %s", camera.name, reader.error)

            self._reap()
            if len(self._readers) >= MAX_READERS:
                logger.warning("Not starting a reader for %s — %d already open",
                               camera.name, len(self._readers))
                return None

            reader = CameraReader(camera)
            self._readers[key] = reader

        reader.start()
        return reader

    def _reap(self) -> None:
        for key, reader in list(self._readers.items()):
            if not reader.alive:
                self._readers.pop(key, None)
        for key, slot in list(self._pushed.items()):
            if not slot.alive:
                self._pushed.pop(key, None)

    def stop_all(self) -> None:
        with self._lock:
            for reader in self._readers.values():
                reader.stop()
            for slot in self._pushed.values():
                slot.stop()
            self._readers.clear()
            self._pushed.clear()
            self._interest.clear()

    @property
    def open_readers(self) -> int:
        with self._lock:
            self._reap()
            return len(self._readers) + len(self._pushed)


broker = Broker()


def first_frame(camera, timeout: float | None = None) -> Frame | None:
    """The newest frame for a camera, starting a reader if there is not one.

    Waits only for the stream to produce its first frame; after that the slot
    is already full and every caller is served from memory.
    """
    if timeout is None:
        timeout = float(settings.CAMPY.get("LIVE_FIRST_FRAME_TIMEOUT", 10.0))
    reader = broker.reader_for(camera)
    if reader is None:
        return None
    frame = reader.latest()
    if frame is not None:
        return frame
    return reader.wait_for_frame(after=0, timeout=timeout)
