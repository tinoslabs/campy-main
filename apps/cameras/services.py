"""Stream ingestion and the per-camera analytics worker."""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone

from apps.aiengine.pipeline import build_pipeline_from_camera
from apps.aiengine.registry import ModelProvider
from apps.aiengine.simulation import SceneSimulator

logger = logging.getLogger("campy.worker")

#: Missing-decoder message, resolved once. Every camera except the built-in
#: simulator is decoded through OpenCV, so when it is absent nothing streams —
#: and the reason has to reach the operator rather than a log file. Blaming the
#: URL and the firewall, as a generic "could not connect" does, sends somebody
#: to check a network that was never the problem.
DECODER_HINT = (
    "The video decoder is not installed, so no camera can be read except the "
    "built-in simulator. Install it with: pip install opencv-python-headless"
)


def decoder_error() -> str | None:
    """Return why video cannot be decoded here, or None when it can."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return DECODER_HINT
    return None


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------
class FrameSource:
    """Yields RGB frames from a camera. Subclasses handle each protocol."""

    def open(self) -> bool:
        return True

    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    @property
    def description(self) -> str:
        return self.__class__.__name__


class OpenCVSource(FrameSource):
    """RTSP / HTTP / HLS / file input via OpenCV.

    OpenCV is used *only* to decode the stream — never for analysis. That stays
    entirely inside Campy AI's own engine.
    """

    def __init__(self, url: str, width: int, height: int, timeout_ms: int = 8000):
        self.url = url
        self.width = int(width)
        self.height = int(height)
        self.timeout_ms = timeout_ms
        self.capture = None

    def open(self) -> bool:
        if decoder_error() is not None:
            return False

        import cv2

        self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        # A small buffer keeps us near real time rather than replaying a backlog.
        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms)
            self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms)
        except Exception:  # noqa: BLE001 - not all builds expose these
            pass
        return bool(self.capture.isOpened())

    def read(self):
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        import cv2

        if self.width and self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    @property
    def description(self) -> str:
        return f"OpenCV({self.url.split('://')[0] if '://' in self.url else 'file'})"


class SimulatedSource(FrameSource):
    """Deterministic synthetic scene.

    Powers demo cameras so a prospect — or a new customer mid-installation —
    can see every analytic working before any hardware is connected.
    """

    SCENARIOS = ("walk", "idle", "crowd", "fire", "abandoned", "restricted", "running")

    def __init__(self, width: int, height: int, scenario: str = "walk", seed: int = 7):
        self.simulator = SceneSimulator(width=width, height=height, seed=seed)
        self.scenario = scenario if scenario in self.SCENARIOS else "walk"
        self.generator = None
        self.loops = 0

    def _make_generator(self):
        simulator = self.simulator
        return {
            "walk": lambda: simulator.walk_across(80),
            "idle": lambda: simulator.idle_person(300),
            "crowd": lambda: simulator.crowd(120, people=14),
            "fire": lambda: simulator.fire_outbreak(90, warmup=20),
            "abandoned": lambda: simulator.abandoned_object(200),
            "restricted": lambda: simulator.restricted_entry(90),
            "running": lambda: simulator.running_person(90),
        }[self.scenario]()

    def open(self) -> bool:
        self.generator = self._make_generator()
        return True

    def read(self):
        if self.generator is None:
            self.open()
        try:
            return next(self.generator)
        except StopIteration:
            self.loops += 1
            self.generator = self._make_generator()
            try:
                return next(self.generator)
            except StopIteration:
                return None

    @property
    def description(self) -> str:
        return f"Simulated({self.scenario})"


def resolve_onvif_url(camera, *, save: bool = True) -> str:
    """Ask an ONVIF camera for its RTSP URL, and remember the answer.

    ONVIF is the reason an installer does not need to know that this
    particular camera wants ``/Streaming/Channels/101``. The discovered URL is
    written back to ``stream_url`` so the operator can see what was found, and
    so a restart does not have to ask again.
    """
    from apps.cameras import onvif

    from .models import Camera

    try:
        url, profile = onvif.discover_stream_url(
            camera.onvif_host or camera.stream_url,
            camera.onvif_port or 80,
            camera.username,
            camera.password,
            # The floor is the analysis width itself: a 704x576 sub-stream is
            # ample for 480x270 analysis, and asking for more only buys a 4K
            # decode whose detail the first resize throws away.
            min_width=int(settings.CAMPY["WORKER_FRAME_WIDTH"]),
        )
    except onvif.OnvifError as exc:
        if camera.stream_url:
            logger.warning("ONVIF discovery failed for camera %s (%s); using the stored URL",
                           camera.pk, exc)
            return camera.connection_url
        raise

    if save and url != camera.stream_url and camera.pk:
        camera.stream_url = url
        Camera.objects.filter(pk=camera.pk).update(
            stream_url=url, status_detail=f"ONVIF profile: {profile.label}"[:300]
        )
    return onvif.with_credentials(url, camera.username, camera.password)


class UnavailableSource(FrameSource):
    """A source that could not be built — it carries the reason instead.

    The worker's contract is that ``open()`` returning False means "mark this
    camera offline with the source description". Reporting a misconfiguration
    that way puts the reason in front of the operator rather than in a
    traceback in a log nobody reads.
    """

    def __init__(self, reason: str):
        self.reason = reason

    def open(self) -> bool:
        return False

    def read(self):
        return None

    @property
    def description(self) -> str:
        return self.reason


def build_source(camera) -> FrameSource:
    """Pick the right frame source for a camera's protocol."""
    from .models import Camera

    width = settings.CAMPY["WORKER_FRAME_WIDTH"]
    height = settings.CAMPY["WORKER_FRAME_HEIGHT"]

    if camera.protocol == Camera.Protocol.DEMO:
        scenario = (camera.analytics_config or {}).get("demo_scenario", "walk")
        return SimulatedSource(width, height, scenario, seed=camera.id or 7)

    problem = decoder_error()
    if problem is not None:
        return UnavailableSource(problem)

    if camera.protocol == Camera.Protocol.ONVIF:
        from apps.cameras import onvif

        try:
            return OpenCVSource(resolve_onvif_url(camera), width, height)
        except onvif.OnvifError as exc:
            return UnavailableSource(f"ONVIF discovery failed — {exc}")

    if not camera.stream_url:
        return UnavailableSource(
            f"No stream URL is set for this camera. Add the "
            f"{camera.get_protocol_display()} address it publishes."
        )
    return OpenCVSource(camera.connection_url, width, height)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
@dataclass
class WorkerStats:
    frames: int = 0
    events: int = 0
    errors: int = 0
    started: float = 0.0
    processing_ms: float = 0.0

    @property
    def elapsed(self) -> float:
        return max(time.time() - self.started, 1e-6)

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed

    def as_dict(self) -> dict:
        return {
            "frames": self.frames,
            "events": self.events,
            "errors": self.errors,
            "fps": round(self.fps, 2),
            "avg_ms": round(self.processing_ms / max(self.frames, 1), 2),
            "uptime_seconds": round(self.elapsed, 1),
        }


class CameraWorker:
    """Runs one camera's stream through the analytics pipeline."""

    def __init__(self, camera, *, max_frames: int | None = None, save_snapshots: bool = True):
        self.camera = camera
        self.max_frames = max_frames
        self.save_snapshots = save_snapshots
        self.pipeline = build_pipeline_from_camera(camera, ModelProvider(camera.organization))
        self.source = build_source(camera)
        self.stats = WorkerStats(started=time.time())
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> WorkerStats:
        if not self.source.open():
            if isinstance(self.source, UnavailableSource):
                reason = self.source.reason
            else:
                # Ask the camera why, so the page says something actionable
                # rather than repeating that the decoder said no.
                reason = explain_stream_failure(self.camera)
            self.camera.mark_offline(reason)
            return self.stats

        self.camera.mark_online(f"Streaming via {self.source.description}")
        interval = 1.0 / max(float(self.camera.target_fps or 6), 0.5)
        consecutive_empty = 0

        try:
            while not self._stop:
                if self.max_frames is not None and self.stats.frames >= self.max_frames:
                    break

                loop_started = time.time()
                frame = self.source.read()
                if frame is None:
                    consecutive_empty += 1
                    self.stats.errors += 1
                    if consecutive_empty >= 15:
                        self.camera.mark_offline("Stream stopped delivering frames")
                        break
                    time.sleep(min(0.5 * consecutive_empty, 3.0))
                    continue
                consecutive_empty = 0

                self.process_frame(frame)

                # Pace to the configured analysis rate.
                remaining = interval - (time.time() - loop_started)
                if remaining > 0 and self.max_frames is None:
                    time.sleep(remaining)
        except KeyboardInterrupt:  # pragma: no cover - interactive use
            logger.info("Worker for camera %s interrupted", self.camera.pk)
        finally:
            self.source.close()
            self._persist_stats()

        return self.stats

    def process_frame(self, frame: np.ndarray):
        """Analyse one frame and persist anything worth keeping."""
        from apps.events.services import dispatch_alerts, record_event

        started = time.perf_counter()
        result = self.pipeline.process(frame)
        self.stats.processing_ms += (time.perf_counter() - started) * 1000
        self.stats.frames += 1

        # Share the frame with anyone watching, so looking at a camera being
        # analysed costs it no second session — and shows the same moment the
        # detectors are working on. Encoding only happens while watched.
        from apps.cameras.live import broker

        if broker.wants_frames(self.camera):
            broker.publish(self.camera, frame)

        if result.events:
            snapshot = self._store_snapshot(frame, result) if self.save_snapshots else None
            for candidate in result.events:
                event = record_event(
                    self.camera, candidate, snapshot=snapshot, frame_index=result.frame_index
                )
                if event is not None:
                    self.stats.events += 1
                    try:
                        dispatch_alerts(event)
                    except Exception:  # noqa: BLE001 - delivery must not stop analysis
                        logger.exception("Alert dispatch failed for event %s", event.pk)
        return result

    def _store_snapshot(self, frame: np.ndarray, result):
        """Save an annotated JPEG as evidence for the events in this frame."""
        from .models import CameraSnapshot

        try:
            from PIL import Image
        except ImportError:
            return None

        try:
            array = np.asarray(frame, dtype=np.uint8)
            if self.camera.privacy_blur_faces or settings.CAMPY["PRIVACY_BLUR_UNKNOWN_FACES"]:
                array = self._blur_unknown_faces(array, result)

            image = Image.fromarray(array)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=82, optimize=True)

            from datetime import timedelta

            retention = settings.CAMPY["SNAPSHOT_RETENTION_DAYS"]
            snapshot = CameraSnapshot(
                camera=self.camera,
                captured_at=timezone.now(),
                width=array.shape[1],
                height=array.shape[0],
                reason=result.events[0].event_type if result.events else "manual",
                annotations={
                    "tracks": [t.to_dict() for t in result.tracks][:20],
                    "events": [e.to_dict() for e in result.events][:10],
                },
                expires_at=timezone.now() + timedelta(days=retention),
            )
            snapshot.image.save(
                f"cam{self.camera.pk}-{int(time.time())}.jpg", ContentFile(buffer.getvalue()), save=False
            )
            snapshot.save()
            return snapshot
        except Exception:  # noqa: BLE001 - evidence is best-effort
            logger.exception("Could not store snapshot for camera %s", self.camera.pk)
            return None

    @staticmethod
    def _blur_unknown_faces(array: np.ndarray, result) -> np.ndarray:
        """Redact faces that were not matched to an enrolled, consenting employee."""
        from apps.aiengine.vision.image import pixelate

        output = array.copy()
        face_result = result.results.get("face")
        if face_result is None:
            return output
        for detection in face_result.detections:
            if detection.label != "face:unknown":
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in detection.box]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, output.shape[1]), min(y2, output.shape[0])
            if x2 > x1 and y2 > y1:
                output[y1:y2, x1:x2] = pixelate(output[y1:y2, x1:x2], blocks=6).astype(np.uint8)
        return output

    def _persist_stats(self) -> None:
        from .models import Camera

        Camera.objects.filter(pk=self.camera.pk).update(
            frames_processed=(self.camera.frames_processed or 0) + self.stats.frames,
            health={**self.stats.as_dict(), **self.pipeline.health()},
            last_frame_at=timezone.now() if self.stats.frames else self.camera.last_frame_at,
            updated_at=timezone.now(),
        )


def explain_stream_failure(camera) -> str:
    """Ask the camera why, when the decoder will only say no.

    OpenCV reports every failure as False, so without this the answer is always
    "check the URL, the credentials and the network" — three places to look
    when the camera knows exactly which one it is.
    """
    url = (camera.stream_url or "").strip().lower()
    from apps.cameras import rtsp_probe

    if url.startswith("rtsp"):
        ask = rtsp_probe.probe
    elif url.startswith("http"):
        ask = rtsp_probe.probe_http
    else:
        return ("Could not open the stream. Check the URL, the credentials, and that "
                "this server can reach the camera on that port.")

    result = ask(camera.connection_url, camera.username, camera.password)
    if result.reachable and result.status and result.status < 400:
        # It answered OPTIONS but would not hand over frames: almost always the
        # path, or a stream the camera will not serve over this transport.
        return (f"{result.detail} It would not start the video, though — check the "
                f"channel in the URL, or try the sub-stream.")
    return result.detail


def check_connection(camera) -> tuple[bool, str]:
    """Try to pull a single frame — behind the 'Test connection' button.

    Not named ``test_connection``: pytest collects anything called ``test_*``
    that reaches a test module's namespace, and then fails asking for a
    ``camera`` fixture.
    """
    source = build_source(camera)
    try:
        if not source.open():
            if isinstance(source, UnavailableSource):
                return False, source.reason
            return False, explain_stream_failure(camera)
        for _ in range(5):
            frame = source.read()
            if frame is not None:
                return True, f"Connected — receiving {frame.shape[1]}x{frame.shape[0]} frames."
        return False, "Connected, but no frames were received."
    except Exception as exc:  # noqa: BLE001
        return False, f"Connection failed: {exc}"
    finally:
        source.close()


def capture_preview(camera) -> np.ndarray | None:
    """Grab one frame for the live-view tile."""
    source = build_source(camera)
    try:
        if not source.open():
            return None
        for _ in range(3):
            frame = source.read()
            if frame is not None:
                return frame
        return None
    finally:
        source.close()
