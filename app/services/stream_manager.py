import logging
import threading

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
from app.core.config import settings
from app.core.database import get_db_sync
from app.services.ppe_detector import PPEDetector
from app.services.zone_monitor import ZoneMonitor
from app.services.fall_detector import FallDetector

logger = logging.getLogger(__name__)


class CameraStream:
    def __init__(self, camera_id: str, rtsp_url: str, detectors: dict):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.detectors = detectors
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"stream-{self.camera_id}")
        self._thread.start()
        logger.info("Stream started: %s", self.camera_id)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Stream stopped: %s", self.camera_id)

    def _run(self) -> None:
        if not _CV2_AVAILABLE:
            logger.warning("opencv not installed — stream %s will not process frames", self.camera_id)
            return

        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error("Cannot open stream: %s (%s)", self.camera_id, self.rtsp_url)
            return

        frame_count = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed for %s — reconnecting", self.camera_id)
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url)
                continue

            frame_count += 1
            if frame_count % settings.FRAME_SKIP != 0:
                continue

            db = get_db_sync()
            try:
                self.detectors["ppe"].process(frame, self.camera_id, db)
                self.detectors["zone"].process(frame, self.camera_id, db)
                self.detectors["fall"].process(frame, self.camera_id, db)
            except Exception:
                logger.exception("Detection error on camera %s", self.camera_id)
            finally:
                db.close()

        cap.release()


class StreamManager:
    def __init__(self):
        self.detectors = {
            "ppe": PPEDetector(),
            "zone": ZoneMonitor(),
            "fall": FallDetector(),
        }
        self._streams: dict[str, CameraStream] = {}

    def add_camera(self, camera_id: str, rtsp_url: str) -> None:
        if camera_id in self._streams:
            self._streams[camera_id].stop()
        stream = CameraStream(camera_id, rtsp_url, self.detectors)
        self._streams[camera_id] = stream
        stream.start()

    def remove_camera(self, camera_id: str) -> None:
        stream = self._streams.pop(camera_id, None)
        if stream:
            stream.stop()

    def stop_all(self) -> None:
        for stream in list(self._streams.values()):
            stream.stop()
        self._streams.clear()

    @property
    def zone_monitor(self) -> ZoneMonitor:
        return self.detectors["zone"]
