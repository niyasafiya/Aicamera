"""
detector.py  —  MJPEG camera feeds with live YOLOv4-tiny object detection.

One CameraFeed thread per camera:
  1. Reads a video file, frame by frame, looping.
  2. Runs YOLOv4-tiny via OpenCV DNN (CPU, no PyTorch).
  3. Draws bounding boxes and confidence labels.
  4. Overlays a CCTV-style HUD (timestamp, camera ID, person/vehicle counts).
  5. JPEG-encodes the frame and stores it so the MJPEG endpoint can serve it.
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from typing import Optional, List, Callable, Dict

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / "models"
_CFG     = str(MODELS_DIR / "yolov4-tiny.cfg")
_WEIGHTS = str(MODELS_DIR / "yolov4-tiny.weights")
_NAMES   = str(MODELS_DIR / "coco.names")

# ---------------------------------------------------------------------------
# COCO class filtering — only what the warehouse scenario needs
# ---------------------------------------------------------------------------
with open(_NAMES) as f:
    _ALL_NAMES = [l.strip() for l in f.readlines()]

TARGETS: Dict[int, str] = {
    i: _ALL_NAMES[i].upper()
    for i in [0, 2, 3, 5, 7]      # person, car, motorcycle, bus, truck
}

VEHICLE_IDS = {2, 3, 5, 7}

# BGR colours for each label
_COLORS = {
    "PERSON":     (50, 220, 100),
    "CAR":        (50, 160, 255),
    "TRUCK":      (30, 100, 255),
    "BUS":        (200,  80, 255),
    "MOTORCYCLE": (255, 180,  50),
}
_DEFAULT_COLOR = (180, 180, 180)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------
class Detection:
    __slots__ = ("label", "confidence", "x1", "y1", "x2", "y2")

    def __init__(self, label, confidence, x1, y1, x2, y2):
        self.label = label
        self.confidence = confidence
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2

    def pixel_distance_to(self, other: "Detection") -> float:
        return ((self.cx - other.cx) ** 2 + (self.cy - other.cy) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# YOLO detector (shared across all camera feeds)
# ---------------------------------------------------------------------------
class YOLODetector:
    """Thread-safe YOLOv4-tiny detector via OpenCV DNN."""

    def __init__(self, conf_thresh: float = 0.35, nms_thresh: float = 0.4):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self._lock = threading.Lock()

        net = cv2.dnn.readNetFromDarknet(_CFG, _WEIGHTS)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = net.getLayerNames()
        out_idx = net.getUnconnectedOutLayers()
        # Flatten scalar or array-of-arrays depending on OpenCV version
        out_idx = out_idx.flatten() if hasattr(out_idx, "flatten") else out_idx
        self._out_layers = [layer_names[i - 1] for i in out_idx]
        self._net = net

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        with self._lock:
            self._net.setInput(blob)
            outputs = self._net.forward(self._out_layers)

        boxes, confs, class_ids = [], [], []
        for output in outputs:
            for det in output:
                scores = det[5:]
                cid = int(np.argmax(scores))
                conf = float(scores[cid])
                if cid not in TARGETS or conf < self.conf_thresh:
                    continue
                cx_n, cy_n, bw_n, bh_n = det[:4]
                bw, bh = int(bw_n * w), int(bh_n * h)
                x1 = int(cx_n * w - bw / 2)
                y1 = int(cy_n * h - bh / 2)
                boxes.append([x1, y1, bw, bh])
                confs.append(conf)
                class_ids.append(cid)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.conf_thresh, self.nms_thresh)
        indices = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]

        results = []
        for i in indices:
            x, y, bw, bh = boxes[i]
            results.append(Detection(
                label=TARGETS[class_ids[i]],
                confidence=confs[i],
                x1=max(0, x), y1=max(0, y),
                x2=min(w, x + bw), y2=min(h, y + bh),
            ))
        return results


# ---------------------------------------------------------------------------
# Camera feed — runs in its own daemon thread
# ---------------------------------------------------------------------------
class CameraFeed:
    def __init__(
        self,
        camera_id: str,
        video_path: str,
        hud_label: str,
        offset_seconds: float = 0,
        on_detection: Optional[Callable] = None,
    ):
        self.camera_id = camera_id
        self.video_path = video_path
        self.hud_label = hud_label
        self.offset_seconds = offset_seconds
        self.on_detection = on_detection

        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._detections: List[Detection] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # -- public API ----------------------------------------------------------

    def start(self, detector: YOLODetector):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, args=(detector,), daemon=True, name=f"cam-{self.camera_id}"
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg or self._make_nosignal()

    def latest_detections(self) -> List[Detection]:
        with self._lock:
            return list(self._detections)

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _make_nosignal() -> bytes:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, "NO SIGNAL", (210, 175), _FONT, 1.3, (80, 80, 80), 2, cv2.LINE_AA)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]):
        for d in detections:
            color = _COLORS.get(d.label, _DEFAULT_COLOR)
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
            text = f"{d.label}  {d.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(text, _FONT, 0.46, 1)
            tag_y = max(d.y1 - 4, th + 6)
            cv2.rectangle(frame, (d.x1, tag_y - th - 4), (d.x1 + tw + 6, tag_y + 2), color, -1)
            cv2.putText(frame, text, (d.x1 + 3, tag_y - 2), _FONT, 0.46, (8, 8, 8), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, detections: List[Detection]):
        h, w = frame.shape[:2]
        ts = time.strftime("%d/%m/%Y  %H:%M:%S")

        # timestamp top-left
        cv2.putText(frame, ts, (8, 18), _FONT, 0.44, (210, 235, 210), 1, cv2.LINE_AA)
        # camera label bottom-left
        cv2.putText(frame, self.hud_label, (8, h - 10), _FONT, 0.4, (170, 210, 170), 1, cv2.LINE_AA)
        # REC indicator top-right
        cv2.circle(frame, (w - 16, 12), 5, (0, 0, 210), -1)
        cv2.putText(frame, "REC", (w - 44, 16), _FONT, 0.4, (0, 0, 200), 1, cv2.LINE_AA)

        # detection summary bottom-right
        persons  = sum(1 for d in detections if d.label == "PERSON")
        vehicles = sum(1 for d in detections if d.label in ("CAR", "TRUCK", "BUS", "MOTORCYCLE"))
        summary  = f"P:{persons}  V:{vehicles}"
        (sw, _), _ = cv2.getTextSize(summary, _FONT, 0.38, 1)
        cv2.putText(frame, summary, (w - sw - 6, h - 10), _FONT, 0.38, (170, 210, 170), 1, cv2.LINE_AA)

    def _fire_events(self, detections: List[Detection]):
        if self.on_detection:
            persons  = [d for d in detections if d.label == "PERSON"]
            vehicles = [d for d in detections if d.label in ("CAR", "TRUCK", "BUS", "MOTORCYCLE")]
            self.on_detection(self.camera_id, persons, vehicles)

    def _loop(self, detector: YOLODetector):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Sentinel] Could not open video: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = min(int(self.offset_seconds * fps), max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        target_delay   = 1.0 / 12    # process at 12 fps (CPU-friendly)
        event_interval = 4.0          # fire detection callback every 4 s
        last_event     = 0.0

        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize for both detection and display
            h, w = frame.shape[:2]
            scale_w = 640
            scale_h = int(h * scale_w / w)
            frame = cv2.resize(frame, (scale_w, scale_h))

            detections = detector.detect(frame)

            self._draw_detections(frame, detections)
            self._draw_hud(frame, detections)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            jpeg = buf.tobytes()

            with self._lock:
                self._jpeg = jpeg
                self._detections = detections

            now = time.time()
            if now - last_event >= event_interval:
                self._fire_events(detections)
                last_event = now

            wait = target_delay - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)

        cap.release()


# ---------------------------------------------------------------------------
# Manager — convenience wrapper used by main.py
# ---------------------------------------------------------------------------
class FeedManager:
    def __init__(self):
        self._detector: Optional[YOLODetector] = None
        self._feeds: Dict[str, CameraFeed] = {}

    def load_detector(self):
        print("[Sentinel] Loading YOLOv4-tiny model …")
        self._detector = YOLODetector()
        print("[Sentinel] Detection model ready.")

    def add_feed(self, camera_id: str, video_path: str, hud_label: str,
                 offset_seconds: float = 0, on_detection: Optional[Callable] = None):
        feed = CameraFeed(camera_id, video_path, hud_label, offset_seconds, on_detection)
        self._feeds[camera_id] = feed
        if self._detector:
            feed.start(self._detector)

    def get_feed(self, camera_id: str) -> Optional[CameraFeed]:
        return self._feeds.get(camera_id)

    def stop_all(self):
        for feed in self._feeds.values():
            feed.stop()
