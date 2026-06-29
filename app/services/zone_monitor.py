import logging
import os
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.events import ZoneIntrusion

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    _LIBS_AVAILABLE = True
except ImportError:
    logger.warning("ultralytics/opencv not installed — zone detection disabled until installed")
    _LIBS_AVAILABLE = False

# camera_id -> [{"name": str, "polygon": [[x,y], ...]}]
_zones: dict[str, list[dict]] = {}

# Per-camera-per-zone cooldown: "camera_id:zone_name" -> last alert datetime
_cooldowns: dict[str, datetime] = {}


class ZoneMonitor:
    def __init__(self):
        self.model = YOLO(settings.PERSON_MODEL_PATH) if _LIBS_AVAILABLE else None
        os.makedirs(settings.ALERT_IMAGE_DIR, exist_ok=True)

    def set_zones(self, camera_id: str, zones: list[dict]) -> None:
        """
        zones: [{"name": "Restricted Zone A", "polygon": [[x1,y1],[x2,y2],...]}]
        Coordinates are in absolute pixels relative to the camera frame.
        """
        _zones[camera_id] = zones

    def process(self, frame, camera_id: str, db: Session) -> None:
        if not _LIBS_AVAILABLE or self.model is None:
            return
        zones = _zones.get(camera_id)
        if not zones:
            return

        # class 0 = person in COCO models
        results = self.model(frame, conf=settings.ZONE_CONFIDENCE, classes=[0], verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            confidence = float(box.conf)

            for zone in zones:
                polygon = np.array(zone["polygon"], dtype=np.int32)
                inside = cv2.pointPolygonTest(polygon, centroid, False) >= 0
                if not inside:
                    continue

                key = f"{camera_id}:{zone['name']}"
                now = datetime.utcnow()
                last = _cooldowns.get(key)
                if last and (now - last).total_seconds() < settings.ALERT_COOLDOWN_SECONDS:
                    continue
                _cooldowns[key] = now

                annotated = _annotate(frame.copy(), polygon, zone["name"], centroid)
                frame_path = _save_frame(annotated, camera_id, "zone")

                db.add(ZoneIntrusion(
                    camera_id=camera_id,
                    timestamp=now,
                    zone_name=zone["name"],
                    confidence=confidence,
                    frame_path=frame_path,
                ))
                db.commit()


def _annotate(frame, polygon, zone_name: str, centroid: tuple):
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.circle(frame, centroid, 6, (0, 0, 255), -1)
    cv2.putText(frame, f"INTRUSION: {zone_name}", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return frame


def _save_frame(frame, camera_id: str, prefix: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(settings.ALERT_IMAGE_DIR, f"{prefix}_{camera_id}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path
