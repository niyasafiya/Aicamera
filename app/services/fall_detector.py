import logging
import os
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.events import FallEvent

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    _LIBS_AVAILABLE = True
except ImportError:
    logger.warning("ultralytics/opencv not installed — fall detection disabled until installed")
    _LIBS_AVAILABLE = False

# A bounding box is considered "fallen" when width/height exceeds this ratio.
_FALL_ASPECT_RATIO = 1.5

# Minimum pixel displacement between frames to count as movement.
_MOVEMENT_THRESHOLD_PX = 20


@dataclass
class _PersonState:
    last_centroid: tuple[int, int]
    last_moved_at: datetime
    motionless_alerted: bool = False
    fall_alerted: bool = False


class FallDetector:
    def __init__(self):
        self.model = YOLO(settings.POSE_MODEL_PATH) if _LIBS_AVAILABLE else None
        self._states: dict[str, dict[int, _PersonState]] = {}
        os.makedirs(settings.ALERT_IMAGE_DIR, exist_ok=True)

    def process(self, frame, camera_id: str, db: Session) -> None:
        if not _LIBS_AVAILABLE or self.model is None:
            return
        results = self.model.track(
            frame, persist=True, conf=settings.FALL_CONFIDENCE,
            classes=[0], verbose=False,
        )[0]

        if results.boxes is None or results.boxes.id is None:
            return

        now = datetime.utcnow()
        camera_states = self._states.setdefault(camera_id, {})
        active_ids: set[int] = set()

        for box, track_id in zip(results.boxes, results.boxes.id):
            tid = int(track_id)
            active_ids.add(tid)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            state = camera_states.get(tid)
            if state is None:
                camera_states[tid] = _PersonState(last_centroid=(cx, cy), last_moved_at=now)
                continue

            moved = np.hypot(cx - state.last_centroid[0], cy - state.last_centroid[1])
            if moved > _MOVEMENT_THRESHOLD_PX:
                state.last_centroid = (cx, cy)
                state.last_moved_at = now
                # Reset alerts when the person starts moving again
                state.motionless_alerted = False
                state.fall_alerted = False

            is_fallen = h > 0 and (w / h) > _FALL_ASPECT_RATIO

            if is_fallen and not state.fall_alerted:
                state.fall_alerted = True
                frame_path = _save_frame(frame, camera_id, "fall")
                db.add(FallEvent(
                    camera_id=camera_id,
                    timestamp=now,
                    event_type="fall",
                    frame_path=frame_path,
                ))
                db.commit()
                continue

            motionless_secs = (now - state.last_moved_at).total_seconds()
            if motionless_secs >= settings.MOTIONLESS_SECONDS and not state.motionless_alerted:
                state.motionless_alerted = True
                frame_path = _save_frame(frame, camera_id, "motionless")
                db.add(FallEvent(
                    camera_id=camera_id,
                    timestamp=now,
                    event_type="motionless",
                    duration_seconds=motionless_secs,
                    frame_path=frame_path,
                ))
                db.commit()

        # Remove tracks that disappeared from the frame
        for tid in list(camera_states.keys()):
            if tid not in active_ids:
                del camera_states[tid]


def _save_frame(frame, camera_id: str, prefix: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(settings.ALERT_IMAGE_DIR, f"{prefix}_{camera_id}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path
