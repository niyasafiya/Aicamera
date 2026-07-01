"""
MJPEG camera-stream endpoints.
Serves synthetic placeholder frames (live RTSP can be wired in later).
"""
from __future__ import annotations

import asyncio
import time
import cv2
import numpy as np
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse

router = APIRouter()

_CAM_CFG = {
    "cam01": {"name": "Main Gate — In",     "id": "CAM-01", "analytics": "LPR · Face"},
    "cam02": {"name": "Facial Recog - Gate", "id": "CAM-02", "analytics": "Face ID"},
    "cam12": {"name": "Loading Bay 2",      "id": "CAM-12", "analytics": "DN-OCR · Count"},
    "cam08": {"name": "Warehouse Aisle 4",  "id": "CAM-08", "analytics": "PPE · Zone"},
    "cam05": {"name": "Turnstile B",        "id": "CAM-05", "analytics": "Tailgate"},
}


def _make_frame(cam_id: str) -> bytes:
    cfg = _CAM_CFG.get(cam_id, {"name": "Unknown", "id": cam_id, "analytics": ""})
    h, w = 360, 640
    frame = np.full((h, w, 3), [10, 16, 22], dtype=np.uint8)

    # Dim grid
    for x in range(0, w, 64):
        cv2.line(frame, (x, 0), (x, h), (16, 24, 32), 1)
    for y in range(0, h, 64):
        cv2.line(frame, (0, y), (w, y), (16, 24, 32), 1)

    # Animated scan-line
    t   = time.time()
    sy  = int((t % 3.5) / 3.5 * h)
    cv2.line(frame, (0, sy), (w, sy), (0, 180, 140), 1)

    # Top-left: green dot + camera name + ID
    cv2.circle(frame, (14, 14), 5, (0, 200, 155), -1)
    cv2.putText(frame, cfg["name"], (26, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (195, 215, 235), 1)
    cv2.putText(frame, cfg["id"],   (26, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 130, 170), 1)

    # Top-right: timestamp + LIVE
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts,      (w - 72, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 200, 220), 1)
    cv2.putText(frame, "LIVE",  (w - 40, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 155), 1)

    # Centre message
    for txt, yp, sc, col in [
        ("NO CAMERA FEED",                      h // 2 - 22, 0.62, (140, 170, 195)),
        ("Connect RTSP / upload demo video",    h // 2 +  4, 0.38, ( 70, 115, 148)),
        (f"Analytics: {cfg['analytics']}",      h // 2 + 20, 0.36, ( 55, 100, 135)),
    ]:
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0]
        cv2.putText(frame, txt, ((w - tw) // 2, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 28), (w, h), (8, 13, 20), -1)
    cv2.putText(frame,
                f"Sentinel AI · Warehouse North · 1080p · {time.strftime('%d %b %Y')}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (55, 95, 130), 1)

    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpg.tobytes()


async def _mjpeg_generator(cam_id: str):
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        jpg = _make_frame(cam_id)
        yield boundary + jpg + b"\r\n"
        await asyncio.sleep(0.10)   # ~10 fps


@router.get("/stream/{cam_id}")
async def stream(cam_id: str):
    if cam_id not in _CAM_CFG:
        return Response(status_code=404, content="Unknown camera")
    return StreamingResponse(
        _mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
