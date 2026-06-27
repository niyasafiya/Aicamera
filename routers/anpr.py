"""
ANPR (Automatic Number Plate Recognition) endpoints.
Video upload → background OCR job → annotated output video + access log.
"""
from __future__ import annotations

import asyncio
import tempfile
import threading
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import db
from services.anpr_service import extract_plates_from_frame

router = APIRouter()

# Use system temp dir so files are NOT inside OneDrive (avoids cloud sync locking)
UPLOADS = Path(tempfile.gettempdir()) / "sentinel_ai"

# Maximum frames to annotate (~10 min @ 30 fps)
_ANNOTATE_MAX_FRAMES = 18_000


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AuthVehicle(BaseModel):
    plate:        str
    owner:        str = "Unknown"
    vehicle_type: str = "Car"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(plate: str) -> str:
    return plate.upper().replace(" ", "").replace("-", "")


def _update_job(job_id: str, status: str, progress: float, error: Optional[str]):
    conn = db.get_conn()
    conn.execute(
        "UPDATE anpr_jobs SET status=?, progress=?, error=? WHERE job_id=?",
        (status, progress, error, job_id),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Frame / video annotation helpers
# ---------------------------------------------------------------------------

def _annotate_frame(
    frame: np.ndarray,
    detections: List[dict],
    auth_map: dict,
) -> np.ndarray:
    """Draw plate bounding boxes with corner brackets and AUTHORIZED/DENIED labels."""
    out = frame.copy()
    fh, fw = out.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    BLACK = (0, 0, 0)

    for d in detections:
        bbox = d.get("bbox")
        norm = _norm(d["plate"])
        auth = auth_map.get(norm)

        color = (50, 220, 90) if auth else (40, 40, 235)   # BGR: green / red
        label = d["plate"] + ("  AUTHORIZED" if auth else "  DENIED")

        if bbox:
            x1, y1, x2, y2 = (
                max(0, bbox[0]), max(0, bbox[1]),
                min(fw - 1, bbox[2]), min(fh - 1, bbox[3]),
            )
            bw = x2 - x1

            # Dark shadow outline — ensures visibility on any background
            cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), BLACK, 5)
            # Main plate rectangle
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            # Corner-bracket markers (L-shaped, CCTV style)
            cl = max(8, min(bw // 4, 22))   # corner arm length
            for cx, cy, sx, sy in [
                (x1, y1,  1,  1),   # top-left
                (x2, y1, -1,  1),   # top-right
                (x1, y2,  1, -1),   # bottom-left
                (x2, y2, -1, -1),   # bottom-right
            ]:
                cv2.line(out, (cx, cy), (cx + sx * cl, cy), color, 4)
                cv2.line(out, (cx, cy), (cx, cy + sy * cl), color, 4)

            # Label banner above the box
            fs = max(0.38, min(0.82, bw / 270))
            (tw, th), bl = cv2.getTextSize(label, font, fs, 2)
            ly2 = y1
            ly1 = max(0, y1 - th - bl - 10)
            lx2 = min(fw, x1 + tw + 14)
            # Shadow behind label background
            cv2.rectangle(out, (x1 - 1, ly1 - 1), (lx2 + 1, ly2 + 1), BLACK, -1)
            cv2.rectangle(out, (x1, ly1), (lx2, ly2), color, -1)
            cv2.putText(out, label, (x1 + 6, ly2 - bl - 2),
                        font, fs, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # No bbox: banner at bottom of frame
            fs = 0.65
            (tw, th), bl = cv2.getTextSize(label, font, fs, 2)
            px, py = 12, fh - 20
            cv2.rectangle(out, (px - 5, py - th - 5), (px + tw + 5, py + bl + 2), BLACK, -1)
            cv2.rectangle(out, (px - 4, py - th - 4), (px + tw + 4, py + bl + 1), color, -1)
            cv2.putText(out, label, (px, py), font, fs, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def _write_annotated_video(
    video_path: str,
    frame_detections: dict,   # frame_num → list[detection]
    auth_map: dict,
    out_path: str,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    job_id: str,
) -> bool:
    """Write MP4 with per-vehicle plate overlays that persist ~1.5 s each."""
    if not frame_detections:
        return False

    persist = max(1, int(fps * 1.5))

    # Build active_at[frame] = list of detections visible at that frame
    active_at: dict = {}
    for fnum, dets in frame_detections.items():
        for f in range(fnum, min(total_frames + 1, fnum + persist)):
            if f not in active_at:
                active_at[f] = []
            seen = {_norm(d["plate"]) for d in active_at[f]}
            for d in dets:
                if _norm(d["plate"]) not in seen:
                    active_at[f].append(d)
                    seen.add(_norm(d["plate"]))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False

    max_frames = min(total_frames, _ANNOTATE_MAX_FRAMES)
    frame_num = 0
    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        dets = active_at.get(frame_num)
        if dets:
            frame = _annotate_frame(frame, dets, auth_map)
        writer.write(frame)
        frame_num += 1
        if frame_num % 150 == 0:
            pct = 60 + (frame_num / max_frames) * 35
            _update_job(job_id, "processing", min(95, pct), None)

    cap.release()
    writer.release()
    return True


# ---------------------------------------------------------------------------
# Background job — runs in Starlette thread pool (sync function)
# ---------------------------------------------------------------------------

def _encode_video_background(
    video_path: str,
    frame_detections: dict,
    auth_map: dict,
    job_id: str,
    total: int,
    fps: float,
    width: int,
    height: int,
):
    """Encode the annotated MP4 after the scan job has already completed.
    Deletes the source video when done."""
    try:
        ann_path = UPLOADS / f"anpr_{job_id}_annotated.mp4"
        _write_annotated_video(
            video_path, frame_detections, auth_map,
            str(ann_path), total, fps, width, height, job_id,
        )
    except Exception:
        import traceback
        print(f"[ANPR encode {job_id}] {traceback.format_exc()}")
    finally:
        Path(video_path).unlink(missing_ok=True)


def _process_video(job_id: str, video_path: str):
    handed_off = False
    try:
        _update_job(job_id, "processing", 0, None)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _update_job(job_id, "error", 0, "Cannot open video file")
            Path(video_path).unlink(missing_ok=True)
            return

        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Sequential read: cap.grab() advances without full pixel decode (3-5× faster
        # than cap.set() seeking in compressed H.264/MP4 video).
        # Short clips get denser sampling so we don't miss the clearest plate frame.
        if total <= int(fps * 30):              # ≤ 30 s video
            step = max(1, int(fps * 0.5))       # every 0.5 s
        else:
            step = max(1, int(fps * 1.0))       # every 1.0 s for longer clips
        max_scan  = min(total, int(fps * 300))  # scan at most 5 minutes

        plates_best:       dict = {}
        frame_detections:  dict = {}
        plate_seen_count:  dict = {}   # norm → frames where it was detected
        frame_num      = 0
        no_new_streak  = 0

        while frame_num < max_scan:
            do_decode = (frame_num % step == 0)
            if do_decode:
                ret, frame = cap.read()
            else:
                ret = cap.grab()

            if not ret:
                break

            if do_decode:
                detections = extract_plates_from_frame(frame)
                improved = False
                if detections:
                    frame_detections[frame_num] = detections
                    for d in detections:
                        norm = _norm(d["plate"])
                        plate_seen_count[norm] = plate_seen_count.get(norm, 0) + 1
                        if norm not in plates_best or d["confidence"] > plates_best[norm]["confidence"] + 0.05:
                            plates_best[norm] = d
                            improved = True

                no_new_streak = 0 if improved else no_new_streak + 1

                # Fast stop: plate confirmed in 2+ frames with decent confidence
                if any(
                    plate_seen_count.get(n, 0) >= 2 and plates_best[n]["confidence"] >= 0.65
                    for n in plates_best
                ):
                    break

                # Fallback stop: no improvement for 5 consecutive decoded frames
                if plates_best and no_new_streak >= 5:
                    break

                if frame_num % (step * 2) == 0:
                    _update_job(job_id, "processing",
                                min(88, frame_num / max_scan * 88), None)

            frame_num += 1

        cap.release()

        # Fallback: pre-filter rejected everything → try first 10 frames without filter
        if not plates_best:
            cap2 = cv2.VideoCapture(video_path)
            for _ in range(min(10, total)):
                ret, frame = cap2.read()
                if not ret:
                    break
                detections = extract_plates_from_frame(frame)
                if detections:
                    frame_detections[frame_num] = detections
                    for d in detections:
                        norm = _norm(d["plate"])
                        if norm not in plates_best or d["confidence"] > plates_best[norm]["confidence"]:
                            plates_best[norm] = d
                    if plates_best:
                        break
            cap2.release()

        # Cross-check against whitelist
        conn = db.get_conn()
        auth_rows = conn.execute(
            "SELECT plate, owner, vehicle_type FROM authorized_vehicles"
        ).fetchall()
        auth_map = {_norm(r["plate"]): dict(r) for r in auth_rows}

        # Persist results to DB immediately
        for norm, data in plates_best.items():
            auth     = auth_map.get(norm)
            decision = "GRANTED" if auth else "DENIED"
            conn.execute(
                "INSERT INTO anpr_log (plate, confidence, authorized, decision) VALUES (?,?,?,?)",
                (data["plate"], data["confidence"], 1 if auth else 0, decision),
            )
            conn.execute(
                "INSERT INTO anpr_plates (job_id, plate, confidence, authorized, owner, vehicle_type) "
                "VALUES (?,?,?,?,?,?)",
                (
                    job_id, data["plate"], data["confidence"],
                    1 if auth else 0,
                    auth["owner"]        if auth else "Unknown",
                    auth["vehicle_type"] if auth else "Unknown",
                ),
            )

        conn.commit()
        conn.close()

        # Mark completed NOW — results are in DB, frontend can show them immediately
        _update_job(job_id, "completed", 100, None)

        # Kick off video encoding in a separate thread so it doesn't block the result
        if frame_detections:
            handed_off = True
            threading.Thread(
                target=_encode_video_background,
                args=(video_path, frame_detections, auth_map,
                      job_id, total, fps, width, height),
                daemon=True,
            ).start()

    except Exception as exc:
        import traceback
        print(f"[ANPR job {job_id}] {traceback.format_exc()}")
        _update_job(job_id, "error", 0, str(exc))
    finally:
        if not handed_off:
            Path(video_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    job_id     = str(uuid.uuid4())
    suffix     = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_path = UPLOADS / f"anpr_{job_id}{suffix}"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    contents   = await video.read()
    await asyncio.to_thread(video_path.write_bytes, contents)

    conn = db.get_conn()
    conn.execute(
        "INSERT INTO anpr_jobs (job_id, status, progress) VALUES (?,?,?)",
        (job_id, "pending", 0),
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(_process_video, job_id, str(video_path))
    return {"job_id": job_id}


@router.get("/job/{job_id}/video")
def get_job_video(job_id: str):
    """Return the annotated output video for a completed job."""
    video_path = UPLOADS / f"anpr_{job_id}_annotated.mp4"
    if not video_path.exists():
        raise HTTPException(404, "No annotated video available for this job")
    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/job/{job_id}/frame")
def get_job_frame(job_id: str):
    """Return a fallback annotated JPEG frame (kept for backwards compat)."""
    frame_path = UPLOADS / f"anpr_{job_id}_frame.jpg"
    if not frame_path.exists():
        raise HTTPException(404, "No annotated frame available for this job")
    return FileResponse(str(frame_path), media_type="image/jpeg")


@router.get("/job/{job_id}")
def get_job(job_id: str):
    conn = db.get_conn()
    job  = conn.execute("SELECT * FROM anpr_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not job:
        conn.close()
        raise HTTPException(404, "Job not found")

    job = dict(job)
    # Return plates at any stage so the UI can show results as they arrive
    rows   = conn.execute(
        "SELECT * FROM anpr_plates WHERE job_id=?", (job_id,)
    ).fetchall()
    plates = [{**dict(r), "authorized": bool(r["authorized"])} for r in rows]
    conn.close()

    has_video = (UPLOADS / f"anpr_{job_id}_annotated.mp4").exists()
    return {**job, "plates": plates, "has_video": has_video}


@router.get("/authorized")
def list_authorized():
    conn  = db.get_conn()
    rows  = conn.execute("SELECT plate, owner, vehicle_type FROM authorized_vehicles").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/authorized", status_code=201)
def add_authorized(body: AuthVehicle):
    norm = _norm(body.plate)
    conn = db.get_conn()
    existing = conn.execute(
        "SELECT 1 FROM authorized_vehicles WHERE plate=?", (norm,)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, f"Plate {norm} already in whitelist")
    conn.execute(
        "INSERT INTO authorized_vehicles (plate, owner, vehicle_type) VALUES (?,?,?)",
        (norm, body.owner, body.vehicle_type),
    )
    conn.commit()
    conn.close()
    return {"plate": norm, "owner": body.owner, "vehicle_type": body.vehicle_type}


@router.delete("/authorized/{plate}", status_code=204)
def remove_authorized(plate: str):
    norm = _norm(plate)
    conn = db.get_conn()
    conn.execute("DELETE FROM authorized_vehicles WHERE plate=?", (norm,))
    conn.commit()
    conn.close()


@router.get("/log")
def access_log(limit: int = Query(30, ge=1, le=200)):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT timestamp, plate, confidence, authorized, decision "
        "FROM anpr_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {**dict(r), "authorized": bool(r["authorized"])} for r in rows
    ]
