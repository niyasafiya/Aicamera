"""
ANPR (Automatic Number Plate Recognition) endpoints.
Video upload → background OCR job → results + access log.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

import cv2
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

import db
from services.anpr_service import extract_plates_from_frame

router = APIRouter()

UPLOADS = Path("uploads")


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
# Background job — runs in Starlette thread pool (sync function)
# ---------------------------------------------------------------------------

def _process_video(job_id: str, video_path: str):
    try:
        _update_job(job_id, "processing", 0, None)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _update_job(job_id, "error", 0, "Cannot open video file")
            Path(video_path).unlink(missing_ok=True)
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25

        # Seek directly to evenly-spaced frames — avoids decoding all intermediate frames.
        # For a 2-min 30fps video this saves decoding ~3500 frames we'd otherwise throw away.
        step    = max(1, int(fps * 2))               # one sample every 2 s
        targets = list(range(step, total, step))[:2]  # max 2 frames
        if not targets:
            targets = [0]                             # very short clip — try first frame

        plates_best: dict = {}

        for i, pos in enumerate(targets):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue

            detections = extract_plates_from_frame(frame)
            for d in detections:
                norm = _norm(d["plate"])
                if norm not in plates_best or d["confidence"] > plates_best[norm]["confidence"]:
                    plates_best[norm] = d

            _update_job(job_id, "processing", min(99, (i + 1) / len(targets) * 100), None)

            # Stop early as soon as any plate is detected
            if plates_best:
                break

        cap.release()

        # Cross-check against whitelist
        conn = db.get_conn()
        auth_rows = conn.execute(
            "SELECT plate, owner, vehicle_type FROM authorized_vehicles"
        ).fetchall()
        auth_map = {_norm(r["plate"]): dict(r) for r in auth_rows}

        plates_out: List[dict] = []
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
            plates_out.append(
                {
                    "plate":        data["plate"],
                    "confidence":   data["confidence"],
                    "authorized":   auth is not None,
                    "owner":        auth["owner"]        if auth else "Unknown",
                    "vehicle_type": auth["vehicle_type"] if auth else "Unknown",
                }
            )

        conn.commit()
        conn.close()
        _update_job(job_id, "completed", 100, None)

    except Exception as exc:
        import traceback
        print(f"[ANPR job {job_id}] {traceback.format_exc()}")
        _update_job(job_id, "error", 0, str(exc))
    finally:
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
    UPLOADS.mkdir(exist_ok=True)
    video_path.write_bytes(await video.read())

    conn = db.get_conn()
    conn.execute(
        "INSERT INTO anpr_jobs (job_id, status, progress) VALUES (?,?,?)",
        (job_id, "pending", 0),
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(_process_video, job_id, str(video_path))
    return {"job_id": job_id}


@router.get("/job/{job_id}")
def get_job(job_id: str):
    conn   = db.get_conn()
    job    = conn.execute("SELECT * FROM anpr_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not job:
        conn.close()
        raise HTTPException(404, "Job not found")

    job = dict(job)
    plates: List[dict] = []
    if job["status"] == "completed":
        rows   = conn.execute(
            "SELECT * FROM anpr_plates WHERE job_id=?", (job_id,)
        ).fetchall()
        plates = [
            {**dict(r), "authorized": bool(r["authorized"])} for r in rows
        ]
    conn.close()
    return {**job, "plates": plates}


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
