"""
Sentinel API — main FastAPI application
Endpoints:
  /api/v1/anpr/*      — Automatic Number Plate Recognition
  /api/v1/biometric/* — Face biometric authentication
  /api/v1/vehicles/*  — Authorized vehicle management
  /api/v1/gate/*      — Access logs
"""
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import aiofiles
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel

HTML_FILE = Path(__file__).parent.parent / "technomak-video-analytics-console.html"

from app.database import get_conn, init_db, UPLOAD_DIR, FACE_DIR
from app import anpr as anpr_module
from app import biometric as bio_module

app = FastAPI(title="Sentinel API", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/console", response_class=FileResponse)
def serve_console():
    return FileResponse(str(HTML_FILE), media_type="text/html")

@app.get("/")
def root():
    return {"status": "ok", "service": "Sentinel API", "version": "2.5.0"}


# ===========================================================================
# ANPR — video upload & processing
# ===========================================================================

@app.post("/api/v1/anpr/upload")
async def anpr_upload(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())
    ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    save_path = UPLOAD_DIR / f"{job_id}{ext}"

    content = await video.read()
    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    conn = get_conn()
    conn.execute(
        "INSERT INTO anpr_jobs (id, status, filename) VALUES (?, 'pending', ?)",
        (job_id, video.filename or ""),
    )
    conn.commit()
    conn.close()

    # Run in background thread (EasyOCR is not async)
    def _run():
        anpr_module.process_video_job(job_id, save_path)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return {"job_id": job_id, "status": "pending"}


@app.get("/api/v1/anpr/job/{job_id}")
def anpr_job_status(job_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM anpr_jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Job not found")

    total = max(row["total_frames"], 1)
    progress = round(row["processed_frames"] / total * 100, 1)
    plates = json.loads(row["plates_found"] or "[]")

    return {
        "job_id": job_id,
        "status": row["status"],
        "filename": row["filename"],
        "progress": progress,
        "total_frames": row["total_frames"],
        "processed_frames": row["processed_frames"],
        "plates": plates,
        "error": row["error"],
    }


# ===========================================================================
# Authorized vehicles CRUD
# ===========================================================================

class VehicleIn(BaseModel):
    plate: str
    owner: str = "Unknown"
    vehicle_type: str = "Car"


@app.get("/api/v1/anpr/authorized")
def list_authorized_vehicles():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, plate, owner, vehicle_type, added_at FROM authorized_vehicles ORDER BY added_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/v1/anpr/authorized", status_code=201)
def add_authorized_vehicle(v: VehicleIn):
    plate_clean = v.plate.upper().replace(" ", "").replace("-", "")
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO authorized_vehicles (plate, owner, vehicle_type) VALUES (?, ?, ?)",
            (plate_clean, v.owner, v.vehicle_type),
        )
        conn.commit()
    except Exception:
        conn.close()
        raise HTTPException(409, "Plate already exists")
    row = conn.execute(
        "SELECT * FROM authorized_vehicles WHERE plate=?", (plate_clean,)
    ).fetchone()
    conn.close()
    return dict(row)


@app.delete("/api/v1/anpr/authorized/{plate}")
def delete_authorized_vehicle(plate: str):
    plate_clean = plate.upper().replace(" ", "").replace("-", "")
    conn = get_conn()
    res = conn.execute(
        "DELETE FROM authorized_vehicles WHERE UPPER(REPLACE(REPLACE(plate,' ',''),'-',''))=?",
        (plate_clean,),
    )
    conn.commit()
    conn.close()
    if res.rowcount == 0:
        raise HTTPException(404, "Vehicle not found")
    return {"deleted": plate_clean}


@app.get("/api/v1/anpr/log")
def access_log(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM access_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ===========================================================================
# Biometric — face registration & verification
# ===========================================================================

@app.post("/api/v1/biometric/register")
async def biometric_register(
    photo: UploadFile = File(...),
    name: str = Form(...),
    employee_id: str = Form(...),
    department: str = Form("General"),
    clearance_level: str = Form("L1"),
):
    img_bytes = await photo.read()
    try:
        img = bio_module._decode_bytes(img_bytes)
    except Exception:
        raise HTTPException(400, "Invalid image file")

    conn = get_conn()
    # Upsert person
    existing = conn.execute(
        "SELECT id FROM registered_persons WHERE employee_id=?", (employee_id,)
    ).fetchone()

    face_path = str(bio_module._face_path(employee_id))

    if existing:
        conn.execute(
            "UPDATE registered_persons SET name=?, department=?, clearance_level=?, face_image=? WHERE employee_id=?",
            (name, department, clearance_level, face_path, employee_id),
        )
    else:
        conn.execute(
            "INSERT INTO registered_persons (name, employee_id, department, clearance_level, face_image) VALUES (?,?,?,?,?)",
            (name, employee_id, department, clearance_level, face_path),
        )
    conn.commit()
    person_row = conn.execute(
        "SELECT * FROM registered_persons WHERE employee_id=?", (employee_id,)
    ).fetchone()
    conn.close()

    reg_result = bio_module.register_face(employee_id, img)
    return {
        "ok": reg_result["ok"],
        "face_detected": reg_result["face_detected"],
        "person": dict(person_row),
    }


@app.post("/api/v1/biometric/verify")
async def biometric_verify(photo: UploadFile = File(...)):
    img_bytes = await photo.read()
    try:
        img = bio_module._decode_bytes(img_bytes)
    except Exception:
        raise HTTPException(400, "Invalid image file")

    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, employee_id, department, clearance_level FROM registered_persons"
    ).fetchall()
    persons = [dict(r) for r in rows]
    conn.close()

    result = bio_module.verify_face(img, persons)

    # Log the attempt
    conn = get_conn()
    person_id = result["person"]["id"] if result["person"] else None
    person_name = result["person"]["name"] if result["person"] else "Unknown"
    conn.execute(
        "INSERT INTO biometric_log (person_id, person_name, confidence, decision) VALUES (?,?,?,?)",
        (person_id, person_name, result["confidence"], result["decision"]),
    )
    conn.commit()
    conn.close()

    return result


@app.get("/api/v1/biometric/persons")
def list_persons():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, employee_id, department, clearance_level, registered_at, "
        "CASE WHEN face_image IS NOT NULL THEN 1 ELSE 0 END AS has_photo "
        "FROM registered_persons ORDER BY name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.delete("/api/v1/biometric/persons/{employee_id}")
def delete_person(employee_id: str):
    conn = get_conn()
    res = conn.execute(
        "DELETE FROM registered_persons WHERE employee_id=?", (employee_id,)
    )
    conn.commit()
    conn.close()
    # Remove face image
    path = bio_module._face_path(employee_id)
    if path.exists():
        path.unlink()
    if res.rowcount == 0:
        raise HTTPException(404, "Person not found")
    return {"deleted": employee_id}


@app.get("/api/v1/biometric/log")
def biometric_log(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM biometric_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ===========================================================================
# Vehicle & Logistics — FR-V1 Turnaround Time
# ===========================================================================

class VehicleGateIn(BaseModel):
    plate: str


def _normalize_plate(plate: str) -> str:
    return plate.upper().replace(" ", "").replace("-", "")


def _lookup_authorized(conn, norm_plate: str):
    return conn.execute(
        "SELECT * FROM authorized_vehicles WHERE UPPER(REPLACE(REPLACE(plate,' ',''),'-',''))=?",
        (norm_plate,),
    ).fetchone()


@app.post("/api/v1/vehicles/entry", status_code=201)
def vehicle_entry(v: VehicleGateIn):
    plate = _normalize_plate(v.plate)
    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM vehicle_visits WHERE plate=? AND status='on_site'", (plate,)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, "Vehicle already on site")
    auth = _lookup_authorized(conn, plate)
    owner = auth["owner"] if auth else "Unknown"
    vtype = auth["vehicle_type"] if auth else "Unknown"
    conn.execute(
        "INSERT INTO vehicle_visits (plate, owner, vehicle_type) VALUES (?,?,?)",
        (plate, owner, vtype),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM vehicle_visits WHERE plate=? AND status='on_site' ORDER BY id DESC LIMIT 1",
        (plate,),
    ).fetchone()
    conn.close()
    return dict(row)


@app.post("/api/v1/vehicles/exit")
def vehicle_exit(v: VehicleGateIn):
    plate = _normalize_plate(v.plate)
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM vehicle_visits WHERE plate=? AND status='on_site' ORDER BY id DESC LIMIT 1",
        (plate,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Vehicle not currently on site")
    entry_dt = datetime.fromisoformat(row["entry_time"])
    exit_dt = datetime.utcnow()
    duration = round((exit_dt - entry_dt).total_seconds() / 60, 1)
    status = "cleared" if duration <= 60 else "over_sla"
    conn.execute(
        "UPDATE vehicle_visits SET exit_time=?, duration_minutes=?, status=? WHERE id=?",
        (exit_dt.strftime("%Y-%m-%d %H:%M:%S"), duration, status, row["id"]),
    )
    conn.commit()
    updated = conn.execute("SELECT * FROM vehicle_visits WHERE id=?", (row["id"],)).fetchone()
    conn.close()
    return dict(updated)


@app.get("/api/v1/vehicles")
def list_vehicles(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM vehicle_visits ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/v1/vehicles/onsite")
def vehicles_onsite():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM vehicle_visits WHERE status='on_site' ORDER BY entry_time DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/v1/vehicles/stats")
def vehicle_stats():
    conn = get_conn()
    on_site = conn.execute(
        "SELECT COUNT(*) FROM vehicle_visits WHERE status='on_site'"
    ).fetchone()[0]
    cleared = conn.execute(
        "SELECT COUNT(*) FROM vehicle_visits WHERE status='cleared'"
    ).fetchone()[0]
    over_sla = conn.execute(
        "SELECT COUNT(*) FROM vehicle_visits WHERE status='over_sla'"
    ).fetchone()[0]
    avg_row = conn.execute(
        "SELECT AVG(duration_minutes) FROM vehicle_visits WHERE duration_minutes IS NOT NULL"
    ).fetchone()
    avg_min = round(avg_row[0], 1) if avg_row[0] else 0
    conn.close()
    return {"on_site": on_site, "cleared": cleared, "over_sla": over_sla, "avg_minutes": avg_min}
