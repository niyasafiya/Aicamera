"""
Biometric face-verification endpoints.
Register → store encoding; Verify → compare against DB.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from typing import Optional

import db
from services import bio_service

router = APIRouter()


class UpdatePerson(BaseModel):
    name:            Optional[str] = None
    department:      Optional[str] = None
    clearance_level: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_persons() -> list[dict]:
    conn  = db.get_conn()
    rows  = conn.execute(
        "SELECT employee_id, name, department, clearance_level, photo_path "
        "FROM persons ORDER BY name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/register", status_code=201)
async def register_person(
    photo:           UploadFile  = File(...),
    name:            str         = Form(...),
    employee_id:     str         = Form(...),
    department:      str         = Form("General"),
    clearance_level: str         = Form("L1"),
):
    image_bytes = await photo.read()

    # Save photo
    photo_path = bio_service.save_photo(employee_id, image_bytes)

    # Compute & store face embedding
    embedding     = bio_service.compute_embedding(image_bytes)
    face_detected = embedding is not None
    if embedding:
        bio_service.save_encoding(employee_id, embedding)

    # Upsert person record
    conn = db.get_conn()
    conn.execute(
        """
        INSERT INTO persons (employee_id, name, department, clearance_level, photo_path)
        VALUES (?,?,?,?,?)
        ON CONFLICT(employee_id) DO UPDATE SET
            name=excluded.name, department=excluded.department,
            clearance_level=excluded.clearance_level, photo_path=excluded.photo_path
        """,
        (employee_id, name, department, clearance_level, photo_path),
    )
    conn.commit()
    conn.close()

    return {
        "success":      True,
        "face_detected": face_detected,
        "message":      (
            f"{name} registered with face embedding."
            if face_detected
            else f"{name} registered (no face detected — use a clearer photo)."
        ),
    }


@router.post("/verify")
async def verify_face(photo: UploadFile = File(...)):
    image_bytes = await photo.read()
    persons     = _list_persons()
    result      = bio_service.verify_face(image_bytes, persons)

    # Log
    decision = "GRANTED" if result["matched"] else "DENIED"
    person   = result.get("person") or {}
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO bio_log (person_name, confidence, decision) VALUES (?,?,?)",
        (person.get("name", "Unknown"), result["confidence"], decision),
    )
    conn.commit()
    conn.close()

    return result


@router.get("/persons")
def list_persons():
    rows = _list_persons()
    return [
        {
            "employee_id":    r["employee_id"],
            "name":           r["name"],
            "department":     r["department"],
            "clearance_level": r["clearance_level"],
            "has_photo":      r["photo_path"] is not None,
        }
        for r in rows
    ]


@router.patch("/persons/{employee_id}")
def update_person(employee_id: str, body: UpdatePerson):
    conn = db.get_conn()
    row = conn.execute("SELECT employee_id FROM persons WHERE employee_id=?", (employee_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Person not found")

    updates: dict = {}
    if body.name is not None:            updates["name"]            = body.name
    if body.department is not None:      updates["department"]      = body.department
    if body.clearance_level is not None: updates["clearance_level"] = body.clearance_level

    if updates:
        set_clause = ", ".join(f"{k}=?" for k in updates)
        conn.execute(
            f"UPDATE persons SET {set_clause} WHERE employee_id=?",
            (*updates.values(), employee_id),
        )
        conn.commit()
    conn.close()
    return {"employee_id": employee_id, **updates}


@router.delete("/persons/{employee_id}", status_code=204)
def delete_person(employee_id: str):
    conn = db.get_conn()
    conn.execute("DELETE FROM persons WHERE employee_id=?", (employee_id,))
    conn.commit()
    conn.close()
    bio_service.delete_encoding(employee_id)


@router.post("/log", status_code=201)
def add_log_entry(
    person_name: str   = Form(""),
    confidence:  float = Form(0.0),
    decision:    str   = Form("GRANTED"),
):
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO bio_log (person_name, confidence, decision) VALUES (?,?,?)",
        (person_name, round(float(confidence), 4), decision.upper()),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@router.get("/log")
def bio_log(limit: int = Query(30, ge=1, le=200)):
    conn  = db.get_conn()
    rows  = conn.execute(
        "SELECT timestamp, person_name, confidence, decision "
        "FROM bio_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
