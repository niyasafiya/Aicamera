"""
Sentinel AI Backend — FastAPI entry point.

Run:
    python main.py
    -or-
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Open http://127.0.0.1:8000/docs for interactive API docs.
"""
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import db
from routers import anpr, biometric, vehicles, streams

app = FastAPI(
    title="Sentinel AI Backend",
    version="2.4.1",
    description="Video-analytics backend for the Technomak Sentinel console.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _warmup_easyocr():
    from services.anpr_service import _get_reader
    print("[Startup] Loading EasyOCR models in background…")
    _get_reader()
    print("[Startup] EasyOCR ready — first scan will be fast.")


@app.on_event("startup")
async def on_startup():
    for d in ("uploads", "data/faces"):
        Path(d).mkdir(parents=True, exist_ok=True)
    db.init_db()
    threading.Thread(target=_warmup_easyocr, daemon=True).start()
    print("\n  Sentinel AI backend ready — http://127.0.0.1:8000\n")


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "version": "2.4.1", "service": "Sentinel AI"}


app.include_router(streams.router,   tags=["Streams"])
app.include_router(anpr.router,      prefix="/api/v1/anpr",      tags=["ANPR"])
app.include_router(biometric.router, prefix="/api/v1/biometric", tags=["Biometric"])
app.include_router(vehicles.router,  prefix="/api/v1/vehicles",  tags=["Vehicles"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
