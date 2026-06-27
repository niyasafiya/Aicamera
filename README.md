# Sentinel — AI Video Analytics Console

A full-stack warehouse security and operations platform built for **Technomak**.  
Single-page control-room UI backed by a FastAPI server with real computer-vision pipelines.

---

## Features

| Module | Capability |
|--------|-----------|
| **ANPR** (Automatic Number Plate Recognition) | Upload gate-camera footage → detect Indian licence plates via EasyOCR → cross-check against an authorised-vehicle whitelist → log every entry/exit decision |
| **Biometric Auth** | Face recognition via DeepFace — enrol employees, verify against stored embeddings |
| **Object Detection** | YOLOv4-tiny & YOLOv8n inference on uploaded video frames — people, PPE, vehicles, forklifts |
| **Live MJPEG Streams** | Per-camera MJPEG endpoints served by OpenCV |
| **Gate & Access** | Whitelist management (add / remove plates), access log, facial recognition + tailgating detection |
| **People & Safety** | PPE compliance monitoring, zone intrusion alerts |
| **Vehicle & Logistics** | Turnaround time tracking, near-miss event intake |
| **Operations Overview** | Real-time KPI dashboard, 4-camera live feed grid, event feed |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| Computer Vision | OpenCV, EasyOCR, DeepFace, YOLOv4-tiny, YOLOv8n (Ultralytics) |
| Database | SQLite (via `db.py`) |
| Frontend | Single self-contained HTML file — no build step |
| ML deps | NumPy, Pillow, SciPy, TensorFlow/Keras |

---

## Project Structure

```
Ai Camera/
├── main.py                          # FastAPI app entry point
├── db.py                            # SQLite connection helper
├── technomak-video-analytics-console.html  # Full frontend UI
├── requirements.txt
├── start.bat / start_server.bat     # Windows one-click launchers
│
├── routers/
│   ├── anpr.py                      # ANPR upload, job polling, whitelist, access log
│   ├── biometric.py                 # Face enrol & verify endpoints
│   ├── streams.py                   # MJPEG camera stream endpoints
│   └── vehicles.py                  # Vehicle turnaround & near-miss endpoints
│
├── services/
│   ├── anpr_service.py              # EasyOCR multi-pass plate extraction
│   ├── bio_service.py               # DeepFace embedding enrol/verify
│   ├── yolo_service.py              # YOLOv4-tiny / YOLOv8n inference
│   └── demo_pipeline.py             # Demo detection pipeline
│
├── app/                             # Alternate standalone app module
│   ├── anpr.py
│   ├── biometric.py
│   ├── database.py
│   └── main.py
│
├── models/
│   ├── yolov4-tiny.weights          # YOLOv4-tiny weights
│   ├── yolov4-tiny.cfg
│   ├── coco.names
│   └── yolov8n.pt                   # YOLOv8n weights
│
├── faces/                           # Enrolled face images
├── uploads/                         # Temporary video uploads (auto-cleaned)
└── data/                            # Static reference data
```

---

## Setup & Run

**Requirements:** Python 3.10+

### 1. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads EasyOCR and DeepFace model weights automatically (~500 MB).

### 3. Start the server

**Windows (one-click):**
```
start_server.bat
```

**Or manually:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the console

Open `technomak-video-analytics-console.html` directly in your browser.  
The UI connects to `http://127.0.0.1:8000` automatically.

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## API Endpoints

### ANPR
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/anpr/upload` | Upload gate video → start background OCR job |
| `GET` | `/api/v1/anpr/job/{job_id}` | Poll job status & results |
| `GET` | `/api/v1/anpr/authorized` | List whitelisted vehicles |
| `POST` | `/api/v1/anpr/authorized` | Add vehicle to whitelist |
| `DELETE` | `/api/v1/anpr/authorized/{plate}` | Remove vehicle from whitelist |
| `GET` | `/api/v1/anpr/log` | Recent access log |

### Biometric
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/biometric/enrol` | Enrol an employee face |
| `POST` | `/api/v1/biometric/verify` | Verify face against enrolled embeddings |

### Vehicles
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/vehicles/entry` | Record vehicle entry |
| `POST` | `/api/v1/vehicles/exit` | Record vehicle exit (calculates turnaround) |
| `GET` | `/api/v1/vehicles` | All vehicle visits |
| `GET` | `/api/v1/vehicles/onsite` | Vehicles currently on site |
| `GET` | `/api/v1/analytics/turnaround` | Average turnaround stats |

### Streams
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/stream/{cam_id}` | MJPEG stream for a camera (cam01, cam12, etc.) |

---

## ANPR Pipeline

1. Video uploaded → saved to `uploads/`
2. Background thread samples up to **5 evenly-spaced frames** (one every 4 s)
3. Each frame is cropped to the bottom 65 % and resized to 480 px wide
4. EasyOCR runs with `canvas_size=480` (fast CRAFT detection)
5. Detected text is cleaned, validated against Indian plate regex, and OCR-confusion characters are corrected (O↔0, I↔1, S↔5, etc.)
6. Plates are fuzzy-matched (Levenshtein ≤ 1) against the whitelist
7. Results written to `anpr_plates` and `anpr_log` tables; job marked `completed`

---

## Biometric Pipeline
 
1. Enrolment: face image uploaded → DeepFace extracts embedding → stored in `faces/`
2. Verification: query image compared against all stored embeddings using cosine similarity
3. Returns match name, confidence score, and GRANTED / DENIED decision


---

## Licence

Internal project — Technomak. Not for public distribution.
