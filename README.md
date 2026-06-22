# Sentinel API — Vehicle & Logistics module

A small **Python backend (API)** for one module of the Technomak video-analytics
project. It is written with **FastAPI** and is designed to be read and run by a
beginner.

This module implements two of your requirements:

| Requirement | What it does | Status |
|-------------|--------------|--------|
| **FR-V1** Vehicle Turnaround Time | Records when a vehicle enters and exits, then calculates how long it stayed | ✅ Fully working |
| **FR-V2** Near-miss collisions | Receives and stores forklift↔pedestrian near-miss events | ✅ Event intake working |

---

## What is an "API" and why is it separate from the cameras?

Think of the system in three layers:

1. **Cameras + AI vision model** — watches the video and *detects* things
   ("a vehicle entered", "a forklift came within 0.8 m of a person").
2. **This API (the backend)** — *receives* those detections, stores them, and
   does the maths (durations, averages, status labels).
3. **The frontend (your UI mockup)** — *displays* the results to the operator.

This project is **layer 2**. It does not look at video itself. That is why you
can build and run it fully in Python without any AI model — the model is a
separate job (that is the part people call "fine-tuning", explained below).

### About "fine-tuning"

Fine-tuning means taking an existing AI vision model and *training it further on
warehouse photos* so it gets good at spotting forklifts, vests, plates, etc.
That needs labelled image datasets and usually a GPU, and it is a separate piece
of work from this API. This backend is built to *consume* whatever the model
produces, so the two can be developed independently.

---

## Folder structure

```
technomak-api/
├── app/
│   ├── __init__.py      <- marks "app" as a Python package (can be empty)
│   └── main.py          <- the whole API lives here
├── requirements.txt     <- the libraries to install
└── README.md            <- this file
```

---

## How to run it (step by step, for beginners)

You need **Python 3.9 or newer** installed. Check by running `python --version`.

Open a terminal **inside the `technomak-api` folder** (in VS Code: Terminal →
New Terminal), then:

**1) Create a virtual environment** (a private box for this project's libraries)

Windows:
```
python -m venv venv
venv\Scripts\activate
```

Mac / Linux:
```
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of your terminal line.

**2) Install the libraries**
```
pip install -r requirements.txt
```

**3) Start the server**
```
uvicorn app.main:app --reload
```

You should see a line like `Uvicorn running on http://127.0.0.1:8000`.
Leave this terminal running. (Press `Ctrl + C` to stop it later.)

**4) Open the interactive docs in your browser**

Go to: **http://127.0.0.1:8000/docs**

This page is created automatically by FastAPI. You can click any endpoint →
**"Try it out"** → **Execute**, and see the real response. No frontend needed
to test it. This page alone is great to show in an internship demo.

---

## The endpoints (what you can call)

Base address: `http://127.0.0.1:8000`

### Health
- `GET /` — check the server is alive.

### FR-V1 — Turnaround time
- `POST /api/v1/vehicles/entry` — record a vehicle entering.
  Send: `{ "plate": "KL07 CK 4521" }`
- `POST /api/v1/vehicles/exit` — record a vehicle leaving (calculates duration).
  Send: `{ "plate": "KL07 CK 4521" }`
- `GET /api/v1/vehicles` — list every visit with duration + status.
- `GET /api/v1/vehicles/onsite` — only vehicles currently inside.
- `GET /api/v1/analytics/turnaround` — average turnaround + chart buckets.

### FR-V2 — Near-miss
- `POST /api/v1/near-miss` — store a near-miss event.
  Send: `{ "forklift_id": "FL-3", "location": "Aisle 6", "gap_meters": 0.8 }`
- `GET /api/v1/near-miss` — list near-miss events.

The server starts with some **sample data already loaded**, so the lists are not
empty the first time you look.

---

## Try it without the docs page (optional)

Using `curl` from another terminal:

```
# See current vehicles
curl http://127.0.0.1:8000/api/v1/vehicles

# Record a new entry
curl -X POST http://127.0.0.1:8000/api/v1/vehicles/entry -H "Content-Type: application/json" -d "{\"plate\": \"KL99 ZZ 0001\"}"

# Record its exit (gives you the turnaround time)
curl -X POST http://127.0.0.1:8000/api/v1/vehicles/exit -H "Content-Type: application/json" -d "{\"plate\": \"KL99 ZZ 0001\"}"
```

---

## How the status colours are decided

- `Cleared` (green) — left within the target time.
- `On site` (amber) — still inside.
- `Over SLA` (red) — stayed longer than the allowed time (`SLA_MINUTES`, default 60).

Near-miss severity from the gap distance:
`critical` if under 1.0 m, `warning` if under 1.5 m, otherwise `info`.

---

## Good next steps (if you want to go further)

1. **Save data in a real database** so it survives a restart (start with SQLite —
   it is built into Python).
2. **Connect the frontend**: replace the hard-coded table on the Vehicle &
   Logistics screen with a `fetch("http://127.0.0.1:8000/api/v1/vehicles")` call.
   (Your UI does not need to change in layout — only where the numbers come from.)
3. **Add the other modules** the same way (Gate, Safety, Material & DN), one
   router file per module.
