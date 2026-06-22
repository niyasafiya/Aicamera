import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "sentinel.db"
FACE_DIR = Path(__file__).parent.parent / "faces"
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"

FACE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)


def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS authorized_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT UNIQUE NOT NULL,
            owner TEXT DEFAULT 'Unknown',
            vehicle_type TEXT DEFAULT 'Car',
            added_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS access_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            confidence REAL DEFAULT 0,
            authorized INTEGER DEFAULT 0,
            decision TEXT DEFAULT 'DENIED',
            source TEXT DEFAULT 'ANPR',
            timestamp TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS registered_persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            employee_id TEXT UNIQUE NOT NULL,
            department TEXT DEFAULT 'General',
            clearance_level TEXT DEFAULT 'L1',
            face_image TEXT,
            registered_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS biometric_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            person_name TEXT DEFAULT 'Unknown',
            confidence REAL DEFAULT 0,
            decision TEXT DEFAULT 'DENIED',
            timestamp TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS anpr_jobs (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'pending',
            filename TEXT DEFAULT '',
            total_frames INTEGER DEFAULT 0,
            processed_frames INTEGER DEFAULT 0,
            plates_found TEXT DEFAULT '[]',
            error TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS vehicle_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            owner TEXT DEFAULT 'Unknown',
            vehicle_type TEXT DEFAULT 'Car',
            entry_time TEXT DEFAULT (datetime('now')),
            exit_time TEXT,
            duration_minutes REAL,
            status TEXT DEFAULT 'on_site'
        );
    """)

    # Seed authorized vehicles
    seed_vehicles = [
        ("KL07CK4521", "Meridian Logistics", "Truck"),
        ("TN38BA1190", "Coastal Freight", "Van"),
        ("KL11AX8830", "Vector Supply", "Car"),
        ("KA05MN2204", "Operations Fleet", "Truck"),
        ("KL10AB1234", "Technomak Internal", "Car"),
        ("TN09CD5678", "Technomak Internal", "Car"),
        ("MH12EF9012", "Guest — Authorized", "Car"),
        ("KL01GH3456", "Vendor A", "Van"),
        ("KL55AB1234", "Security Fleet", "Car"),
        ("TN22XY7890", "Delivery Partner", "Truck"),
    ]
    for plate, owner, vtype in seed_vehicles:
        try:
            c.execute(
                "INSERT OR IGNORE INTO authorized_vehicles (plate, owner, vehicle_type) VALUES (?, ?, ?)",
                (plate, owner, vtype)
            )
        except Exception:
            pass

    # Seed registered persons
    seed_persons = [
        ("Arun Menon", "EMP-2041", "Logistics", "L2"),
        ("Priya Nair", "EMP-1982", "Operations", "L3"),
        ("Rahul Kumar", "EMP-2155", "Security", "L1"),
        ("Deepa Pillai", "EMP-1850", "Administration", "L2"),
        ("Sanjay Thomas", "EMP-2300", "Engineering", "L3"),
    ]
    for name, emp_id, dept, cl in seed_persons:
        try:
            c.execute(
                "INSERT OR IGNORE INTO registered_persons (name, employee_id, department, clearance_level) VALUES (?, ?, ?, ?)",
                (name, emp_id, dept, cl)
            )
        except Exception:
            pass

    # Seed vehicle visits (demo history)
    seed_visits = [
        ("KL07CK4521", "Meridian Logistics", "Truck",  "2026-06-17 09:37:00", None,                  None,  "on_site"),
        ("TN38BA1190", "Coastal Freight",    "Van",    "2026-06-17 09:07:00", "2026-06-17 09:35:00", 28.0,  "cleared"),
        ("KL11AX8830", "Vector Supply",      "Car",    "2026-06-17 08:40:00", "2026-06-17 09:19:00", 39.0,  "cleared"),
        ("KA05MN2204", "Operations Fleet",   "Truck",  "2026-06-17 08:12:00", "2026-06-17 09:24:00", 72.0,  "over_sla"),
        ("KL10AB1234", "Technomak Internal", "Car",    "2026-06-17 07:55:00", "2026-06-17 08:38:00", 43.0,  "cleared"),
    ]
    for plate, owner, vtype, entry, exit_t, dur, status in seed_visits:
        try:
            c.execute(
                "INSERT OR IGNORE INTO vehicle_visits (plate,owner,vehicle_type,entry_time,exit_time,duration_minutes,status) VALUES (?,?,?,?,?,?,?)",
                (plate, owner, vtype, entry, exit_t, dur, status)
            )
        except Exception:
            pass

    conn.commit()
    conn.close()
