import sqlite3
from pathlib import Path

DB_PATH = Path("data/sentinel.db")


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS authorized_vehicles (
            plate        TEXT PRIMARY KEY,
            owner        TEXT NOT NULL DEFAULT 'Unknown',
            vehicle_type TEXT NOT NULL DEFAULT 'Car',
            created_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS anpr_jobs (
            job_id     TEXT PRIMARY KEY,
            status     TEXT NOT NULL DEFAULT 'pending',
            progress   REAL NOT NULL DEFAULT 0,
            error      TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS anpr_plates (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id       TEXT NOT NULL,
            plate        TEXT NOT NULL,
            confidence   REAL NOT NULL DEFAULT 0,
            authorized   INTEGER NOT NULL DEFAULT 0,
            owner        TEXT NOT NULL DEFAULT '',
            vehicle_type TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS anpr_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT DEFAULT (datetime('now')),
            plate      TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            authorized INTEGER NOT NULL DEFAULT 0,
            decision   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS persons (
            employee_id    TEXT PRIMARY KEY,
            name           TEXT NOT NULL,
            department     TEXT NOT NULL DEFAULT 'General',
            clearance_level TEXT NOT NULL DEFAULT 'L1',
            photo_path     TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS bio_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT DEFAULT (datetime('now')),
            person_name TEXT NOT NULL DEFAULT 'Unknown',
            confidence  REAL NOT NULL DEFAULT 0,
            decision    TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS vehicle_visits (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            plate            TEXT NOT NULL,
            owner            TEXT NOT NULL DEFAULT 'Unknown',
            entry_time       TEXT NOT NULL DEFAULT (datetime('now')),
            exit_time        TEXT,
            duration_minutes INTEGER
        );
        CREATE TABLE IF NOT EXISTS vehicle_demo_jobs (
            job_id      TEXT PRIMARY KEY,
            status      TEXT NOT NULL DEFAULT 'pending',
            progress    REAL NOT NULL DEFAULT 0,
            error       TEXT,
            result_json TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );
    """)

    existing = conn.execute("SELECT COUNT(*) FROM authorized_vehicles").fetchone()[0]
    if existing == 0:
        seeds = [
            # Indian plates
            ("KL07CK4521", "Arun Menon",          "Car"),
            ("KA09AB1234", "Meridian Logistics",   "Truck"),
            ("TN32CD5678", "Coastal Freight",      "Van"),
            ("MH12EF9012", "Vector Supply",        "Truck"),
            ("KL07CK0001", "Technomak Security",   "Car"),
            ("AP28CZ1122", "Warehouse Ops",        "Van"),
            # Dubai plates
            ("A12345",  "Sheikh Mohammed",         "Car"),
            ("B1234",   "Dubai Police",            "Car"),
            ("CD123",   "Emirates Logistics",      "Truck"),
            ("X99999",  "Technomak UAE",           "Car"),
            ("AB5678",  "Al Futtaim Transport",    "Van"),
        ]
        for plate, owner, vtype in seeds:
            conn.execute(
                "INSERT OR IGNORE INTO authorized_vehicles (plate, owner, vehicle_type) VALUES (?,?,?)",
                (plate, owner, vtype),
            )
    conn.commit()
    conn.close()
