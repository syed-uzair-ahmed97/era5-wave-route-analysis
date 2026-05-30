"""
server.py
=========
Lightweight Flask API that bridges the web frontend to map1_optimized.py.

Endpoints
---------
  POST /run          — start analysis (body: JSON config), streams progress via SSE
  GET  /status       — current run state
  GET  /results      — returns results.json content
  GET  /download/png    — downloads the PNG
  GET  /download/svg    — downloads the SVG
  GET  /track        — returns track.txt as JSON array
  GET  /ping         — health check

Start
-----
  pip install flask flask-cors
  python server.py
  Then open index.html in your browser.
"""

import os
import sys
import json
import sqlite3
import subprocess
import threading
import time
from functools import wraps
from flask import Flask, jsonify, request, send_file, redirect, session
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-change-this-secret-key")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower() == "true",
)
CORS(app)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
DB_FILE      = os.path.join(SCRIPT_DIR, "app.db")

RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
PNG_FILE     = os.path.join(RESULTS_DIR, "Route_Weighted_Joint_Probability.png")
SVG_FILE     = os.path.join(RESULTS_DIR, "Route_Weighted_Joint_Probability.svg")
TRACK_FILE   = os.path.join(DATA_DIR, "track.txt")
ROUTE_GEOJSON = os.path.join(RESULTS_DIR, "route.geojson")
OPTIMIZED_PY = os.path.join(SCRIPT_DIR, "map1_optimized.py")

PORT_SEED = [
    ("Rizhao", "China", "CNRZH", 37.83, 120.83),
    ("Cape Town", "South Africa", "ZACPT", -33.9249, 18.4241),
    ("Esbjerg", "Denmark", "DKEBJ", 55.46, 8.43),
    ("Shanghai", "China", "CNSHA", 31.23, 121.49),
    ("Singapore", "Singapore", "SGSIN", 1.29, 103.85),
    ("Colombo", "Sri Lanka", "LKCMB", 6.94, 79.84),
    ("Rotterdam", "Netherlands", "NLRTM", 51.92, 4.48),
    ("Antwerp", "Belgium", "BEANR", 51.22, 4.40),
    ("Hamburg", "Germany", "DEHAM", 53.55, 9.99),
    ("Busan", "South Korea", "KRPUS", 35.10, 129.05),
    ("Yokohama", "Japan", "JPYOK", 35.45, 139.64),
    ("Port Said", "Egypt", "EGPSD", 31.26, 32.30),
    ("Jebel Ali", "United Arab Emirates", "AEJEA", 24.99, 55.02),
    ("Los Angeles", "United States", "USLAX", 33.74, -118.25),
    ("New York", "United States", "USNYC", 40.70, -74.01),
]

init_done = False

# Shared run state
run_state = {
    "running"  : False,
    "done"     : False,
    "success"  : False,
    "log"      : [],
    "returncode": None,
    "progress" : 0,
    "message"  : "Idle",
}
run_lock = threading.Lock()


def db_connect():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@example.com").lower()
    admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                country TEXT,
                unlocode TEXT,
                lat REAL NOT NULL,
                lon REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                origin_lon REAL,
                origin_lat REAL,
                destination_lon REAL,
                destination_lat REAL,
                waypoints_json TEXT,
                track_file_path TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        if conn.execute("SELECT COUNT(*) FROM ports").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO ports (name, country, unlocode, lat, lon) VALUES (?, ?, ?, ?, ?)",
                PORT_SEED,
            )

        user = conn.execute("SELECT id FROM users WHERE email = ?", (admin_email,)).fetchone()
        if user is None:
            conn.execute(
                "INSERT INTO users (email, password_hash, role) VALUES (?, ?, 'admin')",
                (admin_email, generate_password_hash(admin_password)),
            )


def ensure_initialized():
    global init_done
    if not init_done:
        init_db()
        init_done = True


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/") or request.path in {
                "/run", "/route/generate", "/status", "/results", "/track"
            } or request.path.startswith("/download/"):
                return jsonify({"error": "login required"}), 401
            return redirect("/login")
        return fn(*args, **kwargs)
    return wrapper


# ─── helpers ──────────────────────────────────────────────────────────────────

def _reset_state():
    with run_lock:
        run_state.update(running=True, done=False,
                         success=False, log=[], returncode=None,
                         progress=0, message="Starting analysis")


def _progress_from_line(line: str):
    checks = [
        ("Loading ERA5 data", 10, "Loading ERA5 data"),
        ("Dataset opened", 18, "ERA5 files opened"),
        ("Loading track", 22, "Loading route track"),
        ("Track loaded", 28, "Route track loaded"),
        ("Selecting nearest ERA5 grid points", 34, "Selecting ERA5 grid points"),
        ("Computing trapezoidal route weights", 42, "Computing route weights"),
        ("Extracting Hs and Tp", 52, "Extracting Hs and Tp along route"),
        ("Data loaded", 66, "Wave data loaded"),
        ("Computing joint probability matrices", 74, "Computing Hs/Tp probabilities"),
        ("Combining matrices", 82, "Combining route-weighted matrix"),
        ("Rendering joint probability", 88, "Rendering figure"),
        ("Writing results.json", 96, "Writing web results"),
        ("All done", 100, "Analysis complete"),
        ("Analysis completed successfully", 100, "Analysis complete"),
    ]
    for needle, progress, message in checks:
        if needle in line:
            return progress, message
    return None, None


def _append_log(line: str):
    with run_lock:
        run_state["log"].append(line)
        progress, message = _progress_from_line(line)
        if progress is not None:
            run_state["progress"] = max(run_state["progress"], progress)
            run_state["message"] = message


def _finish(rc: int):
    with run_lock:
        run_state["running"]    = False
        run_state["done"]       = True
        run_state["success"]    = rc == 0
        run_state["returncode"] = rc
        run_state["progress"]   = 100 if rc == 0 else run_state["progress"]
        run_state["message"]    = "Analysis complete" if rc == 0 else "Analysis failed"


def _run_analysis(config: dict):
    """Run map1_optimized.py in a worker thread and capture progress output."""
    folder = config.get("folder", DATA_DIR)
    year_start = int(config.get("year_start", 2016))
    year_end = int(config.get("year_end", 2025))

    try:
        from storage_cache import ensure_era5_files
        missing = ensure_era5_files(folder, year_start, year_end)
        if missing:
            _append_log("⚠  Missing ERA5 files and object storage is not configured:")
            for path in missing[:8]:
                _append_log(f"   {path}")
            if len(missing) > 8:
                _append_log(f"   ... {len(missing) - 8} more")
    except Exception as exc:
        _append_log(f"✗  Storage fetch error: {exc}")
        _finish(-1)
        return
    cmd = [
        sys.executable, OPTIMIZED_PY,
        "--folder", folder,
        "--track",  config.get("track",  TRACK_FILE),
        "--years",
            str(year_start),
            str(year_end),
        "--output", config.get("output", RESULTS_DIR),
        "--hs-max",  str(config.get("hs_max",  9.0)),
        "--hs-step", str(config.get("hs_step", 1.0)),
        "--tp-max",  str(config.get("tp_max",  21.0)),
        "--tp-step", str(config.get("tp_step", 1.0)),
        "--json",
    ]

    _append_log("▶  Starting analysis...")
    _append_log(f"   Command: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            _append_log(line.rstrip())
        proc.wait()
        _finish(proc.returncode)

        if proc.returncode == 0:
            _append_log("✓  Analysis completed successfully!")
        else:
            _append_log(f"✗  Analysis failed (exit code {proc.returncode})")

    except Exception as exc:
        _append_log(f"✗  Exception: {exc}")
        _finish(-1)


def _as_lon_lat(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [lon, lat]")
    lon = float(value[0])
    lat = float(value[1])
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError(f"{name} is outside valid lon/lat bounds")
    return [lon, lat]


def _build_sea_route(origin, destination, waypoints):
    try:
        import searoute as sr
    except ImportError as exc:
        raise RuntimeError("Python package 'searoute' is not installed. Run: pip install searoute") from exc

    route_points = [origin] + waypoints + [destination]
    all_coords = []
    total_length = 0.0
    units = "km"

    for idx in range(len(route_points) - 1):
        segment = sr.searoute(route_points[idx], route_points[idx + 1])
        coords = segment["geometry"]["coordinates"]
        if idx:
            coords = coords[1:]
        all_coords.extend(coords)
        total_length += float(segment.get("properties", {}).get("length", 0.0))
        units = segment.get("properties", {}).get("units", units)

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": all_coords,
        },
        "properties": {
            "length": total_length,
            "units": units,
            "waypoints": waypoints,
        },
    }


# ─── endpoints ────────────────────────────────────────────────────────────────

@app.before_request
def before_request():
    ensure_initialized()


@app.get("/")
def index():
    if not session.get("user_id"):
        return redirect("/login")
    return redirect("/dashboard")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with db_connect() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = int(user["id"])
            session["email"] = user["email"]
            session["role"] = user["role"]
            return redirect("/dashboard")
        return send_file(os.path.join(SCRIPT_DIR, "login.html"))

    if session.get("user_id"):
        return redirect("/dashboard")
    return send_file(os.path.join(SCRIPT_DIR, "login.html"))


@app.get("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.get("/dashboard")
@app.get("/route-builder")
@app.get("/analysis")
@app.get("/results-page")
@login_required
def dashboard():
    """Serve the frontend dashboard."""
    index_path = os.path.join(SCRIPT_DIR, "index.html")
    if not os.path.isfile(index_path):
        return "<h2>index.html not found next to server.py</h2>", 404
    return send_file(index_path)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e), "hint": "Valid endpoints: /, /ping, /run, /route/generate, /status, /results, /track, /download/png|svg"}), 404


@app.get("/ping")
def ping():
    return jsonify({"ok": True, "server": "wave-analysis"})


@app.get("/api/me")
@login_required
def me():
    return jsonify({
        "id": session.get("user_id"),
        "email": session.get("email"),
        "role": session.get("role"),
    })


@app.get("/api/ports/search")
@login_required
def search_ports():
    query = request.args.get("q", "").strip()
    with db_connect() as conn:
        if query:
            rows = conn.execute(
                """
                SELECT id, name, country, unlocode, lat, lon
                FROM ports
                WHERE name LIKE ? OR country LIKE ? OR unlocode LIKE ?
                ORDER BY name
                LIMIT 30
                """,
                (f"%{query}%", f"%{query}%", f"%{query}%"),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, country, unlocode, lat, lon FROM ports ORDER BY name LIMIT 30"
            ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.post("/run")
@login_required
def run_analysis():
    with run_lock:
        if run_state["running"]:
            return jsonify({"error": "Analysis already running"}), 409

    config = request.get_json(silent=True) or {}
    _reset_state()
    threading.Thread(target=_run_analysis, args=(config,), daemon=True).start()
    return jsonify({"started": True})


@app.post("/route/generate")
@login_required
def generate_route():
    config = request.get_json(silent=True) or {}
    try:
        origin = _as_lon_lat(config.get("origin"), "origin")
        destination = _as_lon_lat(config.get("destination"), "destination")
        waypoints = [_as_lon_lat(p, f"waypoint {i + 1}") for i, p in enumerate(config.get("waypoints") or [])]
        route = _build_sea_route(origin, destination, waypoints)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    coords = route["geometry"]["coordinates"]
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(TRACK_FILE, "w", encoding="utf-8") as f:
        for lon, lat in coords:
            f.write(f"{lon}, {lat}\n")

    with open(ROUTE_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(route, f)

    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO routes (
                user_id, origin_lon, origin_lat, destination_lon, destination_lat,
                waypoints_json, track_file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.get("user_id"), origin[0], origin[1], destination[0], destination[1],
                json.dumps(waypoints), TRACK_FILE,
            ),
        )

    return jsonify({
        "saved_track": TRACK_FILE,
        "saved_geojson": ROUTE_GEOJSON,
        "length": route["properties"]["length"],
        "units": route["properties"]["units"],
        "points": len(coords),
        "coordinates": [{"lon": float(lon), "lat": float(lat)} for lon, lat in coords],
    })


@app.get("/status")
@login_required
def status():
    with run_lock:
        return jsonify({
            "running"   : run_state["running"],
            "done"      : run_state["done"],
            "success"   : run_state["success"],
            "returncode": run_state["returncode"],
            "progress"  : run_state["progress"],
            "message"   : run_state["message"],
            "log"       : run_state["log"],
        })


@app.get("/results")
@login_required
def results():
    if not os.path.isfile(RESULTS_JSON):
        return jsonify({"error": "results.json not found — run analysis first"}), 404
    with open(RESULTS_JSON) as f:
        return jsonify(json.load(f))


@app.get("/download/png")
@login_required
def download_png():
    if not os.path.isfile(PNG_FILE):
        return jsonify({"error": "PNG not found"}), 404
    return send_file(PNG_FILE, as_attachment=True,
                     download_name="Route_Weighted_Joint_Probability.png")


@app.get("/download/svg")
@login_required
def download_svg():
    if not os.path.isfile(SVG_FILE):
        return jsonify({"error": "SVG not found"}), 404
    return send_file(SVG_FILE, as_attachment=True,
                     download_name="Route_Weighted_Joint_Probability.svg")


@app.get("/track")
@login_required
def get_track():
    if not os.path.isfile(TRACK_FILE):
        return jsonify({"error": "track.txt not found"}), 404
    import numpy as np
    data = np.loadtxt(TRACK_FILE, delimiter=",")
    return jsonify([{"lon": float(r[0]), "lat": float(r[1])} for r in data])


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_initialized()
    print()
    print("  Wave Analysis API Server")
    print("  ========================")
    print("  Dashboard -> http://localhost:5050")
    print("  API ping  -> http://localhost:5050/ping")
    print()
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
