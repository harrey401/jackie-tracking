"""rt_logger.py — Stream Firebase RTDB ticks into per-session CSVs.

Each unique session_id seen in the stream gets its own
logs/session_<ts>_<id>.csv file, matching the naming convention used by
motion_logger.py. Historical ticks replayed on connect are routed to their
original session's file rather than being mixed into a single shared CSV.

On clean shutdown (Ctrl-C), every open session CSV is uploaded to Firebase
Storage under sessions/<filename>.csv — the same bucket motion_logger.py
uses for JSONL archives.

Setup:
    pip install requests
    cp .env.example .env   # fill in your Firebase URL + token

Run:
    python rt_logger.py            # reads JACKIE_FIREBASE_URL from env
    # or, if not using a .env loader, source it manually:
    set -a; source .env; set +a; python rt_logger.py

Tail the live CSV from another terminal:
    tail -f logs/session_*.csv
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests  # type: ignore


def _redact(text: str) -> str:
    """Strip ?auth=<token> from anything we're about to print/log."""
    return re.sub(r"([?&])auth=[^&\s]+", r"\1auth=<redacted>", str(text))


# === Config ==================================================================

DATABASE_URL = os.getenv("JACKIE_FIREBASE_URL", "")
COLLECTION   = os.getenv("JACKIE_FIREBASE_COLLECTION", "ticks")
LOG_DIR      = Path(__file__).parent / "logs"
AUTH_TOKEN: str | None = os.getenv("JACKIE_FIREBASE_TOKEN") or None

FIELDNAMES = [
    "timestamp", "session_id", "t_accum", "fsm_state",
    "face_visible", "face_cx", "dist_m", "dx_norm",
    "linear_cmd", "angular_cmd", "track_id",
    "lidar_forward_min", "obstacle_region",
]


# === Per-session CSV tracking ================================================

# session_id -> {"file": f, "writer": w, "path": Path}
_sessions: dict[str, dict] = {}


def _open_session(session_id: str, first_tick_ts: str | None) -> dict:
    """Open a new CSV for session_id and register it in _sessions."""
    LOG_DIR.mkdir(exist_ok=True)

    # Derive wall-clock prefix from the tick's own timestamp when available
    # so the filename reflects when Jackie actually ran, not when rt_logger
    # first processed it.
    try:
        dt = datetime.fromisoformat(first_tick_ts.replace("Z", "+00:00"))
        ts = dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sid_short = session_id[:8] if len(session_id) >= 8 else session_id
    path = LOG_DIR / f"session_{ts}_{sid_short}.csv"
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    f.flush()
    print(f"[rt_logger] session {sid_short} → {path}", flush=True)
    entry = {"file": f, "writer": w, "path": path}
    _sessions[session_id] = entry
    return entry


def _get_or_open(session_id: str, tick: dict) -> dict:
    if session_id not in _sessions:
        return _open_session(session_id, tick.get("ts") or tick.get("timestamp"))
    return _sessions[session_id]


def _write_row(tick: dict) -> None:
    session_id = tick.get("session_id") or "unknown"
    entry = _get_or_open(session_id, tick)
    row = {k: tick.get(k) for k in FIELDNAMES}
    row["timestamp"] = tick.get("ts") or datetime.utcnow().isoformat()
    entry["writer"].writerow(row)
    entry["file"].flush()  # so `tail -f` sees rows immediately


# === Firebase Storage upload =================================================

_DEFAULT_BUCKET = "jackie-493900.firebasestorage.app"


def _upload_csv(path: Path) -> None:
    """Push a session CSV to Firebase Storage, overwriting any existing file."""
    bucket = (os.getenv("JACKIE_FIREBASE_BUCKET") or _DEFAULT_BUCKET).rstrip("/").split("/")[0]
    upload_url = (
        f"https://firebasestorage.googleapis.com/v0/b/"
        f"{bucket}/o?name=sessions%2F{path.name}"
    )
    if AUTH_TOKEN:
        upload_url += f"&auth={AUTH_TOKEN}"
    print(f"[rt_logger] uploading {path.name} to bucket={bucket}", flush=True)
    headers = {"Content-Type": "text/csv"}
    try:
        with open(path, "rb") as f:
            r = requests.post(
                upload_url,
                headers=headers,
                data=f,
                timeout=30,
            )
        if r.ok:
            print(f"[rt_logger] uploaded {path.name} → gs://{bucket}/sessions/",
                  flush=True)
        else:
            print(
                f"[rt_logger] Storage upload failed for {path.name}: "
                f"HTTP {r.status_code} — {r.text[:300]}",
                file=sys.stderr, flush=True,
            )
    except Exception as e:
        print(
            f"[rt_logger] Storage upload error for {path.name}: {_redact(e)}",
            file=sys.stderr, flush=True,
        )


def _close_all() -> None:
    """Flush and close every open session CSV, then upload to Firebase Storage.

    On this run: uploads every CSV in logs/ so Storage is fully in sync
    (overwrites any previous versions of the same filename).
    Going forward: only the most recently created CSV is uploaded — any
    older files are already in Storage from their own session's shutdown.
    """
    for entry in _sessions.values():
        entry["file"].flush()
        entry["file"].close()
    _sessions.clear()

    csv_files = sorted(LOG_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_files:
        return
    # Upload the most recently written CSV (the current session's file).
    # All CSVs in logs/ are uploaded on first run to back-fill Storage;
    # subsequent runs only need the newest file since older ones are already there.
    for path in csv_files:
        _upload_csv(path)


# === Firebase SSE stream =====================================================

def _stream_loop() -> None:
    url = f"{DATABASE_URL}/{COLLECTION}.json"
    if AUTH_TOKEN:
        url += f"?auth={AUTH_TOKEN}"
    headers = {"Accept": "text/event-stream"}

    while True:
        try:
            with requests.get(url, headers=headers, stream=True, timeout=None) as r:
                r.raise_for_status()
                event_type = None
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        event_type = None
                        continue
                    if raw.startswith("event:"):
                        event_type = raw.split(":", 1)[1].strip()
                    elif raw.startswith("data:"):
                        payload = raw[5:].strip()
                        if event_type not in ("put", "patch") or payload in ("null", ""):
                            continue
                        try:
                            msg = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        path  = msg.get("path", "")
                        data  = msg.get("data")
                        if data is None:
                            continue
                        if path == "/" and isinstance(data, dict):
                            # Initial snapshot — replay all existing ticks,
                            # each routed to its own session CSV.
                            for _, v in data.items():
                                if isinstance(v, dict):
                                    _write_row(v)
                        elif isinstance(data, dict):
                            # Incremental tick appended under /ticks.
                            _write_row(data)
        except (requests.RequestException, ConnectionError) as e:
            print(
                f"[rt_logger] disconnected ({_redact(e)}), reconnecting in 2s",
                file=sys.stderr, flush=True,
            )
            time.sleep(2)


# === Entrypoint ==============================================================

def main() -> None:
    if not DATABASE_URL:
        print(
            "[rt_logger] JACKIE_FIREBASE_URL is not set.\n"
            "           Add it to .env or export it before running, e.g.:\n"
            "             export JACKIE_FIREBASE_URL=https://YOUR-PROJECT-default-rtdb.firebaseio.com",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[rt_logger] streaming from {DATABASE_URL}/{COLLECTION}", flush=True)
    try:
        _stream_loop()
    except KeyboardInterrupt:
        print("\n[rt_logger] stopped — uploading session CSVs…", flush=True)
    finally:
        _close_all()


if __name__ == "__main__":
    main()
