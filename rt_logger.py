"""rt_logger.py — Stream Firebase RTDB ticks into a CSV in motion_logger.py's format.

Subscribes to your Realtime Database via Server-Sent Events and appends every
new tick to logs/session_<ts>_<id>.csv with the same columns motion_logger.py
writes. Reconnects automatically on network blips. Stop with Ctrl-C.

Setup:
    pip install requests

Run:
    python rt_logger.py

Tail the live CSV from another terminal:
    tail -f logs/session_*.csv
"""

import csv
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests # type: ignore


# === Config ==================================================================

DATABASE_URL = "https://jackie-493900-default-rtdb.firebaseio.com"
COLLECTION = "ticks"
LOG_DIR = Path(__file__).parent / "logs" # Gives you option to save logs locally to your machine

# When the rules get locked down, set this to your database secret
# (Project Settings → Service Accounts → Database secrets) and the script
# will append it as ?auth=... to the streaming URL. Leave as None for the
# current open-rules setup.
AUTH_TOKEN: str | None = None

# Same column shape as motion_logger.py. Missing keys in a tick become empty
# cells — that's intentional, so you can change Jackie's payload over time
# without breaking the consumer.
FIELDNAMES = [
    "timestamp", "session_id", "t_accum", "fsm_state",
    "face_visible", "face_cx", "dist_m", "dx_norm",
    "linear_cmd", "angular_cmd", "track_id",
    "lidar_forward_min", "obstacle_region",
]


# === CSV plumbing ============================================================

def open_csv():
    LOG_DIR.mkdir(exist_ok=True)
    sid = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"session_{ts}_{sid}.csv"
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    f.flush()
    return f, w, path


def write_row(writer, file, tick):
    """Map a Firebase tick dict into motion_logger.py's CSV row shape."""
    row = {k: tick.get(k) for k in FIELDNAMES}
    # Use the tick's own ts if present; otherwise stamp arrival time so
    # every row has a timestamp.
    row["timestamp"] = tick.get("ts") or datetime.utcnow().isoformat()
    writer.writerow(row)
    file.flush()  # so tail -f sees rows immediately


# === Firebase SSE stream =====================================================

def stream_loop(writer, file):
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
                        # Blank line = end of an SSE event.
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
                        path = msg.get("path", "")
                        data = msg.get("data")
                        if data is None:
                            continue
                        if path == "/" and isinstance(data, dict):
                            # Initial snapshot — replay every existing tick.
                            for _, v in data.items():
                                if isinstance(v, dict):
                                    write_row(writer, file, v)
                        elif isinstance(data, dict):
                            # New child appended under /ticks.
                            write_row(writer, file, data)
        except (requests.RequestException, ConnectionError) as e:
            print(
                f"[rt_logger] disconnected ({e}), reconnecting in 2s",
                file=sys.stderr, flush=True,
            )
            time.sleep(2)


# === entrypoint ==============================================================

def main():
    f, w, path = open_csv()
    print(f"[rt_logger] writing → {path}", flush=True)
    print(f"[rt_logger] streaming from {DATABASE_URL}/{COLLECTION}", flush=True)
    try:
        stream_loop(w, f)
    except KeyboardInterrupt:
        print("\n[rt_logger] stopped by user")
    finally:
        f.close()


if __name__ == "__main__":
    main()
