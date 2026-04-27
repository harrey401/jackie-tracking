import json
import os
import queue
import threading
import uuid
from datetime import datetime
from pathlib import Path

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None  # cloud features become no-ops if requests isn't installed

"""
    A logger for recording all of Jackie's actions during a session where it is active.
    Logs are stored in dictionary form, as can be seen in _out in follow_mode.py.
    Initially, CSV files were considered as the database output, but since dictionaries
    closely resemble JSON objects, JSONL (JSON Lines) format was chosen for its simplicity
    and efficiency in handling large datasets, with the ability to add on more data and info
    using nested keys and values, so now, all of Jackie's actions are written in JSON files.
    JSON has already been used per Jackie's documentation using Java and Kotlin, so it was
    a natural choice to continue using it for the Python implementation as well, ensuring
    consistency across different parts of the project.

    When communicating with the robot and a server, JSON is more organized via key-value pairs,
    and better shows the structure of the data and allows for future nested structured data.

    Reference:
    Swarat. (2023, October 27). *Working with CSV and JSON in Python*.
    Medium. https://medium.com/@swaratvaghela30112003/working-with-csv-and-json-in-python-fc88e49c1c1d


    Cloud dual-write (added):
    -------------------------
    On top of the local JSONL, MotionLogger can also (a) live-stream every row
    to Firebase Realtime Database, and (b) upload the completed JSONL file to
    Firebase Storage when close() is called. Both are opt-in and best-effort —
    if the network drops or requests is not installed, the local file is still
    written normally and Jackie's hot loop is never blocked.

    Enable via either constructor args or environment variables (env vars
    win when no constructor arg is passed, so existing call sites in
    follow_mode.py keep working unchanged):

        JACKIE_FIREBASE_URL    e.g. https://jackie-493900-default-rtdb.firebaseio.com
        JACKIE_FIREBASE_TOKEN  optional auth=… token for locked-down rules
        JACKIE_FIREBASE_BUCKET e.g. jackie-493900.appspot.com

    The RTDB writes go to /ticks (matching what rt_logger.py listens on).
    The Storage upload goes to sessions/<filename>.jsonl in the bucket.
"""


# Background-worker config — tuned for 10 Hz. 2400 rows ≈ 240 s of buffer,
# which is plenty of headroom for short network blips.
_QUEUE_MAX = 2400
_POST_TIMEOUT_S = 2.0
_UPLOAD_TIMEOUT_S = 30.0
_JOIN_TIMEOUT_S = 5.0


class MotionLogger:
    def __init__(
        self,
        log_dir: str = "logs",
        firebase_url: str | None = None,
        firebase_token: str | None = None,
        firebase_bucket: str | None = None,
    ):
        Path(log_dir).mkdir(exist_ok=True)
        self.session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = Path(log_dir) / f"session_{timestamp}_{self.session_id}.jsonl"

        self._file = open(self._filepath, "w")

        # Cloud config — explicit args win, otherwise fall back to env vars
        # so follow_mode.py doesn't need to change.
        self._url = firebase_url or os.getenv("JACKIE_FIREBASE_URL")
        self._token = firebase_token or os.getenv("JACKIE_FIREBASE_TOKEN")
        self._bucket = firebase_bucket or os.getenv("JACKIE_FIREBASE_BUCKET")
        if requests is None:
            # No requests → silently disable cloud paths but keep local writes.
            self._url = None
            self._bucket = None

        # Background POST worker so the 10 Hz loop never blocks on network IO.
        # Hot loop only does queue.put_nowait() (microseconds); the worker
        # thread does the actual HTTP requests.
        self._q: "queue.Queue[dict | None]" = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker: threading.Thread | None = None
        if self._url:
            self._worker = threading.Thread(
                target=self._drain, name="MotionLogger-rtdb", daemon=True
            )
            self._worker.start()

    # ---- local writes -------------------------------------------------------

    def _write(self, row: dict):
        self._file.write(json.dumps(row) + "\n")

    def log_tick(self, row: dict):
        row["timestamp"] = datetime.utcnow().isoformat()
        row["session_id"] = self.session_id
        self._write(row)
        self._enqueue(row)

    def log_state_change(self, from_state: str, to_state: str, t_accum: float):
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "t_accum": t_accum,
            "fsm_state": f"{from_state}→{to_state}",
        }
        self._write(row)
        self._enqueue(row)

    # ---- cloud (RTDB live stream) ------------------------------------------

    def _enqueue(self, row: dict):
        if not self._url:
            return
        try:
            self._q.put_nowait(row)
        except queue.Full:
            # Queue is at capacity (sustained network outage). Drop the row
            # rather than block Jackie. The local JSONL still has everything.
            pass

    def _drain(self):
        """Worker loop: pull rows off the queue and POST them to RTDB."""
        url = f"{self._url.rstrip('/')}/ticks.json"
        if self._token:
            url += f"?auth={self._token}"
        while True:
            row = self._q.get()
            if row is None:  # shutdown sentinel from close()
                return
            try:
                requests.post(url, json=row, timeout=_POST_TIMEOUT_S)
            except Exception:
                # Never crash the worker; the row is just dropped from cloud.
                # Local JSONL still has it.
                pass

    # ---- cloud (file archive on close) -------------------------------------

    def _upload_file(self):
        """Push the completed JSONL to Firebase Storage as the durable record."""
        if not (self._bucket and requests):
            return
        url = (
            f"https://firebasestorage.googleapis.com/v0/b/"
            f"{self._bucket}/o?name=sessions/{self._filepath.name}"
        )
        try:
            with open(self._filepath, "rb") as f:
                requests.post(
                    url,
                    headers={"Content-Type": "application/x-ndjson"},
                    data=f,
                    timeout=_UPLOAD_TIMEOUT_S,
                )
        except Exception:
            # Best-effort. Local file is still on disk for manual upload.
            pass

    # ---- shutdown -----------------------------------------------------------

    def close(self):
        self._file.flush()
        self._file.close()
        if self._worker:
            # Drain any buffered rows before quitting — sentinel goes to the
            # back of the queue, so everything ahead of it gets POSTed first.
            self._q.put(None)
            self._worker.join(timeout=_JOIN_TIMEOUT_S)
        self._upload_file()
