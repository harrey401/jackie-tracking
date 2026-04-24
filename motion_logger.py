import json
import uuid
from datetime import datetime
from pathlib import Path

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

"""



class MotionLogger:
    def __init__(self, log_dir="logs"):
        Path(log_dir).mkdir(exist_ok=True)
        self.session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path(log_dir) / f"session_{timestamp}_{self.session_id}.jsonl"

        self._file = open(filepath, "w")

    def _write(self, row: dict):
        self._file.write(json.dumps(row) + "\n")

    def log_tick(self, row: dict):
        row["timestamp"] = datetime.utcnow().isoformat()
        row["session_id"] = self.session_id
        self._write(row)

    def log_state_change(self, from_state: str, to_state: str, t_accum: float):
        self._write({
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "t_accum": t_accum,
            "fsm_state": f"{from_state}→{to_state}",
        })

    def close(self):
        self._file.flush()
        self._file.close()