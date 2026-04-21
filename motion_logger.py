import csv
import uuid
from datetime import datetime
from pathlib import Path

class MotionLogger:
    def __init__(self, log_dir="logs"):
        Path(log_dir).mkdir(exist_ok=True)
        self.session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path(log_dir) / f"session_{timestamp}_{self.session_id}.csv"

        self._file = open(filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=[
            "timestamp", "session_id", "t_accum", "fsm_state",
            "face_visible", "face_cx", "dist_m", "dx_norm",
            "linear_cmd", "angular_cmd", "track_id",
            "lidar_forward_min", "obstacle_region",
        ])
        self._writer.writeheader()

    def log_tick(self, row: dict):
        row["timestamp"] = datetime.utcnow().isoformat()
        row["session_id"] = self.session_id
        self._writer.writerow(row)

    def log_state_change(self, from_state: str, to_state: str, t_accum: float):
        # Optional — write a marker row for FSM transitions
        self._writer.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "t_accum": t_accum,
            "fsm_state": f"{from_state}→{to_state}",
        })

    def close(self):
        self._file.flush()
        self._file.close()