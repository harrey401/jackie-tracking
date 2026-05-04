"""tick_simulator.py — POST ticks to Firebase RTDB at 10 Hz.

Synthetic mode (default):
    Generates smoothly-drifting sinusoidal values so you can verify
    rt_logger.py end-to-end without running Jackie. Do NOT run synthetic
    mode against a database that is also receiving real Jackie data.

Live mode (--live):
    Subscribes to Jackie's ROS topics and POSTs real sensor values:
      - face_cx, dx_norm, dist_m  ← JACKIE_FACE_TOPIC (std_msgs/String JSON)
      - linear_cmd, angular_cmd   ← JACKIE_CMDVEL_TOPIC (Twist)
      - lidar_forward_min         ← JACKIE_SCAN_TOPIC  (LaserScan, ±25° arc)
    Override any topic name via the corresponding env var (defaults below).

    Expected face topic payload — a JSON string published on JACKIE_FACE_TOPIC:
        {"face_cx": 0.52, "face_visible": true, "dist_m": 1.4}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN THE FULL FIREBASE LOGGING PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prerequisites — put these in .env (gitignored) and source before running:

    export JACKIE_FIREBASE_URL=https://<project>-default-rtdb.firebaseio.com
    export JACKIE_FIREBASE_TOKEN=<auth token>             # optional
    export JACKIE_FIREBASE_BUCKET=<project>.appspot.com   # optional (file archive)

Step 1 — Jackie side  (motion_logger.py is auto-imported by follow_mode.py):
    # MotionLogger writes a local JSONL file AND live-streams every tick to
    # Firebase RTDB in a background thread — Jackie's hot loop is never blocked.
    set -a; source .env; set +a
    python follow_mode.py   # or via smait/navigation/tracking_loader.py

Step 2 — Monitor side  (any machine with Firebase network access):
    set -a; source .env; set +a
    python rt_logger.py
    # Live CSV tail in a second terminal:
    tail -f logs/session_*.csv

Testing without Jackie (synthetic sinusoidal data):
    set -a; source .env; set +a
    python tick_simulator.py             # 10 Hz, run forever
    python tick_simulator.py --rate 5    # 5 Hz
    python tick_simulator.py --count 50  # stop after 50 ticks

Testing with Jackie running (real ROS data):
    set -a; source .env; set +a
    python tick_simulator.py --live      # subscribes to ROS topics (requires rospy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Setup:
    pip install requests

Stop with Ctrl-C.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import requests  # type: ignore


def _redact(text: str) -> str:
    """Strip ?auth=<token> from anything we're about to print/log."""
    return re.sub(r"([?&])auth=[^&\s]+", r"\1auth=<redacted>", str(text))


# === Config ==================================================================

# Pulled from environment so secrets stay out of git. Set them in .env
# (gitignored) and source it before running.
DATABASE_URL = os.getenv("JACKIE_FIREBASE_URL", "")
COLLECTION   = os.getenv("JACKIE_FIREBASE_COLLECTION", "ticks")
AUTH_TOKEN: str | None = os.getenv("JACKIE_FIREBASE_TOKEN") or None

# ROS topic names used in --live mode. Override any of these via env vars.
_CMDVEL_TOPIC    = os.getenv("JACKIE_CMDVEL_TOPIC",    "/cmd_vel_mux/input/navi_override")
_SCAN_TOPIC      = os.getenv("JACKIE_SCAN_TOPIC",      "/scan")
_FACE_TOPIC      = os.getenv("JACKIE_FACE_TOPIC",      "/jackie/face_obs")  # std_msgs/String JSON
_OBSTACLE_TOPIC  = os.getenv("JACKIE_OBSTACLE_TOPIC",  "/obstacle_region")  # std_msgs/Int8

# Forward LIDAR arc for lidar_forward_min — ±25° matches follow_mode.py's
# LIDAR_BUBBLE_ARC_RAD so the logged value is comparable.
_LIDAR_FWD_ARC   = math.radians(25)
_LIDAR_MIN_VALID = 0.05  # reject self-hit returns below this (metres)


# === Live-mode ROS state =====================================================

class _RosState:
    """Thread-safe latest-value store populated by ROS topic callbacks.

    Holds plausible defaults so the first few ticks are valid even before
    a message has arrived on a given topic.
    """

    def __init__(self):
        self._lock              = threading.Lock()
        self.face_cx:           float = 0.5
        self.face_visible:      bool  = False
        self.dist_m:            float = 1.2
        self.linear_cmd:        float = 0.0
        self.angular_cmd:       float = 0.0
        self.lidar_forward_min: float = math.inf
        self.obstacle_region:   int   = 0

    # ------------------------------------------------------------------
    def on_cmdvel(self, msg) -> None:
        with self._lock:
            self.linear_cmd  = msg.linear.x
            self.angular_cmd = msg.angular.z

    def on_scan(self, msg) -> None:
        """Compute minimum range inside the ±25° forward arc from a LaserScan."""
        a0, da = msg.angle_min, msg.angle_increment
        rmax   = msg.range_max
        best   = math.inf
        for i, r in enumerate(msg.ranges):
            theta = a0 + i * da
            if not (-_LIDAR_FWD_ARC <= theta <= _LIDAR_FWD_ARC):
                continue
            if not math.isfinite(r) or r < _LIDAR_MIN_VALID or r > rmax:
                continue
            if r < best:
                best = r
        with self._lock:
            self.lidar_forward_min = best

    def on_face(self, msg) -> None:
        """Parse a std_msgs/String JSON payload carrying face observations."""
        try:
            data = json.loads(msg.data)
            with self._lock:
                self.face_cx      = float(data.get("face_cx",      self.face_cx))
                self.face_visible = bool( data.get("face_visible", False))
                self.dist_m       = float(data.get("dist_m",       self.dist_m))
        except Exception:
            pass

    def on_obstacle(self, msg) -> None:
        """Record the latest chassis obstacle-region signal (std_msgs/Int8)."""
        with self._lock:
            self.obstacle_region = int(msg.data)

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Return a point-in-time copy of all fields for make_tick()."""
        with self._lock:
            face_cx = self.face_cx
            fwd     = self.lidar_forward_min
            return {
                "face_cx":           face_cx,
                "dx_norm":           face_cx - 0.5,
                "face_visible":      self.face_visible,
                "dist_m":            self.dist_m,
                "linear_cmd":        self.linear_cmd,
                "angular_cmd":       self.angular_cmd,
                "lidar_forward_min": fwd if math.isfinite(fwd) else None,
                "obstacle_region":   self.obstacle_region,
            }


def _start_ros(state: _RosState) -> bool:
    """Initialise rospy and subscribe to Jackie's topics. Returns True on success."""
    try:
        import rospy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import String
        from std_msgs.msg import Int8
        rospy.init_node("jackie_tick_sim", anonymous=True, disable_signals=True)
        rospy.Subscriber(_CMDVEL_TOPIC,   Twist,     state.on_cmdvel,  queue_size=1)
        rospy.Subscriber(_SCAN_TOPIC,     LaserScan, state.on_scan,    queue_size=1)
        rospy.Subscriber(_OBSTACLE_TOPIC, Int8,      state.on_obstacle, queue_size=1)
        if _FACE_TOPIC:
            rospy.Subscriber(_FACE_TOPIC, String, state.on_face, queue_size=1)
        print(f"[tick_sim] ROS live: cmd_vel={_CMDVEL_TOPIC}  scan={_SCAN_TOPIC}"
              f"  obstacle={_OBSTACLE_TOPIC}", flush=True)
        if _FACE_TOPIC:
            print(f"[tick_sim] ROS live: face={_FACE_TOPIC}", flush=True)
        return True
    except Exception as e:
        print(
            f"[tick_sim] ROS unavailable ({e}) — ROS fields will use default values",
            file=sys.stderr, flush=True,
        )
        return False


# === Tick builder ============================================================

def make_tick(session_id: str, i: int, t_accum: float,
              live: dict | None = None) -> dict:
    """Build one tick dict.

    When `live` is a snapshot dict from _RosState, real sensor values from
    Jackie's ROS topics are used.  Otherwise sinusoidal synthetic values are
    generated (testing only — see module docstring).
    """
    if live is not None:
        face_cx           = live["face_cx"]
        dx_norm           = live["dx_norm"]
        angular_cmd       = live["angular_cmd"]
        linear_cmd        = live["linear_cmd"]
        dist_m            = live["dist_m"]
        lidar_forward_min = live["lidar_forward_min"]
        face_visible      = live["face_visible"]
        obstacle_region   = live.get("obstacle_region", 0)
        skill             = "tick_sim_live"
    else:
        # Sinusoidal base trajectory with per-channel sensor noise that
        # approximates real Jackie hardware characteristics.
        #
        # face_cx:           ±~10 px detector jitter on 640 px frame → σ=0.015 norm
        # dist_m:            vision width estimate is noisy → σ=0.035 m
        # lidar_forward_min: RPLIDAR ±2 cm ranging noise → σ=0.018 m
        # linear_cmd:        small velocity tracking error → σ=0.004 m/s
        # angular_cmd:       inherits face_cx noise + motor latency → σ=0.008 rad/s
        # face_visible:      ~4 % dropout rate for brief detection misses
        face_visible      = random.random() > 0.04
        face_cx_base      = 0.5 + 0.35 * math.sin(t_accum * 0.6)
        face_cx           = max(0.0, min(1.0, face_cx_base + random.gauss(0, 0.015)))
        dx_norm           = face_cx - 0.5
        angular_cmd       = -1.2 * dx_norm + random.gauss(0, 0.008)
        linear_cmd        = 0.15 + 0.05 * math.sin(t_accum * 0.2) + random.gauss(0, 0.004)
        dist_m            = max(0.1, 1.2 + 0.3 * math.cos(t_accum * 0.3) + random.gauss(0, 0.035))
        lidar_forward_min = max(0.05, 2.5 + 0.4 * math.sin(t_accum * 0.4) + random.gauss(0, 0.018))
        # Derive obstacle region from LiDAR: 1 = obstacle within slow-down
        # threshold (mirrors LIDAR_SLOW_RANGE_M in follow_mode.py), 0 = clear.
        obstacle_region   = 1 if lidar_forward_min < 1.0 else 0
        skill             = "tick_sim"

    fsm_states = ["SEARCH", "ACQUIRE", "TRACK", "TRACK", "TRACK"]

    return {
        "ts":               datetime.now(timezone.utc).isoformat(),
        "session_id":       session_id,
        "skill":            skill,
        "t_accum":          round(t_accum, 3),
        "fsm_state":        fsm_states[i % len(fsm_states)],
        "face_visible":     face_visible,
        "face_cx":          round(float(face_cx), 4),
        "dist_m":           round(float(dist_m), 3),
        "dx_norm":          round(float(dx_norm), 4),
        "linear_cmd":       round(float(linear_cmd), 4),
        "angular_cmd":      round(float(angular_cmd), 4),
        "track_id":         1,
        "lidar_forward_min": (
            round(float(lidar_forward_min), 3)
            if lidar_forward_min is not None else None
        ),
        "obstacle_region":  obstacle_region,
    }


# === POST loop ===============================================================

def post_loop(rate_hz: float, count: int | None, live: bool = False):
    url = f"{DATABASE_URL}/{COLLECTION}.json"
    if AUTH_TOKEN:
        url += f"?auth={AUTH_TOKEN}"

    ros_state: _RosState | None = None
    if live:
        ros_state = _RosState()
        _start_ros(ros_state)

    session_id = uuid.uuid4().hex[:8]
    period     = 1.0 / rate_hz
    mode_label = "LIVE (ROS)" if live else "SYNTHETIC"
    print(f"[tick_sim] session={session_id}  mode={mode_label}  rate={rate_hz:.1f} Hz",
          flush=True)
    print(f"[tick_sim] POSTing to {_redact(url)}", flush=True)
    if count:
        print(f"[tick_sim] will stop after {count} ticks", flush=True)

    i         = 0
    t_accum   = 0.0
    next_tick = time.monotonic()

    while True:
        snap = ros_state.snapshot() if ros_state is not None else None
        tick = make_tick(session_id, i, t_accum, live=snap)
        try:
            r = requests.post(url, json=tick, timeout=2.0)
            if not r.ok:
                print(f"[tick_sim] POST failed {r.status_code}: {r.text[:120]}",
                      file=sys.stderr, flush=True)
        except requests.RequestException as e:
            print(f"[tick_sim] POST error: {_redact(e)}", file=sys.stderr, flush=True)

        i       += 1
        t_accum += period
        if count and i >= count:
            print(f"[tick_sim] sent {i} ticks, exiting", flush=True)
            return

        # Sleep to next tick boundary; skip ahead if we fell behind so we
        # don't accumulate lag forever.
        next_tick += period
        sleep_for  = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()


# === Entrypoint ==============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rate",  type=float, default=10.0,
                    help="ticks per second (default: 10)")
    ap.add_argument("--count", type=int,   default=None,
                    help="stop after N ticks (default: run forever)")
    ap.add_argument("--live",  action="store_true",
                    help="subscribe to ROS topics for real sensor data "
                         "(requires rospy and a running ROS master)")
    args = ap.parse_args()

    if not DATABASE_URL:
        print(
            "[tick_sim] JACKIE_FIREBASE_URL is not set.\n"
            "           Add it to .env or export it before running, e.g.:\n"
            "             export JACKIE_FIREBASE_URL=https://YOUR-PROJECT-default-rtdb.firebaseio.com",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        post_loop(args.rate, args.count, live=args.live)
    except KeyboardInterrupt:
        print("\n[tick_sim] stopped by user")


if __name__ == "__main__":
    main()
