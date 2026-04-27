"""tick_simulator.py — POST synthetic ticks to Firebase RTDB at 10 Hz.

Fakes the data Jackie would publish so you can verify rt_logger.py end-to-end
without touching Jackie's actual code. Every tick gets a smoothly-drifting
face position, a derived angular command, and a slow linear ramp so the
resulting CSV looks plausible when you tail it.

Setup:
    pip install requests

Run:
    python tick_simulator.py            # 10 Hz forever
    python tick_simulator.py --rate 5   # 5 Hz
    python tick_simulator.py --count 100  # stop after 100 ticks

Stop with Ctrl-C. This is a TESTING tool — do not run it against a database
that's also receiving real Jackie data, or you'll pollute your logs.
"""

import argparse
import math
import sys
import time
import uuid
from datetime import datetime, timezone

import requests  # type: ignore


# === Config ==================================================================

DATABASE_URL = "https://jackie-493900-default-rtdb.firebaseio.com"
COLLECTION = "ticks"

# Same story as rt_logger.py: leave None while rules are open, set to your
# database secret once you lock the rules to `auth != null`.
AUTH_TOKEN: str | None = None


# === Synthetic tick generator ================================================

def make_tick(session_id: str, i: int, t_accum: float) -> dict:
    """Build a fake tick that mirrors what face_user / follow_mode would emit."""
    # Smooth sinusoidal face drift across the frame (0..1 normalized).
    face_cx = 0.5 + 0.35 * math.sin(t_accum * 0.6)
    dx_norm = face_cx - 0.5            # error from frame center
    angular_cmd = -1.2 * dx_norm       # fake P controller
    linear_cmd = 0.15 + 0.05 * math.sin(t_accum * 0.2)  # gentle forward bob
    dist_m = 1.2 + 0.3 * math.cos(t_accum * 0.3)        # ~1.2m ± 0.3m

    # Cycle through FSM states so the CSV shows variety.
    fsm_states = ["SEARCH", "ACQUIRE", "TRACK", "TRACK", "TRACK"]
    fsm_state = fsm_states[i % len(fsm_states)]

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "skill": "tick_sim",            # so simulated data is distinguishable
        "t_accum": round(t_accum, 3),
        "fsm_state": fsm_state,
        "face_visible": True,
        "face_cx": round(face_cx, 4),
        "dist_m": round(dist_m, 3),
        "dx_norm": round(dx_norm, 4),
        "linear_cmd": round(linear_cmd, 4),
        "angular_cmd": round(angular_cmd, 4),
        "track_id": 1,
        "lidar_forward_min": round(2.5 + 0.4 * math.sin(t_accum * 0.4), 3),
        "obstacle_region": "none",
    }


# === POST loop ===============================================================

def post_loop(rate_hz: float, count: int | None):
    url = f"{DATABASE_URL}/{COLLECTION}.json"
    if AUTH_TOKEN:
        url += f"?auth={AUTH_TOKEN}"

    session_id = uuid.uuid4().hex[:8]
    period = 1.0 / rate_hz
    print(f"[tick_sim] session_id={session_id}", flush=True)
    print(f"[tick_sim] POSTing to {url} at {rate_hz:.1f} Hz", flush=True)
    if count:
        print(f"[tick_sim] will stop after {count} ticks", flush=True)

    i = 0
    t_accum = 0.0
    next_tick = time.monotonic()
    while True:
        tick = make_tick(session_id, i, t_accum)
        try:
            r = requests.post(url, json=tick, timeout=2.0)
            if not r.ok:
                print(f"[tick_sim] POST failed {r.status_code}: {r.text[:120]}",
                      file=sys.stderr, flush=True)
        except requests.RequestException as e:
            print(f"[tick_sim] POST error: {e}", file=sys.stderr, flush=True)

        i += 1
        t_accum += period
        if count and i >= count:
            print(f"[tick_sim] sent {i} ticks, exiting", flush=True)
            return

        # Sleep to next tick boundary; if we fell behind, skip ahead so we
        # don't accumulate lag forever.
        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()


# === entrypoint ==============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rate", type=float, default=10.0,
                    help="ticks per second (default: 10)")
    ap.add_argument("--count", type=int, default=None,
                    help="stop after N ticks (default: run forever)")
    args = ap.parse_args()

    try:
        post_loop(args.rate, args.count)
    except KeyboardInterrupt:
        print("\n[tick_sim] stopped by user")


if __name__ == "__main__":
    main()
