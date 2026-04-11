"""Face the User — rotation-only face tracking (server-side Python).

Matches the conventions of follow_mode.py so switching between files
feels natural: same `_PID` class, same gain names (`PAN_KP/KI/KD`,
`PAN_FF_GAIN`), same pixel-space error, same frame geometry, same
`Logic.reset() / Logic.step(obs)` contract.

The difference from follow_mode.py:
  - rotation only — linear is always 0, no driving, no distance PID
  - no FSM — there's only one behaviour: face the user
  - no scan when face is lost — just stop rotating (intentional: this
    skill shouldn't spin looking for people)

Contract: Logic.reset() / Logic.step(obs) -> {linear, angular}.
"""

from collections import deque


# ─── Camera geometry (match follow_mode.py) ────────────────────────────────
FRAME_WIDTH_PX = 640

# ─── Pan PID gains (same magnitudes as follow_mode.py) ─────────────────────
PAN_KP = 0.003
PAN_KI = 0.0001
PAN_KD = 0.001

# ─── Velocity feed-forward (same name as follow_mode.py) ──────────────────
PAN_FF_GAIN = 0.002

# ─── Output clamp ──────────────────────────────────────────────────────────
MAX_ANGULAR = 0.8


class _PID:
    """Same PID as in follow_mode.py. Ported from PidController.kt."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error, dt):
        self._integral += error * dt
        deriv = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * deriv

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class Logic:
    def __init__(self):
        self._pan_pid = _PID(PAN_KP, PAN_KI, PAN_KD)
        self._cx_history = deque(maxlen=5)      # (t_accum, cx_px) for velocity FF
        self.reset()

    def reset(self):
        """Called when Face-the-User is turned on."""
        self._pan_pid.reset()
        self._cx_history.clear()
        self._t_accum = 0.0
        self._last_target_vel_x = 0.0

    def step(self, obs):
        dt = max(obs["dt"], 1e-3)
        self._t_accum += dt

        # No face → stop rotating. No scanning.
        if not obs.get("face_visible") or obs.get("face_cx") is None:
            return {"linear": 0.0, "angular": 0.0}

        frame_w = obs.get("frame_width_px") or FRAME_WIDTH_PX
        cx_px = obs["face_cx"] * frame_w

        # Update velocity history for feed-forward
        self._cx_history.append((self._t_accum, cx_px))
        if len(self._cx_history) >= 2:
            t0, c0 = self._cx_history[0]
            t1, c1 = self._cx_history[-1]
            if t1 - t0 > 1e-3:
                self._last_target_vel_x = (c1 - c0) / (t1 - t0)

        # Pan PID on pixel error (same scale as follow_mode.py)
        dx = cx_px - frame_w / 2.0
        ang_z = self._pan_pid.compute(dx, dt)

        # Velocity feed-forward — lead the moving target
        vel_ff = self._last_target_vel_x * PAN_FF_GAIN
        vel_ff = max(-0.3, min(0.3, vel_ff))
        ang_z += vel_ff

        # Final clamp — applied AFTER the feed-forward so MAX_ANGULAR is
        # the true hard ceiling.
        ang_z = max(-MAX_ANGULAR, min(MAX_ANGULAR, ang_z))

        return {"linear": 0.0, "angular": ang_z}
