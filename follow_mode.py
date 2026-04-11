"""Follow Me — server-side Python port of FollowController.kt.

Port notes:
  - Kotlin → Python. Same FSM, same gains, same thresholds.
  - "Updated Rotational Motion" commit tried to add a Kalman-velocity
    feed-forward but referenced a method (`targetTrack.velocityX()`) that
    wasn't exposed on KalmanTrack, so it didn't compile. Here we estimate
    face velocity from `face_cx` history instead — same intent, works now.
  - On-device MediaPipe + DeepSORT + Kalman are gone; the server tracker
    provides an already-associated face in `obs`. Logic is unchanged.
  - OBSTACLE / COLLISION states are kept (as in the original). They have
    no real trigger yet because there's no proximity sensor in `obs` —
    Can wire them up once depth/lidar lands in `obs`.
  - Companion-object `lastTargetVelX` and `scanDirection` are now instance
    state (avoiding the thread-safety issue the Kotlin version had).

This file is hot-reloaded: edit and push, robot picks it up in ~10s.
Contract: Logic.reset() / Logic.step(obs) -> {linear, angular}.
"""

import math
from collections import deque


# ─── Thresholds (metres) ───────────────────────────────────────────────────
FOLLOW_DISTANCE_M        = 2.0    # max distance at which we try to follow
TARGET_FOLLOW_DISTANCE_M = 0.8    # desired stopping distance
COLLISION_DISTANCE_M     = 0.10   # "emergency close"

# ─── Camera geometry ───────────────────────────────────────────────────────
FRAME_WIDTH_PX  = 640             # assumed frame width 
FOCAL_LENGTH_PX = 600.0           # calibrate per camera
FACE_WIDTH_M    = 0.165           # avg adult face ~16.5cm

# ─── Rotation speeds (rad/s) ───────────────────────────────────────────────
TURN_SPEED_RAD_S = 0.4
DEG_45_RAD = math.pi / 4.0
DEG_90_RAD = math.pi / 2.0

# ─── Timing ────────────────────────────────────────────────────────────────
COLLISION_PAUSE_S = 3.0

# ─── PID gains ──────────────────────────────────────────
PAN_KP  = 0.003
PAN_KI  = 0.0001
PAN_KD  = 0.001

# DIST_KP retuned for metre-based distance error
# was calibrated against pixel² area error (magnitudes in the thousands);
# now that we control on meters (magnitudes ~0..1.5), the gain must be ~1000×
# larger to produce sensible linear speed. Rough sizing: at max error 1.2m
# we want ~0.3 m/s forward, so kp ≈ 0.25.
DIST_KP = 0.25
DIST_KI = 0.0
DIST_KD = 0.15

# ─── Feed-forward ───────────────────────────
PAN_FF_GAIN = 0.002               # applied to face_cx velocity (px/s)

# ─── Output clamps ────────────────────────
MAX_ANGULAR = 0.8
MAX_LINEAR  = 0.3


# ─── FSM states ────────────────────────────────────────────────────────────
FOLLOWING      = "FOLLOWING"
SCAN_ROTATE    = "SCAN_ROTATE"
OBSTACLE_TURN  = "OBSTACLE_TURN"
COLLISION_STOP = "COLLISION_STOP"
COLLISION_TURN = "COLLISION_TURN"
CLEAR_CHECK    = "CLEAR_CHECK"


class _PID:
    """Port of PidController.kt. Same math, nothing fancy."""
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
    """Server-side Python port of on-device FollowController."""

    def __init__(self):
        self._pan_pid = _PID(PAN_KP, PAN_KI, PAN_KD)
        self._dist_pid = _PID(DIST_KP, DIST_KI, DIST_KD)
        self._cx_history = deque(maxlen=5)     # (t_accum, cx_px) for velocity
        self.reset()

    def reset(self):
        """Called when Follow Me is turned on."""
        self._pan_pid.reset()
        self._dist_pid.reset()
        self._cx_history.clear()
        self._t_accum = 0.0

        # FSM state — start in SCAN_ROTATE like the Kotlin version
        self._fsm = SCAN_ROTATE
        self._manoeuvre_end_s = 0.0     # when current scan/turn/stop expires
        self._scan_direction = 1.0      # +1 = CCW, -1 = CW
        self._last_target_vel_x = 0.0   # remembered between face losses
        self._start_manoeuvre(DEG_45_RAD, self._scan_direction)

        self._prev_linear = 0.0
        self._prev_angular = 0.0

    # ── Main tick ──────────────────────────────────────────────────────

    def step(self, obs):
        dt = max(obs["dt"], 1e-3)
        self._t_accum += dt

        face_bounds = self._extract_face(obs)
        dist_m = self._estimate_distance(obs) if face_bounds else float("inf")

        # Update face velocity history (used by FF term)
        if face_bounds is not None:
            cx_px = face_bounds["cx_px"]
            self._cx_history.append((self._t_accum, cx_px))
            if len(self._cx_history) >= 2:
                t0, c0 = self._cx_history[0]
                t1, c1 = self._cx_history[-1]
                if t1 - t0 > 1e-3:
                    self._last_target_vel_x = (c1 - c0) / (t1 - t0)

        # FSM
        if self._fsm == FOLLOWING:
            if face_bounds is None or dist_m > FOLLOW_DISTANCE_M:
                # face lost / out of range → scan
                self._enter_state(SCAN_ROTATE)
                return self._out(0.0, 0.0)
            return self._drive_toward_face(face_bounds, dist_m, dt)

        if self._fsm == SCAN_ROTATE:
            if self._t_accum < self._manoeuvre_end_s:
                return self._out(0.0, TURN_SPEED_RAD_S * self._scan_direction)
            # manoeuvre done — check if we have a face now
            if face_bounds is not None and dist_m <= FOLLOW_DISTANCE_M:
                self._enter_state(FOLLOWING)
                return self._out(0.0, 0.0)
            # no face → another 45° scan, same direction
            self._start_manoeuvre(DEG_45_RAD, self._scan_direction)
            return self._out(0.0, 0.0)

        if self._fsm == OBSTACLE_TURN:
            if self._t_accum < self._manoeuvre_end_s:
                return self._out(0.0, -TURN_SPEED_RAD_S)   # CW
            self._enter_state(CLEAR_CHECK)
            return self._out(0.0, 0.0)

        if self._fsm == CLEAR_CHECK:
            # No proximity sensor data yet — just resume following
            self._enter_state(FOLLOWING)
            return self._out(0.0, 0.0)

        if self._fsm == COLLISION_STOP:
            if self._t_accum >= self._manoeuvre_end_s:
                self._enter_state(COLLISION_TURN)
            return self._out(0.0, 0.0)

        if self._fsm == COLLISION_TURN:
            if self._t_accum < self._manoeuvre_end_s:
                return self._out(0.0, -TURN_SPEED_RAD_S)   # CW
            self._enter_state(CLEAR_CHECK)
            return self._out(0.0, 0.0)

        # unknown state — be safe
        return self._out(0.0, 0.0)

    # ── PID drive toward face (meters-based) ─────

    def _drive_toward_face(self, face_bounds, dist_m, dt):
        # Lateral error in PIXELS
        dx = face_bounds["cx_px"] - FRAME_WIDTH_PX / 2.0

        # Distance error in METRES
        dist_error = dist_m - TARGET_FOLLOW_DISTANCE_M

        # PID outputs
        ang_z = self._pan_pid.compute(dx, dt)
        lin_x = self._dist_pid.compute(dist_error, dt)

        # Velocity feed-forward — if the person is drifting right (+vel_x),
        # add extra rotation.
        vel_ff = self._last_target_vel_x * PAN_FF_GAIN
        vel_ff = max(-0.3, min(0.3, vel_ff))
        ang_z += vel_ff

        # Sign flip — Jackie's /cmd_vel_mux/input/navi_override expects
        # negative angular.z to rotate toward a user on the right of frame.
        ang_z = -ang_z

        # Final clamp — applied AFTER the feed-forward so MAX_ANGULAR is the
        # true hard ceiling.
        ang_z = max(-MAX_ANGULAR, min(MAX_ANGULAR, ang_z))
        lin_x = max(-MAX_LINEAR,  min(MAX_LINEAR,  lin_x))

        # Safety: stop linear motion when very close
        if dist_m < COLLISION_DISTANCE_M + 0.05:
            lin_x = 0.0

        return self._out(lin_x, ang_z)

    # ── State transitions ─────────────────────────────────────────────────

    def _enter_state(self, next_state):
        self._fsm = next_state
        if next_state == SCAN_ROTATE:
            # Rotate toward direction the target was last moving
            self._scan_direction = 1.0 if self._last_target_vel_x >= 0 else -1.0
            self._start_manoeuvre(DEG_45_RAD, self._scan_direction)
        elif next_state == OBSTACLE_TURN:
            self._start_manoeuvre(DEG_90_RAD, -1.0)
        elif next_state == COLLISION_STOP:
            self._manoeuvre_end_s = self._t_accum + COLLISION_PAUSE_S
        elif next_state == COLLISION_TURN:
            self._start_manoeuvre(DEG_45_RAD, -1.0)
        # CLEAR_CHECK and FOLLOWING have no setup

    def _start_manoeuvre(self, angle_rad, direction):
        self._scan_direction = direction
        duration_s = angle_rad / TURN_SPEED_RAD_S
        self._manoeuvre_end_s = self._t_accum + duration_s

    # ── Helpers ───────────────────────────────────────────────────────────

    def _extract_face(self, obs):
        """Return dict with cx_px + w_px if face visible, else None."""
        if not obs.get("face_visible") or obs.get("face_cx") is None:
            return None
        frame_w = obs.get("frame_width_px") or FRAME_WIDTH_PX

        # Prefer the server-provided pixel width; fall back to area-derived.
        w_px = obs.get("face_w_px")
        if w_px is None:
            area = obs.get("face_area") or 0
            w_px = math.sqrt(max(area, 1) / 1.3)   # assume 1.3 aspect ratio

        cx_px = obs["face_cx"] * frame_w
        return {"cx_px": cx_px, "w_px": w_px}

    def _estimate_distance(self, obs):
        """Pinhole distance model — estimateDistance()."""
        face_bounds = self._extract_face(obs)
        if face_bounds is None:
            return float("inf")
        w_px = face_bounds["w_px"]
        if w_px <= 0:
            return float("inf")
        return (FACE_WIDTH_M * FOCAL_LENGTH_PX) / w_px

    def _out(self, linear, angular):
        self._prev_linear = linear
        self._prev_angular = angular
        return {"linear": linear, "angular": angular}
