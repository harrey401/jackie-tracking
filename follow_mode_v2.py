"""Follow-me — v2 reference implementation (upgraded, event-grade).

Same contract as follow_mode.py. Drives Jackie toward a person and faces them.

Upgrades over v1:
  - Full PID on both pan (angular) and distance (linear)
  - Distance estimate derived from face_w_norm, NOT face_area
    (width is 1D and far more stable than w×h under head tilt / hair)
  - Separate inverse-distance model for linear gain (not a raw pixel²)
  - Motion coupling: reduce linear speed when angular error is large
    (don't drive forward while turning hard — dignified approach)
  - Reverse gear: back up if user walks too close
  - Distance hysteresis: don't oscillate at the stopping threshold
  - Velocity feed-forward on pan (lead moving users)
  - Input low-pass filters on face_cx AND face_w_norm
  - Size-adaptive deadzone on pan
  - Output slew-rate limits on both linear and angular
  - Graceful lost-face decay (ramp to zero, don't hard-stop)
  - Detection quality gate (reject tiny-face noise)

See UPGRADES.md for the reasoning behind each change.
"""

# === TUNABLES ==================================================================

# --- Pan (rotation) PID ---
PAN_KP          = 1.4
PAN_KI          = 0.25
PAN_KD          = 0.22
PAN_I_LIMIT     = 0.4
PAN_DEADZONE_BASE   = 0.03
PAN_DEADZONE_SCALE  = 0.08
VELOCITY_FF_GAIN    = 0.5   # pan feed-forward from face velocity
VELOCITY_WINDOW     = 5

# --- Distance PID (operates on meters, via face-width model) ---
# We convert face_w_norm → estimated distance in "head widths".
# A head is ~16cm wide. At face_w_norm=0.25 the face fills a quarter
# of the frame → user is ~0.7m away on a 60° HFOV camera. Calibrate on Jackie.
TARGET_DISTANCE_M   = 0.9    # desired stopping distance (meters-ish)
STOP_HYSTERESIS_M   = 0.15   # don't re-engage until error exceeds this
HEAD_WIDTH_M        = 0.16   # assumed head width in meters
FOCAL_LENGTH_PX_EST = 500.0  # f = (px_width * distance) / real_width, calibrate

DIST_KP         = 0.6
DIST_KI         = 0.15
DIST_KD         = 0.05
DIST_I_LIMIT    = 0.3

# --- Input smoothing ---
INPUT_EMA_CX    = 0.35   # EMA alpha for face_cx
INPUT_EMA_W     = 0.55   # stronger filter on width (noisier)

# --- Speed caps + slew rate ---
MAX_LINEAR          = 0.25
MAX_ANGULAR         = 1.0
MAX_LINEAR_ACCEL    = 0.4    # m/s² — gentle starts
MAX_ANGULAR_ACCEL   = 3.5    # rad/s²

# --- Chassis friction floors ---
# Jackie's /cmd_vel_mux/input/navi_override ignores commands below these
# magnitudes (the wheels just don't break static friction). Floored only
# when motion is actually wanted (outside deadzone / outside hysteresis).
MIN_ANGULAR_KICK    = 0.22   # rad/s
MIN_LINEAR_KICK     = 0.08   # m/s

# --- Motion coupling: if angular error is big, slow down linear ---
# linear_scale = max(COUPLING_MIN, 1 - coupling_k * |pan_error|)
COUPLING_K      = 3.0
COUPLING_MIN    = 0.1  # never drop below 10% of commanded linear

# --- Detection gate ---
MIN_FACE_W_NORM = 0.03

# --- Lost-face behaviour ---
LOST_HOLD_S     = 0.6
LOST_DECAY_S    = 0.8
# NOTE: scanning intentionally disabled. If Jackie loses the user, it stops.
# Add it back here if you want, guarded by a time limit + low speed.


# === LOGIC ======================================================================

from collections import deque
import math

class Logic:
    def __init__(self):
        self.reset()

    def reset(self):
        self._cx_filtered = None
        self._w_filtered = None
        self._cx_history = deque(maxlen=VELOCITY_WINDOW)

        self._pan_i = 0.0
        self._dist_i = 0.0
        self._prev_cx = None
        self._prev_dist = None

        self._prev_linear = 0.0
        self._prev_angular = 0.0

        self._at_target = False  # hysteresis flag on distance
        self._t_accum = 0.0

    """
    The step() function 
    """
    def step(self, obs):
        dt = max(obs["dt"], 1e-3)
        self._t_accum += dt

        face_visible = obs["face_visible"] and obs["face_cx"] is not None
        face_area = obs.get("face_area") or 0
        # Derive face_w_norm if the controller only gave us face_area.
        # Area = w * h, heads are ~1:1.3 → w ≈ sqrt(area / 1.3) in pixels.
        # But we need w as a fraction of frame — the server already gives
        # face_w_norm in face_user obs. For follow_mode, we rebuild it.
        face_w_norm = obs.get("face_w_norm")
        if face_w_norm is None and face_area:
            # Rough: assume 320-wide frame. Good enough; filtered heavily below.
            w_px = math.sqrt(max(face_area, 1) / 1.3)
            face_w_norm = w_px / 320.0

        # --- Detection quality gate ---
        if face_visible and (face_w_norm is None or face_w_norm < MIN_FACE_W_NORM):
            face_visible = False

        # --- Face lost: graceful decay ---
        if not face_visible:
            if obs["face_age_s"] < LOST_HOLD_S:
                return {"linear": self._prev_linear, "angular": self._prev_angular}
            decay_elapsed = obs["face_age_s"] - LOST_HOLD_S
            if decay_elapsed < LOST_DECAY_S:
                k = 1.0 - (decay_elapsed / LOST_DECAY_S)
                lin = self._slew_linear(self._prev_linear * k, dt)
                ang = self._slew_angular(self._prev_angular * k, dt)
                self._prev_linear = lin
                self._prev_angular = ang
                return {"linear": lin, "angular": ang}
            # fully lost — zero out everything
            self._pan_i = 0.0
            self._dist_i = 0.0
            self._prev_cx = None
            self._prev_dist = None
            self._cx_filtered = None
            self._w_filtered = None
            self._cx_history.clear()
            self._at_target = False
            lin = self._slew_linear(0.0, dt)
            ang = self._slew_angular(0.0, dt)
            self._prev_linear = lin
            self._prev_angular = ang
            return {"linear": lin, "angular": ang}

        # --- Input filters ---
        raw_cx = obs["face_cx"]
        if self._cx_filtered is None:
            self._cx_filtered = raw_cx
        else:
            self._cx_filtered = INPUT_EMA_CX * self._cx_filtered + (1 - INPUT_EMA_CX) * raw_cx
        cx = self._cx_filtered

        if self._w_filtered is None:
            self._w_filtered = face_w_norm
        else:
            self._w_filtered = INPUT_EMA_W * self._w_filtered + (1 - INPUT_EMA_W) * face_w_norm
        w = self._w_filtered

        self._cx_history.append((self._t_accum, cx))

        # --- Distance estimate (pinhole from face width) ---
        # px_width on a 320-wide frame = w * 320
        w_px = max(w * 320.0, 1.0)
        distance_m = (FOCAL_LENGTH_PX_EST * HEAD_WIDTH_M) / w_px

        # --- Pan PID ---
        deadzone = PAN_DEADZONE_BASE + PAN_DEADZONE_SCALE * max(0.0, 1.0 - w / 0.2)
        pan_err = cx - 0.5
        if abs(pan_err) < deadzone:
            pan_err_eff = 0.0
        else:
            pan_err_eff = (abs(pan_err) - deadzone) * (1.0 if pan_err > 0 else -1.0)

        self._pan_i += pan_err_eff * dt
        max_pan_i = PAN_I_LIMIT / max(PAN_KI, 1e-6)
        self._pan_i = max(-max_pan_i, min(max_pan_i, self._pan_i))

        if self._prev_cx is None:
            pan_d = 0.0
        else:
            pan_d = (cx - self._prev_cx) / dt
        self._prev_cx = cx

        # Pan velocity feed-forward
        pan_ff = 0.0
        if len(self._cx_history) >= 2:
            t0, c0 = self._cx_history[0]
            t1, c1 = self._cx_history[-1]
            if t1 - t0 > 1e-3:
                pan_ff = -VELOCITY_FF_GAIN * (c1 - c0) / (t1 - t0)

        target_angular = -(PAN_KP * pan_err_eff + PAN_KI * self._pan_i) - PAN_KD * pan_d + pan_ff

        # --- Distance PID with hysteresis ---
        dist_err = distance_m - TARGET_DISTANCE_M

        if self._at_target and abs(dist_err) > STOP_HYSTERESIS_M:
            self._at_target = False
            target_linear = 0.0
            self._dist_i *= 0.95  # bleed integral while parked
        else:
            self._dist_i += dist_err * dt
            max_dist_i = DIST_I_LIMIT / max(DIST_KI, 1e-6)
            self._dist_i = max(-max_dist_i, min(max_dist_i, self._dist_i))

            if self._prev_dist is None:
                dist_d = 0.0
            else:
                dist_d = (distance_m - self._prev_dist) / dt

            target_linear = (
                DIST_KP * dist_err
                + DIST_KI * self._dist_i
                + DIST_KD * dist_d
            )

            # re-engage hysteresis on arrival
            if abs(dist_err) < STOP_HYSTERESIS_M * 0.3:
                self._at_target = True
        self._prev_dist = distance_m

        # --- Motion coupling: scale linear down when pan error is big ---
        coupling = max(COUPLING_MIN, 1.0 - COUPLING_K * abs(pan_err))
        target_linear *= coupling

        # --- Clamp ---
        target_linear = max(-MAX_LINEAR, min(MAX_LINEAR, target_linear))
        target_angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, target_angular))

        # --- Chassis friction floor ---
        # Only kick angular when actively trying to rotate (outside deadzone).
        if pan_err_eff != 0.0 and 0.0 < abs(target_angular) < MIN_ANGULAR_KICK:
            target_angular = (
                MIN_ANGULAR_KICK if target_angular > 0 else -MIN_ANGULAR_KICK
            )
        # Only kick linear when actively trying to move (not parked).
        if not self._at_target and 0.0 < abs(target_linear) < MIN_LINEAR_KICK:
            target_linear = (
                MIN_LINEAR_KICK if target_linear > 0 else -MIN_LINEAR_KICK
            )

        # --- Slew-rate limit ---
        linear = self._slew_linear(target_linear, dt)
        angular = self._slew_angular(target_angular, dt)

        self._prev_linear = linear
        self._prev_angular = angular
        return {"linear": linear, "angular": angular}

    def _slew_linear(self, target, dt):
        max_delta = MAX_LINEAR_ACCEL * dt
        delta = target - self._prev_linear
        if delta > max_delta:  return self._prev_linear + max_delta
        if delta < -max_delta: return self._prev_linear - max_delta
        return target

    def _slew_angular(self, target, dt):
        max_delta = MAX_ANGULAR_ACCEL * dt
        delta = target - self._prev_angular
        if delta > max_delta:  return self._prev_angular + max_delta
        if delta < -max_delta: return self._prev_angular - max_delta
        return target
