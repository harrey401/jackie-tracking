"""Face-the-user — v2 production implementation (event-grade).

Same contract as before: Logic.reset / Logic.step(obs) -> {linear, angular}.

Upgrades over the earlier baseline:
  - Full PID (P + I + D) with anti-windup clamping
  - Derivative-on-measurement (no first-tick D-kick)
  - Input low-pass filter on raw face_cx (kills detector jitter UPSTREAM,
    before it turns into angular velocity — this is the main cure for the
    "left-right constantly" shimmy we saw on the baseline)
  - Windowed velocity feed-forward (leads a walking user)
  - Size-adaptive gain + deadzone (gentler on far/noisy faces)
  - Output slew-rate limit (caps angular acceleration, prevents jerk)
  - Detection quality gate (rejects tiny hallucinated faces)
  - Graceful lost-face decay (hold-then-ramp, no abrupt stops)

NO MIN_KICK in v2: PAN_KP=1.6 on normalized error is strong enough that
any meaningful off-center error naturally produces ≥ the chassis friction
threshold. Adding a kick floor causes bang-bang at the deadzone edge
(tracker jitter crosses the edge → output snaps between 0 and ±kick).
Small residual errors produce sub-threshold output and the chassis
quietly parks close-to-centered — which is what you want.

Sign convention: face on the RIGHT of frame (face_cx > 0.5) → Jackie must
rotate RIGHT → NEGATIVE angular.z on /cmd_vel_mux/input/navi_override.
The combine line already flips the sign; do not re-flip elsewhere.
"""

from collections import deque


# === TUNABLES ==================================================================

# Core PID gains (applied to normalized pan error, range -0.5..+0.5)
# D and I are deliberately tiny — both amplify face-detector jitter. Keep
# them small unless you've verified the detector is clean enough to handle
# higher values. P-dominant control with heavy input filtering is what
# actually works against a 3-5 pixel jittering bbox.
PAN_KP = 1.2
PAN_KI = 0.05
PAN_KD = 0.03
INTEGRAL_LIMIT = 0.4

# Target horizontal position of the face in the frame (normalized 0..1).
# 0.5 = exact center. Tune this if Jackie's camera mount is offset from
# the body's rotational axis — Jackie's left eye vs right eye effect.
#   - If the user ends up on Jackie's LEFT EYE SIDE  → raise this (0.55)
#   - If the user ends up on Jackie's RIGHT EYE SIDE → lower this (0.45)
# Think of it as "where in the camera image should I try to keep your face".
TARGET_CX = 0.9

# Deadzone — widens automatically for small (far) faces
# 0.06 normalized = ~38 px on a 640-wide frame. Generous by design to
# absorb detector jitter without Jackie reacting to it.
BASE_DEADZONE = 0.06
DEADZONE_SIZE_SCALE = 0.08

# Input low-pass filter on raw face_cx (alpha = weight of HISTORY,
# 1 - alpha = weight of new sample). 0.6 = strong filter.
INPUT_EMA_ALPHA = 0.6

# Velocity feed-forward — mostly off. Re-enable when the detector is
# clean enough to trust its velocity estimate.
VELOCITY_FF_GAIN = 0.1
VELOCITY_WINDOW = 5

# Size-adaptive gain: scale by clamp(face_w_norm / REF_FACE_W, 0.4, 1.0)
REF_FACE_W = 0.18

# Output slew rate limit (max angular acceleration)
MAX_ANGULAR_ACCEL = 2.0  # rad/s²

# Hard speed ceiling (server also clamps)
MAX_ANGULAR = 0.8

# Detection quality gate — reject faces smaller than this as noise
MIN_FACE_W_NORM = 0.03

# Lost-face behaviour
LOST_HOLD_S = 0.4
LOST_DECAY_S = 0.6


# === LOGIC ======================================================================


class Logic:
    def __init__(self):
        self.reset()

    def reset(self):
        self._cx_filtered = None
        self._cx_history = deque(maxlen=VELOCITY_WINDOW)
        self._integral = 0.0
        self._prev_measurement = None
        self._prev_output = 0.0
        self._t_accumulator = 0.0

    def step(self, obs):
        dt = max(obs["dt"], 1e-3)
        self._t_accumulator += dt

        face_visible = obs["face_visible"] and obs["face_cx"] is not None
        face_w = obs.get("face_w_norm")

        # Detection quality gate
        if face_visible and (face_w is None or face_w < MIN_FACE_W_NORM):
            face_visible = False

        # Face lost: graceful decay
        if not face_visible:
            age = obs.get("face_age_s", float("inf"))
            if age < LOST_HOLD_S:
                return {"linear": 0.0, "angular": self._prev_output}
            decay_elapsed = age - LOST_HOLD_S
            if decay_elapsed < LOST_DECAY_S:
                k = 1.0 - (decay_elapsed / LOST_DECAY_S)
                out = self._rate_limit(self._prev_output * k, dt)
                self._prev_output = out
                return {"linear": 0.0, "angular": out}
            self._integral = 0.0
            self._prev_measurement = None
            self._cx_filtered = None
            self._cx_history.clear()
            self._prev_output = self._rate_limit(0.0, dt)
            return {"linear": 0.0, "angular": self._prev_output}

        # Input low-pass filter — kills detector jitter before the PID
        raw_cx = obs["face_cx"]
        if self._cx_filtered is None:
            self._cx_filtered = raw_cx
        else:
            self._cx_filtered = (
                INPUT_EMA_ALPHA * self._cx_filtered
                + (1.0 - INPUT_EMA_ALPHA) * raw_cx
            )
        cx = self._cx_filtered

        self._cx_history.append((self._t_accumulator, cx))

        # Size-adaptive scaling
        w = face_w or REF_FACE_W
        size_scale = max(0.4, min(1.0, w / REF_FACE_W))
        deadzone = BASE_DEADZONE + DEADZONE_SIZE_SCALE * max(
            0.0, 1.0 - w / REF_FACE_W
        )

        # Pan error — continuous at deadzone edge.
        # TARGET_CX shifts the "zero error" point to compensate for camera
        # mount offset (see TARGET_CX comment at top).
        error = cx - TARGET_CX
        if abs(error) < deadzone:
            error_effective = 0.0
        else:
            error_effective = (abs(error) - deadzone) * (1.0 if error > 0 else -1.0)

        # Proportional
        p_term = PAN_KP * error_effective

        # Integral with anti-windup
        self._integral += error_effective * dt
        max_i = INTEGRAL_LIMIT / max(PAN_KI, 1e-6)
        self._integral = max(-max_i, min(max_i, self._integral))
        i_term = PAN_KI * self._integral

        # Derivative on measurement (not error — avoids first-tick kick)
        if self._prev_measurement is None:
            d_term = 0.0
        else:
            d_measurement = (cx - self._prev_measurement) / dt
            d_term = PAN_KD * d_measurement
        self._prev_measurement = cx

        # Velocity feed-forward — lead the moving target
        ff_term = 0.0
        if len(self._cx_history) >= 2:
            t0, cx0 = self._cx_history[0]
            t1, cx1 = self._cx_history[-1]
            if t1 - t0 > 1e-3:
                cx_vel = (cx1 - cx0) / (t1 - t0)
                ff_term = -VELOCITY_FF_GAIN * cx_vel

        # Combine — sign flip lives here: face right → turn right → negative angular
        raw_output = -(p_term + i_term) - d_term + ff_term
        raw_output *= size_scale

        # Hard clamp
        raw_output = max(-MAX_ANGULAR, min(MAX_ANGULAR, raw_output))

        # Slew-rate limit
        output = self._rate_limit(raw_output, dt)
        self._prev_output = output

        return {"linear": 0.0, "angular": output}

    def _rate_limit(self, target, dt):
        max_delta = MAX_ANGULAR_ACCEL * dt
        delta = target - self._prev_output
        if delta > max_delta:
            return self._prev_output + max_delta
        if delta < -max_delta:
            return self._prev_output - max_delta
        return target
