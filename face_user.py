"""Face-the-user — v3 production implementation (event-grade, Jackie-tuned).

Same contract as before: Logic.reset / Logic.step(obs) -> {linear, angular}.

v3 delta (2026-04-21):
  - 1€ filter (Casiez et al. 2012) replaces the fixed-α EMA on face_cx.
    The adaptive cutoff is heavy when the signal is still (kills the 3-5px
    bbox jitter that was producing residual left-right wobble on Jackie)
    and opens up when the user actually walks (no lag penalty). This is
    the single biggest stability win — same control law, cleaner input.
  - Speed/accel caps halved (MAX_ANGULAR 0.8→0.4, MAX_ANGULAR_ACCEL 2.0→1.2)
    so face-me feels deliberate rather than snappy. Jackie was overshooting
    in the first tick on large errors even with the old slew limit.
  - PAN_KI + PAN_KD slashed — with a clean filtered input, derivative gives
    nothing new and integral windup caused visible drift-back-and-kick.
    P-dominant control is what works when the measurement is clean.
  - Velocity feed-forward halved for the same reason (FF amplifies whatever
    the filter is doing; turn it down when filtering is better).

Retained from v2:
  - Derivative-on-measurement (no first-tick D-kick)
  - Size-adaptive gain + deadzone (gentler on far/noisy faces)
  - Output slew-rate limit (caps angular acceleration, prevents jerk)
  - Detection quality gate (rejects tiny hallucinated faces)
  - Graceful lost-face decay (hold-then-ramp, no abrupt stops)
  - Yaw-veto + sustained-offset hysteresis (shipped 2026-04-21)

Sign convention: face on the RIGHT of frame (face_cx > 0.5) → Jackie must
rotate RIGHT → NEGATIVE angular.z on /cmd_vel_mux/input/navi_override.
The combine line already flips the sign; do not re-flip elsewhere.

NO MIN_KICK: PAN_KP on normalized error is strong enough that a real
off-center face naturally produces ≥ the chassis friction threshold.
Adding a kick floor causes bang-bang at the deadzone edge. Small residual
errors produce sub-threshold output and the chassis quietly parks — which
is what we want.
"""

import math


# === TUNABLES ==================================================================

# Core PID gains (applied to normalized pan error, range -0.5..+0.5).
# With a clean filtered input (1€ filter below), the D and I terms add
# almost nothing but noise amplification and windup. Keep them nominal —
# effectively P-only control.
PAN_KP = 1.2
PAN_KI = 0.005   # was 0.05 — effectively disables the integral kick
PAN_KD = 0.005   # was 0.03 — derivative isn't useful on a clean signal
INTEGRAL_LIMIT = 0.3

# Target horizontal position of the face in the frame (normalized 0..1).
# 0.5 = exact center. Tune this if Jackie's camera mount is offset from
# the body's rotational axis — Jackie's left eye vs right eye effect.
#   - If the user ends up on Jackie's LEFT EYE SIDE  → raise this (0.55)
#   - If the user ends up on Jackie's RIGHT EYE SIDE → lower this (0.45)
# Think of it as "where in the camera image should I try to keep your face".
TARGET_CX = 0.98

# Deadzone — widens automatically for small (far) faces. 0.06 normalized
# = ~38 px on a 640-wide frame. Below this error, output is zero.
BASE_DEADZONE = 0.06
DEADZONE_SIZE_SCALE = 0.08

# 1€ filter parameters (Casiez et al. 2012, "1€ Filter: A Simple Speed-based
# Low-pass Filter"). f_min is the cutoff at rest — low value = heavy filter
# on jitter. beta is the speed sensitivity — raises cutoff when the user
# moves so we don't lag real motion. d_cutoff filters the velocity estimate
# that drives the adaptive cutoff.
#
# Chosen for Jackie's 3-5 px jitter at 10 Hz control rate (15 fps video input):
#   - f_min=0.5 Hz → static jitter attenuation ~30x vs a 5Hz cutoff
#   - beta=3.0    → cutoff rises to ~2 Hz when user walks at ~0.5 norm/s
#   - d_cutoff=1.0 Hz → velocity estimate itself is filtered
ONE_EURO_F_MIN = 0.5
ONE_EURO_BETA = 3.0
ONE_EURO_D_CUTOFF = 1.0

# Velocity feed-forward — light touch. 1€ filter removes most of the noise,
# so FF can still contribute, but we don't need to rely on it.
VELOCITY_FF_GAIN = 0.05   # was 0.1

# Size-adaptive gain: scale by clamp(face_w_norm / REF_FACE_W, 0.4, 1.0)
REF_FACE_W = 0.18

# Output slew rate limit (max angular acceleration). Halved from v2 for
# a more deliberate feel — no sudden pops at the start of a turn.
MAX_ANGULAR_ACCEL = 1.2   # rad/s²  (was 2.0)

# Hard speed ceiling (server also clamps). Halved from v2 — Gow's feedback
# "speed too high" on 2026-04-21. 0.4 rad/s ≈ 23°/s — feels attentive, not
# twitchy. Server-side MAX_ANGULAR_SAFETY is 0.5 for belt + suspenders.
MAX_ANGULAR = 0.4   # rad/s  (was 0.8)

# Detection quality gate — reject faces smaller than this as noise
MIN_FACE_W_NORM = 0.03

# Lost-face behaviour
LOST_HOLD_S = 0.4
LOST_DECAY_S = 0.6


# Head-turn filter — bystander-rejection brakes (shipped 2026-04-21).
# 1. Yaw veto: if |yaw| > YAW_VETO_DEG the user is looking elsewhere; don't
#    chase the resulting face_cx drift. Hysteresis below YAW_RESUME_DEG.
# 2. Sustained offset: only rotate after the offset has been outside the
#    deadzone for HYSTERESIS_SUSTAIN_S continuous seconds. Head glances
#    are brief; body translations persist.
YAW_VETO_DEG = 20.0
YAW_RESUME_DEG = 10.0
HYSTERESIS_SUSTAIN_S = 0.4


# === 1€ FILTER ==================================================================


class _OneEuroFilter:
    """Adaptive low-pass filter for noisy real-time pointer-like signals.

    Mathematically: exponential smoothing with a cutoff frequency that
    rises linearly with the signal's instantaneous velocity.

        cutoff(t) = f_min + beta * |velocity(t)|
        alpha(cutoff, dt) = 1 / (1 + tau/dt)   where tau = 1 / (2π * cutoff)
        x_hat(t) = alpha * x(t) + (1 - alpha) * x_hat(t-1)

    Intuition: when the user holds still, velocity ≈ 0, cutoff = f_min,
    strong smoothing hides the jitter. When the user walks, velocity is
    large, cutoff rises, the filter opens up and we track the motion with
    minimal lag. A fixed-α EMA cannot do both at once.

    Reference: Casiez, Roussel, Vogel. "1€ Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems." CHI 2012.
    """

    def __init__(
        self,
        f_min: float = ONE_EURO_F_MIN,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_D_CUTOFF,
    ) -> None:
        self.f_min = f_min
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0

    def __call__(self, x: float, dt: float) -> float:
        dt = max(dt, 1e-3)
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.f_min + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


# === LOGIC ======================================================================


class Logic:
    def __init__(self):
        self._filter = _OneEuroFilter()
        self.reset()

    def reset(self):
        self._filter.reset()
        self._integral = 0.0
        self._prev_measurement = None
        self._prev_cx_vel = 0.0
        self._prev_output = 0.0
        self._t_accumulator = 0.0
        self._yaw_vetoed = False
        self._offset_sustained_since = None

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
            self._prev_cx_vel = 0.0
            self._filter.reset()
            self._prev_output = self._rate_limit(0.0, dt)
            return {"linear": 0.0, "angular": self._prev_output}

        # 1€ filter — kills detector jitter at rest, tracks real motion.
        raw_cx = obs["face_cx"]
        cx = self._filter(raw_cx, dt)

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
            self._offset_sustained_since = None
        else:
            error_effective = (abs(error) - deadzone) * (1.0 if error > 0 else -1.0)
            if self._offset_sustained_since is None:
                self._offset_sustained_since = self._t_accumulator

        # Yaw veto — user's head is turned away from Jackie. Decay current
        # output toward zero (respecting slew) and freeze PID state. None
        # yaw (MediaPipe failure) falls through — the caller treats missing
        # yaw as "unknown, trust offset logic".
        yaw = obs.get("face_yaw_deg")
        if yaw is not None:
            ayaw = abs(yaw)
            if ayaw > YAW_VETO_DEG:
                self._yaw_vetoed = True
            elif self._yaw_vetoed and ayaw < YAW_RESUME_DEG:
                self._yaw_vetoed = False

        if self._yaw_vetoed:
            self._integral = 0.0
            self._offset_sustained_since = None
            out = self._rate_limit(0.0, dt)
            self._prev_output = out
            self._prev_measurement = cx
            return {"linear": 0.0, "angular": out}

        # Sustained-offset hysteresis — brief glances shouldn't trigger a
        # rotation. Wait for the offset to persist across HYSTERESIS_SUSTAIN_S
        # before engaging the PID. Anti-windup: keep integral frozen during
        # the wait window so we don't store up a kick.
        if (
            self._offset_sustained_since is not None
            and (self._t_accumulator - self._offset_sustained_since)
            < HYSTERESIS_SUSTAIN_S
        ):
            out = self._rate_limit(0.0, dt)
            self._prev_output = out
            self._prev_measurement = cx
            return {"linear": 0.0, "angular": out}

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

        # Velocity feed-forward — already filtered inside 1€, so this is
        # a gentle nudge, not a driver.
        cx_vel = 0.0
        if self._prev_measurement is not None:
            cx_vel = (cx - self._prev_measurement) / dt
        # Low-pass the velocity estimate so a single jumpy frame doesn't
        # produce a ff spike.
        self._prev_cx_vel = 0.5 * self._prev_cx_vel + 0.5 * cx_vel
        ff_term = -VELOCITY_FF_GAIN * self._prev_cx_vel

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
