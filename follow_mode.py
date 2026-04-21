"""Follow Me v2 — backstepping + alpha-beta + LIDAR-aware reactive controller.

Tuned for Jackie (diff-drive, 10 Hz control, server-side face tracking,
`sensor_msgs/LaserScan` on /scan, `std_msgs/Int8` on /obstacle_region).

Why this replaces the old PID loop
----------------------------------
The previous controller ran a raw PID on pixel-space `face_cx` with KD>0 and
no input filter. Derivative-on-error plus unfiltered detection noise gave a
textbook oscillation signature: the robot zigzags hardest when the user walks
straight backward (face parked near cx=0.5, detector jitter ±5 px feeding KD).

This file fixes that and adds real obstacle awareness:

  1. Alpha-beta filter on face_cx, EMA+outlier reject on distance — the PID
     only ever sees smoothed state.
  2. Derivative-on-measurement with first-order LPF on the derivative —
     damping without noise kick (Astrom form).
  3. Dual dead-zone gate: |filtered_dx| AND |filtered_vx| must both exceed
     thresholds before angular output is produced. This is the specific cure
     for straight-backward-walking oscillation.
  4. Asymmetric gains: KP is halved inside a near-center band, restored
     outside. Fast intercept, hard damping.
  5. Backstepping cross-coupling `ω += k_c · v · sin(θ_err)` — compensates
     for translation when bearing error is small (Zidani-Drid).
  6. Output slew-rate limit on both linear and angular — no commanded jerks.
  7. LIDAR (when `obs["lidar_ranges"]` is present):
       - Safety bubble: min range in forward arc < threshold → zero linear.
       - Forward speed scaled by clearance (bubble controller).
       - Follow-the-gap: ω biased toward widest gap when a close obstacle
         blocks the direct heading. Steers AROUND, not just stops.
  8. `obs["obstacle_region"]` (chassis Int8 nav signal) — last-resort veto
     that forces COLLISION_STOP for 2s when nonzero for 3 consecutive ticks.

The FSM is retained but every state finally has a real trigger.

Contract (unchanged):  Logic.reset() / Logic.step(obs) -> {linear, angular}
Hot-reloaded by `smait/navigation/tracking_loader.py` on file mtime change.
"""

import math
from collections import deque
from motion_logger import MotionLogger

# ───── Geometry / speed caps ───────────────────────────────────────────────
FRAME_WIDTH_PX       = 640
FOCAL_LENGTH_PX      = 600.0     # calibrate per camera
FACE_WIDTH_M         = 0.165
TARGET_FOLLOW_DIST_M = 0.5
MAX_FOLLOW_DIST_M    = 3.0
COLLISION_DIST_M     = 0.33      # vision-estimated fallback stop
MAX_LINEAR           = 0.25
MAX_ANGULAR          = 0.5       # matches server-side MAX_ANGULAR_SAFETY
TURN_SPEED           = 0.4

# ───── Filters ─────────────────────────────────────────────────────────────
# Alpha-beta on face_cx (px). α near 1.0 = heavy smoothing (more lag);
# β small = light velocity update. Tuned for ~10-15 Hz updates.
ALPHA_POS            = 0.55
BETA_VEL             = 0.08
DIST_EMA_ALPHA       = 0.3       # EMA on distance
DIST_OUTLIER_RATIO   = 0.5       # reject |Δd| > 0.5·d_filt
DERIV_LPF_ALPHA      = 0.3       # first-order LPF on derivative term
INPUT_STALE_RESET_S  = 0.5       # face lost > this resets filter state

# ───── Pan control ──────────────────────────────────────────────────────────
# Gains work on normalized-pixel error (dx_px / FRAME_WIDTH_PX) so units are
# [-1, 1]. Far easier to reason about than raw-pixel KP=0.003.
PAN_KP_FAR           = 1.2       # outside near-center band
PAN_KP_NEAR          = 0.6       # inside near-center band (anti-osc)
PAN_KI               = 0.0       # disabled — integral on visual servo adds
                                 # drift with no corresponding benefit here
PAN_KD               = 0.25      # derivative-on-measurement, LPF'd
PAN_DEADZONE_NORM    = 0.035     # ±3.5% of frame (~22 px on 640)
PAN_VEL_DEADZONE     = 0.08      # normalized px/s — filtered target velocity
NEAR_CENTER_BAND     = 0.10      # |dx_norm| < 10% → near-center gains

# ───── Distance control ───────────────────────────────────────────────────
DIST_KP              = 0.35
DIST_DEADZONE_M      = 0.08      # ±8 cm around target

# ───── Backstepping cross-coupling ─────────────────────────────────────────
# ω += BACKSTEP_GAIN · v · sin(θ_err). θ_err is the filtered bearing (rad)
# inferred from dx: for a face at cx offset dx_px, bearing ≈ atan(dx_px /
# focal_px). Small-angle → dx_norm · (FRAME_WIDTH_PX / FOCAL_LENGTH_PX).
BACKSTEP_GAIN        = 0.6

# ───── Output slew limits (per second) ─────────────────────────────────────
LINEAR_SLEW          = 0.6       # m/s²
ANGULAR_SLEW         = 2.0       # rad/s²

# ───── LIDAR reactive layer ────────────────────────────────────────────────
LIDAR_BUBBLE_ARC_RAD = math.radians(25)   # ±25° forward = safety cone
LIDAR_GAP_ARC_RAD    = math.radians(70)   # ±70° forward = gap search arc
LIDAR_STOP_RANGE_M   = 0.28               # min clearance → zero linear
LIDAR_SLOW_RANGE_M   = 1.0                # full-speed above this
LIDAR_MIN_VALID_M    = 0.05               # reject returns below this (self-hits)
LIDAR_GAP_BIAS_GAIN  = 0.8                # k in ω += k · (gap_θ - face_θ)
LIDAR_GAP_TRIGGER_M  = 0.6                # invoke gap bias only when forward
                                          # min range falls below this
OBSTACLE_TRIP_COUNT  = 3                  # consecutive ticks of nonzero int
COLLISION_PAUSE_S    = 2.0
SCAN_45_RAD          = math.pi / 4.0
SCAN_90_RAD          = math.pi / 2.0

# ───── FSM ──────────────────────────────────────────────────────────────────
FOLLOWING      = "FOLLOWING"
SCAN_ROTATE    = "SCAN_ROTATE"
OBSTACLE_TURN  = "OBSTACLE_TURN"
COLLISION_STOP = "COLLISION_STOP"
COLLISION_TURN = "COLLISION_TURN"
CLEAR_CHECK    = "CLEAR_CHECK"


# ─── Target-state filter ───────────────────────────────────────────────────
class _AlphaBeta:
    """Classic alpha-beta filter — smooths position, gives calibrated velocity.

    State: (x, v). On each tick with measurement z and timestep dt:
        x_pred = x + v·dt
        residual = z − x_pred
        x_new = x_pred + α·residual
        v_new = v     + (β/dt)·residual
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.x: float | None = None
        self.v: float = 0.0

    def reset(self) -> None:
        self.x = None
        self.v = 0.0

    def update(self, z: float, dt: float) -> tuple[float, float]:
        dt = max(dt, 1e-3)
        if self.x is None:
            self.x, self.v = z, 0.0
            return self.x, self.v
        x_pred = self.x + self.v * dt
        r = z - x_pred
        self.x = x_pred + self.alpha * r
        self.v = self.v + (self.beta / dt) * r
        return self.x, self.v


# ─── PID with derivative-on-measurement + LPF ──────────────────────────────
class _PID:
    """Astrom form: D acts on the filtered measurement, not the error.

    Eliminates derivative kick on setpoint step and the noise amplification
    that plagues D-on-error when the measurement is jittery.
    """

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_meas: float | None = None
        self._d_lpf = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_meas = None
        self._d_lpf = 0.0

    def compute(self, error: float, meas: float, dt: float,
                integral_max: float | None = None) -> float:
        dt = max(dt, 1e-3)
        if self.ki > 0:
            self._integral += error * dt
            if integral_max is not None:
                self._integral = max(-integral_max, min(integral_max, self._integral))
        if self._prev_meas is None:
            d_raw = 0.0
        else:
            d_raw = -(meas - self._prev_meas) / dt     # D on measurement
        self._prev_meas = meas
        self._d_lpf = DERIV_LPF_ALPHA * d_raw + (1 - DERIV_LPF_ALPHA) * self._d_lpf
        return self.kp * error + self.ki * self._integral + self.kd * self._d_lpf


# ─── LIDAR helpers ─────────────────────────────────────────────────────────
def _forward_min_range(obs: dict, arc_rad: float) -> float:
    """Minimum LIDAR return in ±arc_rad around 0 rad (forward).

    Returns math.inf when LIDAR data is missing, stale, or empty.
    """
    ranges = obs.get("lidar_ranges")
    if not ranges:
        return math.inf
    a0 = obs.get("lidar_angle_min")
    da = obs.get("lidar_angle_increment")
    if a0 is None or da is None or da <= 0:
        return math.inf
    n = len(ranges)
    rmax = obs.get("lidar_range_max") or float("inf")
    best = math.inf
    for i in range(n):
        theta = a0 + i * da
        # Forward in the scan frame is θ=0. Assume robot-forward is 0 rad.
        if -arc_rad <= theta <= arc_rad:
            r = ranges[i]
            if r is None:
                continue
            try:
                rv = float(r)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(rv):
                continue
            if rv < LIDAR_MIN_VALID_M or rv > rmax:
                continue
            if rv < best:
                best = rv
    return best


def _widest_gap_bearing(obs: dict, arc_rad: float, min_gap_range_m: float) -> float | None:
    """Return bearing (rad, robot frame) of center of the widest gap where all
    returns exceed `min_gap_range_m`. None if no such gap or no LIDAR.

    Simple Follow-the-Gap (Sezer & Gokasan 2012): threshold the range array,
    find the longest contiguous run of passable bins, return its center angle.
    """
    ranges = obs.get("lidar_ranges")
    if not ranges:
        return None
    a0 = obs.get("lidar_angle_min")
    da = obs.get("lidar_angle_increment")
    if a0 is None or da is None or da <= 0:
        return None

    n = len(ranges)
    start = end = None
    best_start = best_end = None
    best_len = 0

    for i in range(n):
        theta = a0 + i * da
        if theta < -arc_rad or theta > arc_rad:
            if start is not None:
                if end - start > best_len:
                    best_start, best_end, best_len = start, end, end - start
                start = end = None
            continue
        r = ranges[i]
        try:
            rv = float(r) if r is not None else 0.0
        except (TypeError, ValueError):
            rv = 0.0
        passable = math.isfinite(rv) and rv >= min_gap_range_m
        if passable:
            if start is None:
                start = i
            end = i
        else:
            if start is not None:
                if end - start > best_len:
                    best_start, best_end, best_len = start, end, end - start
                start = end = None

    if start is not None and end - start > best_len:
        best_start, best_end = start, end
        best_len = end - start

    if best_start is None or best_len < 3:
        return None
    mid = (best_start + best_end) * 0.5
    return a0 + mid * da


# ─── Main logic ────────────────────────────────────────────────────────────
class Logic:
    """Hot-reloadable follow-me controller.

    Public contract:
      reset()               — called when follow mode activates or on reload
      step(obs) -> dict     — called every tick with latest observations

    obs keys read (missing keys degrade gracefully):
      dt, face_visible, face_cx (norm), face_w_px, face_area, frame_width_px,
      track_id, face_age_s, lidar_ranges, lidar_angle_min,
      lidar_angle_increment, lidar_range_max, obstacle_region
    """

    def __init__(self):
        self._pan = _PID(PAN_KP_FAR, PAN_KI, PAN_KD)
        self._dist = _PID(DIST_KP, 0.0, 0.0)
        self._cx_filter = _AlphaBeta(ALPHA_POS, BETA_VEL)
        self._smooth_dist: float | None = None
        self._last_cx_time: float | None = None

        # FSM
        self._fsm = SCAN_ROTATE
        self._t_accum = 0.0
        self._manoeuvre_end_s = 0.0
        self._scan_direction = 1.0
        self._obstacle_trip = 0
        self._collision_frame_count = 0

        # Output slew state
        self._prev_linear = 0.0
        self._prev_angular = 0.0

        # Locked target (server-side also enforces; belt and suspenders)
        self.locked_track_id: int | None = None
        
        # Create a motion logger instance for logging Jackie's motions
        self._logger = MotionLogger()
        

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._pan.reset()
        self._dist.reset()
        self._cx_filter.reset()
        self._smooth_dist = None
        self._last_cx_time = None
        self._t_accum = 0.0
        self._fsm = SCAN_ROTATE
        self._scan_direction = 1.0
        self._start_manoeuvre(SCAN_45_RAD, self._scan_direction)
        self._obstacle_trip = 0
        self._collision_frame_count = 0
        self._prev_linear = 0.0
        self._prev_angular = 0.0
        self.locked_track_id = None
        self._logger = MotionLogger()  # Start a new log session on reset

    # ------------------------------------------------------------------
    def step(self, obs: dict) -> dict:
        dt = max(float(obs.get("dt") or 0.1), 1e-3)
        self._t_accum += dt

        face = self._extract_face(obs)
        dist_m = self._update_distance(face) if face else float("inf")
        cx_filt_norm, vx_filt_norm = self._update_cx(face, dt, obs)

        # Obstacle-region veto (trips COLLISION_STOP independently of LIDAR).
        region = int(obs.get("obstacle_region") or 0)
        if region != 0:
            self._obstacle_trip += 1
        else:
            self._obstacle_trip = 0
        if self._obstacle_trip >= OBSTACLE_TRIP_COUNT and self._fsm == FOLLOWING:
            self._enter_state(COLLISION_STOP)

        # FSM dispatch.
        if self._fsm == FOLLOWING:
            if face is None or dist_m > MAX_FOLLOW_DIST_M:
                self._enter_state(SCAN_ROTATE)
                return self._out(0.0, 0.0)
            linear, angular = self._follow_step(
                cx_filt_norm, vx_filt_norm, dist_m, dt, obs
            )
            return self._out(linear, angular)

        if self._fsm == SCAN_ROTATE:
            # Acquired a face mid-scan? Stop rotating immediately.
            if face is not None and dist_m <= MAX_FOLLOW_DIST_M:
                self._enter_state(FOLLOWING)
                return self._out(0.0, 0.0)
            if self._t_accum < self._manoeuvre_end_s:
                return self._out(0.0, TURN_SPEED * self._scan_direction)
            self._start_manoeuvre(SCAN_45_RAD, self._scan_direction)
            return self._out(0.0, 0.0)

        if self._fsm == OBSTACLE_TURN:
            if self._t_accum < self._manoeuvre_end_s:
                return self._out(0.0, -TURN_SPEED)
            self._enter_state(CLEAR_CHECK)
            return self._out(0.0, 0.0)

        if self._fsm == CLEAR_CHECK:
            # If LIDAR says the path is clear and we still see the face,
            # resume. Otherwise scan.
            forward_clear = _forward_min_range(obs, LIDAR_BUBBLE_ARC_RAD)
            if forward_clear > LIDAR_STOP_RANGE_M + 0.1 and face is not None:
                self._enter_state(FOLLOWING)
            else:
                self._enter_state(SCAN_ROTATE)
            return self._out(0.0, 0.0)

        if self._fsm == COLLISION_STOP:
            if self._t_accum >= self._manoeuvre_end_s:
                self._enter_state(COLLISION_TURN)
            return self._out(0.0, 0.0)

        if self._fsm == COLLISION_TURN:
            if self._t_accum < self._manoeuvre_end_s:
                return self._out(0.0, -TURN_SPEED)
            self._enter_state(CLEAR_CHECK)
            return self._out(0.0, 0.0)

        return self._out(0.0, 0.0)

    # ── FOLLOWING drive ────────────────────────────────────────────────
    def _follow_step(self, dx_norm: float, vx_norm: float, dist_m: float,
                     dt: float, obs: dict) -> tuple[float, float]:
        # Dual dead-zone gate — kills straight-back oscillation.
        abs_dx = abs(dx_norm)
        ang_error = dx_norm
        if abs_dx < PAN_DEADZONE_NORM and abs(vx_norm) < PAN_VEL_DEADZONE:
            ang_error = 0.0

        # Asymmetric gains: near-center → low KP, far → full KP.
        self._pan.kp = PAN_KP_NEAR if abs_dx < NEAR_CENTER_BAND else PAN_KP_FAR

        # Pan PID. Sign: dx > 0 means face is right of center. On Jackie's
        # cmd_vel_mux positive angular.z is CCW (rotates left) — to face a
        # target on the right we need negative angular.z. So the output is
        # negated below after all additive terms.
        raw_ang = self._pan.compute(ang_error, dx_norm, dt)

        # Distance PID.
        dist_err = dist_m - TARGET_FOLLOW_DIST_M
        if abs(dist_err) < DIST_DEADZONE_M:
            dist_err = 0.0
        raw_lin = self._dist.compute(dist_err, dist_m, dt)

        # Backstepping cross-coupling: keeps the chassis heading aligned to
        # target trajectory when bearing error is small. θ_err ≈ dx_norm ·
        # (FRAME_WIDTH_PX / FOCAL_LENGTH_PX) in the small-angle regime.
        theta_err = dx_norm * (FRAME_WIDTH_PX / FOCAL_LENGTH_PX)
        backstep = BACKSTEP_GAIN * raw_lin * math.sin(theta_err)
        raw_ang += backstep

        # LIDAR modulation (present + fresh).
        lin, ang = raw_lin, raw_ang
        forward_min = _forward_min_range(obs, LIDAR_BUBBLE_ARC_RAD)

        if math.isfinite(forward_min):
            # Safety bubble: stop linear if something's too close.
            if forward_min < LIDAR_STOP_RANGE_M:
                lin = 0.0
                self._collision_frame_count += 1
            else:
                # Smooth speed-scaling between STOP and SLOW thresholds.
                scale = (forward_min - LIDAR_STOP_RANGE_M) / (LIDAR_SLOW_RANGE_M - LIDAR_STOP_RANGE_M)
                scale = max(0.0, min(1.0, scale))
                lin *= scale
                self._collision_frame_count = 0

            # Follow-the-gap heading bias: when a close obstacle sits on the
            # direct heading, add a rotation term toward the widest gap so
            # we steer around it while still trying to track the face.
            if forward_min < LIDAR_GAP_TRIGGER_M:
                gap_theta = _widest_gap_bearing(
                    obs, LIDAR_GAP_ARC_RAD, LIDAR_GAP_TRIGGER_M
                )
                if gap_theta is not None:
                    face_theta = theta_err
                    ang += LIDAR_GAP_BIAS_GAIN * (gap_theta - face_theta)

            # If we've been stuck against the bubble for ~0.5s with no face
            # progress, escalate to OBSTACLE_TURN so we actually break free.
            if self._collision_frame_count >= 5:
                self._enter_state(OBSTACLE_TURN)
                return 0.0, 0.0
        else:
            # No LIDAR — fall back to vision proximity guard.
            if dist_m < COLLISION_DIST_M + 0.025:
                self._collision_frame_count += 1
            else:
                self._collision_frame_count = 0
            if self._collision_frame_count >= 3:
                lin = 0.0

        # Final sign flip + clamp + slew.
        ang = -ang
        lin = max(-MAX_LINEAR, min(MAX_LINEAR, lin))
        ang = max(-MAX_ANGULAR, min(MAX_ANGULAR, ang))
        lin = self._slew(self._prev_linear, lin, LINEAR_SLEW, dt)
        ang = self._slew(self._prev_angular, ang, ANGULAR_SLEW, dt)
        return lin, ang

    # ── Filter updates ─────────────────────────────────────────────────
    def _update_cx(self, face, dt, obs) -> tuple[float, float]:
        """Returns (dx_norm, vx_norm) from the alpha-beta filter."""
        if face is None:
            # Long gap → reset; short gap → coast on predicted state.
            now = self._t_accum
            if self._last_cx_time is None or (now - self._last_cx_time) > INPUT_STALE_RESET_S:
                self._cx_filter.reset()
                return 0.0, 0.0
            # Use coasted state without injecting a measurement.
            if self._cx_filter.x is None:
                return 0.0, 0.0
            cx_px, vx_px = self._cx_filter.x, self._cx_filter.v
        else:
            cx_px, vx_px = self._cx_filter.update(face["cx_px"], dt)
            self._last_cx_time = self._t_accum

        frame_w = float(obs.get("frame_width_px") or FRAME_WIDTH_PX)
        dx_norm = (cx_px - frame_w / 2.0) / frame_w
        vx_norm = vx_px / frame_w
        return dx_norm, vx_norm

    def _update_distance(self, face) -> float:
        w_px = face["w_px"]
        if w_px <= 0:
            return float("inf")
        raw = (FACE_WIDTH_M * FOCAL_LENGTH_PX) / w_px
        if self._smooth_dist is None:
            self._smooth_dist = raw
            return raw
        # Outlier reject — face-width jitter is the dominant error source.
        if abs(raw - self._smooth_dist) > DIST_OUTLIER_RATIO * self._smooth_dist:
            return self._smooth_dist
        self._smooth_dist = DIST_EMA_ALPHA * raw + (1 - DIST_EMA_ALPHA) * self._smooth_dist
        return self._smooth_dist

    # ── FSM plumbing ───────────────────────────────────────────────────
    def _enter_state(self, next_state: str) -> None:
        self._fsm = next_state
        if next_state == FOLLOWING:
            self._pan.reset()
            self._dist.reset()
            self._collision_frame_count = 0
        elif next_state == SCAN_ROTATE:
            self.locked_track_id = None
            self._scan_direction = 1.0 if self._cx_filter.v >= 0 else -1.0
            self._start_manoeuvre(SCAN_45_RAD, self._scan_direction)
        elif next_state == OBSTACLE_TURN:
            self._start_manoeuvre(SCAN_90_RAD, -1.0)
            self._collision_frame_count = 0
        elif next_state == COLLISION_STOP:
            self._manoeuvre_end_s = self._t_accum + COLLISION_PAUSE_S
            self._obstacle_trip = 0
        elif next_state == COLLISION_TURN:
            self._start_manoeuvre(SCAN_45_RAD, -1.0)
        self._logger.log_state_change(self._fsm, next_state, self._t_accum)

    def _start_manoeuvre(self, angle_rad: float, direction: float) -> None:
        self._scan_direction = direction
        self._manoeuvre_end_s = self._t_accum + (angle_rad / TURN_SPEED)

    # ── Helpers ────────────────────────────────────────────────────────
    def _extract_face(self, obs: dict) -> dict | None:
        if not obs.get("face_visible") or obs.get("face_cx") is None:
            return None
        frame_w = float(obs.get("frame_width_px") or FRAME_WIDTH_PX)
        track_id = obs.get("track_id")
        if self.locked_track_id is not None and track_id is not None \
                and track_id != self.locked_track_id:
            return None
        w_px = obs.get("face_w_px")
        if w_px is None:
            area = obs.get("face_area") or 0
            w_px = math.sqrt(max(area, 1) / 1.3)
        cx_px = float(obs["face_cx"]) * frame_w
        return {"cx_px": cx_px, "w_px": float(w_px)}

    @staticmethod
    def _slew(prev: float, target: float, max_rate: float, dt: float) -> float:
        max_step = max_rate * dt
        delta = target - prev
        if delta > max_step:
            return prev + max_step
        if delta < -max_step:
            return prev - max_step
        return target

    def _out(self, linear: float, angular: float, obs: dict= None) -> dict:
        self._prev_linear = linear
        self._prev_angular = angular
        
        face_visible = obs.get("face_visible") if obs is not None else None
        
        self._logger.log_tick({
            "t_accum":          self._t_accum,
            "fsm_state":        self._fsm,
            "face_visible":     face_visible,
            "dist_m":           self._smooth_dist,
            "linear_cmd":       linear,
            "angular_cmd":      angular,
        })
       
        return {"linear": linear, "angular": angular}
