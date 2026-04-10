"""Follow-me tracking logic — drive toward and face the user.

This file controls how Jackie follows a person. Edit freely. The server
hot-reloads this file every time you save it.

Contract:
    Logic.reset()         — called once when follow mode activates
    Logic.step(obs) -> act — called ~10 times per second

`obs` is what the server sees right now:
    face_visible : bool            True if a face was seen in the last 1s
    face_cx      : float or None   Face center X, 0.0=left, 0.5=center, 1.0=right
    face_cy      : float or None   Face center Y, 0.0=top, 1.0=bottom
    face_area    : int or None     Face bbox area in px² (bigger = closer)
    face_age_s   : float           Seconds since last face seen (inf if never)
    dt           : float           Seconds since last step (usually ~0.1)
    robot_theta  : float           Current robot heading in radians

`act` is what you want Jackie to do:
    linear  : float   m/s forward. Positive = forward, negative = backward
    angular : float   rad/s rotation. Positive = LEFT, negative = RIGHT

Safety caps enforced by the server AFTER your step():
    |linear|  <= 0.25 m/s
    |angular| <= 1.50 rad/s
You can return anything, it will be clamped.
"""

# === TUNABLES — change these freely ============================================

# Target distance — bigger face_area means user is closer.
# 10_000 px² ~= comfortable conversation distance on 320px frame.
FACE_AREA_TARGET        = 8000
FACE_AREA_FACE_TO_FACE  = 15000   # "close enough, stop"

# Pan (rotation) gains
PAN_KP       = 1.2
PAN_DEADZONE = 0.05   # ignore small horizontal errors (normalized 0..1)

# Distance (forward) gains
DIST_KP      = 0.00004  # small because error is in pixels²
DIST_DEADZONE= 1500     # don't creep on small errors

# Speed caps (the server also clamps — these are your soft limits)
MAX_ANGULAR  = 1.0
MAX_LINEAR   = 0.25

# Lost-face behaviour
LOST_HOLD_S  = 0.8    # hold last command for this long after face disappears
SCAN_ENABLED = False  # if True, rotate to search for face. If False, just stop.
SCAN_SPEED   = 0.4    # rad/s when scanning (only used if SCAN_ENABLED)

# Smoothing
LINEAR_SMOOTHING  = 0.4
ANGULAR_SMOOTHING = 0.3


# === LOGIC ======================================================================

class Logic:
    def __init__(self):
        self.reset()

    def reset(self):
        """Called when follow mode turns on. Clear any memory between sessions."""
        self._prev_linear = 0.0
        self._prev_angular = 0.0
        self._last_known_linear = 0.0
        self._last_known_angular = 0.0

    def step(self, obs):
        """Decide what velocity to send. Return {'linear': ..., 'angular': ...}."""

        # --- No face right now ---
        if not obs["face_visible"] or obs["face_cx"] is None:
            # Briefly hold last command so we don't jitter on a one-frame dropout
            if obs["face_age_s"] < LOST_HOLD_S:
                return {
                    "linear": self._last_known_linear,
                    "angular": self._last_known_angular,
                }
            # Face truly lost
            if SCAN_ENABLED:
                return {"linear": 0.0, "angular": SCAN_SPEED}
            # Don't spin. Just stop.
            self._prev_linear = 0.0
            self._prev_angular = 0.0
            return {"linear": 0.0, "angular": 0.0}

        # --- Face visible: compute pan + distance ---
        pan_error = obs["face_cx"] - 0.5  # + = face is right of center

        if abs(pan_error) < PAN_DEADZONE:
            target_angular = 0.0
        else:
            target_angular = -PAN_KP * pan_error  # face right → turn right
            target_angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, target_angular))

        # Distance: positive error = too far (face too small) = drive forward
        area = obs["face_area"] or 0
        dist_error = FACE_AREA_TARGET - area

        if abs(dist_error) < DIST_DEADZONE or area >= FACE_AREA_FACE_TO_FACE:
            target_linear = 0.0
        else:
            target_linear = DIST_KP * dist_error
            target_linear = max(-MAX_LINEAR, min(MAX_LINEAR, target_linear))

        # Smooth
        linear  = (1.0 - LINEAR_SMOOTHING)  * target_linear  + LINEAR_SMOOTHING  * self._prev_linear
        angular = (1.0 - ANGULAR_SMOOTHING) * target_angular + ANGULAR_SMOOTHING * self._prev_angular

        self._prev_linear = linear
        self._prev_angular = angular
        self._last_known_linear = linear
        self._last_known_angular = angular

        return {"linear": linear, "angular": angular}
