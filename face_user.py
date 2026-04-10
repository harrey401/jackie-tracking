"""Face-the-user tracking logic — rotation only, no forward motion.

This file controls how Jackie orients toward the user while it is speaking.
Edit freely. The server hot-reloads this file every time you save it.

Contract:
    Logic.reset()         — called once when tracking activates (TTS starts)
    Logic.step(obs) -> act — called ~10 times per second

`obs` is what the server sees right now:
    face_visible : bool            True if a face was seen in the last 1s
    face_cx      : float or None   Face center X, 0.0=left, 0.5=center, 1.0=right
    face_cy      : float or None   Face center Y, 0.0=top, 1.0=bottom
    face_w_norm  : float or None   Face width as fraction of frame (0..1)
    face_age_s   : float           Seconds since last face seen (inf if never)
    dt           : float           Seconds since last step (usually ~0.1)

`act` is what you want Jackie to do:
    linear  : float   m/s forward (KEEP 0 for face-the-user, we don't drive)
    angular : float   rad/s rotation. Positive = turn LEFT. Negative = turn RIGHT.

Safety caps enforced by the server AFTER your step():
    |linear|  <= 0.25 m/s
    |angular| <= 1.50 rad/s
You can return anything, it will be clamped.
"""

# === TUNABLES — change these freely ============================================

PAN_KP       = 1.5    # proportional gain on horizontal error
DEADZONE     = 0.05   # ignore errors smaller than this (face close to center)
MAX_ANGULAR  = 0.6    # rad/s cap — gentle for rotation-only
SMOOTHING    = 0.3    # 0.0 = no smoothing, 1.0 = very smooth (slow to react)
LOST_HOLD_S  = 0.5    # keep last command for this long after face disappears
                      # then stop. Set higher to make Jackie hold its pose longer.


# === LOGIC ======================================================================

class Logic:
    def __init__(self):
        self.reset()

    def reset(self):
        """Called when tracking turns on. Clear any memory between sessions."""
        self._prev_angular = 0.0
        self._last_known_angular = 0.0

    def step(self, obs):
        """Decide what velocity to send. Return {'linear': ..., 'angular': ...}."""

        # --- No face right now ---
        if not obs["face_visible"] or obs["face_cx"] is None:
            # If we saw one very recently, keep holding the last command briefly.
            if obs["face_age_s"] < LOST_HOLD_S:
                return {"linear": 0.0, "angular": self._last_known_angular}
            # Otherwise stop rotating. Do NOT spin to search.
            self._prev_angular = 0.0
            return {"linear": 0.0, "angular": 0.0}

        # --- Face visible: center it ---
        error = obs["face_cx"] - 0.5  # + = face is right of center

        # Deadzone: don't fidget when face is already centered
        if abs(error) < DEADZONE:
            target_angular = 0.0
        else:
            # Proportional control. Negative sign: face right → turn right (neg angular).
            target_angular = -PAN_KP * error
            target_angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, target_angular))

        # Exponential smoothing to avoid jerky motion
        angular = (1.0 - SMOOTHING) * target_angular + SMOOTHING * self._prev_angular
        self._prev_angular = angular
        self._last_known_angular = angular

        return {"linear": 0.0, "angular": angular}
