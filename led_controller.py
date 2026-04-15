"""LED Controller — maps robot state to RGB LED colors.

Reads the result dict from Logic.step() in follow_mode.py or face_user.py
and returns an (R, G, B) tuple for the current state.

Usage:
    from led_controller import LEDController

    led = LEDController()

    # In your main loop:
    result = logic.step(obs)
    color = led.compute(result, fsm_state=logic._fsm, face_visible=obs["face_visible"])
    led.publish(color)   # wire to your ROS topic or GPIO driver

Color scheme:
    Red      (255,   0,   0)  — Collision stop / collision turn (emergency)
    Orange   (255,  60,   0)  — Backing up (negative linear velocity)
    Yellow   (255, 200,   0)  — Scanning for face / face lost
    Green    (  0, 255,   0)  — Following and moving forward
    Blue     (  0, 120, 255)  — Following but stationary (at target distance)
    White    (255, 255, 255)  — Idle / unknown state

Publishing:
    Implement publish() for your hardware. Two common options are shown:
      - ROS:  std_msgs/ColorRGBA on a /led_color topic
      - GPIO: direct PWM output via RPi.GPIO or similar
"""

# ── FSM state strings (must match follow_mode.py) ─────────────────────────
FOLLOWING      = "FOLLOWING"
SCAN_ROTATE    = "SCAN_ROTATE"
OBSTACLE_TURN  = "OBSTACLE_TURN"
COLLISION_STOP = "COLLISION_STOP"
COLLISION_TURN = "COLLISION_TURN"
CLEAR_CHECK    = "CLEAR_CHECK"

# ── Named colors (R, G, B) ─────────────────────────────────────────────────
COLOR_RED    = (255,   0,   0)   # emergency / collision
COLOR_ORANGE = (255,  60,   0)   # reversing
COLOR_YELLOW = (255, 200,   0)   # scanning / face lost
COLOR_GREEN  = (  0, 255,   0)   # moving forward
COLOR_BLUE   = (  0, 120, 255)   # locked on, stationary
COLOR_WHITE  = (255, 255, 255)   # idle / unknown


class LEDController:
    """Computes and publishes LED colors based on robot state."""

    def __init__(self, ros_node=None):
        """
        Args:
            ros_node: Optional ROS node handle. If provided, publish() will
                      send a ColorRGBA message on /led_color. If None,
                      publish() prints to stdout (useful for testing).
        """
        self._ros_node = ros_node
        self._publisher = None

        if ros_node is not None:
            self._setup_ros_publisher()

    # ── Color computation ──────────────────────────────────────────────────

    def compute_follow(self, result: dict, fsm_state: str) -> tuple:
        """
        Determine LED color for follow_mode.py states.

        Args:
            result:    The dict returned by Logic.step() — must contain
                       'linear' and 'angular' keys.
            fsm_state: The current FSM state string (e.g. logic._fsm).

        Returns:
            (R, G, B) tuple with values in 0..255.
        """
        linear = result.get("linear", 0.0)

        if fsm_state in (COLLISION_STOP, COLLISION_TURN):
            return COLOR_RED

        if linear < -0.01:
            return COLOR_ORANGE

        if fsm_state == SCAN_ROTATE:
            return COLOR_YELLOW

        if fsm_state == FOLLOWING:
            if linear > 0.01:
                return COLOR_GREEN
            return COLOR_BLUE  # at target distance, holding position

        return COLOR_WHITE  # OBSTACLE_TURN, CLEAR_CHECK, or unknown

    def compute_face_user(self, result: dict, face_visible: bool) -> tuple:
        """
        Determine LED color for face_user.py states.

        Args:
            result:       The dict returned by Logic.step() — must contain
                          'angular' key.
            face_visible: Whether a face is currently detected.

        Returns:
            (R, G, B) tuple with values in 0..255.
        """
        angular = result.get("angular", 0.0)

        if not face_visible:
            return COLOR_YELLOW  # lost face / searching

        if abs(angular) < 0.01:
            return COLOR_BLUE    # face centered, holding still

        return COLOR_GREEN       # actively rotating to track face

    # ── Publishing ─────────────────────────────────────────────────────────

    def publish(self, color: tuple):
        """
        Send the RGB color to the LED hardware.

        Swap in your real transport below. Two options are shown:
          Option A — ROS ColorRGBA topic  (uncomment if using ROS)
          Option B — RPi GPIO PWM         (uncomment if driving GPIO directly)

        Args:
            color: (R, G, B) tuple with values in 0..255.
        """
        r, g, b = color

        # ── Option A: ROS std_msgs/ColorRGBA ──────────────────────────────
        # Uncomment this block and remove the print() fallback below.
        #
        # if self._publisher is not None:
        #     from std_msgs.msg import ColorRGBA
        #     msg = ColorRGBA()
        #     msg.r = r / 255.0
        #     msg.g = g / 255.0
        #     msg.b = b / 255.0
        #     msg.a = 1.0
        #     self._publisher.publish(msg)
        #     return

        # ── Option B: RPi GPIO PWM (3-channel, active-high) ───────────────
        # Uncomment and set your GPIO pin numbers.
        #
        # PIN_R, PIN_G, PIN_B = 17, 27, 22   # BCM pin numbers
        # self._pwm_r.ChangeDutyCycle(r / 255.0 * 100)
        # self._pwm_g.ChangeDutyCycle(g / 255.0 * 100)
        # self._pwm_b.ChangeDutyCycle(b / 255.0 * 100)
        # return

        # ── Fallback: print to stdout (testing / simulation) ──────────────
        label = self._color_label(color)
        print(f"[LED] RGB=({r:3d}, {g:3d}, {b:3d})  {label}")

    # ── ROS setup (internal) ───────────────────────────────────────────────

    def _setup_ros_publisher(self):
        """Create a ROS publisher on /led_color. Called once at init."""
        try:
            from std_msgs.msg import ColorRGBA
            self._publisher = self._ros_node.create_publisher(
                ColorRGBA, "/led_color", qos_profile=10
            )
        except Exception as exc:
            print(f"[LEDController] ROS publisher setup failed: {exc}")
            self._publisher = None

    # ── Utility ────────────────────────────────────────────────────────────

    @staticmethod
    def _color_label(color: tuple) -> str:
        """Human-readable label for a known color tuple."""
        return {
            COLOR_RED:    "🔴 EMERGENCY / COLLISION",
            COLOR_ORANGE: "🟠 REVERSING",
            COLOR_YELLOW: "🟡 SCANNING / FACE LOST",
            COLOR_GREEN:  "🟢 MOVING FORWARD",
            COLOR_BLUE:   "🔵 LOCKED ON / STATIONARY",
            COLOR_WHITE:  "⚪ IDLE",
        }.get(color, f"CUSTOM {color}")


# ── Integration example ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke-test without real hardware.
    led = LEDController()

    # Simulate follow_mode states
    print("=== follow_mode.py states ===")
    test_cases_follow = [
        ({"linear": 0.2,  "angular": -0.3}, FOLLOWING),
        ({"linear": 0.0,  "angular":  0.0}, FOLLOWING),
        ({"linear": 0.0,  "angular":  0.4}, SCAN_ROTATE),
        ({"linear": -0.1, "angular":  0.0}, FOLLOWING),
        ({"linear": 0.0,  "angular":  0.0}, COLLISION_STOP),
        ({"linear": 0.0,  "angular": -0.4}, COLLISION_TURN),
        ({"linear": 0.0,  "angular":  0.0}, CLEAR_CHECK),
    ]
    for result, state in test_cases_follow:
        color = led.compute_follow(result, state)
        print(f"  FSM={state:<16s}  linear={result['linear']:+.2f}  →  ", end="")
        led.publish(color)

    # Simulate face_user states
    print("\n=== face_user.py states ===")
    test_cases_face = [
        ({"angular":  0.5}, True),   # tracking, rotating
        ({"angular":  0.0}, True),   # face centered
        ({"angular": -0.2}, False),  # face lost, decaying
    ]
    for result, visible in test_cases_face:
        color = led.compute_face_user(result, visible)
        print(f"  face_visible={str(visible):<5s}  angular={result['angular']:+.2f}  →  ", end="")
        led.publish(color)
