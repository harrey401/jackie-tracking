"""temperature_check.py — Face temperature scanner for Jackie's eye camera.

Uses OpenCV to detect and track faces from Jackie's camera feed, then reads
forehead temperature from an attached thermal sensor (MLX90640, AMG8833, or
FLIR Lepton). Falls back to a plausible simulated reading when no thermal
hardware is present (useful for development / offline testing).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT — RGB cameras cannot measure temperature.
Real readings require an IR/thermal camera co-mounted with Jackie's eye.
Supported hardware drivers (install the one you have):
  • MLX90640  (I²C, 32×24 px) → pip install adafruit-circuitpython-mlx90640
  • AMG8833    (I²C, 8×8 px)  → pip install adafruit-circuitpython-amg88xx
  • FLIR Lepton (SPI via PureThermal board) → pip install pylepton
  • Seek Thermal → pip install seekcamera
  • Any OpenCV-accessible thermal cam → set THERMAL_SOURCE to its device index
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Two modes of operation
──────────────────────
1. STANDALONE  — call `run_camera_loop()` directly to open Jackie's camera,
   detect faces with OpenCV, overlay the temperature reading, and display the
   live annotated feed. Good for local testing.

2. INTEGRATED  — the `Logic` class follows the same reset/step(obs) contract
   used by face_user.py and follow_mode.py. The SMAIT server can load it as a
   skill. In this mode Jackie will speak or display the temperature; no live
   OpenCV window is opened on the server (headless).

Sign conventions, obs keys, and return dict format are identical to the
other files in this repo (see README.md).
"""

from __future__ import annotations

import time
import math
import random
import logging
from collections import deque
from typing import Optional

import cv2   # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TUNABLES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Camera ──────────────────────────────────────────────────────────────────
# Device index for Jackie's RGB eye camera (used in standalone mode).
RGB_CAMERA_INDEX = 0

# OpenCV face detector. "haarcascade" is CPU-friendly; "dnn" is more accurate.
# Choices: "haarcascade" | "dnn"
FACE_DETECTOR = "haarcascade"

# DNN model paths (only used when FACE_DETECTOR == "dnn")
# Download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
DNN_PROTOTXT  = "deploy.prototxt"
DNN_CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONFIDENCE_THRESHOLD = 0.55

# ── Thermal sensor ──────────────────────────────────────────────────────────
# Which thermal backend to attempt, in priority order.
# Each driver is imported lazily so missing libraries don't crash import.
# Choices: "mlx90640" | "amg8833" | "flir_lepton" | "seek" | "opencv_thermal" | "simulate"
# "simulate" is always the final fallback.
THERMAL_BACKENDS = ["mlx90640", "amg8833", "flir_lepton", "seek", "simulate"]

# Device index if the thermal camera is exposed as an OpenCV VideoCapture source.
THERMAL_SOURCE = 1  # change if your thermal cam appears on a different index

# ── Temperature thresholds (°C) ─────────────────────────────────────────────
TEMP_NORMAL_LOW  = 36.1   # °C — below this: sub-normal
TEMP_NORMAL_HIGH = 37.5   # °C — above this: elevated / fever
TEMP_FEVER_HIGH  = 38.0   # °C — above this: high fever alert

# ── Display / annotation ─────────────────────────────────────────────────────
DISPLAY_FAHRENHEIT = False   # True → show °F alongside °C
SHOW_THERMAL_OVERLAY = True  # blend pseudo-colour thermal patch over face ROI
OVERLAY_ALPHA = 0.35         # thermal overlay opacity (0=invisible, 1=opaque)

# ── Smoothing ────────────────────────────────────────────────────────────────
TEMP_EMA_ALPHA  = 0.25   # low-pass on raw temp readings (lower = smoother)
TEMP_HISTORY_N  = 8      # rolling window for median filter (additional noise kill)
FACE_EMA_ALPHA  = 0.35   # smoothing on face bbox (kills jitter in the overlay)

# ── Simulation fallback ──────────────────────────────────────────────────────
# When no thermal hardware is found these parameters drive the simulated reading.
SIM_BASE_TEMP   = 36.6   # °C baseline (healthy human forehead)
SIM_NOISE_STD   = 0.15   # °C std-dev (realistic sensor noise)
SIM_DRIFT_SPEED = 0.003  # °C per step drift magnitude (slow random walk)


# ═══════════════════════════════════════════════════════════════════════════════
# THERMAL BACKENDS
# ═══════════════════════════════════════════════════════════════════════════════

class _ThermalBase:
    """Abstract base for thermal sensor drivers."""

    def read_grid(self) -> Optional[np.ndarray]:
        """Return a 2-D float32 array of temperatures (°C), or None on failure."""
        raise NotImplementedError

    def close(self):
        pass


class _MLX90640Driver(_ThermalBase):
    """Adafruit MLX90640 — 32×24 IR array, I²C."""

    def __init__(self):
        import board, busio
        import adafruit_mlx90640
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
        self._mlx = adafruit_mlx90640.MLX90640(i2c)
        self._mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        self._frame = [0.0] * 768  # 32×24

    def read_grid(self) -> Optional[np.ndarray]:
        try:
            self._mlx.getFrame(self._frame)
            return np.array(self._frame, dtype=np.float32).reshape(24, 32)
        except Exception as exc:
            logger.warning("MLX90640 read error: %s", exc)
            return None


class _AMG8833Driver(_ThermalBase):
    """Adafruit AMG8833 — 8×8 IR array, I²C."""

    def __init__(self):
        import board, busio
        import adafruit_amg88xx
        i2c = busio.I2C(board.SCL, board.SDA)
        self._sensor = adafruit_amg88xx.AMG88XX(i2c)

    def read_grid(self) -> Optional[np.ndarray]:
        try:
            return np.array(self._sensor.pixels, dtype=np.float32)  # 8×8
        except Exception as exc:
            logger.warning("AMG8833 read error: %s", exc)
            return None


class _FLIRLeptonDriver(_ThermalBase):
    """FLIR Lepton via PureThermal board (pylepton library)."""

    def __init__(self):
        import pylepton
        self._lepton = pylepton.Capture("/dev/spidev0.0").__enter__()

    def read_grid(self) -> Optional[np.ndarray]:
        try:
            frame = np.zeros((60, 80, 1), dtype=np.uint16)
            self._lepton.capture(frame)
            # Convert raw LWIR counts to approximate °C (Lepton 3.x)
            kelvin = frame[:, :, 0].astype(np.float32) * 0.01
            return kelvin - 273.15
        except Exception as exc:
            logger.warning("FLIR Lepton read error: %s", exc)
            return None

    def close(self):
        try:
            self._lepton.__exit__(None, None, None)
        except Exception:
            pass


class _SeekDriver(_ThermalBase):
    """Seek Thermal camera (seekcamera-python SDK)."""

    def __init__(self):
        import seekcamera
        self._mgr = seekcamera.SeekCameraManager(seekcamera.SeekCameraIOType.USB)
        self._cam: Optional[seekcamera.SeekCamera] = None
        self._frame: Optional[np.ndarray] = None

        def on_event(cam, evt, _mgr):
            if evt == seekcamera.SeekCameraManagerEvent.CONNECT:
                self._cam = cam
                cam.register_frame_available_callback(self._on_frame, None)
                cam.capture_session_start(seekcamera.SeekCameraFrameFormat.THERMOGRAPHY_FLOAT)

        self._mgr.register_event_callback(on_event, None)

    def _on_frame(self, _cam, frame, _user):
        self._frame = frame.thermography_float.data.copy()

    def read_grid(self) -> Optional[np.ndarray]:
        return self._frame  # already °C float32

    def close(self):
        if self._cam:
            self._cam.capture_session_stop()


class _OpenCVThermalDriver(_ThermalBase):
    """Thermal cam exposed as a standard VideoCapture device (raw 16-bit mode)."""

    def __init__(self, device_index: int = THERMAL_SOURCE):
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open thermal cam at index {device_index}")

    def read_grid(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok:
            return None
        # Assume 16-bit grayscale encoding of millikelvins (common for UVC thermal cams)
        if frame.dtype == np.uint16:
            return frame.astype(np.float32) * 0.01 - 273.15
        # Fallback: treat 8-bit as a scaled 0–50 °C range
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return gray.astype(np.float32) / 255.0 * 50.0

    def close(self):
        self._cap.release()


class _SimulatedDriver(_ThermalBase):
    """
    Simulated thermal sensor for development / offline testing.

    Produces a small grid of 'temperatures' centred around a realistic human
    forehead value with sensor noise and a slow random walk. The resulting
    grid is 8×8 to match AMG8833 resolution.

    ⚠️ Values are NOT real. Disable this in production by ensuring real
    thermal hardware is present and listed before "simulate" in THERMAL_BACKENDS.
    """

    def __init__(self):
        self._base = SIM_BASE_TEMP
        self._drift = 0.0
        logger.warning(
            "TemperatureCheck: No real thermal hardware found — using SIMULATION. "
            "Readings are NOT accurate."
        )

    def read_grid(self) -> np.ndarray:
        # Slow random walk
        self._drift += random.gauss(0, SIM_DRIFT_SPEED)
        self._drift = max(-1.0, min(1.0, self._drift))
        center = self._base + self._drift
        # 8×8 grid: hottest in the middle (forehead peak), cooler at edges
        grid = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                dist = math.hypot(r - 3.5, c - 3.5) / 5.0
                grid[r, c] = center - dist * 0.8 + random.gauss(0, SIM_NOISE_STD)
        return grid


def _build_thermal_driver() -> _ThermalBase:
    """Try each backend in THERMAL_BACKENDS order, return the first that works."""
    drivers = {
        "mlx90640":     (_MLX90640Driver,    {}),
        "amg8833":      (_AMG8833Driver,     {}),
        "flir_lepton":  (_FLIRLeptonDriver,  {}),
        "seek":         (_SeekDriver,        {}),
        "opencv_thermal": (_OpenCVThermalDriver, {}),
        "simulate":     (_SimulatedDriver,   {}),
    }
    for name in THERMAL_BACKENDS:
        cls, kwargs = drivers.get(name, (None, {}))
        if cls is None:
            continue
        try:
            drv = cls(**kwargs)
            logger.info("ThermalDriver: using %s", name)
            return drv
        except Exception as exc:
            logger.debug("ThermalDriver %s unavailable: %s", name, exc)
    return _SimulatedDriver()   # guaranteed fallback


# ═══════════════════════════════════════════════════════════════════════════════
# FACE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _build_face_detector():
    """Return a callable (frame_bgr) → list[(x, y, w, h)]."""

    if FACE_DETECTOR == "dnn":
        try:
            net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_CAFFEMODEL)

            def _detect_dnn(frame):
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0,
                    (300, 300), (104.0, 177.0, 123.0)
                )
                net.setInput(blob)
                detections = net.forward()
                faces = []
                for i in range(detections.shape[2]):
                    conf = detections[0, 0, i, 2]
                    if conf < DNN_CONFIDENCE_THRESHOLD:
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
                return faces

            logger.info("FaceDetector: using DNN (SSD ResNet-10)")
            return _detect_dnn
        except Exception as exc:
            logger.warning("DNN detector unavailable (%s), falling back to Haar.", exc)

    # ── Haar cascade (default) ────────────────────────────────────────────────
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def _detect_haar(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return list(cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        ))

    logger.info("FaceDetector: using Haar cascade")
    return _detect_haar


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_forehead_temp(
    thermal_grid: np.ndarray,
    face_cx_norm: float,
    face_cy_norm: float,
    face_w_norm: float,
    frame_w: int,
    frame_h: int,
) -> float:
    """
    Map the face bounding box onto the thermal grid and return the median
    temperature in the estimated forehead region (top third of the face bbox).

    Args:
        thermal_grid:  H×W float32 array of °C values from the thermal sensor.
        face_cx_norm:  face centre X, normalised 0‥1 (matches obs["face_cx"]).
        face_cy_norm:  face centre Y, normalised 0‥1 (matches obs["face_cy"]).
        face_w_norm:   face width, normalised 0‥1 (matches obs["face_w_norm"]).
        frame_w/h:     RGB frame dimensions in pixels.

    Returns:
        Estimated forehead temperature in °C.
    """
    gh, gw = thermal_grid.shape[:2]

    # Face bbox in normalised coordinates
    face_h_norm = face_w_norm * 1.3   # assume 1.3:1 height-to-width ratio
    fx0 = face_cx_norm - face_w_norm / 2
    fy0 = face_cy_norm - face_h_norm / 2

    # Forehead = top third of the face bbox
    forehead_y0 = fy0
    forehead_y1 = fy0 + face_h_norm / 3.0
    forehead_x0 = fx0 + face_w_norm * 0.1
    forehead_x1 = fx0 + face_w_norm * 0.9

    # Map to thermal grid pixel indices
    r0 = max(0, int(forehead_y0 * gh))
    r1 = min(gh, int(forehead_y1 * gh) + 1)
    c0 = max(0, int(forehead_x0 * gw))
    c1 = min(gw, int(forehead_x1 * gw) + 1)

    roi = thermal_grid[r0:r1, c0:c1]
    if roi.size == 0:
        # Fallback: whole-grid median (shouldn't happen in normal use)
        return float(np.median(thermal_grid))

    return float(np.median(roi))


def _temp_label(temp_c: float) -> tuple[str, tuple[int, int, int]]:
    """Return (display_label, BGR_colour) for the given temperature."""
    if temp_c < TEMP_NORMAL_LOW:
        label = f"{temp_c:.1f} °C  LOW"
        colour = (255, 200, 0)      # cyan-ish
    elif temp_c <= TEMP_NORMAL_HIGH:
        label = f"{temp_c:.1f} °C  NORMAL"
        colour = (0, 220, 60)       # green
    elif temp_c <= TEMP_FEVER_HIGH:
        label = f"{temp_c:.1f} °C  ELEVATED"
        colour = (0, 165, 255)      # orange
    else:
        label = f"{temp_c:.1f} °C  FEVER"
        colour = (0, 0, 220)        # red

    if DISPLAY_FAHRENHEIT:
        temp_f = temp_c * 9 / 5 + 32
        label += f" / {temp_f:.1f} °F"

    return label, colour


# ═══════════════════════════════════════════════════════════════════════════════
# ANNOTATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _annotate_frame(
    frame: np.ndarray,
    faces: list,
    temps: list[float],
    thermal_grid: Optional[np.ndarray],
) -> np.ndarray:
    """Draw bounding boxes, temperature labels, and optional thermal overlay."""

    out = frame.copy()

    for (x, y, w, h), temp in zip(faces, temps):
        label, colour = _temp_label(temp)

        # ── Optional thermal pseudo-colour overlay ──────────────────────────
        if SHOW_THERMAL_OVERLAY and thermal_grid is not None:
            # Resize thermal grid to the face ROI
            roi = out[y:y + h, x:x + w]
            if roi.size > 0:
                # Normalise thermal values to 0-255 for colormap
                t_min, t_max = thermal_grid.min(), thermal_grid.max()
                span = max(t_max - t_min, 1.0)
                norm = ((thermal_grid - t_min) / span * 255).astype(np.uint8)
                coloured = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
                coloured_resized = cv2.resize(coloured, (w, h))
                blended = cv2.addWeighted(roi, 1 - OVERLAY_ALPHA, coloured_resized, OVERLAY_ALPHA, 0)
                out[y:y + h, x:x + w] = blended

        # ── Bounding box ────────────────────────────────────────────────────
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)

        # ── Forehead crosshair ──────────────────────────────────────────────
        fh_cy = y + h // 6          # top sixth = forehead centre
        fh_cx = x + w // 2
        cv2.drawMarker(out, (fh_cx, fh_cy), colour,
                       cv2.MARKER_CROSS, markerSize=12, thickness=1)

        # ── Temperature label ────────────────────────────────────────────────
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        lx = max(0, x)
        ly = max(th + baseline + 4, y - 6)
        # Background pill
        cv2.rectangle(out,
                      (lx - 2, ly - th - baseline - 4),
                      (lx + tw + 2, ly + baseline),
                      (30, 30, 30), cv2.FILLED)
        cv2.putText(out, label, (lx, ly - baseline), font, font_scale, colour, thickness)

    # ── Sensor mode watermark ────────────────────────────────────────────────
    mode = "THERMAL" if not isinstance(_thermal_driver_singleton, _SimulatedDriver) else "SIMULATED"
    cv2.putText(out, f"Temp sensor: {mode}", (8, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS  (initialised lazily)
# ═══════════════════════════════════════════════════════════════════════════════

_thermal_driver_singleton: Optional[_ThermalBase] = None
_face_detector_singleton = None


def _get_thermal() -> _ThermalBase:
    global _thermal_driver_singleton
    if _thermal_driver_singleton is None:
        _thermal_driver_singleton = _build_thermal_driver()
    return _thermal_driver_singleton


def _get_detector():
    global _face_detector_singleton
    if _face_detector_singleton is None:
        _face_detector_singleton = _build_face_detector()
    return _face_detector_singleton


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE CAMERA LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_camera_loop(camera_index: int = RGB_CAMERA_INDEX):
    """
    Open Jackie's RGB camera, detect faces, read temperatures, display live.

    Press 'q' or ESC to quit.
    This function blocks until the window is closed.

    Example:
        python temperature_check.py
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB camera at index {camera_index}.")

    detect = _get_detector()
    thermal = _get_thermal()

    # Per-face EMA state (keyed by detection index — simple, stable enough)
    smoothed_bbox: list[tuple] = []
    temp_history: deque = deque(maxlen=TEMP_HISTORY_N)
    ema_temp: Optional[float] = None

    logger.info("Starting temperature check loop. Press q or ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.error("Frame capture failed — camera disconnected?")
            break

        faces = detect(frame)
        h, w = frame.shape[:2]

        # ── Read thermal grid once per RGB frame ─────────────────────────────
        grid = thermal.read_grid()

        temps: list[float] = []
        for i, (fx, fy, fw, fh) in enumerate(faces):
            cx_norm = (fx + fw / 2) / w
            cy_norm = (fy + fh / 2) / h
            fw_norm = fw / w

            if grid is not None:
                raw_temp = _extract_forehead_temp(grid, cx_norm, cy_norm, fw_norm, w, h)
            else:
                raw_temp = SIM_BASE_TEMP  # sensor read failed

            # EMA smoothing
            if ema_temp is None:
                ema_temp = raw_temp
            else:
                ema_temp = TEMP_EMA_ALPHA * raw_temp + (1 - TEMP_EMA_ALPHA) * ema_temp

            # Rolling median
            temp_history.append(ema_temp)
            final_temp = float(np.median(list(temp_history)))
            temps.append(final_temp)

        annotated = _annotate_frame(frame, faces, temps, grid)
        cv2.imshow("Jackie — Temperature Check", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    _get_thermal().close()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED LOGIC CLASS  (SMAIT server contract)
# ═══════════════════════════════════════════════════════════════════════════════

class Logic:
    """
    SMAIT server skill contract — same shape as face_user.py / follow_mode.py.

    step(obs) returns the usual {linear, angular} dict so the chassis does
    not move (this skill is sensors-only), and also populates obs with
    'temperature_c' and 'temperature_status' for downstream consumers (e.g.,
    Jackie's speech / display skills).

    Activation:
        Jackie touchscreen → Tracking Tests → Temperature Check → START
    """

    def __init__(self):
        self._thermal = _get_thermal()
        self._detect  = _get_detector()
        self._ema_temp: Optional[float] = None
        self._temp_history: deque = deque(maxlen=TEMP_HISTORY_N)
        self._last_reading_time: float = 0.0
        self._reading_interval_s: float = 0.5   # don't hammer the thermal sensor

    def reset(self):
        self._ema_temp = None
        self._temp_history.clear()
        self._last_reading_time = 0.0

    def step(self, obs: dict) -> dict:
        """
        Called ~10 Hz by the SMAIT server.

        Reads temperature when a face is visible and the sensor cooldown has
        elapsed. Writes results back into obs so other skills can read them.
        Jackie does NOT move — linear and angular are always 0.
        """
        now = time.monotonic()

        if not obs.get("face_visible"):
            # No face → no reading; clear last result
            obs["temperature_c"] = None
            obs["temperature_status"] = "NO_FACE"
            return {"linear": 0.0, "angular": 0.0}

        # Throttle sensor reads
        if now - self._last_reading_time < self._reading_interval_s:
            # Return last known values without blocking
            obs["temperature_c"] = (
                float(np.median(list(self._temp_history)))
                if self._temp_history else None
            )
            obs["temperature_status"] = self._status_from_temp(obs["temperature_c"])
            return {"linear": 0.0, "angular": 0.0}

        self._last_reading_time = now

        frame_w = obs.get("frame_width_px", 640)
        frame_h = obs.get("frame_height_px", 480)
        cx_norm = obs.get("face_cx", 0.5)
        cy_norm = obs.get("face_cy", 0.5)
        fw_norm = obs.get("face_w_norm", 0.2)

        grid = self._thermal.read_grid()
        if grid is not None:
            raw_temp = _extract_forehead_temp(
                grid, cx_norm, cy_norm, fw_norm, frame_w, frame_h
            )
        else:
            logger.warning("Thermal sensor read returned None — skipping this tick.")
            obs["temperature_c"] = None
            obs["temperature_status"] = "SENSOR_ERROR"
            return {"linear": 0.0, "angular": 0.0}

        # EMA + median filter
        if self._ema_temp is None:
            self._ema_temp = raw_temp
        else:
            self._ema_temp = TEMP_EMA_ALPHA * raw_temp + (1 - TEMP_EMA_ALPHA) * self._ema_temp

        self._temp_history.append(self._ema_temp)
        final_temp = float(np.median(list(self._temp_history)))

        obs["temperature_c"] = final_temp
        obs["temperature_status"] = self._status_from_temp(final_temp)

        logger.info(
            "Temperature check: %.1f °C — %s",
            final_temp, obs["temperature_status"]
        )

        return {"linear": 0.0, "angular": 0.0}

    @staticmethod
    def _status_from_temp(temp_c: Optional[float]) -> str:
        if temp_c is None:
            return "UNKNOWN"
        if temp_c < TEMP_NORMAL_LOW:
            return "LOW"
        if temp_c <= TEMP_NORMAL_HIGH:
            return "NORMAL"
        if temp_c <= TEMP_FEVER_HIGH:
            return "ELEVATED"
        return "FEVER"


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_camera_loop()
