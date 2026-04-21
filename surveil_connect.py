# ══════════════════════════════════════════════════════════════════════════════════════════
# surveil_connect.py
#
# Shared camera layer for Jackie's four surveillance cameras (mounted 90° apart).
# Provides: CameraStream (RTSP connection + health tracking), tile composition
# helpers, and a standalone 2×2 grid viewer.
#
# Imported by stream_check.py for panorama / health-check functionality.
#
# Camera layout (front = Jackie's face direction):
#       ┌──────────────┬──────────────┐
#       │   FRONT (0°) │  RIGHT (90°) │
#       ├──────────────┼──────────────┤
#       │  LEFT (270°) │  REAR (180°) │
#       └──────────────┴──────────────┘
#
# Press 'q' to quit the standalone viewer.
# ══════════════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import time
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CHASSIS_IP = "192.168.20.22"
RTSP_PORT  = 554
USERNAME   = ""   # e.g. "admin" — leave blank if unauthenticated
PASSWORD   = ""   # e.g. "jackie123"

# Per-camera tile size in the grid (each feed is resized to this before stitching)
TILE_W, TILE_H = 640, 360

# Reconnect tuning
MAX_CONSECUTIVE_FAILS = 30   # ~1 s at 30 fps before tear-down + reopen
RECONNECT_BACKOFF_S   = 2.0


def build_url(channel: int) -> str:
    """Build an RTSP URL for a given channel on the chassis proxy."""
    auth = f"{USERNAME}:{PASSWORD}@" if USERNAME or PASSWORD else ""
    return f"rtsp://{auth}{CHASSIS_IP}:{RTSP_PORT}/live/ch{channel}"


# Camera definitions — (label, rtsp url, grid position).
# Consumed by both the standalone viewer and stream_check.py.
CAMERA_DEFS = [
    {"label": "FRONT  (0°)",   "url": build_url(0), "pos": (0, 0)},
    {"label": "RIGHT  (90°)",  "url": build_url(1), "pos": (0, 1)},
    {"label": "LEFT   (270°)", "url": build_url(2), "pos": (1, 0)},
    {"label": "REAR   (180°)", "url": build_url(3), "pos": (1, 1)},
]


# ──────────────────────────────────────────────────────────────────────────────
# Camera stream  (with auto-reconnect and health tracking)
# ──────────────────────────────────────────────────────────────────────────────

class CameraStream:
    """Wraps a single RTSP stream with reconnect logic and health tracking."""

    def __init__(self, label: str, url: str):
        self.label = label
        self.url   = url
        self.cap: Optional[cv2.VideoCapture] = None
        self.alive = False
        self._fail_count     = 0
        self._last_reconnect = 0.0
        self._fps_samples:  list[float] = []
        self._last_frame_t  = 0.0
        self._open()

    # ── Public API ─────────────────────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame, or None if the stream is unavailable."""
        if self.cap is None or not self.cap.isOpened():
            self._maybe_reconnect()
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._fail_count += 1
            if self._fail_count >= MAX_CONSECUTIVE_FAILS:
                logger.warning("[%s] %d consecutive read failures — reconnecting",
                               self.label, self._fail_count)
                self.alive = False
                self._maybe_reconnect()
            return None

        now = time.monotonic()
        if self._last_frame_t > 0:
            dt = now - self._last_frame_t
            if dt > 0:
                self._fps_samples.append(1.0 / dt)
                if len(self._fps_samples) > 30:
                    self._fps_samples.pop(0)
        self._last_frame_t = now

        self._fail_count = 0
        self.alive = True
        return frame

    @property
    def fps(self) -> float:
        """Approximate measured FPS (0 if no frames received yet)."""
        return float(np.mean(self._fps_samples)) if self._fps_samples else 0.0

    def resolution(self) -> tuple[int, int]:
        """Return (width, height) reported by the capture, or (0, 0)."""
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    def health_str(self) -> str:
        w, h = self.resolution()
        status = "OK" if self.alive else "DEAD"
        return f"{self.label}: {status}  {w}×{h}  {self.fps:.1f} fps"

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.alive = False

    # ── Internals ──────────────────────────────────────────────────────────────

    def _open(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = self.cap.isOpened()
        self.alive = ok
        self._fail_count     = 0
        self._last_reconnect = time.monotonic()
        if ok:
            w, h = self.resolution()
            logger.info("[%s] connected — %d×%d", self.label, w, h)
        else:
            logger.warning("[%s] failed to open %s", self.label, self.url)
        return ok

    def _maybe_reconnect(self):
        if time.monotonic() - self._last_reconnect >= RECONNECT_BACKOFF_S:
            self._open()


# ──────────────────────────────────────────────────────────────────────────────
# Frame composition helpers  (used by standalone viewer and stream_check.py)
# ──────────────────────────────────────────────────────────────────────────────

def placeholder_tile(label: str) -> np.ndarray:
    """Black tile with the camera label — shown when a feed is unavailable."""
    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    cv2.putText(tile, label,       (20, TILE_H // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    cv2.putText(tile, "no signal", (20, TILE_H // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)
    return tile


def annotate(frame: np.ndarray, label: str) -> np.ndarray:
    """Resize a frame to the tile size and draw its label."""
    tile = cv2.resize(frame, (TILE_W, TILE_H))
    cv2.rectangle(tile, (0, 0), (TILE_W, 28), (0, 0, 0), cv2.FILLED)
    cv2.putText(tile, label, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 60), 2)
    return tile


def compose_grid(tiles_by_pos: dict) -> np.ndarray:
    """Stitch a (row, col) → tile dict into a 2×2 grid."""
    blank = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    top = np.hstack([tiles_by_pos.get((0, 0), blank), tiles_by_pos.get((0, 1), blank)])
    bot = np.hstack([tiles_by_pos.get((1, 0), blank), tiles_by_pos.get((1, 1), blank)])
    return np.vstack([top, bot])


# ──────────────────────────────────────────────────────────────────────────────
# Standalone viewer
# ──────────────────────────────────────────────────────────────────────────────

def main():
    streams = [(cam, CameraStream(cam["label"], cam["url"])) for cam in CAMERA_DEFS]

    window = "Jackie's 360° Feed"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, TILE_W * 2, TILE_H * 2)

    try:
        while True:
            tiles = {}
            for cam, stream in streams:
                frame = stream.read()
                tiles[cam["pos"]] = (
                    annotate(frame, cam["label"]) if frame is not None
                    else placeholder_tile(cam["label"])
                )

            cv2.imshow(window, compose_grid(tiles))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for _, stream in streams:
            stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
