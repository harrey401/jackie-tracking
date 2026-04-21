"""stream_check.py — Live-stream health check and 360° panorama saver for Jackie.

Jackie has four HD ICR night-vision cameras (2 MP each) mounted 90° apart on
her head.  The ICR (infrared cut-filter removable) mechanism lets each camera
switch between colour mode in daylight and monochrome IR-illuminated mode at
night by detecting reflected near-infrared light — not heat emission.

This module does two things:
  1. Stream check  — verify that each RTSP feed is alive and report resolution
                     and approximate frame rate per camera.
  2. Panorama save — every PANO_INTERVAL_S seconds, stitch all four feeds into
                     a horizontal 360° strip and write it to PANO_OUTPUT_DIR.

Camera and stream primitives (CameraStream, tile helpers, CAMERA_DEFS) live in
surveil_connect.py and are imported here to avoid duplication.

Two modes of operation
──────────────────────
STANDALONE  — `python stream_check.py` opens a live 2×2 grid window and saves
              panoramas in the background.
              `python stream_check.py --check-only` prints a one-shot health
              report and exits without opening a window.

INTEGRATED  — the `Logic` class follows the same reset / step(obs) contract
              used by face_user.py and follow_mode.py.  step() populates obs
              with per-camera health info and returns {linear:0, angular:0}.

Camera layout (front = Jackie's face direction)
───────────────────────────────────────────────
       ┌──────────────┬──────────────┐
       │   FRONT (0°) │  RIGHT (90°) │
       ├──────────────┼──────────────┤
       │  LEFT (270°) │  REAR (180°) │
       └──────────────┴──────────────┘

Panorama stitch order (continuous left → right sweep):
  FRONT (0°) → RIGHT (90°) → REAR (180°) → LEFT (270°)
"""

from __future__ import annotations

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from surveil_connect import (
    CAMERA_DEFS,
    TILE_W, TILE_H,
    CameraStream,
    placeholder_tile,
    annotate,
    compose_grid,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# How often to save a panorama (seconds; also the tick interval in Logic.step).
PANO_INTERVAL_S = 10.0

# Where panoramas are written.  Created automatically if it doesn't exist.
PANO_OUTPUT_DIR = Path("pano_captures")

# Indices into CAMERA_DEFS for a continuous left-to-right 360° sweep.
PANO_ORDER = [0, 1, 3, 2]   # FRONT → RIGHT → REAR → LEFT


# ═══════════════════════════════════════════════════════════════════════════════
# PANORAMA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compose_pano(tiles: list[Optional[np.ndarray]], labels: list[str]) -> np.ndarray:
    """Stitch tiles horizontally in PANO_ORDER for a 360° strip."""
    strips = [
        annotate(tile, label) if tile is not None else placeholder_tile(label)
        for tile, label in zip(tiles, labels)
    ]
    return np.hstack(strips)


def _save_pano(pano: np.ndarray) -> Path:
    PANO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = PANO_OUTPUT_DIR / f"pano_{ts}.jpg"
    cv2.imwrite(str(path), pano)
    logger.info("Panorama saved → %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM HEALTH CHECK  (one-shot, no window)
# ═══════════════════════════════════════════════════════════════════════════════

def check_streams(timeout_s: float = 5.0) -> list[dict]:
    """
    Open all four feeds, collect frames for `timeout_s` seconds, then return a
    health dict per camera with keys: label, url, alive, width, height, fps.
    """
    streams = [CameraStream(cam["label"], cam["url"]) for cam in CAMERA_DEFS]
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        for s in streams:
            s.read()
        time.sleep(0.033)   # ~30 Hz poll

    results = []
    for s in streams:
        w, h = s.resolution()
        results.append({
            "label":  s.label,
            "url":    s.url,
            "alive":  s.alive,
            "width":  w,
            "height": h,
            "fps":    round(s.fps, 1),
        })
        logger.info(s.health_str())
        s.release()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_camera_loop():
    """
    Display a live 2×2 grid of all four camera feeds.
    Saves a 360° panorama to PANO_OUTPUT_DIR every PANO_INTERVAL_S seconds.
    Press 'q' or ESC to quit.
    """
    streams = [CameraStream(cam["label"], cam["url"]) for cam in CAMERA_DEFS]

    window = "Jackie — 360° Stream Check"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, TILE_W * 2, TILE_H * 2)

    last_pano_t = 0.0

    print("Stream health:")
    for s in streams:
        print(" ", s.health_str())
    print(f"Panoramas will be saved every {PANO_INTERVAL_S:.0f} s → {PANO_OUTPUT_DIR}/")
    print("Press 'q' or ESC to quit.\n")

    try:
        while True:
            latest = {i: s.read() for i, s in enumerate(streams)}

            tiles_by_pos = {
                cam["pos"]: (
                    annotate(latest[i], cam["label"]) if latest[i] is not None
                    else placeholder_tile(cam["label"])
                )
                for i, cam in enumerate(CAMERA_DEFS)
            }
            grid = compose_grid(tiles_by_pos)

            now = time.monotonic()
            secs_until = max(0.0, PANO_INTERVAL_S - (now - last_pano_t))
            bar = (
                "  ".join(
                    f"{'OK' if streams[i].alive else 'DEAD'} {CAMERA_DEFS[i]['label'].split()[0]}"
                    for i in range(4)
                )
                + f"   |   next pano in {secs_until:.0f}s"
            )
            cv2.rectangle(grid, (0, grid.shape[0] - 24), (grid.shape[1], grid.shape[0]),
                          (20, 20, 20), cv2.FILLED)
            cv2.putText(grid, bar, (10, grid.shape[0] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow(window, grid)

            if now - last_pano_t >= PANO_INTERVAL_S:
                pano = _compose_pano(
                    [latest[i] for i in PANO_ORDER],
                    [CAMERA_DEFS[i]["label"] for i in PANO_ORDER],
                )
                _save_pano(pano)
                last_pano_t = now

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    finally:
        for s in streams:
            s.release()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED LOGIC CLASS  (SMAIT server contract)
# ═══════════════════════════════════════════════════════════════════════════════

class Logic:
    """
    SMAIT server skill — same reset/step(obs) shape as face_user.py / follow_mode.py.

    step() checks stream health and saves panoramas every PANO_INTERVAL_S ticks.
    Jackie does not move; linear and angular are always 0.

    obs keys populated:
        streams_alive    — list[bool], one per camera (FRONT, RIGHT, LEFT, REAR)
        streams_fps      — list[float], measured FPS per camera
        pano_last_saved  — str path of the most recently saved panorama, or ""
    """

    def __init__(self):
        self._streams = [CameraStream(cam["label"], cam["url"]) for cam in CAMERA_DEFS]
        self._last_pano_t    = 0.0
        self._pano_last_saved = ""

    def reset(self):
        self._last_pano_t    = 0.0
        self._pano_last_saved = ""

    def step(self, obs: dict) -> dict:
        now    = time.monotonic()
        latest = [s.read() for s in self._streams]

        obs["streams_alive"] = [s.alive     for s in self._streams]
        obs["streams_fps"]   = [round(s.fps, 1) for s in self._streams]

        if now - self._last_pano_t >= PANO_INTERVAL_S:
            pano = _compose_pano(
                [latest[i] for i in PANO_ORDER],
                [CAMERA_DEFS[i]["label"] for i in PANO_ORDER],
            )
            self._pano_last_saved = str(_save_pano(pano))
            self._last_pano_t = now

        obs["pano_last_saved"] = self._pano_last_saved
        return {"linear": 0.0, "angular": 0.0}

    def shutdown(self):
        for s in self._streams:
            s.release()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Jackie 360° stream checker")
    parser.add_argument(
        "--check-only", action="store_true",
        help="Print a one-shot stream health report and exit (no window).",
    )
    args = parser.parse_args()

    if args.check_only:
        results = check_streams()
        print("\n── Stream Health Report ──────────────────────────")
        for r in results:
            status = "OK  " if r["alive"] else "DEAD"
            print(f"  {status}  {r['label']:20s}  {r['width']}×{r['height']}  {r['fps']} fps")
        print()
    else:
        run_camera_loop()
