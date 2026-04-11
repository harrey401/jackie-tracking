# Jackie Tracking Logic

This repository is the **playground** for two tracking skills on **Jackie** (the SJSU lab service robot). Both run server-side. You edit the Python here, push to GitHub, and the running robot picks up the change in ~10 seconds — no SSH, no restart, no access to the rest of the SMAIT codebase.

| File | Skill | What it should do |
|------|-------|-------------------|
| `face_user.py` | **Face the User** | Jackie rotates to keep the user centered. Rotation only — no driving. |
| `follow_mode.py` | **Follow Me** | Jackie drives toward the user and faces them. Full mobility. |

`follow_mode.py` is a Python port of your original on-device Kotlin `FollowController.kt` (`smait-jackie-app`, main branch, plus the intent from your unmerged "Updated Rotational Motion" commit). Same FSM states (`FOLLOWING`, `SCAN_ROTATE`, `OBSTACLE_TURN`, `COLLISION_STOP`, `COLLISION_TURN`, `CLEAR_CHECK`), same constants, same pan PID gains you already tuned (`PAN_KP = 0.003`, etc.), same pinhole distance model. The on-device MediaPipe / DeepSORT / Kalman stack is gone — the SMAIT server provides an already-tracked face in `obs`.

`face_user.py` uses the **same `_PID` class and the same gain names** as `follow_mode.py` (`PAN_KP`, `PAN_KI`, `PAN_KD`, `PAN_FF_GAIN`, `MAX_ANGULAR`, `FRAME_WIDTH_PX`), so what you learn tuning one file transfers directly to the other. The difference is just that Face the User has no distance PID, no FSM, and doesn't scan when the face is lost.

---

## How you trigger them

Neither skill auto-fires. They're activated **explicitly** from a test page in the ENG192 Lab app on Jackie's touchscreen:

```
   Jackie touchscreen          Server                 This repo
   ┌────────────────┐          ┌──────────┐          ┌──────────────┐
   │ Tracking Tests │          │  SMAIT   │          │  GitHub      │
   │                │          │          │          │              │
   │ [Face the User]│ ──START─►│ runs the │ ◄──pull──│ face_user.py │
   │     [STOP]     │          │ logic    │   ~10s   │ follow_mode  │
   │                │          │ from     │          │              │
   │  [Follow Me]   │ ──START─►│ this     │          │              │
   │     [STOP]     │          │ repo     │          │              │
   └────────────────┘          └──────────┘          └──────────────┘
```

1. Open the lab app on Jackie → tap **Tracking Tests**.
2. The page has two cards: *Face the User* and *Follow Me*. Each has a START/STOP button.
3. Tap START → the server starts calling your `Logic.step()` ~10 times per second and sending the resulting velocity to the chassis.
4. Tap STOP → the server stops the loop and zeroes the chassis.
5. Only one skill runs at a time. Starting one stops the other.

That's the only way they activate. They will not run during normal Ask-Jackie conversation.

---

## How edits propagate

```
 ┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
 │  You edit code  │  →    │  GitHub (this    │  →    │  SMAIT server   │
 │  on github.com  │       │  repo, main)     │       │  auto-pulls     │
 └─────────────────┘       └──────────────────┘       └────────┬────────┘
                                                                │
                                                                │ file changed?
                                                                ▼
                                                      ┌─────────────────┐
                                                      │ importlib       │
                                                      │ reloads module  │
                                                      └────────┬────────┘
                                                                │
                                                                ▼
                                                      ┌─────────────────┐
                                                      │ Jackie uses new │
                                                      │ logic next tick │
                                                      └─────────────────┘
```

Every ~10 seconds the SMAIT server runs `git pull` on this repo. If a file changed, the server calls `importlib.reload()` on it. The next time Jackie needs a velocity command (~10 Hz), it uses your new code.

**If you commit a syntax error**, the server catches the exception, logs it, and keeps running the **last good version**. You won't brick the robot. You'll see the error in the server log. Fix it, push again, Jackie recovers.


---

## The contract: `Logic.reset()` and `Logic.step(obs)`

Both files expose the **same shape**: a `Logic` class with two methods.

### `reset()`

Called once when the behaviour turns on. Use it to clear any memory you keep between sessions.

```python
def reset(self):
    self._prev_angular = 0.0
```

### `step(obs)` — called ~10 Hz

This is where you make decisions. You receive what Jackie can see right now, and you return what you want the wheels to do.

**Input — `obs` (dict):**

| Key               | Type          | Meaning |
|-------------------|---------------|---------|
| `face_visible`    | `bool`        | `True` if a face was seen in the last 1 second |
| `face_cx`         | `float│None`  | Face center X, `0.0`=left edge, `0.5`=center, `1.0`=right edge |
| `face_cy`         | `float│None`  | Face center Y, `0.0`=top, `1.0`=bottom |
| `face_w_px`       | `float│None`  | Face width in pixels (for the pinhole distance model) |
| `face_w_norm`     | `float│None`  | Face width as fraction of frame width (0..1) |
| `face_area`       | `int│None`    | Face bounding box area in pixels². Bigger = closer. |
| `frame_width_px`  | `int`         | Camera frame width in pixels (currently 640) |
| `face_age_s`      | `float`       | Seconds since last face was seen. `inf` if none ever. |
| `dt`              | `float`       | Seconds since the previous `step()` call. Usually ~0.1. |
| `robot_theta`     | `float`       | Current robot heading in radians (follow_mode only) |

Both files convert `face_cx` to pixel coordinates the same way: `cx_px = face_cx * frame_width_px`. Your PID gains (`PAN_KP = 0.003` etc.) are in pixel-space, matching your Kotlin `FRAME_WIDTH_PX = 640` convention.

**Output — `act` (dict):**

| Key       | Type    | Meaning |
|-----------|---------|---------|
| `linear`  | `float` | Forward speed, m/s. Positive = forward. **Keep 0 in `face_user.py`** — we're rotation-only there. |
| `angular` | `float` | Rotation speed, rad/s. **Positive = turn LEFT, negative = turn RIGHT.** |

The server clamps whatever you return to safe limits (`|linear| ≤ 0.25 m/s`, `|angular| ≤ 1.5 rad/s`), so you can't hurt anyone by typing a big number.

---

## A minimal example

```python
class Logic:
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def step(self, obs):
        if not obs["face_visible"]:
            return {"linear": 0.0, "angular": 0.0}   # face gone → stop

        error = obs["face_cx"] - 0.5     # how far off-center the face is
        angular = -1.5 * error           # turn toward it
        return {"linear": 0.0, "angular": angular}
```

That's the whole program. Everything in `face_user.py` and `follow_mode.py` is just this pattern with more polish — deadzone, smoothing, speed caps, lost-face behaviour.

---

## Tuning tips

All the knobs live at the **top of each file** as UPPER_CASE constants. You don't have to touch the `Logic` class to change behaviour — just edit the numbers.

**If rotation is too slow to react (Jackie lags behind you):**
- Raise `PAN_KP` — the proportional term
- Raise `PAN_FF_GAIN` — the feed-forward term (helps with *moving* users especially)
- Raise `MAX_ANGULAR` if the output is saturating

**If rotation overshoots and oscillates back-and-forth:**
- Lower `PAN_KP`
- Raise `PAN_KD` — the derivative term damps overshoot
- Consider lowering `PAN_FF_GAIN` if the lead is too aggressive

**If Jackie sits slightly off-center when the user stops moving (steady-state error):**
- Raise `PAN_KI` — the integral term eliminates steady-state offset

**In `follow_mode.py`, if forward motion is too slow / too fast:**
- Raise/lower `DIST_KP`. Quick sizing: if the user is `N` metres past `TARGET_FOLLOW_DISTANCE_M` and you want Jackie moving at `V` m/s, set `DIST_KP ≈ V / N`.

**If Jackie drives into you (doesn't stop at the target distance):**
- Raise `DIST_KD` (adds damping so it eases into the target)
- Or tighten `TARGET_FOLLOW_DISTANCE_M` with more margin

**If you hate the spin-to-find-face behaviour when the target is lost:**
- In `follow_mode.py`, change the transition out of `FOLLOWING` so that face-lost goes somewhere else instead of `SCAN_ROTATE` (e.g. just stop). It's your code, rewrite the FSM however you want.
- `face_user.py` already doesn't scan — when the face is gone it just stops rotating.

---

## Adding new behaviour

You can add helper functions, track extra state on `self`, import `math` or `time` — normal Python. You cannot import modules that aren't in SMAIT's venv (numpy/scipy/etc are fine, random libs are not).

---

## How to edit from anywhere

**Option A — github.com in a browser** (easiest, works from your phone):
1. Go to https://github.com/harrey401/jackie-tracking
2. Click the file you want to edit
3. Click the pencil icon ✏️
4. Make changes, scroll down, click "Commit changes"
5. Wait ~10 seconds — the running server picks it up automatically

**Option B — git locally**:
```bash
git clone https://github.com/harrey401/jackie-tracking.git
cd jackie-tracking
# edit files
git add .
git commit -m "tune pan_kp"
git push
```

---

## Test loop, end to end

1. Edit `face_user.py` (or `follow_mode.py`) on github.com in a browser. Commit.
2. Wait ~10 seconds. The server pulls and reloads. It logs one of:
   ```
   [tracking] face_user.py reloaded OK
   [tracking] face_user.py RELOAD FAILED: SyntaxError at line 42 — keeping previous version
   ```
3. On Jackie's touchscreen: **Tracking Tests → Face the User → START**.
4. Move around, see what Jackie does.
5. Tap **STOP**. Jackie zeroes its wheels.
6. Back to step 1.

You can iterate as fast as you can commit. The server never restarts.

**Heads up:** if you tap START while the *other* skill is running, the server stops the other one first. You can't have both running at the same time — they both want the chassis.

