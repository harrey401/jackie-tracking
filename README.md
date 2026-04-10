# Jackie Tracking Logic

This repository contains the logic that controls how **Jackie** (the service robot) tracks people. Two behaviours live here:

| File | What it does | When it runs |
|------|--------------|--------------|
| `face_user.py` | Rotates Jackie to face the user while Jackie is speaking. No forward movement. | Automatically starts on every TTS response in the ENG192 Lab app |
| `follow_mode.py` | Drives Jackie toward a person and faces them. Full mobility. | When the "Follow Me" button is pressed |

Both files are **hot-reloaded** on the server. You edit them, save (commit + push to GitHub), and the running robot picks up the change within ~10 seconds — no restart.

---

## How it works

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

Every ~10 seconds the server runs `git pull` on this repo. If a file changed, the server calls `importlib.reload()` on it. The next time Jackie needs a velocity command (about 10 times per second), it uses your new code.

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

| Key            | Type          | Meaning |
|----------------|---------------|---------|
| `face_visible` | `bool`        | `True` if a face was seen in the last 1 second |
| `face_cx`      | `float│None`  | Face center X, `0.0`=left edge, `0.5`=center, `1.0`=right edge |
| `face_cy`      | `float│None`  | Face center Y, `0.0`=top, `1.0`=bottom |
| `face_w_norm`  | `float│None`  | Face width as fraction of frame (face_user only) |
| `face_area`    | `int│None`    | Face bounding box area in pixels² (follow_mode only). Bigger = closer. |
| `face_age_s`   | `float`       | Seconds since last face was seen. `inf` if none ever. |
| `dt`           | `float`       | Seconds since the previous `step()` call. Usually ~0.1. |
| `robot_theta`  | `float`       | Current robot heading in radians (follow_mode only) |

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

**If Jackie is too jittery:**
- Increase `SMOOTHING` (closer to 1.0)
- Increase `DEADZONE` so small errors are ignored
- Lower `PAN_KP` so the response is gentler

**If Jackie is too slow to react:**
- Decrease `SMOOTHING` (closer to 0.0)
- Raise `PAN_KP`
- Raise `MAX_ANGULAR`

**If Jackie overshoots and oscillates:**
- Lower `PAN_KP`
- Increase `SMOOTHING`

**If you hate the spinning-when-face-lost behaviour:**
- In `follow_mode.py`, set `SCAN_ENABLED = False` (this is the default now)
- In `face_user.py`, there is no scan — it just stops

**If the robot stops too abruptly when a face blinks out for one frame:**
- Increase `LOST_HOLD_S` — this holds the last command for a bit longer before giving up

---

## Adding new behaviour

You can add helper functions, track extra state on `self`, import `math` or `time` — normal Python. You cannot import modules that aren't in SMAIT's venv (numpy/scipy/etc are fine, random libs are not).

If you need a new field in `obs` (e.g. the user's distance in meters, or the session state), ask Gow — that's on the server side.

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

## Checking it worked

The server log prints a line every time it reloads a file:

```
[tracking] face_user.py reloaded OK
[tracking] follow_mode.py reloaded OK
```

Or if you broke it:

```
[tracking] face_user.py RELOAD FAILED: SyntaxError at line 42 — keeping previous version
```

Ask Gow for the server log URL if you want to watch it live.
