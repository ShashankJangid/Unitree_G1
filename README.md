# Unitree G1 Simulation & Control

MuJoCo simulation for pretrained walking, whole-body dance playback, and
scripted G1 actions. This branch integrates the current Documents `unitree`
inference build (local source snapshot `e75d758`) into the existing project.
The contribution is maintained in [Thorn-probably/Unitree_G1-sim](https://github.com/Thorn-probably/Unitree_G1-sim),
a fork of the original project.

The upstream hardware tools and their documentation remain in
[`unitree_g1_vibes/`](unitree_g1_vibes/). Simulation checks do not establish
real-robot safety. No training or hardware commands were run for this integration.

## Setup

Use Python **3.11 or 3.12** and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run python main.py --help
```

Inference dependencies are declared in `pyproject.toml`; `uv.lock` pins the
resolved environment. The optional `train` extra is separate. The inherited
pickup trainer is experimental and has known blockers described below.

## Run

On Linux/macOS, the launcher uses `.venv/bin/python` when available:

```bash
./run.sh teleop                         # Interactive walking
./run.sh walk --headless --duration 15  # Bounded walking check
./run.sh wander --headless --auto-push --push-force 200
./run.sh gangnam                       # Repeated Gangnam Style playback
./run.sh gangnam --headless             # One 20.3-second excerpt
./run.sh gangnam --headless --duration 45
./run.sh dance --headless               # GRIT K-pop reference, not Gangnam
./run.sh wave --headless --duration 5
```

The portable Python entrypoint is `uv run python main.py <mode> [options]`.
Existing Windows launchers `run.bat` and `train.bat` remain available; Windows
and macOS runtime behavior was not exercised for this integration.

All nine demo modes accept `--headless` and optional positive finite
`--duration` in **simulated seconds**. Headless mode disables graphics before
MuJoCo imports, requires no display/keyboard, and skips real-time sleeps.
The defaults are 10 seconds for ordinary demos and one 20.3-second excerpt
for Gangnam. Headless `teleop` runs nominal 0.4 m/s walking, not interactive input.
Without `--headless`, demos use a graphical window.

## Modes and limitations

| Mode | Behavior |
| --- | --- |
| `teleop` | Verified 12-DoF recurrent walking checkpoint with keyboard bursts |
| `walk`, `wander` | Commanded or deterministic bounded random walking; optional push |
| `gangnam` | Unitree RL Lab 29-DoF Gangnam Style ONNX policy, no gantry |
| `dance` | Published GRIT whole-body K-pop reference, no gantry |
| `wave`, `bow`, `punch` | Scripted joint trajectories with the support gantry |
| `reach` | Existing reach routine; not a validated dynamic grasp-and-lift task |
| `train` | Inherited experimental pickup trainer; known blockers, not validated |

Gangnam uses reference seconds 5.2–25.5 and repeats until Esc/Q/window close
or Ctrl+C. Left-drag orbits, right-drag pans, and the scroll wheel zooms.
Its dedicated main-thread GLFW renderer avoids the passive-viewer shutdown
race observed on X11/XWayland. No audio is included.

**Replay is not seamless:** the excerpt endpoints differ, so each logged
replay boundary restores the starting pose, velocities, and policy state.
Metrics accumulate across cycles. Falls terminate playback; they are not
silently hidden by a recovery reset. `--duration` also bounds graphical playback.

Teleop controls: W/up sprint, Z turbo, S/down brake/reverse, A/D or left/right
turn, Q/E strafe, Space/X cruise, R reset, F sustained disturbance. Terminal
input and viewer-key input are supported. The dedicated walk/wander controller
instead uses W/A/S/D for strafe commands and Q/E for turning.

## Implementation and assets

| File | Purpose |
| --- | --- |
| [`main.py`](main.py) | Lazy CLI routing; headless and duration validation |
| [`g1_teleop.py`](g1_teleop.py) | Cross-platform interactive walking |
| [`g1_velocity.py`](g1_velocity.py) | Walking, wandering, bounded push checks |
| [`g1_gangnam.py`](g1_gangnam.py), [`gangnam_controller.py`](gangnam_controller.py) | Gangnam inference, mapping, PD control and metrics |
| [`gangnam_viewer.py`](gangnam_viewer.py) | Synchronous graphics lifecycle and camera |
| [`g1_expressive.py`](g1_expressive.py), [`grit_controller.py`](grit_controller.py) | GRIT inference and supported scripted actions |
| [`g1_train.py`](g1_train.py) | Unmodified inherited experimental trainer |

Required model files and meshes remain tracked, not hidden by `.gitignore`.
See [Gangnam sources, hashes and Apache-2.0 notice](models/gangnam/README.md)
and [GRIT sources and hashes](models/grit/README.md). GRIT's source repository
has no GitHub-detected license; redistribution terms need maintainer review
before merging/publishing those third-party artifacts. Their inclusion here
does not establish permission to relicense them under this project's MIT license.

## Checks and open work

```bash
uv run pytest -q
env -u DISPLAY -u WAYLAND_DISPLAY ./run.sh gangnam --headless --duration 45
```

All **53 tests** passed in this fork checkout on Linux, Python 3.11.15,
MuJoCo 3.16.0, using the source build's existing interpreter/environment.
The suite covers policy contracts, unsupported dance dynamics, multi-cycle
replay, mocked viewer lifecycle, and display-free CLI routes. Four additional
controlled-input tests check manual/fall policy resets, sustained force
expiry, and pending-force clearing on reset. The training CLI check is a
routing stub, **not a training run**. A fresh environment installation was
not exercised; the lockfile consistency check passed.

The source build completed all default-duration headless demos. This fork
also completed a 45-second Gangnam run (22,500 physics steps, two complete cycles plus a partial
third, minimum pelvis height 0.6327 m, minimum up-axis 0.9460, tracking RMS
0.1167 rad, zero torque saturation). These are simulation results on one
environment, not general robustness or hardware guarantees.
The fork's headless wander check completed 5,000 steps with a 200 N, 250 ms
disturbance and zero recovery resets.

This integration addresses upstream startup, packaging, headless routing,
normal policy reset, and disturbance-duration problems tracked in issues
[#1](https://github.com/ShashankJangid/Unitree_G1/issues/1)–
[#5](https://github.com/ShashankJangid/Unitree_G1/issues/5).

The unchanged pickup trainer still has:
- [#6: constructor module-shadowing error](https://github.com/ShashankJangid/Unitree_G1/issues/6).
- [#7: 33-value declared / 36-value actual observation mismatch](https://github.com/ShashankJangid/Unitree_G1/issues/7).
- [#8: nonphysical pickup objective](https://github.com/ShashankJangid/Unitree_G1/issues/8).

Do not interpret successful inference/headless checks as fixes for those
training or manipulation issues. Scripted action modes retain their gantry;
the current default expressive scene does not establish a physical object grasp.
