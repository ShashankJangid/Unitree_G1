# Official Gangnam Style inference assets

Source: [Unitree Robotics unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab),
pinned commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`.
The upstream repository supplies an Apache-2.0 license, retained as `LICENCE`.

Files are unmodified downloads from the upstream directory:
`deploy/robots/g1_29dof/config/policy/mimic/gangnam_style/`.

| Local file | Upstream path within that directory | SHA-256 |
| --- | --- | --- |
| `policy.onnx` | `exported/policy.onnx` | `c09a6088fbf85aecd96b1fa89528d7462d4a571a981408d5ff2afb69f0375b7d` |
| `deploy.yaml` | `params/deploy.yaml` | `164137b43e8834d7c53828d5697aa993911fb2fca5dcb749e737912daf3ca7bb` |
| `motion.csv` | `params/G1_gangnam_style_V01.bvh_60hz.csv` | `d74f54662546969826a25386a21f429b6b7f8fd4793578c9923ca5ce4e9ee0aa` |

The adapter verifies all three hashes before loading. No training, checkpoint
resume, or network requests are performed during playback.

## Deployment contract

- ONNX float32 input `obs [1,154]`, output `actions [1,29]`.
- Policy inference at 50 Hz; joint-space PD at 500 Hz.
- 1,942 CSV frames at 60 Hz: world root XYZ, root quaternion **XYZW**,
  then 29 depth-first motor-order joint positions.
- `joint_ids_map` gathers XML/motor positions into policy order; policy
  default positions, action offsets, and action scales are already policy order.
  Stiffness and damping are motor/XML order.
- Observation: reference joint positions/velocities, 6D reference torso
  orientation relative to actual torso, body angular velocity, relative
  joint positions, joint velocities, previous raw policy action.
- Reference torso rotation is `R_root @ Rz(waist_yaw) @ Rx(waist_roll) @
  Ry(waist_pitch)`, with initial yaw alignment as in upstream `State_Mimic.cpp`.
- Sampling uses linear interpolation and quaternion SLERP in native motion
  time. This deliberately avoids upstream MotionLoader_'s rounded-frame
  interpolation/extrapolation quirk.
- The upstream FSM selects reference seconds 5.2–25.5. The graphical runner
  repeats this 20.3-second excerpt until close; headless defaults to one cycle.
- The reference endpoints do not match. At each scheduled replay boundary,
  restore the starting physical pose/velocities and reset the policy's previous
  action and yaw alignment. This visible restart is not a seamless transition
  or a fall-recovery mechanism. Falls still stop playback with an error.
- Physics steps, elapsed time, balance minima, torque saturation, and tracking
  metrics accumulate across replay boundaries; `completed_cycles` counts full
  excerpts, excluding the partial final cycle.

Run `./run.sh gangnam` from the repository root; Esc/Q/window close or Ctrl+C
stops the graphical loop. Use `--headless` for a full single-excerpt dynamics
check with metrics, or `--headless --duration 45` for multiple cycles.
`--duration` also bounds graphical playback in simulated seconds. The local model is
`models/g1/scene_29dof.xml` with its support constraint disabled.

## Verification

Full excerpt: 10,150 physics steps; min pelvis height 0.6327 m;
min pelvis up-axis 0.9460; tracking RMS 0.1174 rad; torque saturation 0.
No falls, recovery resets, gantry support, or root-position overrides within
an excerpt. Scheduled replay boundaries explicitly restore the reference
starting state. Four policy-driven poses at elapsed 2, 6, 10, and 14 seconds
were also rendered and visually inspected. No audio is included.

Multi-cycle check: 45 simulated seconds, 22,500 steps, two complete cycles
plus part of a third; min pelvis height 0.6327 m, min up-axis 0.9460, tracking
RMS 0.1167 rad, no torque saturation or falls. A native graphical check ran
21.3 simulated seconds into the second cycle, then injected KeyboardInterrupt;
the renderer cleaned up and the process exited successfully.

Tests: `uv run pytest -q tests/test_gangnam_policy.py tests/test_gangnam_runtime.py`.
These simulation checks do not establish real-hardware safety or robustness
to arbitrary perturbations.
