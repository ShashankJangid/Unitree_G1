# UNITREE G1 SIMULATION & OPERATION INSTRUCTIONS

This document provides complete instructions for operating the Unitree G1 humanoid simulation, interactive 3D manual teleoperation, bipedal walking, autonomous ACT policies, and the real-time telemetry dashboard.

---

## 1. Quick Start

Run the simulation from the `g1_physics_sim` directory using the project's Python virtual environment:

```powershell
cd "c:\Users\D&T\Documents\Shashank\Unitree G1\g1_physics_sim"
..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe g1_sandbox.py
```

To run headless (without opening a GUI window, useful for scripts/servers):

```powershell
..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe g1_sandbox.py --headless
```

---

## 2. Live Telemetry & Motor Activity Dashboard

When `g1_sandbox.py` launches, it automatically starts a background HTTP telemetry server at:

```
http://localhost:8080
```

Open this URL in any web browser to view:
- **Live 50 Hz orientation**: Real-time Yaw, Pitch, Roll in degrees.
- **IK Tracking Error**: Sub-millimeter accuracy readout in millimeters.
- **End-Effector Target vs Hand**: 3D coordinate comparison.
- **Real-Time Rolling Charts**:
  - Arm motors position tracking ($q_{\text{des}}$ vs $q_{\text{act}}$)
  - Leg joint angles
  - Motor joint torques ($\tau$ in Nm)
  - 3D Cartesian tracking error & Euler angles
- **Complete 29-Motor Status Table**: Desired angle, actual angle, error, and applied torque for all 29 joints.

---

## 3. Interactive Keyboard Controls

While the simulation viewer window or terminal is focused, use the following key commands:

### Mode Switching & humanoid control
- **`0` or `V`**: Toggle **Dynamic Ground Walking Mode** (activates whole-body bipedal locomotion policy).
- **`1`**: Control **Left Arm** in 3D Cartesian Space.
- **`2`**: Control **Right Arm** in 3D Cartesian Space.
- **`3`**: Control **Waist Orientation** (`W`/`S` pitch, `A`/`D` yaw, `Q`/`E` roll).
- **`4`**: Control **Squat & Pelvis Elevation** (`W`/`S` up/down).
- **`5` or `C`**: Cycle and trigger pre-programmed choreography scripts (Wave, Circle, Square, Punch, Bow).
- **`6`**: Control **Left Leg** joints directly.
- **`7`**: Control **Right Leg** joints directly.
- **`8`**: Control **Torso Twist** and whole-body articulation.
- **`9`**: Engage/Disengage **Autonomous ACT Neural Policy** (learned pick-and-place task).
- **`J`**: Trigger official Unitree high-level arm actions (Salute, Hug, Kiss, Clap, Wave).
- **`K`**: Toggle official hand grip clamp (Dexterous Hand SDK torque control).

### 3D Cartesian Point Tracking & Manual Teleoperation (When Arm Mode `1` or `2` Active)
- **`Mouse Cursor`**: Hold **`Ctrl` + Right-Click and drag** the green marker sphere directly in 3D space.
- **`W` / `S`**: Nudge target vertically (Z-axis Up / Down).
- **`A` / `D`**: Nudge target laterally (Y-axis Left / Right).
- **`Q` / `E`**: Nudge target longitudinally (X-axis Forward / Backward).
- **`[` / `]`**: Decrease / Increase nudge step size.
- **`R`**: Reset robot to nominal standing posture and re-initialize IK target to current hand position.

### Dynamic Bipedal Walking (When Walking Mode `0` or `V` Active)
- **`W` / `Up Arrow`**: Walk forward (+0.1 m/s up to 1.2 m/s).
- **`S` / `Down Arrow`**: Slow down / stop (down to 0.0 m/s).
- **`A` / `Left Arrow`**: Turn / rotate left (+0.2 rad/s).
- **`D` / `Right Arrow`**: Turn / rotate right (-0.2 rad/s).
- **`Q` / `E`**: Strafe laterally (Left / Right).
- **`X`**: Immediate full stop (zeros all linear and angular velocities).

### Balance, Physics & Ground Plane
- **`B`**: Toggle **Free-Standing Ground Balance** vs **Suspended Gantry Harness**.
- **`F`**: Apply a physical 60 N perturbation push to the pelvis to test balance recovery.
- **`G`**: Toggle ground plane appearance (Checkerboard Partition Plane vs Normal Solid Floor).

### Motion Recording & Playback
- **`M`**: Start/Stop 50 Hz motion recording. Saves full 29-joint telemetry to `trajectories/motion_<timestamp>.json` and formats datasets for Unitree LeRobot.
- **`P`**: Play back the most recently recorded trajectory.

### Safety & Hardware Interlocks
- **`O` or `T`**: Toggle live hardware DDS motor streaming (when connected to a physical robot).
- **`Spacebar`**: **Emergency Stop** (cuts motor torque immediately; press `R` to re-arm).
- **`H`**: Display the interactive help command menu in the console.

---

## 4. Clean Terminal Output

Terminal logging displays clean, essential telemetry on a single line:

```
Motors: Yaw= 26.4° Pitch= 59.6° Roll= 29.4° | Target: [0.03, 0.24, 0.75]m | Hand: [0.03, 0.24, 0.75]m | Err:  0.0mm
```

When walking mode is active:

```
Motors: Yaw=  0.0° Pitch=  0.0° Roll=  0.0° | Cmd: vx=0.30 vy=0.00 yaw=0.00 | Pos: [0.12, 0.00, 0.77]m | Err:  0.0mm
```

---

## 5. Running Standalone Trajectory Playback

To replay a previously recorded trajectory:

```powershell
..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe play_motion.py trajectories/demo_wave_left.json
```

To loop playback continuously:

```powershell
..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe play_motion.py trajectories/demo_wave_left.json --loop
```

---

## 6. Running Autonomous ACT Neural Policy Training

To train the multi-robot behavioral cloning ACT policy on GPU:

```powershell
..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe g1_act_trainer.py --train --epochs 50 --device cuda
```

To evaluate the policy inside the simulation:

```powershell
..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe g1_act_trainer.py --eval
```
