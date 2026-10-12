# Unitree G1 Real-Physics Simulation, Motion Recorder & Hardware Deployment Framework (`g1_physics_sim`)

A clean, production-grade simulation, real-time Cartesian teleoperation, motion recording, and hardware execution stack built specifically for the **Unitree G1 EDU (29-DoF)** humanoid robot.

---

## 🌟 Key New Features

### 1. 🖱️ Interactive Mouse Cursor Drag & Drop
- The 3D target now features a green translucent handle sphere and RGB Cartesian coordinate axes:
  - **Red Axis**: $+X$ (Forward)
  - **Green Axis**: $+Y$ (Lateral)
  - **Blue Axis**: $+Z$ (Vertical)
- **How to Drag with Mouse**: Hold **`Ctrl` + Right-Click and drag** the green sphere in the 3D viewer window to freely move the target in 3D Cartesian space. The robot arm will smoothly track your mouse cursor in real time!

### 2. 📹 Motion Recording & Activity Capture
- Press **`M`** at any moment to start recording live motion.
- Captures all 29 motor joint angles ($q$), velocities ($\dot{q}$), end-effector positions, and timestamps at 50 Hz.
- Press **`M`** again to save. The trajectory is automatically exported as a structured JSON file in `trajectories/motion_<timestamp>.json`.
- Works during simulation teleoperation **or** during live robot kinesthetic teaching!

### 3. ⏯️ Trajectory Playback Engine (`play_motion.py`)
- Replay any recorded motion or movement script with smooth interpolation and hardware slew protection:
  ```powershell
  # Play in simulation
  python play_motion.py trajectories/demo_wave_left.json

  # Play on physical Unitree G1 EDU robot
  python play_motion.py trajectories/demo_wave_left.json --hardware --iface Ethernet
  ```
- Or press **`P`** in the live viewer to instantly replay the last recorded motion!

### 4. 🎭 Movement Choreography Scripts
- Press **`C`** on your keyboard to cycle and execute pre-programmed movement scripts:
  1. **Friendly Wave Gesture**: Human-like waving motion with the wrist and elbow.
  2. **Cartesian Circle Trace**: Traces a smooth circle in the $Y-Z$ plane.
  3. **Cartesian Square Trace**: Traces a 3D box pattern in space.
- You can also launch them via CLI: `python run_simulation.py --script wave`.

### 5. 🛡️ 5-Layer Super-Safe Kinematic Governor
- **Spherical Workspace Governor ($R_{\max} = 34.0\text{ cm}$)**: Target cannot be commanded beyond physical arm reach.
- **Anti-Hyperextension Guard**: $q_{\text{elbow}} \ge +0.05\text{ rad}$. The elbow **never bends backwards**.
- **Nullspace Posture Projector**: Redundant 7th DoF continuously biases the elbow into a natural human posture.
- **Zero-Jerk Hardware Sync**: Reads physical motor angles on startup so initial tracking error is $0.0\text{ cm}$.
- **Hardware Slew-Rate Limiter**: Caps joint velocities at $1.20\text{ rad/s}$ to prevent sudden motor jerks.
- **Hardware Streaming Clutch**: Starts in `[STANDBY]` mode; press **`O`** to engage motors, hit **`Spacebar`** for immediate emergency stop.

---

## 📁 Repository Structure

```
g1_physics_sim/
├── g1_constants.py          # Official 29-DoF motor indices (0..28), torque saturation limits & nominal stance
├── g1_robot.py              # Physics engine: Forward kinematics, Jacobians, and PD actuator models
├── g1_env.py                # Gymnasium environment with 500 Hz physics and 50 Hz DDS control loop
├── g1_hardware_bridge.py    # Zero-code-change DDS bridge to physical G1 EDU (`rt/arm_sdk` & `rt/lowcmd`)
├── g1_trajectory_manager.py # Motion Recorder, Trajectory Player & Movement Script generator
├── run_simulation.py        # Interactive 3D MuJoCo viewer with mouse drag, recording & teleoperation
├── play_motion.py           # Standalone trajectory player utility for Simulation & Real Hardware
├── test_robot_connection.py # Non-destructive hardware diagnostic & interface checker
├── verify_physics.py        # Automated physics verification suite (5/5 tests passing)
├── trajectories/            # Saved motion JSON recordings and demo choreographies
└── models/g1/               # Clean 29-DoF G1 meshes, materials, and physics scene definitions
```

---

## 🎮 Interactive Controls Reference

| Control | Action | Details |
| :---: | :---: | :---: |
| **`Ctrl` + Right-Click Drag** | **Mouse 3D Drag** | Drag target sphere directly with mouse cursor |
| **`W` / `S`** (or `↑`/`↓`) | Up / Down ($+z / -z$) | Step target up/down in Cartesian space |
| **`A` / `D`** (or `←`/`→`) | Left / Right ($+y / -y$) | Step target left/right in Cartesian space |
| **`Q` / `E`** | Forward / Backward ($+x / -x$) | Step target forward/backward |
| **`M`** | **Record Motion** | Start / Stop recording all 29 motors to JSON |
| **`P`** | **Play Trajectory** | Replay last recorded motion or loaded trajectory |
| **`C`** | **Run Choreography** | Execute built-in script (Wave -> Circle -> Square) |
| **`O`** (or `T`) | **Hardware Clutch** | Toggle live motor streaming to real G1 robot |
| **`Space`** | **Emergency Stop** | Instantly disengage hardware streaming & pause |
| **`1`** / **`2`** | Switch Arm | `1` = Left Arm, `2` = Right Arm |
| **`R`** | Reset Posture | Return arm to nominal safe rest pose |
| **`]`** / **`[`** | Adjust Step Size | Increase / decrease Cartesian step size |
| **`Esc`** | Exit | Safely close simulation and DDS channels |

---

## 🚀 How to Run

### 1. Interactive Simulation (Mouse Drag + Recording + Teleop)
```powershell
cd "c:\Users\D&T\Documents\Shashank\Unitree G1\g1_physics_sim"
..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe run_simulation.py
```

### 2. Run Pre-Built Choreography Demo
```powershell
# Play the friendly wave gesture
..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe play_motion.py trajectories/demo_wave_left.json
```

### 3. Deploy to Physical Unitree G1 EDU Robot
```powershell
# Live interactive control with mouse / keys
..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe run_simulation.py --hardware --iface Ethernet

# Or replay any recorded motion directly on physical robot
..\unitree_g1_vibes\RL-shenanigans\.venv\Scripts\python.exe play_motion.py trajectories/demo_wave_left.json --hardware --iface Ethernet
```
