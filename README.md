# Unitree G1 Humanoid Robot Framework

A comprehensive, production-ready software and simulation suite for the **Unitree G1 (29-DoF EDU)** humanoid robot. This repository brings together high-fidelity MuJoCo physics simulation, real-time Cartesian teleoperation with interactive 3D mouse tracking, motion recording and trajectory playback, hardware deployment bridges, reinforcement learning environments, and sensor processing pipelines (LiDAR SLAM, RealSense depth streaming, and GUI teleoperation).

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Architecture](#repository-architecture)
- [Modules Breakdown](#modules-breakdown)
  - [1. Real-Physics Simulation & Hardware Bridge (`g1_physics_sim`)](#1-real-physics-simulation--hardware-bridge-g1_physics_sim)
  - [2. Vision, SLAM, Teleop & RL Stack (`unitree_g1_vibes`)](#2-vision-slam-teleop--rl-stack-unitree_g1_vibes)
- [Safety & Kinematic Governors](#safety--kinematic-governors)
- [Interactive Controls & Teleoperation](#interactive-controls--teleoperation)
- [Installation & Quick Start](#installation--quick-start)
- [Physics Verification](#physics-verification)
- [License](#license)

---

## 🌟 Overview

The **Unitree G1 Humanoid Robot Framework** provides an end-to-end platform bridging theoretical robotics simulation with physical deployment on the 29-DoF Unitree G1 humanoid platform. Designed with modularity and hardware safety in mind, it enables researchers and robotics engineers to rapidly prototype motor controllers, record complex whole-body choreographies, train RL policies, process Livox Mid-360 LiDAR and Intel RealSense visual data, and execute commands safely over Unitree SDK2 DDS channels.

---

## 🚀 Key Features

- **Interactive 3D Simulation & Teleoperation**:
  - Drag-and-drop 3D Cartesian target handles directly inside the MuJoCo viewer using mouse interactions (`Ctrl + Right-Click Drag`).
  - Real-time inverse kinematics (Damped Least Squares & Jacobian Transpose) for dual 7-DoF arm manipulation.
- **5-Layer Super-Safe Kinematic Governor**:
  - Spherical workspace bounding ($R_{\max} = 34.0\text{ cm}$) preventing physical overextension.
  - Anti-hyperextension guard ($q_{\text{elbow}} \ge +0.05\text{ rad}$) ensuring natural forward bending.
  - Slew-rate velocity limiting capped at $1.20\text{ rad/s}$ to eliminate hardware shock.
  - Zero-jerk initial state synchronization and hardware streaming clutch (`[STANDBY]` mode with spacebar emergency stop).
- **Motion Recording & Trajectory Engine**:
  - 50 Hz trajectory recorder capturing all 29 joint positions, velocities, end-effector coordinates, and timestamps.
  - Standalone player (`play_motion.py`) with cubic spline interpolation for simulation and hardware replay.
  - Pre-built choreographies (friendly wave gesture, Cartesian circle, and square tracing).
- **Reinforcement Learning Suite**:
  - Gymnasium-compliant environments (`g1_arm_rl_env.py`) integrated with Stable-Baselines3 PPO.
  - Reward structures for precision end-effector reaching, smoothness, and torque minimization.
- **LiDAR SLAM & Perception**:
  - Real-time 3D point cloud processing and SLAM with Livox Mid-360 LiDAR sensors.
  - GStreamer pipelines for low-latency Intel RealSense depth and color video streaming over Jetson hardware.
- **Hardware DDS Bridge**:
  - Zero-code-change DDS bridge communicating via Unitree SDK2 `rt/arm_sdk` and `rt/lowcmd` topics.

---

## 📁 Repository Architecture

```
Unitree G1/
├── g1_physics_sim/              # Real-Physics simulation, teleoperation & hardware bridge
│   ├── g1_constants.py          # Official 29-DoF joint indices, limits & nominal stance
│   ├── g1_robot.py              # MuJoCo physics engine, FK, Jacobians & PD actuator models
│   ├── g1_env.py                # Gym environment with 500 Hz physics & 50 Hz control loops
│   ├── g1_hardware_bridge.py    # Unitree SDK2 DDS bridge (rt/arm_sdk & rt/lowcmd)
│   ├── g1_trajectory_manager.py # Motion recorder, player & trajectory generator
│   ├── run_simulation.py        # Interactive 3D viewer with mouse drag & teleoperation
│   ├── play_motion.py           # Standalone motion replayer for Sim & Physical Hardware
│   ├── test_robot_connection.py # Non-destructive DDS network diagnostic tool
│   ├── verify_physics.py        # Automated physics verification suite (5/5 tests passing)
│   ├── models/g1/               # Clean 29-DoF MuJoCo MJCF XMLs, meshes & materials
│   └── trajectories/            # Exported motion recordings & demo gestures
│
├── unitree_g1_vibes/            # Perception, SLAM, GUI teleoperation & RL pipelines
│   ├── run_geoff_gui.py         # Full-featured teleoperation GUI & robot monitor
│   ├── run_geoff_stack.py       # Multi-process stack launcher (SLAM + Video + Control)
│   ├── live_slam.py             # Livox Mid-360 LiDAR 3D SLAM visualizer
│   ├── live_points.py           # Raw point cloud streamer and visualizer
│   ├── livox_python.py          # Livox SDK Python interface
│   ├── stream_realsense.py      # RealSense camera capture & GStreamer transmitter
│   ├── receive_realsense_gst.py # GStreamer depth/color video stream receiver
│   ├── keyboard_controller.py   # Keyboard teleoperation client for locomotion
│   ├── hanger_boot_sequence.py  # Safe motorized boot & calibration routine
│   ├── g1_loco_client_example.py# Unitree locomotion API interface examples
│   └── RL-shenanigans/          # Reinforcement learning experiments
│       ├── g1_arm_rl_env.py     # Custom Gym reaching environment
│       ├── train_g1_arm_policy.py# PPO training pipeline using Stable-Baselines3
│       ├── eval_policy.py       # Policy evaluation benchmarks
│       └── demo_sim_viewer.py   # Visual demonstration of trained RL policies
│
├── LICENSE                      # MIT Open Source License
├── .gitignore                   # Standard ignore rules (venv, cache, temporary files)
└── README.md                    # Project documentation
```

---

## 🛠️ Modules Breakdown

### 1. Real-Physics Simulation & Hardware Bridge (`g1_physics_sim`)
The simulation subsystem models the Unitree G1 EDU robot using MuJoCo with explicit 29-DoF dynamics:
- **Kinematic Formulation**: Dual 7-DoF arms with full analytical and numerical forward kinematics, Jacobian computation, and nullspace posture projection.
- **Actuation Models**: Motor PD controller modeled with realistic torque saturation limits (Hip: 88 N·m, Knee: 139 N·m, Arm joints: 25 N·m, Wrist: 5 N·m).
- **Interactive Teleop**: Users can manipulate end-effectors via keyboard shortcuts or by directly dragging 3D target coordinates with a mouse.
- **Trajectory Engine**: Real-time 50 Hz trajectory capture and playback across simulation and live DDS robot bridges.

### 2. Vision, SLAM, Teleop & RL Stack (`unitree_g1_vibes`)
The perception and telemetry subsystem handles real-world sensor streams and locomotion:
- **Perception Pipeline**: Ingests point clouds from Livox Mid-360 LiDAR via UDP broadcasts, maintaining real-time occupancy and point cloud maps.
- **Video Streaming**: Hardware-accelerated H.264/RTP streaming using GStreamer for low-latency visual feedback from onboard Jetson computers.
- **Reinforcement Learning**: Reinforcement learning environment modeling reaching tasks with PPO policies in MuJoCo.

---

## 🛡️ Safety & Kinematic Governors

Deploying high-torque humanoid robots requires strict safety constraints to protect both hardware and human operators. The framework incorporates five integrated safety layers:

1. **Spherical Workspace Governor**: Commands outside a radial distance $R_{\max} = 0.34\text{ m}$ from the shoulder pivot are clamped automatically.
2. **Anti-Hyperextension Guard**: The elbow joint is strictly constrained ($q_{\text{elbow}} \ge +0.05\text{ rad}$) to prevent mechanical lockup or reverse bending.
3. **Nullspace Posture Projection**: Redundant degrees of freedom utilize nullspace projection $(I - J^\dagger J)$ to preserve a natural human-like rest posture.
4. **Hardware Slew-Rate Limiter**: Maximum joint velocity is capped at $1.20\text{ rad/s}$ during real-time teleoperation and trajectory playback.
5. **DDS Streaming Clutch**: Real-world motor streaming starts in `[STANDBY]` mode; pressing `O` engages active hardware streaming, and `Spacebar` acts as an immediate software emergency stop.

---

## 🎮 Interactive Controls Reference

When running the interactive simulation viewer (`run_simulation.py`):

| Control | Action | Details |
| :---: | :---: | :---: |
| **`Ctrl` + Right-Click Drag** | **Mouse 3D Drag** | Drag target sphere directly with mouse cursor |
| **`W` / `S`** (or `↑` / `↓`) | Up / Down ($+z / -z$) | Step target up/down in Cartesian space |
| **`A` / `D`** (or `←` / `→`) | Left / Right ($+y / -y$) | Step target left/right in Cartesian space |
| **`Q` / `E`** | Forward / Backward ($+x / -x$) | Step target forward/backward |
| **`M`** | **Record Motion** | Start / Stop recording all 29 motors to JSON |
| **`P`** | **Play Trajectory** | Replay last recorded motion or loaded trajectory |
| **`C`** | **Run Choreography** | Execute built-in script (Wave -> Circle -> Square) |
| **`O`** (or `T`) | **Hardware Clutch** | Toggle live motor streaming to real G1 robot |
| **`Spacebar`** | **Emergency Stop** | Instantly disengage hardware streaming & pause |
| **`1`** / **`2`** | Switch Arm | `1` = Left Arm, `2` = Right Arm |
| **`R`** | Reset Posture | Return arm to nominal safe rest pose |
| **`]`** / **`[`** | Adjust Step Size | Increase / decrease Cartesian step size |
| **`Esc`** | Exit | Safely close simulation and DDS channels |

---

## 📦 Installation & Quick Start

### Prerequisites
- Python 3.10+
- MuJoCo 3.0+
- NumPy, SciPy, Matplotlib
- Gymnasium, Stable-Baselines3 (optional, for RL module)
- PyQt5, Open3D, OpenCV (optional, for GUI and SLAM viewer)

### Setup
```bash
git clone https://github.com/ShashankJangid/Unitree_G1.git
cd Unitree_G1
pip install numpy scipy mujoco gymnasium stable-baselines3
```

### Launch Interactive Simulation
```bash
cd g1_physics_sim
python run_simulation.py
```

### Replay a Trajectory
```bash
# Playback in MuJoCo simulation
python play_motion.py trajectories/demo_wave_left.json

# Playback on physical G1 robot via DDS Ethernet interface
python play_motion.py trajectories/demo_wave_left.json --hardware --iface eth0
```

### Run Automated Physics Verification
```bash
python verify_physics.py
```

---

## 🔬 Physics Verification

The automated verification suite (`verify_physics.py`) asserts model integrity against 5 rigorous benchmarks:
- **Model Compilation**: Compiles 29 actuators and 35 system DoFs without collision or constraint errors.
- **Torque Limit Saturation**: Confirms computed motor torques strictly adhere to official torque limits under full gravitational loading.
- **Sub-Millimeter Convergence**: Validates that Cartesian reaching tasks achieve $< 1\text{ mm}$ terminal position error across random reachable coordinates.
- **Out-of-Reach Saturation**: Ensures out-of-reach Cartesian targets are safely clamped to the $34\text{ cm}$ spherical boundary.
- **Elbow Non-Inversion**: Verifies the elbow angle remains strictly positive ($q > 0$) without hyperextension under all reaching conditions.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
