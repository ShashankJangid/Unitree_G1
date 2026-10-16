# Unitree G1 Humanoid Simulation & RL Framework

Clean, high-performance simulation and Reinforcement Learning framework for the Unitree G1 humanoid robot in MuJoCo with NVIDIA GPU acceleration.

---

## Minimal 3-Script Architecture

The entire codebase is structured into three clean, self-contained Python scripts:

| File | Purpose |
| :--- | :--- |
| **[`main.py`](file:///c:/Users/D&T/Documents/Shashank/Unitree%20G1/main.py)** | Unified entrypoint to run reactive locomotion teleoperation or whole-body RL training. |
| **[`g1_teleop.py`](file:///c:/Users/D&T/Documents/Shashank/Unitree%20G1/g1_teleop.py)** | Reactive omnidirectional locomotion teleoperation runner with high-speed bursts and perturbation recovery. |
| **[`g1_train.py`](file:///c:/Users/D&T/Documents/Shashank/Unitree%20G1/g1_train.py)** | Whole-body bending down & object pickup RL environment (`G1PickupEnv`) + PPO training loop with live 3D rendering. |

---

## Directory Structure

```
Unitree G1/
├── models/
│   ├── motion.pt                     # Official Unitree pretrained locomotion policy
│   └── g1/
│       ├── scene.xml                 # 12-DOF locomotion scene
│       ├── g1_12dof.xml              # 12-DOF robot model
│       ├── g1_pickup_scene.xml       # Bending & pickup training scene with table & props
│       ├── g1_29dof_with_hand.xml    # 29-DOF humanoid with dexterous hands
│       └── meshes/                   # STL robot surface meshes
├── g1_teleop.py                      # Reactive omnidirectional locomotion runner
├── g1_train.py                       # Whole-body bending & object pickup RL trainer
├── main.py                           # Unified CLI runner
├── run.bat                           # 1-Click launcher for teleoperation
└── train.bat                         # 1-Click launcher for RL training
```

---

## Quick Start

### 1. Run Reactive Locomotion Teleoperation
Double-click `run.bat` or run:
```powershell
.\run.bat
```
*(Or `.\.venv\Scripts\python.exe main.py teleop`)*

#### Live Controls (Works in MuJoCo 3D Window & Terminal)
- **`W` / `Up Arrow`**: Fast Sprint burst ($1.00\text{ m/s}$)
- **`Z`**: Turbo Sprint burst ($1.40\text{ m/s}$)
- **`S` / `Down Arrow`**: Active Decel / Brake ($-0.35\text{ m/s}$)
- **`A` / `Left Arrow`**: Responsive Sharp Left Turn ($+0.85\text{ rad/s}$)
- **`D` / `Right Arrow`**: Responsive Sharp Right Turn ($-0.85\text{ rad/s}$)
- **`Q` / `E`**: Responsive Lateral Strafe ($\pm 0.45\text{ m/s}$)
- **`Space` / `X`**: Reset to Steady Cruise ($0.40\text{ m/s}$)
- **`R`**: Instant Stance Reset
- **`F`**: Apply $50\text{ N}$ torso push perturbation

---

### 2. Train Whole-Body Bending & Object Pickup RL
Double-click `train.bat` or run:
```powershell
.\train.bat
```
*(Or `.\.venv\Scripts\python.exe main.py train`)*

#### Controlled Degrees of Freedom (12 DOF)
- **Waist Pitch**: Bends torso forward down (up to $0.9\text{ rad} \approx 51^\circ$)
- **Waist Yaw & Roll**: Torso twist and lateral tilt
- **Knee Crouch**: Symmetrical leg flexion to lower the pelvis toward low objects
- **Right Arm (7 DOF)**: Shoulder pitch/roll/yaw, elbow flexion, wrist orientation
- **Dexterous Hand**: Multi-finger closure for grasping objects