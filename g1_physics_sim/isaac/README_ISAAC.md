# NVIDIA Isaac Sim & Isaac Lab Integration for Unitree G1 EDU (29-DoF)

This suite enables photorealistic, GPU-accelerated PhysX 5 simulation of the Unitree G1 EDU humanoid robot using **NVIDIA Isaac Sim** and **NVIDIA Isaac Lab**.

---

## 🚀 Features

- **PhysX 5 GPU Acceleration**: Full rigid-body articulation dynamics running on NVIDIA CUDA/Tensor cores with the Temporal Gauss-Seidel (TGS) solver.
- **Photorealistic RTX Ray Tracing**: Real-time lighting, reflections, and camera sensors for vision-based learning and teleoperation.
- **29-DoF Direct Joint Actuation**: Matches official hardware torque limits ($100\text{ Nm}$) and PD gains ($K_p = 60.0, K_d = 1.5$).
- **Integrated Full-Spectrum Safety**: S-curve jerk limiting, self-collision capsule protection, and thermal watchdog actively running during simulation.
- **Massively Parallel Isaac Lab Environments**: Train or benchmark hundreds of G1 robots in parallel on a single GPU.

---

## 🛠️ Prerequisites & Installation

1. **System Requirements**:
   - Windows 10/11 or Ubuntu 20.04/22.04
   - NVIDIA RTX GPU (RTX 3070 / 4070 / 4080 / 4090 / A5000+ recommended)
   - NVIDIA Driver $\ge 535.129$

2. **Install NVIDIA Isaac Sim**:
   - Download the **Omniverse Launcher** from [NVIDIA Developer](https://www.nvidia.com/en-us/omniverse/).
   - In the Launcher Exchange tab, search for **Isaac Sim** (v4.0 or v4.1) and click **Install**.
   - Default install path on Windows:
     ```
     C:\Users\<User>\AppData\Local\ov\pkg\isaac-sim-4.1.0\
     ```

---

## 📦 Step-by-Step Execution Guide

### Step 1: Generate Standard G1 URDF
Generate the clean 29-DoF URDF from the MuJoCo kinematics definitions:
```powershell
cd "c:\Users\D&T\Documents\Shashank\Unitree G1\g1_physics_sim\isaac"
python g1_urdf_generator.py
```
*Output: `models/g1/g1_29dof.urdf`*

### Step 2: Convert URDF to Optimized USD Stage (Optional)
```powershell
# Using Isaac Sim Python interpreter:
C:\Users\<User>\AppData\Local\ov\pkg\isaac-sim-4.1.0\python.bat convert_urdf_to_usd.py
```

### Step 3: Launch NVIDIA Isaac Sim Simulation
Run the full simulation with PhysX 5 GPU physics and active safety controls:
```powershell
# Interactive GUI mode with RTX Ray Tracing:
C:\Users\<User>\AppData\Local\ov\pkg\isaac-sim-4.1.0\python.bat g1_isaac_sim.py

# Headless mode for background execution:
C:\Users\<User>\AppData\Local\ov\pkg\isaac-sim-4.1.0\python.bat g1_isaac_sim.py --headless

# Free-standing mode (unlatching gantry harness):
C:\Users\<User>\AppData\Local\ov\pkg\isaac-sim-4.1.0\python.bat g1_isaac_sim.py --free-standing
```

### Step 4: Run NVIDIA Isaac Lab (GPU Parallel RL)
To run 64–2048 parallel robot instances on a single GPU:
```powershell
isaaclab.bat -p g1_isaac_lab_env.py --num_envs 64
```

---

## 🛡️ Integrated Safety in Isaac Sim

The Isaac Sim runner continuously communicates with `g1_safety.py`:
- **Self-Collision Capsule Interlock**: Prevents wrists from penetrating the torso or colliding with each other.
- **S-Curve Jerk Limiter**: Filters velocity ($1.2\text{ rad/s}$), acceleration ($10.0\text{ rad/s}^2$), and jerk ($50.0\text{ rad/s}^3$).
- **Thermal $I^2t$ Watchdog**: Simulates actuator winding heating and alerts if overloaded.
- **Base Tilt & Fall Sensor**: Detects robot tipping over (>25°) and triggers emergency hold.
