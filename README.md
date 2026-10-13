# Unitree G1 - Sim-to-Real Robotics Framework

A production-ready Python framework for controlling the Unitree G1 humanoid robot in both simulation (MuJoCo) and on real hardware. Features a unified sim-to-real pipeline with identical code paths.

## Features

- **Unified Sim-to-Real**: Same policy code runs in MuJoCo simulation and on real G1 hardware
- **Two-Thread Architecture**: 500Hz control loop + configurable policy rate (default 50Hz)
- **DDS Domain Separation**: Simulation uses domain 1, real robot uses domain 0 (prevents accidental cross-talk)
- **Safety First**: Emergency stop, joint limit enforcement, fall detection, velocity clamping
- **Elastic Band Support**: Virtual gantry for safe simulation of free-standing humanoid
- **MotionSwitcher Integration**: Automatic release of onboard sport service before low-level control
- **Arm SDK Support**: Low-level arm control via `rt/arm_sdk` DDS topic
- **Trajectory Recording/Playback**: Record and replay full-body motions
- **Policy Framework**: Abstract `Policy` interface with examples (stand, wave, ankle swing, locomotion)
- **RL Environments**: Gymnasium-compatible environments for arm reaching and locomotion

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Policy                                │
│  (reset, step, get_observation) - User implemented          │
└─────────────────────┬───────────────────────────────────────┘
                      │ G1State / G1Action
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   UnitreeG1Robot                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │  Policy Thread      │    │  Control Thread (500 Hz)    │ │
│  │  @ policy_rate Hz   │    │  Reads target actions       │ │
│  │  Calls policy.step()│    │  Publishes to DDS           │ │
│  └──────────┬──────────┘    └──────────────┬──────────────┘ │
│             │                                │              │
│             ▼                                ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              G1Bridge (Abstract)                     │   │
│  │  ┌─────────────────┐         ┌─────────────────┐    │   │
│  │  │ G1SimBridge     │         │ G1RealHardware  │    │   │
│  │  │ (MuJoCo)        │         │ Bridge (DDS)    │    │   │
│  │  └─────────────────┘         └─────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Core Dependencies
```bash
pip install -r requirements.txt
```

### 2. Unitree SDK2 Python (Required for real robot)
```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

### 3. CycloneDDS (Required for DDS communication)
```bash
# Option 1: pip (if available)
pip install cyclonedds

# Option 2: Build from source
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
export CYCLONEDDS_HOME=/path/to/cyclonedds/install
```

### 4. MuJoCo Models
The framework includes G1 MJCF models in `g1_physics_sim/models/g1/`.

## Quick Start

### Simulation (Safe Development)
```bash
cd g1_physics_sim

# Run ankle swing demo (Unitree's official demo)
python run_sim.py --policy ankle_swing

# Run wave gesture
python run_sim.py --policy wave --right-arm

# Interactive Cartesian control
python run_simulation.py

# Free-standing (no gantry)
python run_simulation.py --free-standing
```

### Real Robot Deployment
```bash
# 1. Connect robot via Ethernet
# 2. Set PC IP to 192.168.123.x (robot is 192.168.123.99)
# 3. Find your interface: ifconfig (Linux) or ipconfig (Windows)
# 4. Robot MUST be suspended from gantry!
# 5. Wireless controller with E-stop in hand!

cd g1_physics_sim

# Stand policy (safest first test)
python run_real.py --policy stand --interface eth0

# Ankle swing (requires gantry)
python run_real.py --policy ankle_swing --interface eth0

# Wave gesture
python run_real.py --policy wave --interface eth0 --right-arm
```

## Configuration

Copy `config.yaml` to `config_local.yaml` and modify:

```yaml
robot:
  num_motors: 29              # 29 (body) or 43 (body + Dex3 hands)
  fix_base: true
  enable_elastic_band: true

dds:
  domain_sim: 1
  domain_real: 0
  interface_sim: "lo"
  interface_real: "Ethernet"   # Your ethernet interface

control:
  physics_freq: 500
  policy_freq: 50

safety:
  max_joint_velocity: 1.2
  fall_height_threshold: 0.45
```

## Writing Custom Policies

```python
from g1_policy import Policy
from g1_state import G1State, G1Action, G1Observation
import numpy as np

class MyCustomPolicy(Policy):
    def __init__(self, my_param: float = 1.0):
        self.my_param = my_param

    def reset(self, state: G1State) -> None:
        """Called at episode start"""
        pass

    def step(self, state: G1State) -> G1Action:
        """Compute action at policy rate (e.g., 50 Hz)"""
        action = G1Action.zero(num_body=29)
        action.q_des[:] = state.q  # Hold current position
        action.kp[:] = 60.0
        action.kd[:] = 1.5
        action.enable_arm_sdk = True
        return action

    def get_observation(self, state: G1State) -> G1Observation:
        """For logging/debugging"""
        return G1Observation(
            timestamp=state.timestamp,
            joint_pos=state.q,
            joint_vel=state.dq,
            # ...
        )

    @property
    def action_dim(self) -> int:
        return 29

    @property
    def obs_dim(self) -> int:
        return 24  # Your observation dimension
```

Run your policy:
```python
from g1_policy import run_policy
from my_policy import MyCustomPolicy

policy = MyCustomPolicy(my_param=0.5)
robot = run_policy(policy, mode="sim", duration=10.0)
```

## Key Classes

### G1State
Complete robot state including joint positions, velocities, torques, base pose, IMU, foot contacts, and hand poses.

### G1Action
Robot action with joint position/velocity/torque targets, PD gains, and motor modes.

### G1Bridge / G1SimBridge / G1RealHardwareBridge
Abstract interface for sim/real robot communication. Hardware bridge handles:
- DDS publisher/subscriber for LowCmd/LowState
- MotionSwitcherClient for service management
- CRC calculation for message integrity
- Two-thread architecture (500Hz control + policy rate)

### UnitreeG1Robot
Main entry point. Handles policy execution, safety monitoring, and sim/real abstraction.

## Trajectory Recording

```python
from g1_trajectory_manager import TrajectoryRecorder, TrajectoryPlayer
from g1_state import G1State, G1Action

recorder = TrajectoryRecorder()
recorder.start()

# During control loop:
state = robot.get_state()
action = policy.step(state)
recorder.record_state(state, action, target_pos)

# Save
path = recorder.stop_and_save("my_motion.json")

# Playback
player = TrajectoryPlayer(path)
player.start_playback(loop=False)
action, done = player.get_action()
```

## Safety Checklist (Real Robot)

Before running on hardware:
- [ ] Robot suspended from certified gantry/strap
- [ ] Wireless controller paired and E-stop tested
- [ ] Clear 2m radius around robot
- [ ] Battery > 50%
- [ ] Correct network interface specified
- [ ] DDS domain = 0 (real robot default)
- [ ] Start with `--num-motors 29` (body only)
- [ ] Test `stand` policy first
- [ ] Have colleague ready with E-stop

## Sim-to-Real Tips

1. **Use Domain 1 for Sim**: Prevents accidental commands to real robot
2. **Match PD Gains**: Use same KP/KD in sim and real
3. **Arm SDK Enable**: Joint index 29 (`kNotUsedJoint`) must be set to 1.0
4. **Release Motion Mode**: Always call `release_motion_mode()` before low-level control
5. **CRC Required**: Real robot requires valid CRC on LowCmd messages
6. **Velocity Clamping**: Hardware bridge clamps joint velocity to 1.2 rad/s

## Directory Structure

```
g1_physics_sim/
├── __init__.py              # Package exports
├── config.yaml              # Default configuration
├── config_local.yaml        # Local overrides (gitignored)
├── config_loader.py         # Configuration management
├── g1_constants.py          # Joint indices, gains, limits, DDS topics
├── g1_robot.py              # MuJoCo robot model
├── g1_hardware_bridge.py    # Sim/Real bridge implementations
├── g1_state.py              # State/Action/Observation dataclasses
├── g1_policy.py             # Policy interface + robot runner
├── g1_policies.py           # Example policies
├── g1_trajectory_manager.py # Recording/playback utilities
├── g1_env.py                # Gymnasium RL environments
├── run_sim.py               # Simple policy runner (sim)
├── run_real.py              # Real robot deployment script
├── run_simulation.py        # Interactive Cartesian control
├── requirements.txt         # Python dependencies
└── models/g1/               # MJCF models and meshes
```

## RL Training

```bash
cd unitree_g1_vibes/RL-shenanigans
python train_g1_arm_policy.py --algo ppo --total-steps 1000000
```

## Resources

- [Unitree G1 SDK Documentation](https://support.unitree.com/home/en/G1_developer/get_sdk)
- [Unitree MuJoCo Simulator](https://github.com/unitreerobotics/unitree_mujoco)
- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [G1-Playground (AlexandreBrown)](https://github.com/AlexandreBrown/G1-Playground)
- [G1 Deploy Python (TheoBounac)](https://github.com/TheoBounac/G1_deploy_python)

## License

MIT License - See LICENSE file for details.

## Citation

If you use this framework in research, please cite:
```
@software{unitree_g1_framework,
  title = {Unitree G1 Sim-to-Real Robotics Framework},
  author = {Shashank},
  year = {2026},
  url = {https://github.com/...}
}
```