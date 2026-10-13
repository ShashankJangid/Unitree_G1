from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class RobotConfig:
    num_motors: int = 29
    fix_base: bool = True
    enable_elastic_band: bool = True


@dataclass
class DDSConfig:
    domain_sim: int = 1
    domain_real: int = 0
    interface_sim: str = "lo"
    interface_real: str = "Ethernet"


@dataclass
class ControlConfig:
    physics_freq: int = 500
    policy_freq: int = 50
    control_freq: int = 500


@dataclass
class SafetyConfig:
    max_joint_velocity: float = 1.2
    max_joint_acceleration: float = 10.0
    max_torque: float = 100.0
    fall_height_threshold: float = 0.45
    joint_limit_margin: float = 0.03


@dataclass
class GainsConfig:
    body_kp: float = 60.0
    body_kd: float = 1.5
    hand_kp: float = 20.0
    hand_kd: float = 0.5


@dataclass
class SimulationConfig:
    timestep: float = 0.002
    gravity: list = None
    elastic_band_stiffness: float = 5000.0
    elastic_band_damping: float = 100.0
    elastic_band_height: float = 0.793

    def __post_init__(self):
        if self.gravity is None:
            self.gravity = [0, 0, -9.81]


@dataclass
class HardwareConfig:
    auto_release_motion_mode: bool = True
    wait_for_robot_timeout: float = 5.0
    arm_sdk_enable_delay: float = 0.2
    crc_enabled: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    save_trajectories: bool = True
    trajectory_dir: str = "trajectories"


@dataclass
class PolicyConfig:
    default_policy: str = "stand"
    checkpoint_dir: str = "models"
    onnx_models_dir: str = "models/onnx"


@dataclass
class G1Config:
    robot: RobotConfig = None
    dds: DDSConfig = None
    control: ControlConfig = None
    safety: SafetyConfig = None
    gains: GainsConfig = None
    simulation: SimulationConfig = None
    hardware: HardwareConfig = None
    logging: LoggingConfig = None
    policy: PolicyConfig = None

    def __post_init__(self):
        if self.robot is None:
            self.robot = RobotConfig()
        if self.dds is None:
            self.dds = DDSConfig()
        if self.control is None:
            self.control = ControlConfig()
        if self.safety is None:
            self.safety = SafetyConfig()
        if self.gains is None:
            self.gains = GainsConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()
        if self.hardware is None:
            self.hardware = HardwareConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.policy is None:
            self.policy = PolicyConfig()


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
LOCAL_CONFIG_PATH = Path(__file__).parent / "config_local.yaml"


def load_config(config_path: Optional[Path] = None) -> G1Config:
    """Load configuration from YAML file."""
    config = G1Config()

    paths_to_try = []
    if config_path:
        paths_to_try.append(config_path)
    paths_to_try.extend([LOCAL_CONFIG_PATH, DEFAULT_CONFIG_PATH])

    for path in paths_to_try:
        if path.exists():
            print(f"[Config] Loading configuration from {path}")
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            if data:
                _apply_dict_to_config(config, data)
            break
    else:
        print("[Config] No config file found, using defaults")

    return config


def _apply_dict_to_config(config: G1Config, data: Dict[str, Any]) -> None:
    """Recursively apply dictionary values to config dataclass."""
    for key, value in data.items():
        if hasattr(config, key):
            attr = getattr(config, key)
            if hasattr(attr, '__dataclass_fields__'):
                _apply_dict_to_config(attr, value)
            else:
                setattr(config, key, value)


def save_config(config: G1Config, path: Path) -> None:
    """Save configuration to YAML file."""
    data = _config_to_dict(config)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[Config] Saved configuration to {path}")


def _config_to_dict(config: G1Config) -> Dict[str, Any]:
    """Convert config dataclass to dictionary."""
    result = {}
    for key in config.__dataclass_fields__:
        value = getattr(config, key)
        if hasattr(value, '__dataclass_fields__'):
            result[key] = _config_to_dict(value)
        else:
            result[key] = value
    return result


def create_local_config() -> None:
    """Create a local config file from defaults."""
    config = G1Config()
    save_config(config, LOCAL_CONFIG_PATH)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        create_local_config()
    else:
        config = load_config()
        print(yaml.dump(_config_to_dict(config), default_flow_style=False))
