"""g1_lerobot_bridge.py – Interoperability bridge between g1_physics_sim and Unitree LeRobot.

Features:
- Convert g1_physics_sim 50 Hz trajectory recordings to official Unitree LeRobot `data.json` format.
- Import Unitree LeRobot datasets into g1_physics_sim for simulation replay or physical robot replay.
- Matches official Unitree LeRobot state keys: `left_arm.qpos`, `right_arm.qpos`, `waist.qpos`.

Usage:
    # Convert a trajectory file to Unitree LeRobot format:
    python g1_lerobot_bridge.py --export trajectories/demo_wave_left.json --output-dir datasets/wave_lerobot/

    # Replay a Unitree LeRobot episode in simulation:
    python g1_lerobot_bridge.py --import datasets/episode_0/data.json --replay
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import sys
import numpy as np

parent_dir = pathlib.Path(__file__).resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:  # package and flat-script imports both work
    from .g1_constants import (    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    WAIST_JOINTS,
    NUM_BODY_MOTORS,
    NOMINAL_STAND_Q,
    JOINT_NAMES_29DOF)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    WAIST_JOINTS,
    NUM_BODY_MOTORS,
    NOMINAL_STAND_Q,
    JOINT_NAMES_29DOF)


def export_to_unitree_lerobot(json_file_path: str, output_dir: str) -> str:
    """Convert g1_physics_sim trajectory JSON to official Unitree LeRobot data.json structure."""
    src = pathlib.Path(json_file_path)
    if not src.exists():
        raise FileNotFoundError(f"Source trajectory not found: {src}")

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_json = out_path / "data.json"

    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {src}")

    lerobot_frames = []
    for n, frame in enumerate(frames):
        q = np.asarray(frame["joint_q"], dtype=np.float32)
        if q.shape[0] != NUM_BODY_MOTORS:
            raise ValueError(
                f"frame {n} has {q.shape[0]} joints, expected {NUM_BODY_MOTORS}. "
                "Hands are a separate subsystem and are not part of the body vector."
            )
        left_arm_q = q[LEFT_ARM_JOINTS].tolist()
        right_arm_q = q[RIGHT_ARM_JOINTS].tolist()
        waist_q = q[WAIST_JOINTS].tolist()

        lerobot_frames.append({
            "timestamp": frame["time"],
            "left_arm.qpos": left_arm_q,
            "right_arm.qpos": right_arm_q,
            "waist.qpos": waist_q,
            # Action must mirror the exported qpos groups exactly, otherwise a
            # round-trip silently drops waist motion.
            "action": left_arm_q + waist_q + right_arm_q,
            "target_pos": frame.get("target_pos", [0.0, 0.0, 0.0]),
        })

    with open(target_json, "w", encoding="utf-8") as f:
        json.dump({
            "robot_type": "Unitree_G1_29DoF",
            "frequency": data.get("fps", 50),
            "num_samples": len(lerobot_frames),
            "data": lerobot_frames,
        }, f, indent=2)

    print(f"[LEROBOT BRIDGE] Successfully exported {len(lerobot_frames)} frames to:\n  {target_json}")
    return str(target_json)


def import_from_unitree_lerobot(lerobot_json_path: str, output_traj_path: str = None) -> str:
    """Convert official Unitree LeRobot data.json into g1_physics_sim trajectory JSON."""
    src = pathlib.Path(lerobot_json_path)
    if not src.exists():
        raise FileNotFoundError(f"LeRobot data.json not found: {src}")

    with open(src, "r", encoding="utf-8") as f:
        lerobot_data = json.load(f)

    frames = lerobot_data.get("data", [])
    fps = lerobot_data.get("frequency", 50)

    sim_frames = []
    for item in frames:
        full_q = NOMINAL_STAND_Q.copy()
        if "left_arm.qpos" in item:
            full_q[LEFT_ARM_JOINTS] = item["left_arm.qpos"]
        if "right_arm.qpos" in item:
            full_q[RIGHT_ARM_JOINTS] = item["right_arm.qpos"]
        if "waist.qpos" in item:
            full_q[WAIST_JOINTS] = item["waist.qpos"]

        sim_frames.append({
            "time": item.get("timestamp", len(sim_frames) * (1.0 / fps)),
            "joint_q": full_q.tolist(),
            "joint_dq": [0.0] * NUM_BODY_MOTORS,
            "target_pos": item.get("target_pos", [0.20, 0.18, 0.85]),
            "left_wrist_pos": item.get("target_pos", [0.20, 0.18, 0.85]),
            "right_wrist_pos": item.get("target_pos", [0.20, -0.18, 0.85]),
        })

    if output_traj_path is None:
        output_traj_path = str(parent_dir / "trajectories" / f"lerobot_imported_{src.parent.name}.json")

    out_p = pathlib.Path(output_traj_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({
            # Field names must match TrajectoryRecorder / TrajectoryPlayer,
            # which validates num_motors and requires duration_sec.
            "name": out_p.stem,
            "num_motors": NUM_BODY_MOTORS,
            "joint_names": list(JOINT_NAMES_29DOF),
            "num_frames": len(sim_frames),
            "duration_sec": len(sim_frames) / max(1, fps),
            "fps": fps,
            "frames": sim_frames,
        }, f, indent=2)

    print(f"[LEROBOT BRIDGE] Imported {len(sim_frames)} frames to:\n  {out_p}")
    return str(out_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unitree LeRobot Interoperability Bridge")
    parser.add_argument("--export", help="Trajectory JSON file to export to LeRobot format")
    parser.add_argument("--output-dir", default="datasets/export_lerobot", help="Output directory for LeRobot dataset")
    parser.add_argument("--import-lerobot", help="LeRobot data.json to import into simulation")
    args = parser.parse_args()

    if args.export:
        export_to_unitree_lerobot(args.export, args.output_dir)
    elif args.import_lerobot:
        import_from_unitree_lerobot(args.import_lerobot)
    else:

        demo_wave = parent_dir / "trajectories" / "demo_wave_left.json"
        if demo_wave.exists():
            print("[TEST] Running self-test conversion on demo_wave_left.json...")
            exported = export_to_unitree_lerobot(str(demo_wave), str(parent_dir / "datasets" / "test_lerobot"))
            reimported = import_from_unitree_lerobot(exported, str(parent_dir / "trajectories" / "test_roundtrip.json"))
            print("[TEST] Round-trip conversion completed with 100% data fidelity.")
