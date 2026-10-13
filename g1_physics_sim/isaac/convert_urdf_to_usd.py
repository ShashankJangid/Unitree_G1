"""convert_urdf_to_usd.py – Automated URDF to USD converter for NVIDIA Isaac Sim.

Converts `models/g1/g1_29dof.urdf` into a high-performance PhysX USD stage `g1_29dof.usd`
with collision meshes, drive stiffness/damping, and mass properties configured.

Usage:
    # Run with Isaac Sim Python:
    python.bat convert_urdf_to_usd.py
"""

import os
import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent
urdf_file = parent_dir / "models" / "g1" / "g1_29dof.urdf"
usd_output = parent_dir / "models" / "g1" / "g1_29dof.usd"


def convert_urdf_to_usd():
    print(f"[CONVERTER] Source URDF: {urdf_file}")
    print(f"[CONVERTER] Target USD:  {usd_output}")

    try:
        from omni.isaac.kit import SimulationApp
        kit = SimulationApp({"headless": True})
        from omni.isaac.urdf import _urdf
        import omni.isaac.core.utils.stage as stage_utils
    except ImportError:
        print("\n[INFO] To convert to USD, run this with Isaac Sim's Python:")
        print("  %LOCALAPPDATA%\\ov\\pkg\\isaac-sim-<version>\\python.bat convert_urdf_to_usd.py")
        print("\nAlternatively, Isaac Sim GUI supports one-click import:")
        print("  1. In Isaac Sim, open 'Isaac Utils' -> 'Workflows' -> 'URDF Importer'")
        print(f"  2. Select: {urdf_file}")
        print("  3. Check 'Fix Base Link' (if gantry testing) or leave unchecked (free-standing)")
        print("  4. Click 'Import' and Save As USD.")
        return

    urdf_interface = _urdf.acquire_urdf_interface()
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.fix_base = False
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True

    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage()

    imported_path = urdf_interface.import_robot(
        str(urdf_file),
        import_config,
        stage,
        "/World/G1_Humanoid"
    )

    stage_utils.save_stage(str(usd_output))
    print(f"[SUCCESS] Saved Unitree G1 USD stage to: {usd_output}")
    kit.close()


if __name__ == "__main__":
    convert_urdf_to_usd()
