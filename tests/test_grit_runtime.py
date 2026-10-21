from pathlib import Path
import sys

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_expressive import G1ExpressiveController
from grit_controller import GRIT_JOINT_NAMES, GritDancePolicy


def test_grit_contract_produces_finite_targets():
    policy = GritDancePolicy(
        ROOT / "models/grit/policy.onnx",
        ROOT / "models/grit/kpop_dance.npz",
        GRIT_JOINT_NAMES,
    )
    qvel = np.zeros(35, dtype=np.float32)
    target = policy.infer(policy.default_qpos, qvel, np.array([1, 0, 0, 0], dtype=np.float32))

    assert target.shape == (29,)
    assert np.isfinite(target).all()


def test_grit_runs_without_support_gantry():
    controller = G1ExpressiveController(use_grit=True)
    pelvis_id = mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    assert controller.gantry_eqid >= 0
    assert controller.data.eq_active[controller.gantry_eqid] == 0

    for _ in range(500):
        controller.step_physics()

    pelvis_up = controller.data.xmat[pelvis_id].reshape(3, 3)[2, 2]
    assert controller.data.qpos[2] > 0.45
    assert pelvis_up > 0.65
    assert np.all(controller.target_q >= controller.model.jnt_range[controller.jids, 0])
    assert np.all(controller.target_q <= controller.model.jnt_range[controller.jids, 1])


def test_grit_keeps_gantry_disabled_when_toggled():
    controller = G1ExpressiveController(use_grit=True)
    controller.toggle_gantry()

    assert controller.data.eq_active[controller.gantry_eqid] == 0


def test_grit_reference_loops_after_last_frame():
    policy = GritDancePolicy(
        ROOT / "models/grit/policy.onnx",
        ROOT / "models/grit/kpop_dance.npz",
        GRIT_JOINT_NAMES,
    )
    policy.reference_index = len(policy.features) - 1
    policy.infer(policy.default_qpos, np.zeros(35, dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32))

    assert policy.reference_index == 0
