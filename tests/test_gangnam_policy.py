"""Contract checks against the published Gangnam inference assets."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gangnam_controller import DEFAULT, JOINT_MAP, SCALE, GangnamPolicy


@pytest.fixture(scope="module")
def policy():
    return GangnamPolicy()


def test_contract_and_sampling(policy):
    assert (policy.dt, policy.start_time, policy.end_time, policy.duration) == (.02, 5.2, 25.5, 20.3)
    assert policy.kp.shape == policy.kd.shape == (29,)
    np.testing.assert_allclose(policy.kp[:6], [40.2, 99.1, 40.2, 99.1, 28.5, 28.5])
    assert sorted(JOINT_MAP.tolist()) == list(range(29))
    pos, quat, q, dq = policy.sample(5.2)
    np.testing.assert_allclose(pos, policy.root_pos[312])
    np.testing.assert_allclose(q, policy.joint_pos[312])
    np.testing.assert_allclose(dq, (policy.joint_pos[313] - q) * 60)
    np.testing.assert_allclose(quat, np.roll(policy.root_quat[312], 1))
    midpoint = policy.sample((312.5) / 60)
    np.testing.assert_allclose(midpoint[2], (policy.joint_pos[312] + policy.joint_pos[313]) / 2)
    np.testing.assert_allclose(policy.joint_vel[-1], policy.joint_vel[-2])


def test_observation_mapping_alignment_and_action(policy):
    class CaptureSession:
        def __init__(self):
            self.obs = None

        def run(self, outputs, feeds):
            self.obs = feeds[policy.input_name]
            return [np.arange(29, dtype=np.float32)[None]]

    session, policy.session = policy.session, CaptureSession()
    try:
        ref_q = policy.sample(policy.start_time)[2]
        ref_root = Rotation.from_quat(policy.root_quat[312])
        rz = Rotation.from_rotvec([0, 0, ref_q[12]])
        rx = Rotation.from_rotvec([ref_q[13], 0, 0])
        ry = Rotation.from_rotvec([0, ref_q[14], 0])
        ref_torso = (ref_root * rz * rx * ry).as_matrix()
        yaw = Rotation.from_euler("z", .5)
        policy.reset(np.roll((yaw * ref_root).as_quat(), 1), np.zeros(29))
        q, dq = np.arange(29, dtype=float), np.arange(29, dtype=float) / 10
        target = policy.infer(0, q, dq, [1, 2, 3], yaw.as_matrix() @ ref_torso)
        obs = policy.session.obs
        assert obs.shape == (1, 154) and obs.dtype == np.float32
        np.testing.assert_allclose(obs[0, :29], ref_q[JOINT_MAP])
        np.testing.assert_allclose(obs[0, 29:58], policy.sample(policy.start_time)[3][JOINT_MAP])
        np.testing.assert_allclose(obs[0, 58:64], np.eye(3)[:, :2].reshape(6), atol=1e-5)
        np.testing.assert_allclose(obs[0, 64:67], [1, 2, 3])
        np.testing.assert_allclose(obs[0, 67:96], q[JOINT_MAP] - DEFAULT)
        np.testing.assert_allclose(obs[0, 96:125], dq[JOINT_MAP])
        np.testing.assert_array_equal(obs[0, 125:], 0)
        np.testing.assert_allclose(target[JOINT_MAP], np.arange(29) * SCALE + DEFAULT, rtol=1e-5)
        policy.infer(0, q, dq, [0, 0, 0], yaw.as_matrix() @ ref_torso)
        np.testing.assert_array_equal(policy.session.obs[0, 125:], np.arange(29))
    finally:
        policy.session = session


def test_invalid_assets_and_outputs(policy, tmp_path):
    with pytest.raises(FileNotFoundError):
        GangnamPolicy(tmp_path)
    class BadSession:
        def run(self, outputs, feeds):
            return [np.full((1, 29), np.nan, dtype=np.float32)]
    session, policy.session = policy.session, BadSession()
    try:
        with pytest.raises(ValueError, match="action output"):
            policy.infer(0, np.zeros(29), np.zeros(29), np.zeros(3), np.eye(3))
        with pytest.raises(ValueError, match="joint_pos_xml"):
            policy.infer(0, np.zeros(28), np.zeros(29), np.zeros(3), np.eye(3))
    finally:
        policy.session = session


def test_default_offsets_are_already_policy_order(policy):
    # The YAML defaults alternate left/right hips in policy order, unlike
    # XML's depth-first leg chain. Catch accidental double-reordering.
    assert DEFAULT[0] == pytest.approx(-.302)
    assert DEFAULT[1] == pytest.approx(-.319)
    assert DEFAULT[9] == pytest.approx(.672)
    xml_default = np.empty(29)
    xml_default[JOINT_MAP] = DEFAULT
    assert xml_default[6] == pytest.approx(-.319)
    assert xml_default[3] == pytest.approx(.672)


def test_reset_alignment_uses_robot_torso_yaw(policy):
    q = np.zeros(29)
    q[12:15] = [.4, .2, -.3]
    root = Rotation.from_euler("xyz", [.15, -.1, .6])
    torso = (root * Rotation.from_rotvec([0, 0, .4]) *
             Rotation.from_rotvec([.2, 0, 0]) * Rotation.from_rotvec([0, -.3, 0]))
    policy.reset(np.roll(root.as_quat(), 1), q)
    ref_root = Rotation.from_quat(policy.root_quat[312])
    yaw_robot = np.arctan2(torso.as_matrix()[1, 0], torso.as_matrix()[0, 0])
    yaw_ref = np.arctan2(ref_root.as_matrix()[1, 0], ref_root.as_matrix()[0, 0])
    np.testing.assert_allclose(policy.alignment.as_matrix(),
                               Rotation.from_euler("z", yaw_robot-yaw_ref).as_matrix(), atol=1e-7)
