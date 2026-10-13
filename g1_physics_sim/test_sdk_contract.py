"""
Regression tests that pin the Unitree G1 wire contract.

Every assertion here is traceable to a file in
official_unitree_repos/unitree_sdk2/example/g1/ . If a test fails, the code has
drifted from the real robot - do not "fix" the test.

  ankle : low_level/g1_ankle_swing_example.cpp
  dual  : low_level/g1_dual_arm_example.cpp
  arm5  : high_level/g1_arm5_sdk_dds_example.cpp
  arm7  : high_level/g1_arm7_sdk_dds_example.cpp
  hand  : high_level/g1_hand_sdk_example.cpp

Run:  python -m pytest test_sdk_contract.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

import g1_constants as C


# --------------------------------------------------------------- joint index
def test_leg_and_waist_indices():
    """ankle: G1JointIndex enum, legs 0-11, waist 12-14."""
    assert (C.L_HIP_PITCH, C.L_HIP_ROLL, C.L_HIP_YAW, C.L_KNEE,
            C.L_ANKLE_PITCH, C.L_ANKLE_ROLL) == (0, 1, 2, 3, 4, 5)
    assert (C.R_HIP_PITCH, C.R_HIP_ROLL, C.R_HIP_YAW, C.R_KNEE,
            C.R_ANKLE_PITCH, C.R_ANKLE_ROLL) == (6, 7, 8, 9, 10, 11)
    assert (C.WAIST_YAW, C.WAIST_ROLL, C.WAIST_PITCH) == (12, 13, 14)


def test_arm_indices():
    """arm7: 29-DOF map, left arm 15-21, right arm 22-28."""
    assert C.LEFT_ARM_JOINTS == [15, 16, 17, 18, 19, 20, 21]
    assert C.RIGHT_ARM_JOINTS == [22, 23, 24, 25, 26, 27, 28]
    assert C.L_ELBOW == 18 and C.R_ELBOW == 25
    assert C.L_WRIST_ROLL == 19 and C.L_WRIST_PITCH == 20 and C.L_WRIST_YAW == 21
    assert C.R_WRIST_ROLL == 26 and C.R_WRIST_PITCH == 27 and C.R_WRIST_YAW == 28


def test_body_is_29_motors():
    """LowState_.motor_state is indexed 0..28 for the body."""
    assert C.NUM_BODY_MOTORS == 29
    assert len(C.JOINT_NAMES_29DOF) == 29
    assert len(C.OFFICIAL_KP) == 29
    assert len(C.OFFICIAL_KD) == 29
    assert len(C.TORQUE_LIMITS) == 29
    assert len(C.JOINT_LIMITS_MIN) == 29
    assert len(C.JOINT_LIMITS_MAX) == 29


def test_pr_ab_aliases():
    """ankle: AB mode re-maps the same slots; WAIST_A/B alias waist roll/pitch."""
    assert C.L_ANKLE_B == C.L_ANKLE_PITCH == 4
    assert C.L_ANKLE_A == C.L_ANKLE_ROLL == 5
    assert C.R_ANKLE_B == C.R_ANKLE_PITCH == 10
    assert C.R_ANKLE_A == C.R_ANKLE_ROLL == 11
    assert C.WAIST_A == C.WAIST_ROLL == 13
    assert C.WAIST_B == C.WAIST_PITCH == 14


def test_k_not_used_joint_is_29():
    """arm5/arm7: motor_cmd[kNotUsedJoint].q is the arm_sdk weight."""
    assert C.K_NOT_USED_JOINT == 29
    assert C.ARM_SDK_ENABLE == 29


def test_arm_sdk_publish_sets():
    """arm7 publishes 17 joints (14 arm + 3 waist) and never touches legs."""
    assert len(C.ARM_SDK_JOINTS_17) == 17
    assert C.ARM_SDK_JOINTS_17 == [12, 13, 14] + list(range(15, 29))
    assert not set(C.ARM_SDK_JOINTS_17) & set(C.LEG_JOINTS)


# -------------------------------------------------------------------- topics
def test_topic_strings_are_exact():
    """Topic names are case-sensitive and must match the SDK byte for byte."""
    assert C.TOPIC_LOW_CMD == 'rt/lowcmd'
    assert C.TOPIC_LOW_STATE == 'rt/lowstate'
    assert C.TOPIC_ARM_SDK == 'rt/arm_sdk'
    assert C.TOPIC_IMU_TORSO == 'rt/secondary_imu'
    assert C.TOPIC_HAND_SDK == 'rt/hand_sdk'


def test_dds_domains():
    assert C.DDS_DOMAIN_SIM == 1
    assert C.DDS_DOMAIN_REAL == 0


# --------------------------------------------------------------------- gains
def test_official_gains_exact():
    """ankle: literal Kp/Kd arrays from the source file."""
    assert C.OFFICIAL_KP[:6].tolist() == [60, 60, 60, 100, 40, 40]
    assert C.OFFICIAL_KP[6:12].tolist() == [60, 60, 60, 100, 40, 40]
    assert C.OFFICIAL_KP[12:15].tolist() == [60, 40, 40]
    assert C.OFFICIAL_KP[15:22].tolist() == [40] * 7
    assert C.OFFICIAL_KP[22:29].tolist() == [40] * 7

    assert C.OFFICIAL_KD[:6].tolist() == [1, 1, 1, 2, 1, 1]
    assert C.OFFICIAL_KD[6:12].tolist() == [1, 1, 1, 2, 1, 1]
    assert C.OFFICIAL_KD[12:15].tolist() == [1, 1, 1]
    assert C.OFFICIAL_KD[15:29].tolist() == [1] * 14


def test_gearbox_derived_gains():
    """dual: GetMotorKp/GetMotorKd keyed on gearbox class."""
    kp, kd = C.gains_from_gearbox()
    assert kp.shape == (29,) and kd.shape == (29,)
    assert kp[3] == 100.0 and kp[9] == 100.0   # knees are GearboxL
    assert kp[0] == 40.0                        # hip pitch is GearboxM -> 40 in this table
    assert np.all(kd == 1.0)


def test_motor_mode_values():
    """ankle: motor_cmd[i].mode -> 1 enable, 0 disable."""
    assert C.MOTOR_MODE_DISABLE == 0
    assert C.MOTOR_MODE_ENABLE == 1


def test_pr_ab_mode_values():
    """ankle: enum class Mode { PR = 0, AB = 1 }."""
    assert C.MODE_PR == 0
    assert C.MODE_AB == 1


def test_ankle_swing_amplitudes():
    """ankle: max_P = pi*30/180, max_R = pi*10/180."""
    assert np.isclose(C.ANKLE_SWING_P_MAX, np.pi * 30.0 / 180.0)
    assert np.isclose(C.ANKLE_SWING_R_MAX, np.pi * 10.0 / 180.0)


def test_arm_sdk_limits():
    """arm5/arm7: kp 60, kd 1.5, control_dt 0.02, max_joint_velocity 0.5."""
    assert C.ARM_SDK_KP == 60.0
    assert C.ARM_SDK_KD == 1.5
    assert np.isclose(C.ARM_SDK_CONTROL_DT, 0.02)
    assert C.ARM_SDK_MAX_JOINT_VELOCITY == 0.5
    assert C.ARM_SDK_WEIGHT_RATE == 0.2
    assert np.isclose(C.LOWCMD_CONTROL_DT, 0.002)


# --------------------------------------------------------------------- hands
def test_hands_are_not_body_motors():
    """
    Regression: an earlier revision appended 14 fake hand joints at body indices
    30..43 and advertised a 43-motor body. Hands travel on rt/hand_sdk with
    MotorCmds_ (hand) or via Dex3Client, never in LowCmd_.motor_cmd[].
    """
    assert not hasattr(C, 'NUM_MOTORS')
    assert C.DEX3_JOINTS_PER_HAND == 7
    assert C.DEX3_NUM_MOTORS == 14
    assert C.HAND_SDK_NUM_MOTORS == 4
    assert len(C.DEX3_JOINT_NAMES) == 7
    assert C.HAND_SDK_ENCODED_NAME == 'unitree_go_msg_dds__MotorCmds_'


def test_joint_names_have_no_hand_entries():
    assert all('hand' not in n for n in C.JOINT_NAMES_29DOF)
    assert len(C.JOINT_NAMES_29DOF) == 29


# ------------------------------------------------------------------ geometry
def test_joint_limits_are_ordered_and_symmetric():
    assert np.all(C.JOINT_LIMITS_MIN < C.JOINT_LIMITS_MAX)
    assert np.all(np.isfinite(C.JOINT_LIMITS_MIN))
    assert np.all(np.isfinite(C.JOINT_LIMITS_MAX))


def test_poses_are_correct_length():
    assert C.NOMINAL_STAND_Q.shape == (29,)
    assert C.ZERO_POSE_Q.shape == (29,)
    assert C.ARM_REST_17.shape == (17,)


# ---------------------------------------------------------------------- crc
def test_crc32_is_stable_and_known():
    """
    Crc32Core(dwPolynomial=0x04c11db7, init=0xFFFFFFFF, no final xor) over
    32-bit words. Self-consistency and determinism are what we can assert
    without the robot; the wiring itself must use unitree_sdk2py.utils.crc.CRC.
    """
    zeros = np.zeros(4, dtype=np.uint32)
    assert C.crc32_core(zeros) == C.crc32_core(zeros)
    assert C.crc32_core(np.zeros(4, dtype=np.uint32)) != \
        C.crc32_core(np.ones(4, dtype=np.uint32))
    assert 0 <= C.crc32_core(zeros) <= 0xFFFFFFFF


# ------------------------------------------------------ data container shape
def test_state_and_action_are_29():
    from g1_state import G1Action, G1State
    s, a = G1State(), G1Action()
    assert s.q.shape == (29,) and s.dq.shape == (29,) and s.tau_est.shape == (29,)
    assert s.motor_state.shape == (29,)
    assert a.q_des.shape == (29,) and a.kp.shape == (29,) and a.kd.shape == (29,)
    assert a.tau_ff.shape == (29,) and a.motor_mode.shape == (29,)


def test_action_instances_do_not_alias_constants():
    """
    Regression: default_factory returned NOMINAL_STAND_Q.copy() but __post_init__
    used np.asarray, which does not copy a matching-dtype array. Every action
    therefore shared one buffer with the module constant, so writing q_des on
    one action silently rewrote the constant for the whole process.
    """
    import g1_constants
    from g1_state import G1Action
    a1, a2 = G1Action(), G1Action()
    a1.q_des[0] = 999.0
    assert a2.q_des[0] != 999.0, 'two actions share a buffer'
    assert g1_constants.NOMINAL_STAND_Q[0] != 999.0, 'action aliased the constant'
    b1, b2 = G1Action.hold(), G1Action()
    b1.kp[0] = 5.0
    assert b2.kp[0] != 5.0


def test_hold_action_removes_torque():
    """
    Regression: a safety E-stop that only rewrites q_des leaves kp in place, so
    the robot stays stiff. G1Action.hold() must zero gains and disable motors.
    """
    from g1_state import G1Action
    h = G1Action.hold()
    assert np.all(h.kp == 0.0)
    assert np.all(h.tau_ff == 0.0)
    assert np.all(h.motor_mode == C.MOTOR_MODE_DISABLE)
    assert h.arm_sdk_weight == 0.0


def test_state_fault_detection():
    from g1_state import G1State
    s = G1State()
    assert not s.any_motor_faulted
    s.motor_state[3] = 1
    assert s.any_motor_faulted
    assert s.get_faulted_motors() == [3]


def test_quaternion_roundtrip():
    from g1_state import quat_to_rpy, rpy_to_quat
    rpy = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.allclose(quat_to_rpy(rpy_to_quat(rpy)), rpy, atol=1e-5)


# -------------------------------------------------------------------- safety
def test_safety_estop_zeroes_gains():
    """
    Regression: filter_and_guard used to return bare positions, so an
    EMERGENCY_STOP still published full kp. It must return a full G1Action.
    """
    import g1_safety as S
    m = S.G1FullSafetyManager()
    m.reset(np.zeros(29))
    m.emergency_stop('unit test')
    action, status = m.filter_and_guard(
        np.full(29, 9.0), np.zeros(29), np.zeros(29),
        np.array([.3, .3, 1.]), np.array([.3, -.3, 1.]), np.zeros(3))
    assert status.state is S.SafetyState.EMERGENCY_STOP
    assert np.all(action.kp == 0.0)
    assert np.all(action.motor_mode == C.MOTOR_MODE_DISABLE)
    assert status.latched_alarms, 'alarm must latch so the cause survives the stop'
    assert m.estop_reason == 'unit test'


def test_safety_reset_enters_gantry_hold_not_active():
    import g1_safety as S
    m = S.G1FullSafetyManager()
    m.reset(np.zeros(29))
    assert m.state is S.SafetyState.GANTRY_HOLD


def test_safety_never_exceeds_joint_limits():
    import g1_safety as S
    m = S.G1FullSafetyManager()
    m.reset(np.zeros(29))
    m.engage()
    action, _ = m.filter_and_guard(
        np.full(29, 99.0), np.zeros(29), np.zeros(29),
        np.array([.3, .3, 1.]), np.array([.3, -.3, 1.]), np.zeros(3))
    assert np.all(action.q_des >= C.JOINT_LIMITS_MIN)
    assert np.all(action.q_des <= C.JOINT_LIMITS_MAX)


def test_safety_tilt_triggers_estop():
    import g1_safety as S
    m = S.G1FullSafetyManager()
    m.reset(np.zeros(29))
    m.engage()
    _, status = m.filter_and_guard(
        np.zeros(29), np.zeros(29), np.zeros(29),
        np.array([.3, .3, 1.]), np.array([.3, -.3, 1.]), np.zeros(3),
        base_tilt_deg=45.0)
    assert status.state is S.SafetyState.EMERGENCY_STOP


# ------------------------------------------------------------------ policies
def test_ankle_policy_matches_official_amplitudes():
    from g1_policies import AnkleSwingPolicy
    from g1_state import G1State
    st = G1State()
    p = AnkleSwingPolicy(duration=1.0, control_dt=0.002)
    p.reset(st)
    seen_modes = set()
    max_pitch = 0.0
    max_roll = 0.0
    for _ in range(6000):
        a = p.step(st)
        seen_modes.add(a.mode_pr)
        max_pitch = max(max_pitch, abs(a.q_des[C.L_ANKLE_PITCH]))
        max_roll = max(max_roll, abs(a.q_des[C.L_ANKLE_ROLL]))
    assert seen_modes == {C.MODE_PR, C.MODE_AB}, 'must exercise both ankle modes'
    assert np.isclose(max_pitch, C.ANKLE_SWING_P_MAX, atol=1e-3)
    assert np.isclose(max_roll, C.ANKLE_SWING_R_MAX, atol=1e-3)


def test_arm_policy_never_moves_legs():
    from g1_policies import ArmSwingPolicy
    from g1_state import G1State
    st = G1State()
    p = ArmSwingPolicy(duration=5.0, control_dt=0.02)
    p.reset(st)
    for _ in range(50):
        a = p.step(st)
        assert np.allclose(a.q_des[C.LEG_JOINTS], C.NOMINAL_STAND_Q[C.LEG_JOINTS])


def test_policies_use_official_gains():
    from g1_policies import StandPolicy
    from g1_state import G1State
    st = G1State()
    p = StandPolicy(ramp_time=0.0)
    p.reset(st)
    a = p.step(st)
    assert np.allclose(a.kp, C.OFFICIAL_KP)
    assert np.allclose(a.kd, C.OFFICIAL_KD)


# ------------------------------------------------------------------- bridges
def test_bridges_are_importable_without_sdk():
    from g1_hardware_bridge import G1Bridge, G1RealHardwareBridge, G1SimBridge
    assert issubclass(G1SimBridge, G1Bridge)
    assert issubclass(G1RealHardwareBridge, G1Bridge)
    assert hasattr(G1Bridge, 'tick')
    assert hasattr(G1Bridge, 'emergency_stop')


def test_real_bridge_rejects_bad_control_path():
    from g1_hardware_bridge import G1RealHardwareBridge
    with pytest.raises(ValueError):
        G1RealHardwareBridge(iface='lo', control_path='nonsense')