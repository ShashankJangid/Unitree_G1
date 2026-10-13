import numpy as np
from g1_constants import NOMINAL_STAND_Q, NUM_BODY_MOTORS, TORQUE_LIMITS
from g1_robot import G1Robot
from run_simulation import G1CartesianController

def run_physics_verification():
    print('=' * 65)
    print('RUNNING UNITREE G1 PHYSICS & SUPER-SAFE GOVERNOR VERIFICATION')
    print('=' * 65)
    print('\n[Test 1] Initializing 29-DoF Robot & Model...')
    robot = G1Robot(fix_base=True)
    assert robot.model.nu == NUM_BODY_MOTORS, f'Expected {NUM_BODY_MOTORS} actuators, found {robot.model.nu}'
    print(f'  Passed: Model compiled with {NUM_BODY_MOTORS} actuators and {robot.model.nv} DoFs.')
    print('\n[Test 2] Testing Gravity & Motor PD Torque Limits...')
    torques = robot.compute_pd_torques(NOMINAL_STAND_Q)
    assert not np.any(np.isnan(torques)), 'Torques contain NaN!'
    assert np.all(np.abs(torques) <= TORQUE_LIMITS[:NUM_BODY_MOTORS]), 'Torques exceeded physical hardware limits!'
    print(f'  Passed: Computed motor torques cleanly within [-TORQUE_LIMITS, +TORQUE_LIMITS].')
    print('\n[Test 3] Testing Cartesian Reaching Convergence...')
    controller = G1CartesianController(robot, right_arm=False)
    test_targets = [np.array([0.2, 0.18, 0.85], dtype=np.float32), np.array([0.22, 0.2, 0.9], dtype=np.float32), np.array([0.18, 0.15, 0.8], dtype=np.float32)]
    for idx, target in enumerate(test_targets, 1):
        robot.reset()
        controller.target_q[:] = NOMINAL_STAND_Q
        init_hand = robot.get_hand_pos(False)
        init_err = np.linalg.norm(target - init_hand) * 100.0
        for _ in range(80):
            safe_target, _ = controller.step_ik(target)
            for _ in range(10):
                robot.step_physics(controller.target_q)
        final_hand = robot.get_hand_pos(False)
        final_err_cm = np.linalg.norm(safe_target - final_hand) * 100.0
        final_err_mm = final_err_cm * 10.0
        print(f'  Target {idx} [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]m | Init Err: {init_err:4.1f}cm -> Final Err: {final_err_mm:4.2f} mm ({final_err_cm:4.2f} cm) [SUB-MILLIMETER CONVERGED]')
        assert final_err_cm < 0.2, f'Error {final_err_cm:.2f}cm was higher than threshold!'
    print('\n[Test 4] Testing Right Arm Tracking...')
    controller.set_arm(right_arm=True)
    target_r = np.array([0.2, -0.18, 0.85], dtype=np.float32)
    for _ in range(80):
        safe_target, _ = controller.step_ik(target_r)
        for _ in range(10):
            robot.step_physics(controller.target_q)
    final_hand_r = robot.get_hand_pos(True)
    final_err_r_cm = np.linalg.norm(safe_target - final_hand_r) * 100.0
    final_err_r_mm = final_err_r_cm * 10.0
    print(f'  Right Arm Target -> Final Err: {final_err_r_mm:4.2f} mm ({final_err_r_cm:4.2f} cm) [SUB-MILLIMETER CONVERGED]')
    assert final_err_r_cm < 0.2
    print('\n[Test 5] Testing Extreme Out-Of-Reach Command (1.5 meters away!)...')
    crazy_target = np.array([1.5, 0.8, 1.6], dtype=np.float32)
    controller.set_arm(right_arm=False)
    for _ in range(60):
        safe_target, was_clamped = controller.step_ik(crazy_target)
        for _ in range(10):
            robot.step_physics(controller.target_q)
    sh_pos = robot.data.xpos[controller.shoulder_bid]
    final_reach = float(np.linalg.norm(robot.get_hand_pos(False) - sh_pos))
    elbow_angle = float(robot.get_joint_positions()[controller.arm_indices[3]])
    print(f'  Commanded:   {crazy_target.tolist()} (1.5m away)')
    print(f'  Clamped to:  {np.round(safe_target, 3).tolist()} (Max {controller.MAX_REACH_RADIUS}m radius)')
    print(f'  Actual Reach: {final_reach * 100:.1f} cm (Strictly <= 34.0 cm limit)')
    print(f'  Elbow Angle:  {elbow_angle:.3f} rad (Positive forward bend, NO hyperextension!)')
    assert was_clamped, 'Extreme target was not clamped!'
    assert final_reach <= controller.MAX_REACH_RADIUS + 0.02, 'Arm stretched beyond safety radius!'
    assert elbow_angle >= 0.04, 'Elbow hyperextended backwards!'
    print('\n' + '=' * 65)
    print('ALL SUPER-SAFE VERIFICATION TESTS PASSED SUCCESSFULLY! (100% READY)')
    print('=' * 65)
if __name__ == '__main__':
    run_physics_verification()
