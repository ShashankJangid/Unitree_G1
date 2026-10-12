from __future__ import annotations
import argparse
import curses
import math
import pathlib
import threading
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple
import numpy as np

def load_policy(path: pathlib.Path):
    from stable_baselines3 import PPO
    print(f'[policy] Loading PPO model from {path} …')
    policy = PPO.load(str(path), device='cpu')
    policy.set_parameters(policy.get_parameters())
    policy.policy.set_training_mode(False)
    return policy

def make_env(render: bool, right_arm: bool):
    import g1_arm_rl_env as _env
    mode = 'human' if render else 'none'
    return _env.G1ArmReachEnv(render_mode=mode, right_arm=right_arm)

class RobotBridge:

    def __init__(self, iface: str, domain: int):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        except Exception:
            print('[robot] SDK-2 not present – robot output disabled')
            self.ok = False
            return
        try:
            ChannelFactoryInitialize(domain, iface)
            self._pub = ChannelPublisher('rt/arm_sdk', LowCmd_)
            self._pub.Init()
            self._cmd = unitree_hg_msg_dds__LowCmd_()
            for mc in self._cmd.motor_cmd:
                mc.mode = 0
                mc.kp = 40.0
                mc.kd = 1.0
            if 29 < len(self._cmd.motor_cmd):
                self._cmd.motor_cmd[29].q = 1.0
            try:
                from unitree_sdk2py.utils.crc import CRC
                self._crc = CRC()
            except Exception:
                self._crc = None
            self.ok = True
        except Exception as e:
            print(f'[robot] DDS initialisation failed – robot disabled ({e})')
            self.ok = False

    def send_qpos(self, q: Dict[int, float]) -> None:
        if not self.ok:
            return
        for idx, val in q.items():
            if idx >= 29:
                continue
            if idx < len(self._cmd.motor_cmd):
                self._cmd.motor_cmd[idx].q = float(val)
        if self._crc is not None:
            if hasattr(self._crc, 'Crc'):
                self._cmd.crc = self._crc.Crc(self._cmd)
            elif hasattr(self._crc, 'calculate_crc'):
                self._cmd.crc = self._crc.calculate_crc(self._cmd)
        self._pub.Write(self._cmd)
JOINTS: List[Tuple[int, str, str]] = [(15, 'L shoulder-pitch', 'left_shoulder_pitch'), (16, 'L shoulder-roll', 'left_shoulder_roll'), (17, 'L shoulder-yaw', 'left_shoulder_yaw'), (18, 'L elbow', 'left_elbow'), (19, 'L wrist-roll', 'left_wrist_roll'), (20, 'L wrist-pitch', 'left_wrist_pitch'), (21, 'L wrist-yaw', 'left_wrist_yaw'), (22, 'R shoulder-pitch', 'right_shoulder_pitch'), (23, 'R shoulder-roll', 'right_shoulder_roll'), (24, 'R shoulder-yaw', 'right_shoulder_yaw'), (25, 'R elbow', 'right_elbow'), (26, 'R wrist-roll', 'right_wrist_roll'), (27, 'R wrist-pitch', 'right_wrist_pitch'), (28, 'R wrist-yaw', 'right_wrist_yaw')]
IDX2LABEL = {idx: lbl for idx, lbl, _ in JOINTS}

def viewer_sync(env, stop_evt: threading.Event, speed, lock: threading.Lock):
    while not stop_evt.is_set():
        with lock:
            env.render()
        time.sleep(speed.dt)

def run_curses(stdscr, env, policy, robot: RobotBridge, out_sim: bool, out_robot: bool, speed, lock: threading.Lock):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)
    STEP = 0.02
    with lock:
        obs, _ = env.reset()
    try:
        from g1_initial_pose import POSE_DICT
        import mujoco as _mj

        def _joint_name(idx: int) -> str | None:
            left = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
            right = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
            if idx == 12:
                return 'waist_yaw_joint'
            if 15 <= idx <= 21:
                return left[idx - 15]
            if 22 <= idx <= 28:
                return right[idx - 22]
            return None

        def apply_pose(goal_too: bool=True) -> None:
            for midx, q in POSE_DICT.items():
                jname = _joint_name(midx)
                if jname is None:
                    continue
                jid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_JOINT, jname)
                if jid == -1:
                    continue
                qadr = int(env.model.jnt_qposadr[jid])
                env.data.qpos[qadr] = float(q)
            _mj.mj_forward(env.model, env.data)
            if goal_too:
                p_h = env._fk()
                env.p_goal[:] = p_h
                if env._goal_mid != -1:
                    env.data.mocap_pos[env._goal_mid] = env.p_goal
            try:
                if hasattr(env, '_park_qadr') and hasattr(env, '_park_rest_q'):
                    new_rest = []
                    for qadr in env._park_qadr:
                        new_rest.append(float(env.data.qpos[qadr]))
                    env._park_rest_q = np.array(new_rest, dtype=np.float32)
            except Exception:
                pass
        apply_pose(goal_too=True)
    except Exception:
        pass
    p_goal = env.p_goal.copy()
    last_robot_send = 0.0
    last_safe_qpos = None
    hold_mode = False
    collision_freeze = False
    while True:
        ch = stdscr.getch()
        if ch != -1:
            key = chr(ch & 255)
            if key in ('x', '\x1b'):
                break
            elif key == 'w':
                p_goal[2] += STEP
            elif key == 's':
                p_goal[2] -= STEP
            elif key == 'a':
                p_goal[1] += STEP
            elif key == 'd':
                p_goal[1] -= STEP
            elif key == 'q':
                p_goal[0] += STEP
            elif key == 'e':
                p_goal[0] -= STEP
            elif key == 'r':
                out_robot = not out_robot and robot.ok
            elif key == 't':
                out_sim = not out_sim
            elif key == ']':
                speed.dt = max(0.005, speed.dt * 0.5)
            elif key == '[':
                speed.dt = min(0.5, speed.dt * 2.0)
            elif key in ('>', '.'):
                speed.mult = min(4.0, speed.mult * 2.0)
            elif key in ('<', ','):
                speed.mult = max(0.25, speed.mult * 0.5)
            elif key == ']':
                speed.dt = max(0.005, speed.dt * 0.5)
            elif key == '[':
                speed.dt = min(0.5, speed.dt * 2.0)
        p_goal = np.clip(p_goal, [-0.1, -0.6, 0.4], [0.6, 0.6, 1.4])
        env.p_goal[:] = p_goal
        if env._goal_mid != -1:
            env.data.mocap_pos[env._goal_mid] = env.p_goal
        if out_sim or out_robot:
            collided = False
            if hasattr(env, '_arm_gids') and hasattr(env, '_protect_gids'):
                arm_gids = env._arm_gids
                prot_gids = env._protect_gids
                mj = env._mujoco if hasattr(env, '_mujoco') else None
                for i in range(env.data.ncon):
                    c = env.data.contact[i]
                    if mj is not None:
                        b1 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_BODY, int(env.model.geom_bodyid[c.geom1]))
                        b2 = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_BODY, int(env.model.geom_bodyid[c.geom2]))
                        if b1 and 'hand' in b1 or (b2 and 'hand' in b2):
                            continue
                    if c.geom1 in arm_gids and c.geom2 in prot_gids or (c.geom2 in arm_gids and c.geom1 in prot_gids):
                        penetration = max(0.0, -c.dist)
                        if penetration < 0.002:
                            continue
                        collided = True
                        break
            if collided:
                collision_freeze = True
            with lock:
                dist = np.linalg.norm(env.p_goal - env._fk())
            if collision_freeze:
                hold_mode = True
            elif not hold_mode and dist < 0.03:
                hold_mode = True
            elif hold_mode and dist > 0.05:
                hold_mode = False
                collision_freeze = False
            if hold_mode:
                with lock:
                    obs, _, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                env._step_count = 0
            else:
                action, _ = policy.predict(obs, deterministic=True)
                action = np.clip(action * speed.mult, env.action_space.low, env.action_space.high)
                with lock:
                    obs, _, done, _, info = env.step(action)
                if not collided:
                    last_safe_qpos = env.data.qpos.copy()
                if done:
                    with lock:
                        obs, _ = env.reset()
                        if last_safe_qpos is not None:
                            env.data.qpos[:] = last_safe_qpos
                            env.data.qvel[:] = 0.0
                            env._mujoco.mj_forward(env.model, env.data)
                    env.p_goal[:] = p_goal
                    if env._goal_mid != -1:
                        env.data.mocap_pos[env._goal_mid] = env.p_goal
        if out_robot and robot.ok and (time.time() - last_robot_send > 0.02):
            if not hasattr(env, '_motor_qadr'):
                import mujoco as _mj
                qadr = {}
                for idx, _lbl, mj_short in JOINTS:
                    jname_joint = mj_short + '_joint'
                    jid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_JOINT, jname_joint)
                    if jid != -1:
                        qadr[idx] = int(env.model.jnt_qposadr[jid])
                env._motor_qadr = qadr
            qpos = {idx: float(env.data.qpos[adr]) for idx, adr in env._motor_qadr.items()}
            robot.send_qpos(qpos)
            last_robot_send = time.time()
        stdscr.erase()
        stdscr.addstr(0, 0, 'G-1 RL Reach Controller – w/a/s/d/q/e goal, [] speed, t(sim) r(robot), Esc quit')
        stdscr.addstr(1, 0, f"Output → sim:[{('X' if out_sim else ' ')}]  robot:[{('X' if out_robot else '-' if robot.ok else '-')}]   rate:{speed.dt * 1000:.0f} ms  mult:{speed.mult:.2f}")
        status = 'HOLD' if hold_mode else 'RUN '
        if collision_freeze:
            status = 'COLL'
        stdscr.addstr(3, 0, f'Goal: x={p_goal[0]:.3f}  y={p_goal[1]:.3f}  z={p_goal[2]:.3f}  (m)   mode:{status}')
        stdscr.refresh()
        time.sleep(speed.dt)

def main() -> None:
    ap = argparse.ArgumentParser(description='Interactive RL reach controller for Unitree G-1 arm')
    ap.add_argument('--model', default='models/ppo_g1_left_53178k.zip', help='Path to trained .zip model')
    ap.add_argument('--right-arm', action='store_true', help='Use policy for the RIGHT arm instead of left')
    ap.add_argument('--iface', default='enp68s0f1', help='DDS network interface (robot)')
    ap.add_argument('--domain', type=int, default=0, help='DDS domain ID')
    ap.add_argument('--sim-only', action='store_true', help='Disable robot output entirely')
    ap.add_argument('--rate', type=float, default=0.04, help='Render / control loop interval (s)')
    args = ap.parse_args()
    model_path = pathlib.Path(args.model).expanduser()
    if not model_path.exists():
        raise SystemExit(f'Model file not found: {model_path}')
    policy = load_policy(model_path)
    env = make_env(render=True, right_arm=args.right_arm)
    try:
        from g1_initial_pose import POSE_DICT
        import mujoco as _mj

        def _joint_name(idx: int) -> str | None:
            left = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']
            right = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
            if idx == 12:
                return 'waist_yaw_joint'
            if 15 <= idx <= 21:
                return left[idx - 15]
            if 22 <= idx <= 28:
                return right[idx - 22]
            return None
        for midx, q in POSE_DICT.items():
            jname = _joint_name(midx)
            if jname is None:
                continue
            jid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                continue
            qadr = int(env.model.jnt_qposadr[jid])
            env.data.qpos[qadr] = float(q)
        _mj.mj_forward(env.model, env.data)
        if hasattr(env, '_fk'):
            p_hand = env._fk()
            env.p_goal[:] = p_hand
            if env._goal_mid != -1:
                env.data.mocap_pos[env._goal_mid] = env.p_goal
    except Exception:
        pass
    robot = RobotBridge(args.iface, args.domain) if not args.sim_only else RobotBridge('', 0)
    out_sim = True
    out_robot = False
    speed = SimpleNamespace(dt=max(0.005, args.rate), mult=0.25)
    lock = threading.Lock()
    stop_evt = threading.Event()
    threading.Thread(target=viewer_sync, args=(env, stop_evt, speed, lock), daemon=True).start()
    try:
        curses.wrapper(run_curses, env, policy, robot, out_sim, out_robot, speed, lock)
    finally:
        stop_evt.set()
if __name__ == '__main__':
    main()
