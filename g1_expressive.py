from __future__ import annotations
import argparse
import json
import math
import os
import pathlib
import sys
import time
import numpy as np

if __name__ == "__main__" and "--headless" in sys.argv[1:]:
    os.environ["MUJOCO_GL"] = "disable"

if sys.platform.startswith("linux") and os.environ.get("DISPLAY") and "PYGLFW_LIBRARY_VARIANT" not in os.environ:
    os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"

import mujoco
from g1_teleop import CrossPlatformKeyboard
from grit_controller import GRIT_KD, GRIT_KP, GritDancePolicy

root_dir = pathlib.Path(__file__).resolve().parent
motion_dir = root_dir / "models" / "motions"
xml_scene = root_dir / "models" / "g1" / "scene_29dof.xml"
if not xml_scene.exists():
    xml_scene = root_dir / "models" / "g1" / "scene_29dof_with_hand.xml"


class G1ExpressiveController:
    """Controls the full 29-DOF Unitree G1 humanoid for expressive motions:

    Disco Fever, tabletop block reaching, greeting waves, respectful bows, and boxing punches.
    Supported by a virtual gantry harness so the humanoid stands rock-solid on its feet without falling.
    """

    JOINT_NAMES_29 = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]

    DEFAULT_STAND_29 = np.array([
        # 12 legs
        -0.15, 0.0, 0.0, 0.35, -0.2, 0.0,
        -0.15, 0.0, 0.0, 0.35, -0.2, 0.0,
        # 3 waist
        0.0, 0.0, 0.0,
        # 7 left arm
        0.2, 0.25, 0.0, 0.6, 0.0, 0.0, 0.0,
        # 7 right arm
        0.2, -0.25, 0.0, 0.6, 0.0, 0.0, 0.0,
    ], dtype=np.float32)

    def __init__(self, scene_path: pathlib.Path = xml_scene, use_grit: bool = False):
        self.scene_path = scene_path
        self.grit_active = bool(use_grit)
        self.grit_policy = None
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Scene not found: {self.scene_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002

        # Check for support gantry equality constraint
        self.gantry_eqid = -1
        for eq_id in range(self.model.neq):
            eq_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id)
            if eq_name and "support_gantry" in eq_name:
                self.gantry_eqid = eq_id
                break

        if self.gantry_eqid != -1:
            self.data.eq_active[self.gantry_eqid] = 0 if self.grit_active else 1
            if self.grit_active:
                print("[STABILITY] GRIT whole-body policy active; support gantry DISABLED.")
            else:
                print("[STABILITY] Support gantry harness ACTIVE (robot stands firmly without falling).")

        # Map joint names to qpos / qvel indices
        self.jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.JOINT_NAMES_29]
        self.qadrs = [self.model.jnt_qposadr[jid] for jid in self.jids]
        self.dofadrs = [self.model.jnt_dofadr[jid] for jid in self.jids]

        # Check for finger joints
        finger_names = [
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint",
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_index_0_joint", "left_hand_index_1_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        ]
        self.finger_qadrs = []
        for fn in finger_names:
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, fn)
            if fid != -1:
                self.finger_qadrs.append(self.model.jnt_qposadr[fid])

        # PD gains for 29 DOFs
        kp_legs = [140, 110, 110, 180, 60, 50] * 2
        kd_legs = [4, 3, 3, 5, 3, 3] * 2
        kp_waist = [160, 130, 130]
        kd_waist = [4, 3, 3]
        kp_arms = [80, 80, 60, 80, 30, 20, 20] * 2
        kd_arms = [2, 2, 2, 2, 1, 1, 1] * 2

        self.kps = np.array(kp_legs + kp_waist + kp_arms, dtype=np.float32)
        self.kds = np.array(kd_legs + kd_waist + kd_arms, dtype=np.float32)
        self.scripted_kps = self.kps.copy()
        self.scripted_kds = self.kds.copy()

        # Current targets
        self.target_q = self.DEFAULT_STAND_29.copy()
        self.current_motion = None
        self.motion_data = {}
        self.motion_frame_idx = 0
        self.motion_start_time = 0.0
        self.sim_time = 0.0
        self.loop_motion = True

        if self.grit_active:
            self.grit_policy = GritDancePolicy(
                root_dir / "models" / "grit" / "policy.onnx",
                root_dir / "models" / "grit" / "kpop_dance.npz",
                self.JOINT_NAMES_29,
            )
            self.kps = GRIT_KP.copy()
            self.kds = GRIT_KD.copy()

        self.grit_target_q = None
        self.grit_control_counter = 0

        # Grasp state (0.0 = open, 1.0 = closed)
        self.grasp_target = 0.0

        self.reset_posture()

    def reset_posture(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.793
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        posture = self.grit_policy.default_qpos if self.grit_active and self.grit_policy else self.DEFAULT_STAND_29
        for adr, val in zip(self.qadrs, posture):
            self.data.qpos[adr] = val
        self.data.qvel[:] = 0.0

        for adr in self.finger_qadrs:
            self.data.qpos[adr] = 0.0

        self.target_q[:] = posture
        self.grasp_target = 0.0
        if self.gantry_eqid != -1:
            self.data.eq_active[self.gantry_eqid] = 0 if self.grit_active else 1
        mujoco.mj_forward(self.model, self.data)
        self.grit_target_q = posture.copy() if self.grit_active else None
        self.grit_control_counter = 0
        if self.grit_active and self.grit_policy:
            qpos = np.array([self.data.qpos[adr] for adr in self.qadrs], dtype=np.float32)
            self.grit_policy.reset(qpos, self.data.qvel.copy(), self.data.qpos[3:7].copy())

    def toggle_gantry(self):
        if self.grit_active:
            print("\n[STABILITY] Gantry toggle ignored while GRIT is active; dance requires unsupported dynamics.")
            sys.stdout.flush()
            return
        if self.gantry_eqid != -1:
            cur = bool(self.data.eq_active[self.gantry_eqid])
            new_state = int(not cur)
            self.data.eq_active[self.gantry_eqid] = new_state
            state_str = "ACTIVE (Robot is firmly supported)" if new_state else "DISABLED (Warning: robot will fall without balance policy)"
            print(f"\n[STABILITY] >>> Support gantry is now: {state_str}")
        else:
            print("\n[STABILITY] >>> No support gantry constraint found in scene.")
        sys.stdout.flush()

    def set_grit_active(self, active: bool):
        if active and self.grit_policy is None:
            self.grit_policy = GritDancePolicy(
                root_dir / "models" / "grit" / "policy.onnx",
                root_dir / "models" / "grit" / "kpop_dance.npz",
                self.JOINT_NAMES_29,
            )
            self.kps = GRIT_KP.copy()
            self.kds = GRIT_KD.copy()
        self.grit_active = bool(active)
        if self.grit_active:
            self.kps = GRIT_KP.copy()
            self.kds = GRIT_KD.copy()
        else:
            self.kps = self.scripted_kps.copy()
            self.kds = self.scripted_kds.copy()
        if self.gantry_eqid != -1:
            self.data.eq_active[self.gantry_eqid] = 0 if self.grit_active else 1
        self.reset_posture()
        state = "ON (gantry disabled)" if self.grit_active else "OFF (scripted PD mode)"
        print(f"[CONTROL] GRIT whole-body policy: {state}")
        sys.stdout.flush()

    def has_fallen(self) -> bool:
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_id < 0:
            return not np.isfinite(self.data.qpos[2])
        up_z = float(self.data.xmat[pelvis_id].reshape(3, 3)[2, 2])
        return not np.isfinite(self.data.qpos[2]) or float(self.data.qpos[2]) < 0.45 or up_z < 0.65

    def load_motion(self, motion_name: str) -> bool:
        path = motion_dir / f"{motion_name}.json"
        if not path.exists():
            print(f"[ERROR] Motion file not found: {path}")
            return False

        with open(path, "r") as f:
            data = json.load(f)

        self.motion_data = data
        self.current_motion = motion_name
        self.motion_frame_idx = 0
        self.motion_start_time = self.sim_time
        source = " (GRIT reference is used for dance)" if self.grit_active else ""
        print(f"[MOTION] Loaded: {motion_name.upper()} ({len(data.get('frames', []))} frames, {data.get('duration_sec', 0)}s){source}")
        return True

    def update_motion(self, sim_time: float):
        if self.grit_active:
            return
        if not self.motion_data or "frames" not in self.motion_data:
            return

        frames = self.motion_data["frames"]
        n_frames = len(frames)
        if n_frames == 0:
            return

        fps = self.motion_data.get("fps", 50)
        duration = self.motion_data.get("duration_sec", n_frames / fps)

        t_elapsed = max(0.0, sim_time - self.motion_start_time)
        if self.loop_motion:
            t_in_motion = t_elapsed % duration
        else:
            t_in_motion = min(t_elapsed, duration - 1e-4)

        frame_pos = t_in_motion * fps
        frame_idx = int(frame_pos) % n_frames
        next_idx = (frame_idx + 1) % n_frames
        blend = float(frame_pos - int(frame_pos))
        q_frame = np.asarray(frames[frame_idx]["joint_q"], dtype=np.float32)
        q_next = np.asarray(frames[next_idx]["joint_q"], dtype=np.float32)

        if len(q_frame) >= 29:
            self.target_q[:] = q_frame[:29] * (1.0 - blend) + q_next[:29] * blend
        elif len(q_frame) == 14:
            self.target_q[15:29] = q_frame * (1.0 - blend) + q_next * blend

    def step_reach_and_pick(self, t: float):
        """Dynamic tabletop block reaching and picking motion with whole-body torso lean and arm manipulation."""
        cycle = t % 8.0
        q = self.DEFAULT_STAND_29.copy()

        if cycle < 2.5:
            # Phase 1: Torso leans smoothly forward, knees flex, right arm extends forward down to block
            prog = cycle / 2.5
            smooth = 0.5 - 0.5 * math.cos(math.pi * prog)
            q[14] = 0.18 * smooth             # waist pitch forward
            q[3] += 0.08 * smooth             # left knee bend
            q[9] += 0.08 * smooth             # right knee bend
            q[0] -= 0.04 * smooth             # hips compensate
            q[6] -= 0.04 * smooth
            # Left arm back for balance
            q[15] = 0.2 + 0.20 * smooth
            q[18] = 0.6 + 0.40 * smooth
            # Right arm reaches down to tabletop
            q[22] = 0.2 - 0.75 * smooth       # right shoulder pitch (-0.55)
            q[23] = -0.25 + 0.15 * smooth     # right shoulder roll
            q[25] = 0.6 + 0.50 * smooth       # right elbow bend (1.10)
            q[27] = -0.30 * smooth            # wrist pitch tilt down to target
            self.grasp_target = 0.0
        elif cycle < 4.0:
            # Phase 2: Grasp block and prepare to lift
            prog = (cycle - 2.5) / 1.5
            q[14] = 0.18
            q[3] += 0.08
            q[9] += 0.08
            q[0] -= 0.04
            q[6] -= 0.04
            q[15] = 0.40
            q[18] = 1.00
            q[22] = -0.55
            q[23] = -0.10
            q[25] = 1.10
            q[27] = -0.30
            self.grasp_target = min(1.0, prog * 1.5)
        elif cycle < 6.5:
            # Phase 3: Lift block up to chest height and rotate wrist to inspect
            prog = (cycle - 4.0) / 2.5
            smooth = 0.5 - 0.5 * math.cos(math.pi * prog)
            # Torso rises back upright
            q[14] = 0.18 * (1.0 - smooth)
            q[3] += 0.08 * (1.0 - smooth)
            q[9] += 0.08 * (1.0 - smooth)
            # Right arm lifts object up to eye level
            q[22] = -0.55 + 0.20 * smooth     # shoulder pitch (-0.35)
            q[23] = -0.10 - 0.20 * smooth     # shoulder roll
            q[25] = 1.10 + 0.45 * smooth      # elbow bent (1.55)
            q[26] = 0.70 * smooth             # wrist roll to present block face
            q[27] = -0.30 + 0.30 * smooth
            q[15] = 0.40 * (1.0 - smooth) + 0.2 * smooth
            q[18] = 1.00 * (1.0 - smooth) + 0.6 * smooth
            self.grasp_target = 1.0
        else:
            # Phase 4: Place block down and settle back to stand
            prog = (cycle - 6.5) / 1.5
            smooth = 0.5 - 0.5 * math.cos(math.pi * prog)
            q[22] = -0.35 + 0.55 * smooth     # return to 0.2
            q[23] = -0.30 + 0.05 * smooth     # return to -0.25
            q[25] = 1.55 - 0.95 * smooth      # return to 0.6
            q[26] = 0.70 * (1.0 - smooth)     # unroll wrist
            self.grasp_target = max(0.0, 1.0 - prog * 2.0)

        self.target_q[:] = q

    def step_physics(self):
        cur_q = np.array([self.data.qpos[adr] for adr in self.qadrs], dtype=np.float32)
        cur_dq = np.array([self.data.qvel[adr] for adr in self.dofadrs], dtype=np.float32)

        if self.grit_active and self.grit_policy:
            if self.gantry_eqid != -1:
                self.data.eq_active[self.gantry_eqid] = 0
            if self.grit_target_q is None or self.grit_control_counter % 10 == 0:
                self.grit_target_q = self.grit_policy.infer(
                    cur_q,
                    self.data.qvel.copy(),
                    self.data.qpos[3:7].copy(),
                )
            self.target_q[:] = self.grit_target_q
            self.grit_control_counter += 1

        self.target_q[:] = np.clip(
            self.target_q,
            self.model.jnt_range[self.jids, 0],
            self.model.jnt_range[self.jids, 1],
        )

        tau = self.kps * (self.target_q - cur_q) - self.kds * cur_dq
        if not np.all(np.isfinite(tau)) or not np.all(np.isfinite(self.target_q)):
            raise FloatingPointError("Non-finite expressive control target")

        n_act = min(len(tau), self.model.nu)
        lower = self.model.actuator_ctrlrange[:n_act, 0]
        upper = self.model.actuator_ctrlrange[:n_act, 1]
        self.data.ctrl[:n_act] = np.clip(tau[:n_act], lower, upper)

        for adr in self.finger_qadrs:
            if not math.isfinite(self.grasp_target):
                raise FloatingPointError("Non-finite finger target")
            self.data.qpos[adr] = self.grasp_target * 1.2

        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.model.opt.timestep


def run_expressive_demo(mode: str = "dance", headless: bool = False, duration: float | None = None):
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be positive and finite")
    print("\n" + "=" * 76)
    print(">>> UNITREE G1 29-DOF EXPRESSIVE ACTION DEMONSTRATOR <<<")
    print("=" * 76)
    print("  Controls in 3D Window:")
    print("    [1] Friendly Greeting Wave")
    print("    [2] Saturday Night Disco Fever")
    print("    [3] Tabletop Block Reach & Pick with Dexterous Hands")
    print("    [4] Respectful Martial Bow")
    print("    [5] Boxing Punch Combination")
    print("    [G] Toggle Support Gantry (scripted modes only; GRIT dance runs unsupported)")
    print("    [Space] Return to Neutral Stance")
    print("=" * 76 + "\n")

    current_mode = mode.lower()
    use_grit = current_mode in ("dance", "disco")
    controller = G1ExpressiveController(use_grit=use_grit)
    if current_mode in ("dance", "disco"):
        current_mode = "dance"
        controller.load_motion("dance")
    elif current_mode in ("wave", "wave_right"):
        current_mode = "wave"
        controller.load_motion("wave_right")
    elif current_mode in ("wave_left",):
        current_mode = "wave"
        controller.load_motion("wave_left")
    elif current_mode in ("reach", "pick"):
        current_mode = "reach"
        controller.current_motion = "reach"
        print("[ACTION] Mode: Tabletop Block Reach & Pick")
    elif current_mode in ("bow",):
        current_mode = "bow"
        controller.load_motion("bow")
    elif current_mode in ("punch", "box"):
        current_mode = "punch"
        controller.load_motion("punch")
    else:
        controller.load_motion("dance")
        current_mode = "dance"

    def handle_action_key(keycode_or_char: int | str):
        nonlocal current_mode
        if isinstance(keycode_or_char, int):
            ch = chr(keycode_or_char).lower() if 0 <= keycode_or_char <= 255 else ""
        else:
            ch = str(keycode_or_char).lower()

        if ch == '1':
            current_mode = "wave"
            controller.set_grit_active(False)
            controller.load_motion("wave_right")
            print("\n[ACTION] >>> Switched to: Friendly Greeting Wave")
        elif ch == '2':
            current_mode = "dance"
            controller.set_grit_active(True)
            controller.load_motion("dance")
            print("\n[ACTION] >>> Switched to: Saturday Night Disco Fever")
        elif ch == '3':
            current_mode = "reach"
            controller.set_grit_active(False)
            controller.current_motion = "reach"
            print("\n[ACTION] >>> Switched to: Tabletop Block Reach & Pick")
        elif ch == '4':
            current_mode = "bow"
            controller.set_grit_active(False)
            controller.load_motion("bow")
            print("\n[ACTION] >>> Switched to: Respectful Martial Bow")
        elif ch == '5':
            current_mode = "punch"
            controller.set_grit_active(False)
            controller.load_motion("punch")
            print("\n[ACTION] >>> Switched to: Boxing Punch Combination")
        elif ch == 'g':
            controller.toggle_gantry()
        elif ch in (' ', 'r'):
            current_mode = "idle"
            controller.current_motion = None
            controller.set_grit_active(False)
            controller.reset_posture()
            print("\n[ACTION] >>> Neutral Standing Stance (Reset)")
        sys.stdout.flush()

    if headless:
        seconds = duration if duration is not None else 10.0
        budget = math.ceil(seconds / controller.model.opt.timestep)
        print(f"[HEADLESS] Running {seconds:g}-second simulation demonstration...")
        resets = 0
        for step in range(1, budget + 1):
            now = controller.sim_time
            if current_mode == "reach":
                controller.step_reach_and_pick(now)
            elif controller.current_motion:
                controller.update_motion(now)
            controller.step_physics()
            if controller.grit_active and controller.has_fallen():
                resets += 1
                raise RuntimeError(f"GRIT fell after {step} steps; resets={resets}")
            if step % 500 == 0:
                print(f"Step {step:5d} | Time {now:4.1f}s | Mode: {current_mode.upper()}")
        print(f"[HEADLESS] Finished {budget} steps; resets={resets}.")
        return {"steps": budget, "resets": resets}

    keyboard = CrossPlatformKeyboard()
    last_render = time.time()
    render_interval = 1.0 / 60.0

    try:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(controller.model, controller.data, key_callback=handle_action_key) as viewer:
            while viewer.is_running():
                if duration is not None and controller.sim_time >= duration:
                    break
                step_start = time.time()
                sim_time = controller.sim_time

                # Poll terminal non-blockingly (works whether terminal or viewer has focus)
                term_ch = keyboard.poll()
                if term_ch:
                    handle_action_key(term_ch)

                if current_mode == "reach":
                    controller.step_reach_and_pick(sim_time)
                elif controller.current_motion:
                    controller.update_motion(sim_time)

                controller.step_physics()
                if controller.grit_active and controller.has_fallen():
                    print(f"[AUTO-RESET] GRIT recovery reset at t={controller.sim_time:.2f}s")
                    controller.reset_posture()

                now = time.time()
                if now - last_render >= render_interval:
                    viewer.sync()
                    last_render = now

                elapsed = time.time() - step_start
                sleep_time = controller.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[G1] Simulation stopped cleanly by user.")
    finally:
        keyboard.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unitree G1 29-DOF Expressive Actions")
    parser.add_argument("mode", nargs="?", default="dance", choices=["dance", "wave", "reach", "bow", "punch"], help="Action mode")
    parser.add_argument("--headless", action="store_true", help="Run without 3D viewer")
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")
    run_expressive_demo(mode=args.mode, headless=args.headless, duration=args.duration)
