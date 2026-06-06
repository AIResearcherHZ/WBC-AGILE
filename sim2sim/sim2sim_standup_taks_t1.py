# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Taks_T1 StandUp Sim2Sim (32 DOF) — 自包含 MuJoCo 部署/验证脚本。

把 IsaacLab 训练的站起策略（任务 ``StandUp-Taks-T1-v0``，rsl_rl ``ActorCritic`` MLP）
直接从训练 checkpoint 加载，在 MuJoCo 里复现，并 **逐字节** 还原训练时的观测/动作管线。
结构对标 ``sim2sim/sim2sim_standup_t1.py``（Booster T1，23 DOF），主要差异：

  * 32 DOF 全身（腿+腰+双臂+腕+脖），关节顺序为 IsaacLab/PhysX 顺序（与配置声明序不同）。
  * 执行器是 **ImplicitActuatorCfg**（不是 Booster 的 DelayedDCMotor）：力矩 = 逐关节 PD，
    再按 ``effort_limit_sim`` **硬截断**；**没有** DC 电机的力矩-转速饱和包络。
  * MuJoCo 模型的关节自带被动 damping/frictionloss，而 IsaacLab 里 damping 已并入 PD(kd)、
    friction=0 —— 加载后会把这些被动项清零、并把 armature 对齐 IsaacLab，避免双重计入。

  观测 (510 = 5 帧历史, **term-major**, 每项内部帧从旧到新, IsaacLab 关节顺序):
      [ base_ang_vel ×0.2      (5×3 =15) ]   # 体坐标系角速度(陀螺仪 imu_gyro)
      [ projected_gravity      (5×3 =15) ]   # 重力方向投影到体坐标系(根四元数)
      [ joint_pos_rel          (5×32=160) ]  # q - default_q
      [ joint_vel_rel ×0.05    (5×32=160) ]
      [ last_action (raw, clip±100) (5×32=160) ]
  动作 (RelativeJointPositionAction): target = q_now + clip(raw·0.1, -1, 1)
  执行器: 逐关节 PD(kp/kd) + 逐关节力矩上限(effort)
  初始化: default/supine/prone 姿态 + settle, 用于触发站起策略

ground-truth 常量由 ``scripts/export_IODescriptors.py --task StandUp-Taks-T1-v0`` 一次性导出
（见 logs/.../exported/standup_taks_t1_v0_IO_descriptors.yaml）后烘焙；effort 上限取自
``agile/rl_env/assets/robots/taks.py`` 的 ``effort_limit_sim``。

用法::

    python sim2sim/sim2sim_standup_taks_t1.py \
        --load_model logs/rsl_rl/stand_up_taks_t1/<run>/model_19000.pt \
        --init-state supine --duration 20.0

    # 先验证站立保持(健全性检查), 再验证站起:
    python sim2sim/sim2sim_standup_taks_t1.py --load_model <ckpt> --init-state default
    python sim2sim/sim2sim_standup_taks_t1.py --load_model <ckpt> --init-state supine
"""

import argparse
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Ground-truth 常量 (由 scripts/export_IODescriptors.py --task StandUp-Taks-T1-v0 导出,
# 见 logs/.../exported/standup_taks_t1_v0_IO_descriptors.yaml).
# IsaacLab/PhysX 关节顺序与 MuJoCo 树(DFS)顺序不同 —— 本列表是策略观测/动作的关节顺序.
# ---------------------------------------------------------------------------
ISAACLAB_JOINT_NAMES = [
    "waist_roll_joint",          # 0
    "waist_pitch_joint",         # 1
    "waist_yaw_joint",           # 2
    "left_shoulder_pitch_joint",  # 3
    "right_shoulder_pitch_joint",  # 4
    "neck_yaw_joint",            # 5
    "left_hip_pitch_joint",      # 6
    "right_hip_pitch_joint",     # 7
    "left_shoulder_roll_joint",  # 8
    "right_shoulder_roll_joint",  # 9
    "neck_roll_joint",           # 10
    "left_hip_roll_joint",       # 11
    "right_hip_roll_joint",      # 12
    "left_shoulder_yaw_joint",   # 13
    "right_shoulder_yaw_joint",  # 14
    "neck_pitch_joint",          # 15
    "left_hip_yaw_joint",        # 16
    "right_hip_yaw_joint",       # 17
    "left_elbow_joint",          # 18
    "right_elbow_joint",         # 19
    "left_knee_joint",           # 20
    "right_knee_joint",          # 21
    "left_wrist_roll_joint",     # 22
    "right_wrist_roll_joint",    # 23
    "left_ankle_pitch_joint",    # 24
    "right_ankle_pitch_joint",   # 25
    "left_wrist_yaw_joint",      # 26
    "right_wrist_yaw_joint",     # 27
    "left_ankle_roll_joint",     # 28
    "right_ankle_roll_joint",    # 29
    "left_wrist_pitch_joint",    # 30
    "right_wrist_pitch_joint",   # 31
]
NUM_JOINTS = len(ISAACLAB_JOINT_NAMES)  # 32


def _by_regex(rules, names):
    """按第一个匹配的正则给每个关节名赋值 (镜像 IsaacLab 的 actuator regex 配置)."""
    out = {}
    for n in names:
        for pat, val in rules:
            if re.search(pat, n):
                out[n] = float(val)
                break
        else:
            raise KeyError(f"没有规则匹配关节 {n!r}")
    return out


# 逐关节属性 (name -> value), 与 agile/rl_env/assets/robots/taks.py 的 TAKS_T1_CFG 执行器一致.
# 腿 hip_yaw/hip_roll 与 waist 同组(589/37.5/0.149/97); hip_pitch/knee 同组(219/13.97/0.0556/120);
# ankle 与 arm(shoulder/elbow) 同组(112/7.16/0.0285/27); wrist(6.63/0.42/0.00168/7); neck(4.94/0.157/0.000313/3).
KP = _by_regex(
    [
        ("waist", 589.409607),
        ("hip_(yaw|roll)", 589.409607),
        ("(hip_pitch|knee)", 219.499985),
        ("(ankle|shoulder|elbow)", 112.434517),
        ("wrist", 6.632374),
        ("neck", 4.936697),
    ],
    ISAACLAB_JOINT_NAMES,
)
KD = _by_regex(
    [
        ("waist", 37.522984),
        ("hip_(yaw|roll)", 37.522984),
        ("(hip_pitch|knee)", 13.973804),
        ("(ankle|shoulder|elbow)", 7.157804),
        ("wrist", 0.422230),
        ("neck", 0.157140),
    ],
    ISAACLAB_JOINT_NAMES,
)
# 逐关节力矩上限 (= effort_limit_sim). ImplicitActuator 用它对 PD 力矩做硬截断.
EFFORT_LIMIT = _by_regex(
    [
        ("waist", 97.0),
        ("hip_(yaw|roll)", 97.0),
        ("(hip_pitch|knee)", 120.0),
        ("(ankle|shoulder|elbow)", 27.0),
        ("wrist", 7.0),
        ("neck", 3.0),
    ],
    ISAACLAB_JOINT_NAMES,
)
# 关节电机转子等效转动惯量 (armature). 与 MuJoCo 模型一致, 加载后会强制对齐.
ARMATURE = _by_regex(
    [
        ("waist", 0.149299),
        ("hip_(yaw|roll)", 0.149299),
        ("(hip_pitch|knee)", 0.055600),
        ("(ankle|shoulder|elbow)", 0.028480),
        ("wrist", 0.001680),
        ("neck", 0.000313),
    ],
    ISAACLAB_JOINT_NAMES,
)
# 速度上限 (= soft_joint_vel_limits). ImplicitActuator 不用它整形力矩, 仅留作参考/绘图.
VELOCITY_LIMIT = _by_regex(
    [
        ("waist", 25.0),
        ("hip_(yaw|roll)", 25.0),
        ("(hip_pitch|knee)", 25.0),
        ("(ankle|shoulder|elbow|wrist|neck)", 8.0),
    ],
    ISAACLAB_JOINT_NAMES,
)

# 默认关节角 (= articulation init_state.joint_pos, 也是 joint_pos_rel 的零点).
DEFAULT_Q = {n: 0.0 for n in ISAACLAB_JOINT_NAMES}
DEFAULT_Q.update(
    {
        "left_shoulder_pitch_joint": 0.16,
        "right_shoulder_pitch_joint": 0.16,
        "left_shoulder_roll_joint": 0.16,
        "right_shoulder_roll_joint": -0.16,
        "left_hip_pitch_joint": -0.14,
        "right_hip_pitch_joint": -0.14,
        "left_elbow_joint": 1.10,
        "right_elbow_joint": 1.10,
        "left_knee_joint": 0.36,
        "right_knee_joint": 0.36,
        "left_ankle_pitch_joint": -0.20,
        "right_ankle_pitch_joint": -0.20,
    }
)

# 烘焙成 IsaacLab 顺序的数组 (策略侧的全部计算都在这个顺序下进行).
KP_ISAAC = np.array([KP[n] for n in ISAACLAB_JOINT_NAMES], dtype=np.float64)
KD_ISAAC = np.array([KD[n] for n in ISAACLAB_JOINT_NAMES], dtype=np.float64)
EFFORT_ISAAC = np.array([EFFORT_LIMIT[n] for n in ISAACLAB_JOINT_NAMES], dtype=np.float64)
ARMATURE_ISAAC = np.array([ARMATURE[n] for n in ISAACLAB_JOINT_NAMES], dtype=np.float64)
VLIM_ISAAC = np.array([VELOCITY_LIMIT[n] for n in ISAACLAB_JOINT_NAMES], dtype=np.float64)
DEFAULT_Q_ISAAC = np.array([DEFAULT_Q[n] for n in ISAACLAB_JOINT_NAMES], dtype=np.float64)

# 观测/动作/时序常量 (params/env.yaml + 导出的 IO descriptor).
ACTION_SCALE = 0.1
ACTION_CLIP = 1.0          # clip(raw*scale, -1, 1)
LAST_ACTION_CLIP = 100.0   # last_action 观测项的 clip
HISTORY_LEN = 5
ANG_VEL_SCALE = 0.2
JOINT_VEL_SCALE = 0.05
SINGLE_FRAME_DIM = 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS  # 102
OBS_DIM = SINGLE_FRAME_DIM * HISTORY_LEN  # 510
# 单帧内各项在 102 维向量里的切片 (term-major 展平时按此切片各取 5 帧).
TERM_SLICES = [
    (0, 3),                          # base_ang_vel
    (3, 6),                          # projected_gravity
    (6, 6 + NUM_JOINTS),             # joint_pos_rel
    (6 + NUM_JOINTS, 6 + 2 * NUM_JOINTS),    # joint_vel_rel
    (6 + 2 * NUM_JOINTS, 6 + 3 * NUM_JOINTS),  # last_action
]

PHYSICS_DT = 1.0 / 200.0
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION  # 1/50 s, 50 Hz
SPAWN_HEIGHT = 0.75   # TAKS_T1_CFG init_state.pos[2]
TARGET_HEIGHT = 0.68  # TAKS_T1_DEFAULT_TRUNK_HEIGHT (仅用于绘图参考)
IMU_GYRO_SENSOR = "imu_gyro"  # MuJoCo 体坐标系陀螺仪传感器名


def _self_check():
    """对照导出值断言烘焙常量正确 (防止后续误改偏离训练策略)."""
    # 顺序均为 ISAACLAB_JOINT_NAMES (32). 数值取自 IO descriptor / taks.py.
    ref_kp = [589.409607] * 3 + [
        112.434517, 112.434517, 4.936697, 219.499985, 219.499985, 112.434517, 112.434517,
        4.936697, 589.409607, 589.409607, 112.434517, 112.434517, 4.936697, 589.409607,
        589.409607, 112.434517, 112.434517, 219.499985, 219.499985, 6.632374, 6.632374,
        112.434517, 112.434517, 6.632374, 6.632374, 112.434517, 112.434517, 6.632374, 6.632374,
    ]
    ref_kd = [37.522984] * 3 + [
        7.157804, 7.157804, 0.157140, 13.973804, 13.973804, 7.157804, 7.157804, 0.157140,
        37.522984, 37.522984, 7.157804, 7.157804, 0.157140, 37.522984, 37.522984, 7.157804,
        7.157804, 13.973804, 13.973804, 0.422230, 0.422230, 7.157804, 7.157804, 0.422230,
        0.422230, 7.157804, 7.157804, 0.422230, 0.422230,
    ]
    ref_eff = [97, 97, 97, 27, 27, 3, 120, 120, 27, 27, 3, 97, 97, 27, 27, 3, 97, 97,
               27, 27, 120, 120, 7, 7, 27, 27, 7, 7, 27, 27, 7, 7]
    ref_arm = [0.149299] * 3 + [
        0.028480, 0.028480, 0.000313, 0.055600, 0.055600, 0.028480, 0.028480, 0.000313,
        0.149299, 0.149299, 0.028480, 0.028480, 0.000313, 0.149299, 0.149299, 0.028480,
        0.028480, 0.055600, 0.055600, 0.001680, 0.001680, 0.028480, 0.028480, 0.001680,
        0.001680, 0.028480, 0.028480, 0.001680, 0.001680,
    ]
    ref_vel = [25, 25, 25, 8, 8, 8, 25, 25, 8, 8, 8, 25, 25, 8, 8, 8, 25, 25,
               8, 8, 25, 25, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    ref_q = [0, 0, 0, 0.16, 0.16, 0, -0.14, -0.14, 0.16, -0.16, 0, 0, 0, 0, 0, 0, 0, 0,
             1.10, 1.10, 0.36, 0.36, 0, 0, -0.20, -0.20, 0, 0, 0, 0, 0, 0]
    assert len(ISAACLAB_JOINT_NAMES) == 32
    assert np.allclose(KP_ISAAC, ref_kp), KP_ISAAC.tolist()
    assert np.allclose(KD_ISAAC, ref_kd), KD_ISAAC.tolist()
    assert np.allclose(EFFORT_ISAAC, ref_eff), EFFORT_ISAAC.tolist()
    assert np.allclose(ARMATURE_ISAAC, ref_arm), ARMATURE_ISAAC.tolist()
    assert np.allclose(VLIM_ISAAC, ref_vel), VLIM_ISAAC.tolist()
    assert np.allclose(DEFAULT_Q_ISAAC, ref_q), DEFAULT_Q_ISAAC.tolist()
    assert OBS_DIM == 510 and SINGLE_FRAME_DIM == 102


_self_check()


# ---------------------------------------------------------------------------
# 策略加载 (从训练 checkpoint 重建 actor MLP; 也支持 TorchScript .pt / .onnx)
# ---------------------------------------------------------------------------
class _ActorMLP(nn.Module):
    """rsl_rl ActorCritic 的 actor 部分: Linear/激活 堆叠 (无观测归一化)."""

    def __init__(self, in_dim, hidden_dims, out_dim, activation):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), activation()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}


def load_policy(path, device, activation="elu"):
    """返回一个可调用对象 obs(np[510]) -> action(np[32]).

    优先把 ``path`` 当作 rsl_rl 训练 checkpoint (含 ``model_state_dict``) 重建 actor;
    否则尝试 TorchScript; ``.onnx`` 用 onnxruntime.
    """
    if str(path).endswith(".onnx"):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(path), providers=providers)
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        def _run(obs_np):
            out = sess.run([out_name], {in_name: obs_np[None, :].astype(np.float32)})[0]
            return out[0].astype(np.float64)

        print(f"  已加载 ONNX 策略: {path}")
        return _run

    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ms = ckpt["model_state_dict"]
        # 取出 actor.layers.{i}.weight, 推断结构.
        wkeys = sorted(
            [k for k in ms if k.startswith("actor.layers.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[2]),
        )
        in_dim = ms[wkeys[0]].shape[1]
        out_dim = ms[wkeys[-1]].shape[0]
        hidden = [ms[k].shape[0] for k in wkeys[:-1]]
        assert in_dim == OBS_DIM, f"actor 输入维度 {in_dim} != 期望 {OBS_DIM}"
        assert out_dim == NUM_JOINTS, f"actor 输出维度 {out_dim} != 期望 {NUM_JOINTS}"
        assert "obs_norm_state_dict" not in ckpt, "该 checkpoint 含观测归一化, 本脚本未实现"
        model = _ActorMLP(in_dim, hidden, out_dim, _ACTIVATIONS[activation])
        actor_state = {k[len("actor.") :]: v for k, v in ms.items() if k.startswith("actor.layers.")}
        model.load_state_dict(actor_state)
        model.eval().to(device)
        print(f"  已从训练 checkpoint 重建 actor MLP (iter={ckpt.get('iter', '?')}, hidden={hidden}, elu, 无归一化)")

        @torch.no_grad()
        def _run(obs_np):
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            return model(obs_t).cpu().numpy().astype(np.float64)

        return _run

    # TorchScript
    model = torch.jit.load(path, map_location=device).eval()
    print(f"  已加载 TorchScript 策略: {path}")

    @torch.no_grad()
    def _run(obs_np):
        obs_t = torch.as_tensor(obs_np[None, :], dtype=torch.float32, device=device)
        return model(obs_t).squeeze(0).cpu().numpy().astype(np.float64)

    return _run


# ---------------------------------------------------------------------------
# MuJoCo <-> IsaacLab 索引映射 (运行时从模型读取, 对关节/执行器顺序完全鲁棒)
# ---------------------------------------------------------------------------
class RobotIndex:
    """缓存 IsaacLab 关节顺序到 MuJoCo qpos/qvel/ctrl 地址的映射."""

    def __init__(self, model):
        import mujoco

        # 浮动基座 (free joint): qpos[0:7] = [x,y,z, qw,qx,qy,qz], qvel[0:6] = [v(world), ω(body)].
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert len(free) == 1, f"期望 1 个 free joint, 实际 {len(free)}"
        self.root_qadr = int(model.jnt_qposadr[free[0]])  # 通常为 0

        # 每个 IsaacLab 关节 -> MuJoCo qpos/qvel(dof)/actuator 地址.
        self.qpos_idx = np.zeros(NUM_JOINTS, dtype=np.int32)
        self.qvel_idx = np.zeros(NUM_JOINTS, dtype=np.int32)
        self.ctrl_idx = np.zeros(NUM_JOINTS, dtype=np.int32)
        # 关节 id -> 执行器 id (motor 直驱, gear=1).
        jnt_to_act = {}
        for a in range(model.nu):
            jnt_to_act[int(model.actuator_trnid[a, 0])] = a
        for i, name in enumerate(ISAACLAB_JOINT_NAMES):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"MuJoCo 模型缺少关节 {name!r}"
            self.qpos_idx[i] = model.jnt_qposadr[jid]
            self.qvel_idx[i] = model.jnt_dofadr[jid]
            assert jid in jnt_to_act, f"关节 {name!r} 没有对应执行器"
            self.ctrl_idx[i] = jnt_to_act[jid]

        # IMU 角速度传感器 (体坐标系陀螺仪).
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, IMU_GYRO_SENSOR)
        assert sid >= 0, f"MuJoCo 模型缺少 {IMU_GYRO_SENSOR!r} 传感器"
        self.gyro_adr = int(model.sensor_adr[sid])

        # 一致性自检: 映射必须覆盖全部 32 个铰接关节且一一对应.
        assert len(set(self.qpos_idx.tolist())) == NUM_JOINTS
        assert len(set(self.ctrl_idx.tolist())) == NUM_JOINTS

    def read_joints(self, data):
        """返回 IsaacLab 顺序的 (q, dq)."""
        return data.qpos[self.qpos_idx].copy(), data.qvel[self.qvel_idx].copy()

    def write_torque(self, data, tau_isaac):
        data.ctrl[self.ctrl_idx] = tau_isaac


def align_dynamics_to_isaaclab(model, idx):
    """把 MuJoCo 关节动力学对齐到 IsaacLab 训练设置, 消除 sim2sim 差距.

    IsaacLab(ImplicitActuator) 里 kd 已作为 PD 阻尼项施加、关节 friction=0;
    而本 MuJoCo 模型的铰接关节自带被动 damping/frictionloss —— 若不清零会与控制器 kd
    **双重计入** 阻尼. armature(电机转子惯量) 二者本应一致, 这里强制对齐以防模型被替换.
    只动 32 个铰接关节的 dof, 不碰浮动基座(自由关节)的 6 个 dof.
    并把执行器 ctrlrange 设为 ±effort, 使本脚本的力矩截断成为唯一权威.
    """
    for i in range(NUM_JOINTS):
        dof = int(idx.qvel_idx[i])
        model.dof_damping[dof] = 0.0
        model.dof_frictionloss[dof] = 0.0
        model.dof_armature[dof] = float(ARMATURE_ISAAC[i])
        act = int(idx.ctrl_idx[i])
        model.actuator_ctrlrange[act] = [-float(EFFORT_ISAAC[i]), float(EFFORT_ISAAC[i])]
        model.actuator_ctrllimited[act] = 1


# ---------------------------------------------------------------------------
# 观测构建 (复现训练: raw -> (noise 关) -> clip -> scale, 然后压入历史)
# ---------------------------------------------------------------------------
def projected_gravity_b(root_quat_wxyz):
    """重力方向 [0,0,-1] 投影到体坐标系 (= IsaacLab projected_gravity)."""
    q_xyzw = root_quat_wxyz[[1, 2, 3, 0]]
    rot = R.from_quat(q_xyzw)
    return rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True)


def build_single_frame(idx, data, last_action_raw):
    """组装单帧 102 维观测 [ang_vel·0.2, proj_grav, q-q0, dq·0.05, last_action]."""
    q_isaac, dq_isaac = idx.read_joints(data)
    ang_vel = data.sensordata[idx.gyro_adr : idx.gyro_adr + 3].copy()
    root_quat = data.qpos[idx.root_qadr + 3 : idx.root_qadr + 7].copy()  # [w,x,y,z]
    grav = projected_gravity_b(root_quat)

    frame = np.empty(SINGLE_FRAME_DIM, dtype=np.float64)
    frame[0:3] = ang_vel * ANG_VEL_SCALE
    frame[3:6] = grav
    frame[6 : 6 + NUM_JOINTS] = q_isaac - DEFAULT_Q_ISAAC
    frame[6 + NUM_JOINTS : 6 + 2 * NUM_JOINTS] = dq_isaac * JOINT_VEL_SCALE
    frame[6 + 2 * NUM_JOINTS : 6 + 3 * NUM_JOINTS] = np.clip(last_action_raw, -LAST_ACTION_CLIP, LAST_ACTION_CLIP)
    return frame


class ObsHistory:
    """(HISTORY_LEN, 102) 环形历史, 索引 0 最旧, 末尾最新.

    展平为 510 维 **term-major**: 依次拼接每一项的 5 帧 (帧从旧到新),
    与 rsl_rl ``flatten_dict(cat([td[k].flatten(1) ...]))`` + IsaacLab CircularBuffer 一致.
    """

    def __init__(self):
        self.buf = None

    def reset(self):
        self.buf = None

    def push(self, frame):
        if self.buf is None:
            # 首帧: 复制填满 (镜像 CircularBuffer 首次 push 填满所有槽).
            self.buf = np.tile(frame, (HISTORY_LEN, 1))
        else:
            self.buf = np.vstack([self.buf[1:], frame[None, :]])

    def flatten(self):
        return np.concatenate([self.buf[:, a:b].reshape(-1) for (a, b) in TERM_SLICES])


# ---------------------------------------------------------------------------
# 执行器: 逐关节 PD + effort 硬截断 (复现 ImplicitActuator; 无 DC 电机饱和包络)
# ---------------------------------------------------------------------------
def pd_torque(target_q, q, dq):
    """显式 PD (目标速度 0) + 逐关节力矩上限, 全部 IsaacLab 顺序."""
    tau = KP_ISAAC * (target_q - q) + KD_ISAAC * (-dq)
    return np.clip(tau, -EFFORT_ISAAC, EFFORT_ISAAC)


# ---------------------------------------------------------------------------
# 初始姿态 + settle
# ---------------------------------------------------------------------------
_HALF_SQRT2 = 0.7071067811865476


def init_pose(model, data, idx, mode, settle_steps, fallen_height):
    """设置初始姿态并 settle, 返回快照 (qpos, qvel) 供键盘复位用."""
    import mujoco

    mujoco.mj_resetData(model, data)
    qadr = idx.root_qadr
    if mode == "default":
        data.qpos[qadr : qadr + 3] = [0.0, 0.0, SPAWN_HEIGHT]
        data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    elif mode == "supine":  # 仰卧(背朝下): 绕体 y 轴 -90°
        data.qpos[qadr : qadr + 3] = [0.0, 0.0, fallen_height]
        data.qpos[qadr + 3 : qadr + 7] = [_HALF_SQRT2, 0.0, -_HALF_SQRT2, 0.0]
    elif mode == "prone":   # 俯卧(脸朝下): 绕体 y 轴 +90°
        data.qpos[qadr : qadr + 3] = [0.0, 0.0, fallen_height]
        data.qpos[qadr + 3 : qadr + 7] = [_HALF_SQRT2, 0.0, _HALF_SQRT2, 0.0]
    else:
        raise ValueError(f"未知 init-state: {mode}")

    # 关节置默认角.
    data.qpos[idx.qpos_idx] = DEFAULT_Q_ISAAC
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # settle: 用 PD 保持默认角, 让机器人落到地面并衰减瞬态, 使首帧是稳定的(摔倒)姿态.
    if mode != "default":
        for _ in range(settle_steps):
            q, dq = idx.read_joints(data)
            idx.write_torque(data, pd_torque(DEFAULT_Q_ISAAC, q, dq))
            mujoco.mj_step(model, data)

    return data.qpos.copy(), data.qvel.copy()


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
class ViewerState:
    """passive viewer 的键盘状态 (空格暂停, R 复位)."""

    def __init__(self):
        self.paused = False
        self.reset_requested = False

    def key_callback(self, keycode):
        if keycode == 32:  # space
            self.paused = not self.paused
            print(f"[viewer] {'暂停' if self.paused else '继续'}")
        elif keycode in (82, 261):  # R / Delete
            self.reset_requested = True
            print("[viewer] 请求复位")


def run(args):
    import mujoco
    import mujoco.viewer

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"设备: {device}")

    print(f"\n加载策略: {args.load_model}")
    policy = load_policy(args.load_model, device)

    print(f"\n加载 MuJoCo 模型: {args.mjcf}")
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model)
    idx = RobotIndex(model)
    if not args.no_dyn_override:
        align_dynamics_to_isaaclab(model, idx)
        print("  已对齐关节动力学到 IsaacLab (清零被动 damping/frictionloss, armature 对齐, ctrlrange=±effort)")
    print(f"  关节数(铰接): {NUM_JOINTS}, 控制 {1.0 / CONTROL_DT:.0f} Hz / 物理 {1.0 / PHYSICS_DT:.0f} Hz")

    settle_steps = int(args.settle_time / PHYSICS_DT)
    snap_qpos, snap_qvel = init_pose(model, data, idx, args.init_state, settle_steps, args.fallen_height)
    print(f"  初始姿态: {args.init_state} (settle {args.settle_time:.2f}s)")

    history = ObsHistory()
    last_action_raw = np.zeros(NUM_JOINTS, dtype=np.float64)

    def do_reset():
        nonlocal last_action_raw
        data.qpos[:] = snap_qpos
        data.qvel[:] = snap_qvel
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        history.reset()
        last_action_raw = np.zeros(NUM_JOINTS, dtype=np.float64)

    do_reset()

    # 数据记录 (用于绘图).
    log = {"t": [], "target_q": [], "actual_q": [], "tau": [], "height": [], "grav_z": []}

    # viewer.
    headless = args.no_viewer or "DISPLAY" not in os.environ
    vstate = ViewerState()
    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=vstate.key_callback)
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 120.0
        viewer.cam.elevation = -20.0
    else:
        print("  无 DISPLAY 或 --no-viewer: 无头运行")

    num_control_steps = int(args.duration / CONTROL_DT)
    real_time = not args.no_real_time and not headless
    render_dt = 1.0 / 30.0
    last_render = time.time()
    wall_start = time.time()

    print(f"\n开始评估 {args.duration:.1f}s ({num_control_steps} 控制步)... (空格暂停, R 复位)")
    print("-" * 80)
    try:
        for step in range(num_control_steps):
            # 暂停 (viewer 仍响应).
            while vstate.paused and viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
                if vstate.reset_requested:
                    break
            if vstate.reset_requested:
                vstate.reset_requested = False
                do_reset()
                wall_start = time.time() - step * CONTROL_DT

            # 观测 (用当前 q/dq + 上一动作), 推理.
            frame = build_single_frame(idx, data, last_action_raw)
            history.push(frame)
            raw_action = policy(history.flatten())  # IsaacLab 顺序, raw
            last_action_raw = raw_action

            # 相对位置动作: target = q_now + clip(raw*0.1, -1, 1).
            q_now, _ = idx.read_joints(data)
            processed = np.clip(raw_action * ACTION_SCALE, -ACTION_CLIP, ACTION_CLIP)
            target_q = q_now + processed

            # 物理步进 (PD 力矩在每个物理步重算, 目标在 decimation 内保持不变).
            tau_last = None
            for _ in range(DECIMATION):
                q, dq = idx.read_joints(data)
                tau = pd_torque(target_q, q, dq)
                idx.write_torque(data, tau)
                mujoco.mj_step(model, data)
                tau_last = tau

            # 记录.
            q_after, _ = idx.read_joints(data)
            log["t"].append(step * CONTROL_DT)
            log["target_q"].append(target_q.copy())
            log["actual_q"].append(q_after)
            log["tau"].append(tau_last.copy())
            log["height"].append(float(data.qpos[idx.root_qadr + 2]))
            log["grav_z"].append(float(projected_gravity_b(data.qpos[idx.root_qadr + 3 : idx.root_qadr + 7])[2]))

            # 渲染 + 实时节流.
            if viewer is not None:
                if not viewer.is_running():
                    print("\nviewer 已关闭, 退出.")
                    break
                now = time.time()
                if now - last_render >= render_dt:
                    viewer.sync()
                    last_render = now
            if real_time:
                sleep = wall_start + (step + 1) * CONTROL_DT - time.time()
                if sleep > 0:
                    time.sleep(sleep)

            if step % 50 == 0:
                print(
                    f"步 {step:4d} | 高度 {log['height'][-1]:5.3f} | grav_z {log['grav_z'][-1]:+5.2f} "
                    f"| |action| {np.abs(raw_action).mean():.3f}"
                )
        print("-" * 80)
        print(f"评估结束, 共 {len(log['t'])} 步.")
    except KeyboardInterrupt:
        print("\n被用户中断 (Ctrl+C).")
    finally:
        if viewer is not None:
            viewer.close()

    if args.save_plots and log["t"]:
        save_plots(log, args.plot_dir)


def save_plots(log, out_dir):
    import matplotlib

    if "DISPLAY" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    t = np.array(log["t"])
    target_q = np.array(log["target_q"])
    actual_q = np.array(log["actual_q"])

    # 关节: 命令 vs 实际.
    n_cols = 4
    n_rows = (NUM_JOINTS + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes1 = axes1.flatten()
    for i in range(NUM_JOINTS):
        ax = axes1[i]
        ax.plot(t, target_q[:, i], "--", label="cmd")
        ax.plot(t, actual_q[:, i], label="actual")
        ax.set_title(ISAACLAB_JOINT_NAMES[i], fontsize=8)
        ax.grid(True)
    for i in range(NUM_JOINTS, len(axes1)):
        fig1.delaxes(axes1[i])
    axes1[0].legend()
    fig1.suptitle("Joint positions: commanded vs actual (IsaacLab order)")
    fig1.tight_layout()
    p1 = os.path.join(out_dir, "standup_taks_t1_joint_positions.png")
    fig1.savefig(p1, dpi=110)

    # 站起进度: 根高度 + 直立度.
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes2[0].plot(t, log["height"], label="root height")
    axes2[0].axhline(TARGET_HEIGHT, color="r", ls="--", label=f"target {TARGET_HEIGHT}")
    axes2[0].set_ylabel("height [m]")
    axes2[0].legend()
    axes2[0].grid(True)
    axes2[1].plot(t, log["grav_z"], label="proj_gravity_z")
    axes2[1].axhline(-1.0, color="g", ls="--", label="fully upright (-1)")
    axes2[1].set_ylabel("projected gravity z")
    axes2[1].set_xlabel("time [s]")
    axes2[1].legend()
    axes2[1].grid(True)
    fig2.suptitle(f"Stand-up progress: root height (->{TARGET_HEIGHT}) and uprightness (grav_z->-1)")
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "standup_taks_t1_progress.png")
    fig2.savefig(p2, dpi=110)
    print(f"已保存图像:\n  {p1}\n  {p2}")


def main():
    p = argparse.ArgumentParser(description="Taks_T1 StandUp Sim2Sim (MuJoCo, 32 DOF)")
    p.add_argument(
        "--load_model",
        type=str,
        default="logs/rsl_rl/stand_up_taks_t1/2026-06-05_19-14-08_stand_up_taks_t1/model_19000.pt",
        help="策略路径: rsl_rl 训练 checkpoint(model_*.pt) / TorchScript(.pt) / .onnx "
        "(也可指向 exported/policy.pt)",
    )
    p.add_argument(
        "--mjcf",
        type=str,
        default="agile/rl_env/assets/robot_menagerie/taks/t1/mujoco/Taks_T1.xml",
        help="MuJoCo 场景 (含地面+IMU+电机)",
    )
    p.add_argument("--init-state", dest="init_state", choices=["default", "supine", "prone"], default="supine")
    p.add_argument("--duration", type=float, default=60.0, help="评估时长 (秒)")
    p.add_argument("--settle-time", type=float, default=0.6, help="初始姿态 settle 时长 (秒)")
    p.add_argument("--fallen-height", type=float, default=0.45, help="supine/prone 初始根高度 (米)")
    p.add_argument("--no-dyn-override", action="store_true", help="不把关节动力学对齐到 IsaacLab (用模型原值, 调试用)")
    p.add_argument("--device", type=str, default="cpu", help="cpu / cuda / auto")
    p.add_argument("--no-viewer", action="store_true", help="无头运行 (不开 viewer)")
    p.add_argument("--no-real-time", action="store_true", help="不做实时节流 (尽快跑)")
    p.add_argument("--save-plots", action="store_true", help="结束后保存关节/站起进度图")
    p.add_argument("--plot-dir", type=str, default="logs/sim2mujoco/standup_taks_t1", help="图像输出目录")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
