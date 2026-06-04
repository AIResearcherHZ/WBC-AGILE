import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from agile.rl_env.mdp.actuators import DelayedDCMotorCfg

_TAKS_T1_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "taks", "Taks_T1.usd"
)

_SEMI_TAKS_T1_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "taks", "Semi_Taks_T1.usd"
)

WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM_JOINT_NAMES = [
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint", ".*_elbow_joint",
]
WRIST_JOINT_NAMES = [".*_wrist_roll_joint", ".*_wrist_yaw_joint", ".*_wrist_pitch_joint"]
NECK_JOINT_NAMES = ["neck_yaw_joint", "neck_roll_joint", "neck_pitch_joint"]
HEAD_JOINT_NAMES = NECK_JOINT_NAMES
LEG_JOINT_NAMES: list[str] = []
FEET_LINK_NAMES: list[str] = []

# 无腿机器人：base_link 是最底链节，静止离地 ~0.07m。吊带把 base 托到该高度并扶正朝向。
# 实测把 base 拉到 0.45/0.65 只会整机悬空+躯干瘫倒；0.2m=略离地、靠扶正力矩保持竖直。
DEFAULT_TRUNK_HEIGHT = 0.2

UNDESIRED_CONTACTS_LINKS = [
    "base_link", "torso_link", "waist_.*_link",
    ".*_shoulder_.*_link", ".*_elbow_link", "neck_.*_link",
]

MIN_DELAY_STEPS = 0
MAX_DELAY_STEPS = 8


##
# Full-body Taks_T1: 腿 + 腰 + 双臂 + 脖子，自由浮动基座，用于运动控制 / 站立任务。
##

# ── 全身 Taks_T1 站立任务专用常量 ──
# 与上面无腿 Semi 的同名常量（LEG_JOINT_NAMES=[]/FEET_LINK_NAMES=[]/DEFAULT_TRUNK_HEIGHT=0.2/
# UNDESIRED_CONTACTS_LINKS）区分；ARM/WRIST/NECK/WAIST_JOINT_NAMES 两者通用，直接复用上面的。
TAKS_T1_LEG_JOINT_NAMES = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
# 脚（腿最末端的链节）。Taks 无单独 foot_link，踝 roll 链节即落地脚。
TAKS_T1_FEET_LINK_NAMES = [".*_ankle_roll_link"]
# 站立时 pelvis（浮动基座/根链节）的目标离地高度。几何：零位时脚在 pelvis 下方 0.706m；
# TAKS_T1_CFG 以微屈腿姿态 spawn 于 z=0.75 → 取 0.68（与 Booster 0.65/0.71≈0.91 同比例）。
TAKS_T1_DEFAULT_TRUNK_HEIGHT = 0.68
# 站立时不应触地的链节（上半身 + 髋 + 骨盆）。故意不含膝/踝/脚——起身过程中它们会合理着地。
TAKS_T1_UNDESIRED_CONTACTS_LINKS = [
    "pelvis",
    "torso_link",
    "waist_.*_link",
    ".*_hip_.*_link",
    ".*_shoulder_.*_link",
    ".*_elbow_link",
    ".*_wrist_.*_link",
    "neck_.*_link",
]

TAKS_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_TAKS_T1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_shoulder_roll_joint": 0.16,
            "right_shoulder_roll_joint": -0.16,
            ".*_shoulder_pitch_joint": 0.16,
            ".*_elbow_joint": 1.10,
            ".*_hip_pitch_joint": -0.14,
            ".*_knee_joint": 0.36,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    # 10Hz, 阻尼比2.0
    actuators={
        "legs_hip_yaw_roll": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 97.0,
                ".*_hip_roll_joint": 97.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 25.0,
                ".*_hip_roll_joint": 25.0,
            },
            stiffness=589.409607,
            damping=37.522984,
            armature=0.149299,
        ),
        "legs_hip_pitch_knee": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 25.0,
                ".*_knee_joint": 25.0,
            },
            stiffness=219.499985,
            damping=13.973804,
            armature=0.055600,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=112.434517,
            damping=7.157804,
            armature=0.028480,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim=97.0,
            velocity_limit_sim=25.0,
            stiffness=589.409607,
            damping=37.522984,
            armature=0.149299,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=112.434517,
            damping=7.157804,
            armature=0.028480,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=7.0,
            velocity_limit_sim=8.0,
            stiffness=6.632374,
            damping=0.422230,
            armature=0.001680,
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=NECK_JOINT_NAMES,
            effort_limit_sim=3.0,
            velocity_limit_sim=8.0,
            stiffness=4.936697,
            damping=0.157140,
            armature=0.000313,
        ),
    },
)


SEMI_TAKS_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_SEMI_TAKS_T1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    articulation_root_prim_path="/base_link",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, DEFAULT_TRUNK_HEIGHT),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim=97.0,
            velocity_limit_sim=4.19,
            stiffness=589.409607,
            damping=37.522984,
            armature=0.149299,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=5.4454,
            stiffness=112.434517,
            damping=7.157804,
            armature=0.028480,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
                "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
            ],
            effort_limit_sim=9.0,
            velocity_limit_sim=20.944,
            stiffness=6.632374,
            damping=0.422230,
            armature=0.001680,
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=NECK_JOINT_NAMES,
            effort_limit_sim=3.0,
            velocity_limit_sim=15.71,
            stiffness=1.234174,
            damping=0.078570,
            armature=0.000313,
        ),
    },
)

SEMI_TAKS_T1_DELAYED_DC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_SEMI_TAKS_T1_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    articulation_root_prim_path="/base_link",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, DEFAULT_TRUNK_HEIGHT),
        joint_pos={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all": DelayedDCMotorCfg(
            max_delay=MAX_DELAY_STEPS,
            min_delay=MIN_DELAY_STEPS,
            saturation_effort=130.0,
            joint_names_expr=[".*"],
            stiffness={
                "waist_.*": 100.0,
                ".*_shoulder_.*": 20.0,
                ".*_elbow_joint": 20.0,
                ".*_wrist_.*": 5.0,
                "neck_.*": 10.0,
            },
            damping={
                "waist_.*": 2.5,
                ".*_shoulder_.*": 0.5,
                ".*_elbow_joint": 0.5,
                ".*_wrist_.*": 0.2,
                "neck_.*": 0.2,
            },
            velocity_limit_sim={
                "waist_.*": 4.19,
                ".*_shoulder_.*": 5.4454,
                ".*_elbow_joint": 5.4454,
                ".*_wrist_.*": 20.944,
                "neck_.*": 15.71,
            },
            friction=0.01,
            armature=0.02,
            effort_limit_sim={
                "waist_yaw_joint": 97.0,
                "waist_roll_joint": 97.0,
                "waist_pitch_joint": 97.0,
                "left_shoulder_pitch_joint": 27.0,
                "left_shoulder_roll_joint": 27.0,
                "left_shoulder_yaw_joint": 27.0,
                "left_elbow_joint": 27.0,
                "right_shoulder_pitch_joint": 27.0,
                "right_shoulder_roll_joint": 27.0,
                "right_shoulder_yaw_joint": 27.0,
                "right_elbow_joint": 27.0,
                "left_wrist_roll_joint": 9.0,
                "left_wrist_yaw_joint": 9.0,
                "left_wrist_pitch_joint": 9.0,
                "right_wrist_roll_joint": 9.0,
                "right_wrist_yaw_joint": 9.0,
                "right_wrist_pitch_joint": 9.0,
                "neck_yaw_joint": 3.0,
                "neck_roll_joint": 3.0,
                "neck_pitch_joint": 3.0,
            },
        ),
    },  # type: ignore
)
