import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from agile.rl_env.mdp.actuators import DelayedDCMotorCfg

_SEMI_TAKS_T1_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "taks", "Semi_Taks_T1", "Semi_Taks_T1.usd"
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

DEFAULT_TRUNK_HEIGHT = 0.65

UNDESIRED_CONTACTS_LINKS = [
    "base_link", "torso_link", "waist_.*_link",
    ".*_shoulder_.*_link", ".*_elbow_link", "neck_.*_link",
]

MIN_DELAY_STEPS = 0
MAX_DELAY_STEPS = 8


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
        joint_pos={
            ".*_shoulder_pitch_joint": 0.16,
            ".*_elbow_joint": 1.10,
            "waist_.*": 0.0,
            "neck_.*": 0.0,
        },
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
        joint_pos={
            ".*_shoulder_pitch_joint": 0.16,
            "left_shoulder_roll_joint": 0.20,
            "right_shoulder_roll_joint": -0.20,
            ".*_elbow_joint": 1.10,
            "waist_.*": 0.0,
            "neck_.*": 0.0,
        },
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
