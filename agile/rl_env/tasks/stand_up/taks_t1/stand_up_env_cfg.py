"""全身 Taks_T1 站立任务（StandUp-Taks-T1-v0）。

结构对标 Booster T1 站立任务（agile/rl_env/tasks/stand_up/t1/stand_up_env_cfg.py）：
有腿、能自己撑起，吊带只是“临时辅助”并由 adaptive_force_decay 课程逐步撤掉，
粗糙地形 + terrain_levels 课程。机器人 / 关节 / 链节命名按 Taks_T1 替换：
  Trunk → pelvis（浮动基座/根）、H2(吊带) → torso_link（脖子太弱，吊躯干）、
  Waist(脚朝向参考) → torso_link、*foot_link* → *_ankle_roll_link、
  *hand_link* → *_wrist_*_link、*Ankle* → *_ankle_*_joint。
"""

import pathlib

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from agile.rl_env import mdp
from agile.rl_env.assets.robots import taks as taks_t1
from agile.rl_env.mdp.terrains import STAND_UP_ROUGH_TERRAIN_CFG  # noqa: F401
from agile.rl_env.termination_cfg import DoneTermCfg as DoneTermEx

FILE_DIR = pathlib.Path(__file__).parent
REPO_DIR = FILE_DIR.parent.parent.parent

from_scratch = 1.0
with_curriculum = 1.0


@configclass
class SceneCfg(InteractiveSceneCfg):
    """有腿机器人，使用粗糙地形（对标 Booster T1）。"""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAND_UP_ROUGH_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                f"TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots：全身 Taks_T1（隐式执行器配置，含腿）
    robot = taks_t1.TAKS_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # 高度传感器挂在浮动根 pelvis 上
    height_measurement_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )


@configclass
class CommandsCfg:
    """无指令。"""


@configclass
class ObservationsCfg:

    @configclass
    class PolicyObservationCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = False
            self.flatten_history_dim = False

    @configclass
    class CriticObservationsCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        contact_forces = ObsTerm(
            func=mdp.contact_force_norm,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            },
            scale=5e-3,
            clip=(-25_000.0, 25_000.0),
        )
        base_height = ObsTerm(
            func=mdp.base_height_from_sensor,
            params={"sensor_cfg": SceneEntityCfg("height_measurement_sensor")},
            clip=(-2, 2),
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = False
            self.flatten_history_dim = False

    policy: PolicyObservationCfg = PolicyObservationCfg()
    critic: CriticObservationsCfg = CriticObservationsCfg()


@configclass
class ActionsCfg:

    joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.1,
        clip={".*": (-1.0, 1.0)},
        use_zero_offset=True,
        preserve_order=True,
    )

    lift = mdp.LiftActionCfg(
        asset_name="robot",
        # Taks 脖子执行器极弱（3Nm），吊头会把头拽脱位 → 吊在 torso_link（经 97Nm 腰关节连到
        # 骨盆，靠近竖直轴）。LiftAction 测的是“根(pelvis)高度”，吊点只决定施力位置。
        link_to_lift="torso_link",
        stiffness_forces=5000.0,
        damping_forces=500.0,
        force_limit=300.0,  # ≈1.2×体重(254N)
        damping_torques=100.0,  # 仅阻尼 yaw，防自旋
        torque_limit=250.0,
        height_sensor="height_measurement_sensor",
        target_height=taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT,
        # 有腿机器人靠腿+策略自己扶正，不开吊带扶正力矩（stiffness_torques=0，默认）；
        # 吊带由 adaptive_lift 课程随站立率逐步撤掉。
        start_lifting_time_s=3.0,
        lifting_duration_s=10.0,
    )


@configclass
class RewardsCfg:
    # 正则
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)
    joint_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-0.01, params={"soft_ratio": 0.8})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 任务：从粗到细的高度跟踪（测 pelvis 根高度）
    base_height_rough = RewTerm(
        func=mdp.base_height_exp,
        weight=2.0,
        params={
            "target_height": taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT,
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    base_height_medium = RewTerm(
        func=mdp.base_height_exp,
        weight=8.0,
        params={
            "target_height": taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT,
            "std": 0.25,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    base_height_fine = RewTerm(
        func=mdp.base_height_exp,
        weight=16.0,
        params={
            "target_height": taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT,
            "std": 0.1,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_if_standing,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "standing_height_threshold": taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "mode": "l1",
        },
    )

    # 上半身关节（双臂 + 腕 + 脖 + 腰）单独罚偏离
    joint_deviation_l1_upper_body = RewTerm(
        func=mdp.joint_deviation_if_standing,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=taks_t1.ARM_JOINT_NAMES
                + taks_t1.WRIST_JOINT_NAMES
                + taks_t1.NECK_JOINT_NAMES
                + taks_t1.WAIST_JOINT_NAMES,
            ),
            "standing_height_threshold": taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "mode": "l1",
        },
    )

    ankle_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_.*_joint")},
    )

    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"])},
    )

    not_moving = RewTerm(
        func=mdp.moving_if_standing,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "weight_lin": 1.0,
            "weight_ang": 1.0,
            "standing_height_threshold": taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    # 美观/姿态
    illegal_contacts = RewTerm(
        func=mdp.illegal_contact,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=taks_t1.TAKS_T1_UNDESIRED_CONTACTS_LINKS),
            "threshold": 1.0,
        },
    )

    feet_distance = RewTerm(
        func=mdp.feet_distance_from_ref_if_standing,
        weight=-50.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=taks_t1.TAKS_T1_FEET_LINK_NAMES),
            "ref_distance": 0.3,  # Taks 两踝默认间距 ~0.33m
            "standing_height_threshold": taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT * 0.8,
            "norm": "l1",
        },
    )

    feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean_vs_base_if_standing,
        weight=-5.0,
        params={
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=taks_t1.TAKS_T1_FEET_LINK_NAMES),
            "base_body_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "standing_height_threshold": taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT * 0.8,
        },
    )

    root_acc = RewTerm(
        func=mdp.body_acc_l2,  # type: ignore
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
    )

    no_height_progress_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-5.0,
        params={"term_keys": "no_height_progress"},
    )


@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    no_height_progress = DoneTermEx(
        func=mdp.no_height_progress,
        termination_type="bad",
        sigma=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "height_increase_threshold": 0.2,
            "time_limit_s": 10.0,
        },
    )


@configclass
class EventCfg:

    # startup 域随机化
    randomize_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 2.0),
            "operation": "scale",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_distribution_params": (0.0, 0.005),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    randomize_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "armature_distribution_params": (0.0, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_bodies_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    randomize_bodies_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.15, 0.15), "y": (-0.05, 0.05), "z": (-0.15, 0.15)},
        },
    )

    # 周期扰动
    apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    apply_external_force_torque_extremities = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wrist_.*_link", ".*_ankle_roll_link"]),
            "force_range": (-5.0, 5.0),
            "torque_range": (-0.5, 0.5),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            }
        },
    )

    # reset：从随机姿态 fallen-state 数据集采样
    reset_base = EventTerm(
        func=mdp.reset_from_fallen_dataset,
        mode="reset",
        params={
            "standing_ratio": 0.1,
        },
    )


@configclass
class CurriculumCfg:

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_standing_at_timeout,
        params={
            "min_height": taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT * 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "n_successes": 5,
            "n_failures": 5,
        },
    )

    # 站立率达标后逐步撤掉吊带辅助力（有腿机器人最终应能自己站住）
    adaptive_lift = CurrTerm(
        func=mdp.adaptive_force_decay,
        params={
            "action_name": "lift",
            "standing_height_threshold": taks_t1.TAKS_T1_DEFAULT_TRUNK_HEIGHT - 0.1,
            "threshold": 0.7,
            "ema_alpha": 0.01,
            "disable_threshold": 0.01,
        },
    )

    increase_action_rate_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate",
            "start_step": 25_000 * from_scratch,
            "num_steps": 50_000 * with_curriculum,
            "terminal_weight": -0.1,
            "use_log_space": True,
        },
    )

    increase_no_progress_penalty = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "no_height_progress_termination",
            "start_step": 60_000 * from_scratch,
            "num_steps": 60_000 * with_curriculum,
            "terminal_weight": -100,
            "use_log_space": True,
        },
    )

    increase_joint_deviation_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "joint_deviation_l1",
            "start_step": 50_000 * from_scratch,
            "num_steps": 50_000 * with_curriculum,
            "terminal_weight": -0.5,
            "use_log_space": False,
        },
    )

    increase_joint_deviation_upper_body_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "joint_deviation_l1_upper_body",
            "start_step": 50_000 * from_scratch,
            "num_steps": 50_000 * with_curriculum,
            "terminal_weight": -0.5,
            "use_log_space": False,
        },
    )


@configclass
class ViewerCfg:
    eye: tuple[float, float, float] = (0.0, -3.0, 1.5)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.7)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type = "asset_root"
    asset_name: str = "robot"
    env_index: int = 0


@configclass
class TaksT1StandUpEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    viewer: ViewerCfg = ViewerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt

        self.sim.physx.gpu_max_rigid_patch_count = 40 * 2**15

        if self.scene.height_measurement_sensor is not None:
            self.scene.height_measurement_sensor.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
