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

FILE_DIR = pathlib.Path(__file__).parent
REPO_DIR = FILE_DIR.parent.parent.parent

from_scratch = 1.0
with_curriculum = 1.0


@configclass
class SceneCfg(InteractiveSceneCfg):
    # 半身机器人无脚，平地即可
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
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

    robot = taks_t1.SEMI_TAKS_T1_DELAYED_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # 高度传感器挂在浮动根 base_link 上
    height_measurement_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )


@configclass
class CommandsCfg:
    pass


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
        # 吊带作用在 torso_link（在 base 之上、靠近上身）：base 自然垂在其下方 → 像被
        # 拎着的木偶一样稳定竖直（若作用在 base 上，上身成倒立摆会翻成头朝下）。
        link_to_lift="torso_link",
        # 高度 PD 调温和：力上限≈1.3×体重(153N)、降低刚度，避免把它顶过冲后弹起。
        stiffness_forces=3000.0,
        damping_forces=500.0,
        force_limit=200.0,
        # 朝向扶正：把 base 的 z 轴 PD 扶到竖直（roll/pitch），并阻尼 yaw 防自旋。
        # 力作用在 torso（上方）→ base 垂在其下不会翻；扶正力矩作用在 base（righting_link）
        # → 直接把测量/奖励所用的 base 朝向扶竖直。这是无腿机器人真正“站直”的关键。
        # 关键：力矩作用在很轻的 base_link（Imin≈0.0078），显式积分下无过冲阻尼上限 c≈I/dt：
        # roll/pitch≈1.5、yaw≈2.9。之前 50/100 远超 → 每步反超速度、注入能量 → 狂抖飞。
        # 这里用“deadbeat”阻尼（最大不过冲），刚度适度（P 项 k<1250 安全），兼顾扶正力与不震荡。
        stiffness_torques=250.0,
        orientation_damping=1.5,
        righting_link="base_link",
        damping_torques=2.5,
        torque_limit=250.0,
        height_sensor="height_measurement_sensor",
        target_height=taks_t1.DEFAULT_TRUNK_HEIGHT,
        # 跌倒姿态已由 reset 数据集给定，无需等待下落；尽快起吊并快速扶正。
        start_lifting_time_s=0.5,
        lifting_duration_s=3.0,
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

    # 任务：从粗到细的高度跟踪
    base_height_rough = RewTerm(
        func=mdp.base_height_exp,
        weight=2.0,
        params={
            "target_height": taks_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    base_height_medium = RewTerm(
        func=mdp.base_height_exp,
        weight=8.0,
        params={
            "target_height": taks_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.25,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    base_height_fine = RewTerm(
        func=mdp.base_height_exp,
        weight=16.0,
        params={
            "target_height": taks_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.1,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_if_standing,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "standing_height_threshold": taks_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "mode": "l1",
        },
    )

    # 上半身关节（除腰外全部）单独罚偏离
    joint_deviation_l1_upper_body = RewTerm(
        func=mdp.joint_deviation_if_standing,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=taks_t1.ARM_JOINT_NAMES
                + taks_t1.WRIST_JOINT_NAMES
                + taks_t1.NECK_JOINT_NAMES,
            ),
            "standing_height_threshold": taks_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "mode": "l1",
        },
    )

    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
    )

    not_moving = RewTerm(
        func=mdp.moving_if_standing,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "weight_lin": 1.0,
            "weight_ang": 1.0,
            "standing_height_threshold": taks_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    illegal_contacts = RewTerm(
        func=mdp.illegal_contact,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=taks_t1.UNDESIRED_CONTACTS_LINKS),
            "threshold": 1.0,
        },
    )

    root_acc = RewTerm(
        func=mdp.body_acc_l2,  # type: ignore
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )

    # SUCCESS 信号：回合超时（走完整段）时仍保持在站立高度则给奖励，
    # 鼓励"站起来并稳住整段 15s"，而不是只在某一帧达到高度。
    standing_at_timeout = RewTerm(
        func=mdp.standing_at_timeout,
        weight=5.0,
        params={
            "min_height": taks_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )


@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 防飞/防爆护栏：被吊带弹飞过高、速度爆炸或出现 NaN 时立即结束该回合，
    # 避免飞行状态污染整批训练数据。target≈0.45m，正常过冲也远低于 1.2m。
    # 注：移除了原 no_height_progress —— 它对 10% 站立初始化的环境几乎必然触发
    # （初始就 0.45，要求再升 +0.2=0.65 不可能），且 sigma 罚 + 奖励项 -5/-100 双重惩罚，
    # 考的是吊带抬没抬而非策略学没学会；吊带已是永久托举，无需该项。
    invalid_state = DoneTerm(
        func=mdp.invalid_state,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_root_height": 1.2,
            "max_lin_vel": 10.0,
            "max_ang_vel": 40.0,
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
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
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
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.15, 0.15), "y": (-0.05, 0.05), "z": (-0.15, 0.15)},
        },
    )

    # 周期扰动
    apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    apply_external_force_torque_extremities = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*_wrist_.*_link", ".*_elbow_link", "neck_.*_link"],
            ),
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
    # 注：移除了 adaptive_lift（adaptive_force_decay）—— 它会在站立率达标后把吊带力
    # 单调衰减到 0 且永不恢复，但无腿机器人一旦撤力必塌。此处吊带为永久虚拟支撑。

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
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type = "asset_root"
    asset_name: str = "robot"
    env_index: int = 0


@configclass
class SemiTaksT1StandUpEnvCfg(ManagerBasedRLEnvCfg):
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

        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.height_measurement_sensor is not None:
            self.scene.height_measurement_sensor.update_period = self.sim.dt

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
