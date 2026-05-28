from isaaclab.utils import configclass

from agile.rl_env.mdp.events import FallenStateDatasetCfg
from agile.rl_env.mdp.symmetry import lr_mirror_TAKS_T1  # noqa: F401
from agile.rl_env.rsl_rl import (  # noqa: F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRewardNormalizationCfg,
    RslRlSymmetryCfg,
)


@configclass
class SemiTaksT1StandUpPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 100_000
    save_interval = 250
    experiment_name = "stand_up_semi_taks_t1"
    run_name = "stand_up_semi_taks_t1"
    wandb_project = "StandUp-SemiTaksT1"
    empirical_normalization = False
    enable_entropy_coef_annealing = False
    entropy_coef_annealing_start_progress = 0.2
    enable_entropy_coef_annealing_success_rate = 0.9

    fallen_state_dataset_cfg: FallenStateDatasetCfg | None = FallenStateDatasetCfg(
        spawn_orientation="random",
        spawn_joint_mode="random",
        spawn_height_offset=1.0,
    )

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=lr_mirror_TAKS_T1,
        ),
        reward_normalization_cfg=RslRlRewardNormalizationCfg(
            decay=0.999,
            epsilon=1e-2,
        ),
    )
