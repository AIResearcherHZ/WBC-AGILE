"""训练前 fallen-state 数据集采集钩子（结构与 stand_up/t1/pre_learn.py 一致）。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agile.rl_env.mdp.events import (
    FallenStateDataset,
    FallenStateDatasetCfg,
    compute_fallen_state_cache_key,
    get_fallen_state_cache_path,
    reset_from_fallen_dataset,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pre_learn(env: ManagerBasedRLEnv, task_name: str, agent_cfg) -> None:
    dataset_cfg: FallenStateDatasetCfg | None = getattr(agent_cfg, "fallen_state_dataset_cfg", None)
    if dataset_cfg is None:
        return

    reset_event = _get_reset_event(env)

    dataset = FallenStateDataset(cfg=dataset_cfg)
    _populate_dataset(env, task_name, dataset, dataset_cfg, label="primary")
    reset_event.set_dataset(dataset)

    # 可选次级数据集（不同朝向，配合 curriculum 的 random_fallen_ratio）
    secondary_cfg: FallenStateDatasetCfg | None = getattr(
        agent_cfg, "fallen_state_dataset_secondary_cfg", None
    )
    if secondary_cfg is not None:
        secondary = FallenStateDataset(cfg=secondary_cfg)
        _populate_dataset(env, task_name + "_secondary", secondary, secondary_cfg, label="secondary")
        reset_event.set_secondary_dataset(secondary)


def _populate_dataset(
    env: ManagerBasedRLEnv,
    task_name: str,
    dataset: FallenStateDataset,
    dataset_cfg: FallenStateDatasetCfg,
    label: str,
) -> None:
    if dataset_cfg.cache_enabled:
        cache_path = _get_cache_path(env, task_name, dataset_cfg)
        if dataset.load(cache_path):
            logger.info(f"Loaded {label} fallen state dataset from cache: {cache_path}")
            return
        logger.info(f"No valid cache for {label} dataset at {cache_path}; collecting fresh states.")

    logger.info(f"Starting {label} fallen state collection (orientation={dataset_cfg.spawn_orientation})...")
    dataset.collect(env, verbose=True)

    if dataset_cfg.cache_enabled:
        cache_path = _get_cache_path(env, task_name, dataset_cfg)
        dataset.save(cache_path)
        logger.info(f"Saved {label} fallen state dataset to cache: {cache_path}")


def _get_reset_event(env: ManagerBasedRLEnv) -> reset_from_fallen_dataset:
    event_manager = env.event_manager

    if "reset" in event_manager.active_terms:
        for term_name in event_manager.active_terms["reset"]:
            term_cfg = event_manager.get_term_cfg(term_name)
            if isinstance(term_cfg.func, reset_from_fallen_dataset):
                return term_cfg.func

    raise AssertionError(
        "No reset_from_fallen_dataset event term found in environment. "
        "Add EventTerm(func=mdp.reset_from_fallen_dataset, ...) to your EventsCfg."
    )


def _get_cache_path(env: ManagerBasedRLEnv, task_name: str, dataset_cfg: FallenStateDatasetCfg) -> str:
    terrain_cfg = None
    if hasattr(env.scene, "terrain") and env.scene.terrain is not None:
        terrain_gen = env.scene.terrain.cfg.terrain_generator
        if terrain_gen is not None:
            terrain_cfg = _serialize_terrain_config(terrain_gen)

    cache_key = compute_fallen_state_cache_key(task_name, terrain_cfg)
    return get_fallen_state_cache_path(dataset_cfg.cache_dir, cache_key)


def _serialize_terrain_config(terrain_gen) -> dict:
    terrain_dict = terrain_gen.to_dict()

    for key in ("class_type", "use_cache", "cache_dir"):
        terrain_dict.pop(key, None)

    if "sub_terrains" in terrain_dict and terrain_dict["sub_terrains"]:
        for sub_terrain_cfg in terrain_dict["sub_terrains"].values():
            if isinstance(sub_terrain_cfg, dict):
                sub_terrain_cfg.pop("class_type", None)
                sub_terrain_cfg.pop("function", None)

    return terrain_dict
