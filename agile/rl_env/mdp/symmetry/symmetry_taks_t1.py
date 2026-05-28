# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0


from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from tensordict.tensordict import TensorDict

from .observations import (
    lr_mirror_base_ang_vel,
    lr_mirror_base_lin_vel,
    lr_mirror_projected_gravity,
    mirror_base_com,
    mirror_external_force_torque,
    mirror_material,
    mirror_velocity_commands,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lr_mirror_TAKS_T1(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",  # noqa: ARG001
) -> tuple[TensorDict | None, torch.Tensor | None]:
    if actions is not None:
        mirrored_actions = mirror_actions_TAKS_T1(actions, env)
        augmented_actions = torch.cat([actions, mirrored_actions], dim=0)
    else:
        augmented_actions = None

    if obs is not None:
        mirrored_obs = TensorDict(
            {name: OBS_TO_MIRROR[name](obs[name], env) for name in obs.keys()},
            batch_size=obs.batch_size,
        )
        augmented_obs = torch.cat([obs, mirrored_obs], dim=0)
    else:
        augmented_obs = None

    return augmented_obs, augmented_actions


def mirror_actions_TAKS_T1(
    actions: torch.Tensor, env: ManagerBasedRLEnv, action_term_name: str = "joint_pos"
) -> torch.Tensor:
    """Mirror the action vector along the robot's sagittal plane."""
    mirrored_indices, neg_indices = resolve_joint_names_taks_t1(
        tuple(env.unwrapped.action_manager._terms[action_term_name]._joint_names)
    )
    mirrored = actions.clone()
    mirrored[..., mirrored_indices] = actions
    mirrored[..., neg_indices] *= -1
    return mirrored


def mirror_joints_TAKS_T1(joints: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mirror a tensor that follows the full articulation joint order."""
    mirrored_indices, neg_indices = resolve_joint_names_taks_t1(
        tuple(env.unwrapped.scene.articulations["robot"].joint_names)
    )
    mirrored = joints.clone()
    mirrored[..., mirrored_indices] = joints
    mirrored[..., neg_indices] *= -1
    return mirrored


def mirror_bodies_TAKS_T1(bodies: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mirror a per-body tensor (e.g. contact forces)."""
    mirrored_indices = resolve_body_names_taks_t1(
        tuple(env.unwrapped.scene.articulations["robot"].body_names)
    )
    mirrored = bodies.clone()
    mirrored[..., mirrored_indices] = bodies
    return mirrored


@lru_cache(maxsize=10)
def resolve_joint_names_taks_t1(action_joint_names: tuple[str, ...]) -> tuple[list[int], list[int]]:
    """Resolve mirrored joint indices and sign-flipping joints.

    The mid-line joints (``waist_*``, ``neck_*``) map to themselves;
    ``waist_roll/yaw`` and ``neck_roll/yaw`` flip sign, ``*_pitch`` does not.
    """
    mirrored_indices: list[int] = []
    for source_joint_name in action_joint_names:
        if "left" in source_joint_name:
            mirrored_joint_name = source_joint_name.replace("left", "right")
        elif "right" in source_joint_name:
            mirrored_joint_name = source_joint_name.replace("right", "left")
        else:
            mirrored_joint_name = source_joint_name

        if mirrored_joint_name not in action_joint_names:
            raise ValueError(
                f"Mirrored joint name {mirrored_joint_name} not found in action joint names"
            )
        mirrored_indices.append(action_joint_names.index(mirrored_joint_name))

    neg_indices: list[int] = []
    neg_indicators = ("roll", "yaw")
    for joint_name in action_joint_names:
        if any(indicator in joint_name for indicator in neg_indicators):
            neg_indices.append(action_joint_names.index(joint_name))
    return mirrored_indices, neg_indices


@lru_cache(maxsize=10)
def resolve_body_names_taks_t1(body_names: tuple[str, ...]) -> list[int]:
    """Resolve mirrored body indices."""
    mirrored_indices: list[int] = []
    for source_body_name in body_names:
        if "left" in source_body_name:
            mirrored_body_name = source_body_name.replace("left", "right")
        elif "right" in source_body_name:
            mirrored_body_name = source_body_name.replace("right", "left")
        else:
            mirrored_body_name = source_body_name

        if mirrored_body_name not in body_names:
            raise ValueError(
                f"Mirrored body name {mirrored_body_name} not found in body names"
            )
        mirrored_indices.append(body_names.index(mirrored_body_name))
    return mirrored_indices


def mirror_actuator_gains(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    mirrored_indices, _ = resolve_joint_names_taks_t1(
        tuple(env.unwrapped.scene.articulations["robot"].joint_names)
    )
    mirrored = obs.clone()
    mirrored[..., mirrored_indices, :] = obs
    return mirrored


def mirror_joint_parameters(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:
    mirrored_indices, _ = resolve_joint_names_taks_t1(
        tuple(env.unwrapped.scene.articulations["robot"].joint_names)
    )
    mirrored = obs.clone()
    mirrored[..., mirrored_indices, :] = obs
    return mirrored


def identity(obs: torch.Tensor, env: ManagerBasedRLEnv) -> torch.Tensor:  # noqa: ARG001
    return obs


OBS_TO_MIRROR: dict[str, Callable] = {
    "projected_gravity": lr_mirror_projected_gravity,
    "base_lin_vel": lr_mirror_base_lin_vel,
    "base_ang_vel": lr_mirror_base_ang_vel,
    "joint_pos": mirror_joints_TAKS_T1,
    "joint_vel": mirror_joints_TAKS_T1,
    "actions": mirror_actions_TAKS_T1,
    "controlled_joint_pos": mirror_actions_TAKS_T1,
    "controlled_joint_vel": mirror_actions_TAKS_T1,
    "velocity_commands": mirror_velocity_commands,
    "height_commands": identity,
    "base_height": identity,
    "material": mirror_material,
    "actuator_gains": mirror_actuator_gains,
    "joint_parameters": mirror_joint_parameters,
    "joint_friction": mirror_joint_parameters,
    "joint_armature": mirror_joint_parameters,
    "external_force_torque": mirror_external_force_torque,
    "base_com": mirror_base_com,
    "base_mass": identity,
    "is_env_inactive": identity,
    "action_smoothing": mirror_actions_TAKS_T1,
    "contact_forces": mirror_bodies_TAKS_T1,
}
"""Hashmap of observation names to functions to mirror the observations."""
