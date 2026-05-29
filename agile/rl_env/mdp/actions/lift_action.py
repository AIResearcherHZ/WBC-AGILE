# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import LiftActionCfg


class LiftAction(ActionTerm):
    """
    Lift action to help a bipedal robot to stand up.

    Applies external forces to lift the robot up by a simple PD law on a target height
    that increases linearly over time. Also applies angular velocity damping to prevent
    spinning.

    Use a curriculum (e.g., `remove_harness` or `adaptive_force_decay`) to reduce
    the forces over time as the robot learns to stand up on its own.
    """

    cfg: LiftActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: LiftActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        self.stiffness_forces = cfg.stiffness_forces
        self.damping_forces = cfg.damping_forces
        self._force_limit = cfg.force_limit
        self.damping_torques = cfg.damping_torques
        self._torque_limit = cfg.torque_limit
        # Roll/pitch "righting" PD (toward upright). Disabled when stiffness_torques == 0.
        self.stiffness_torques = cfg.stiffness_torques
        self.orientation_damping = cfg.orientation_damping
        self._world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device)
        # height sensor
        self._height_sensor: RayCaster = env.scene.sensors[cfg.height_sensor]

        # Override force limit based on robot weight if configured
        if cfg.force_limit_weight_fraction is not None:
            total_mass = self._asset.data.default_mass.sum(dim=1).mean().item()
            gravity = abs(env.cfg.sim.gravity[2]) if hasattr(env.cfg.sim, "gravity") else 9.81
            self._force_limit = cfg.force_limit_weight_fraction * total_mass * gravity
        # Store base force limit for curriculum scaling
        self._base_force_limit = self._force_limit
        self._lift_link_id, _ = self._asset.find_bodies(cfg.link_to_lift)
        # Link the righting torque acts on (defaults to the lifted link; must be resolved AFTER
        # _lift_link_id exists). Applying the lift force at a high link while righting a lower
        # link (the base) lets a legless body hang upright instead of toppling like an inverted
        # pendulum.
        if cfg.righting_link is not None:
            self._righting_link_id, _ = self._asset.find_bodies(cfg.righting_link)
        else:
            self._righting_link_id = self._lift_link_id
        self._is_disabled = False

        # Force application offset in body frame
        self._force_offset = torch.tensor(cfg.force_offset, device=env.device, dtype=torch.float32)

        # Resolve height command if configured (otherwise use time-based ramp)
        if cfg.height_command is not None:
            self._height_command = env.command_manager.get_term(cfg.height_command)
        else:
            self._height_command = None

        # Force scale for curriculum (1.0 = full force, 0.0 = disabled)
        self._force_scale = 1.0

        # Track max height achieved per environment during episode (for curriculum)
        self._max_heights = torch.zeros(env.num_envs, device=env.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.empty(0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.empty(0, device=self.device)

    @property
    def force_scale(self) -> float:
        """Current force scale for logging/monitoring."""
        return self._force_scale

    @property
    def max_heights(self) -> torch.Tensor:
        """Max height achieved per environment during current episode.

        Used by adaptive_force_decay (metric_type="standing_ratio") to determine if robot successfully
        stood up at any point (even if it fell afterwards).
        """
        return self._max_heights

    def scale_forces(self, scale: float) -> None:
        """Scale all force and torque parameters by the given scale.

        Called by curriculum terms to adjust the lift assistance over training.

        Args:
            scale: Scale factor in [0, 1]. 0 = disabled, 1 = full force.
        """
        self._force_scale = scale
        self.stiffness_forces = self.cfg.stiffness_forces * self._force_scale
        self.damping_forces = self.cfg.damping_forces * self._force_scale
        self._force_limit = self._base_force_limit * self._force_scale
        self.damping_torques = self.cfg.damping_torques * self._force_scale
        self.stiffness_torques = self.cfg.stiffness_torques * self._force_scale
        self.orientation_damping = self.cfg.orientation_damping * self._force_scale
        self._torque_limit = self.cfg.torque_limit * self._force_scale
        self._is_disabled = self._force_scale <= 0

    def process_actions(self, actions: torch.Tensor) -> None:
        # store the raw actions
        self._raw_actions = actions

    def _measure_height(self) -> torch.Tensor:
        """Measure the current height for PD control.

        When a height command with an offset is used, measures the offset point height
        (matching what the command tracks). Otherwise measures root height.
        """
        if self._height_command is not None:
            return self._height_command.measured_height
        else:
            height = self._asset.data.root_pos_w[:, 2].unsqueeze(1) - self._height_sensor.data.ray_hits_w[..., 2]
            return torch.mean(height, dim=-1)

    def apply_actions(self) -> None:
        # Always compute and track height (even when disabled) for curriculum
        height = self._measure_height()

        # Track max height achieved during episode
        self._max_heights = torch.maximum(self._max_heights, height)

        if self._is_disabled:
            return

        # find current desired height above ground
        if self._height_command is not None:
            target_height = self._height_command.target_height
        else:
            # Default: time-based linear ramp
            time_passed = self._env.episode_length_buf * self._env.step_dt
            ratio = torch.clamp(
                (time_passed - self.cfg.start_lifting_time_s) / self.cfg.lifting_duration_s, min=0.0, max=1.0
            )
            target_height = ratio * self.cfg.target_height

        # find the error in local frame of root
        forces = torch.zeros_like(self._asset.data.root_lin_vel_b)
        # calculate the height error
        height_error = target_height - height  # (N,)
        # PD：P 来自高度误差，D 来自 base 的 z 速度（抑制过冲，原本只算 P）
        forces[:, 2] = (
            self.stiffness_forces * height_error
            - self.damping_forces * self._asset.data.root_lin_vel_w[:, 2]
        )
        # Disable the lift assist for negative commanded heights: "lie flat" is a
        # joint-configuration task (the policy splays the legs), not a force-support
        # task, and the assist should neither pull the robot up nor push it into the ground.
        if self._height_command is not None:
            forces[:, 2] = torch.where(target_height >= 0, forces[:, 2], torch.zeros_like(forces[:, 2]))
        # limit the forces
        if self.cfg.allow_push_down:
            forces = torch.clamp(forces, -self._force_limit, self._force_limit).unsqueeze(1)
        else:
            forces = torch.clamp(forces, 0.0, self._force_limit).unsqueeze(1)

        # Angular torque (world frame), acting on the righting link:
        #   - roll/pitch: optional PD "righting" toward upright (only if stiffness_torques > 0)
        #   - yaw:        velocity damping to prevent spinning (original behavior)
        torques_w = torch.zeros_like(self._asset.data.root_ang_vel_w)
        ang_vel_w = self._asset.data.root_ang_vel_w

        if self.stiffness_torques > 0:
            # Righting: rotate the righting link's local z-axis back to world up.
            # cross(up_w, z_hat) = (up_y, -up_x, 0) -> a purely horizontal axis (z-component 0),
            # so this drives roll/pitch toward upright while leaving yaw untouched.
            n_envs = self._asset.data.root_quat_w.shape[0]
            z_hat = self._world_up.expand(n_envs, 3)
            right_quat = self._asset.data.body_quat_w[:, self._righting_link_id].squeeze(1)
            up_w = math_utils.quat_apply(right_quat, z_hat)
            righting_axis_w = torch.cross(up_w, z_hat, dim=-1)
            torques_w[:, :2] = (
                self.stiffness_torques * righting_axis_w[:, :2] - self.orientation_damping * ang_vel_w[:, :2]
            )

        if self.damping_torques > 0:
            # Damp only the z-component (yaw); allows roll/pitch when righting is disabled.
            torques_w[:, 2] = -self.damping_torques * ang_vel_w[:, 2]

        # Clamp each component to the torque limit
        torques_w = torch.clamp(torques_w, -self._torque_limit, self._torque_limit)

        composer = self._asset.permanent_wrench_composer
        if self._righting_link_id == self._lift_link_id:
            # Single body: lift force + torque on the same (lifted) link, in its local frame.
            link_quat = self._asset.data.body_quat_w[:, self._lift_link_id].squeeze(1)
            forces_b = math_utils.quat_apply_inverse(link_quat, forces)
            torques_b = math_utils.quat_apply_inverse(link_quat, torques_w.unsqueeze(1))
            positions = self._force_offset.unsqueeze(0).unsqueeze(0).expand(forces_b.shape[0], 1, 3)
            composer.set_forces_and_torques(
                forces=forces_b, torques=torques_b, positions=positions, body_ids=self._lift_link_id
            )
        else:
            # Two bodies (global frame): lift force on the lifted link, righting torque on the
            # righting link. Applied in one call so the composer sums them this step.
            n_envs = forces.shape[0]
            body_ids = list(self._lift_link_id) + list(self._righting_link_id)
            forces_g = torch.zeros(n_envs, 2, 3, device=self.device)
            forces_g[:, 0, :] = forces.squeeze(1)
            torques_g = torch.zeros(n_envs, 2, 3, device=self.device)
            torques_g[:, 1, :] = torques_w
            composer.set_forces_and_torques(forces=forces_g, torques=torques_g, body_ids=body_ids, is_global=True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset max heights for environments that are resetting
        if env_ids is None:
            self._max_heights[:] = 0.0
        else:
            self._max_heights[env_ids] = 0.0
