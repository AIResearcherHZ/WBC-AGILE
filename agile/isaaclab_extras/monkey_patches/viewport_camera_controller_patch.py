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

from isaaclab.envs.ui.viewport_camera_controller import ViewportCameraController

# Keep a handle to the original method so we only wrap it with error handling.
_original_update_tracking_callback = ViewportCameraController._update_tracking_callback


def _safe_update_tracking_callback(self, event):
    """Monkey patch for :meth:`ViewportCameraController._update_tracking_callback`.

    When the viewer follows an asset (``ViewerCfg.origin_type`` is ``"asset_root"`` or
    ``"asset_body"``), this callback is subscribed to the app's post-update event stream and
    runs on *every render frame*, reading the tracked asset's root pose via
    ``scene[asset_name].data.root_pos_w`` -> ``_root_physx_view.get_root_transforms()``.

    During environment teardown / ``close()``, ``reset`` with view rebuilds, or any moment the
    PhysX tensor view is (re)created, ``_root_physx_view`` becomes a *dead weak-reference*. The
    render stream can still fire this callback in that window, so the original method raises
    ``ReferenceError: weakly-referenced object no longer exists`` (or a PhysX ``RuntimeError``)
    on every frame and floods the console with identical tracebacks.

    This wrapper swallows those transient view errors and simply skips the camera update for
    that frame; tracking resumes automatically once the view is valid again. It deliberately
    does NOT catch ``ValueError`` (raised for a genuinely missing/misconfigured ``asset_name``),
    so real configuration errors still surface. Only the camera view is affected -- physics,
    stepping, and training are untouched.
    """
    try:
        _original_update_tracking_callback(self, event)
    except (ReferenceError, RuntimeError):
        # PhysX view torn down / not yet valid this frame -> skip; will recover when valid.
        pass


ViewportCameraController._update_tracking_callback = _safe_update_tracking_callback
