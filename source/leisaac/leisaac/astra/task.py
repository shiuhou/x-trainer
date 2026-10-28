"""Original X-Trainer PickCube scene with right-arm IK and three observation views."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from leisaac.tasks.pick_cube.xtrainer_pick_cube_env_cfg import PickCubeEnvCfg
from leisaac.tasks.template.xtrainer_arm_env_cfg import (
    XTrainerArmActionsCfg,
    euler_to_quat,
)

# Root-aligned right pedestal reference, NOT the shoulder-joint origin.
# USD J2_1 localPos0 is (1.06, 0, 0.2234): its joint anchor is 223.4 mm
# above this reference. Preserve the existing public coordinate convention.
RIGHT_BASE_IN_ROOT = (1.06, 0.0, 0.0)
# Same grasp-center convention as upstream pick_cube/mdp/observations.py.
# This is a reference convention, not a measured center of contact pressure.
GRASP_IN_WRIST = (0.0, 0.0, 0.16)
# Narrow distal collision-mesh region in the pinned upstream USD, metres.
# Engineering geometry only: neither a calibrated TCP nor object/task state.
GRIPPING_REGION_WRIST_Z = (0.168291953, 0.197000248)
FIXED_VIEWS = {
    "front_rgb": ((0.55, -1.0, 0.8), (0.60, 0.0, 0.15)),
    "right_rgb": ((1.3, -0.7, 0.8), (0.60, 0.0, 0.15)),
}


def camera(path: str, position: tuple, width: int, height: int) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=path,
        offset=TiledCameraCfg.OffsetCfg(pos=position, convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=36.0,
            clipping_range=(0.01, 10.0),
        ),
        width=width,
        height=height,
        update_period=0.0,
    )


class GravityCompensatedIK(DifferentialInverseKinematicsAction):
    """Isaac differential IK with native PhysX gravity feedforward.

    Keep the authored masses, gravity and drive gains. Without feedforward,
    differential IK continually references the sagged measured joint positions,
    leaving a static Cartesian error under gravity.
    """

    def apply_actions(self):
        super().apply_actions()
        gravity = self._asset.root_physx_view.get_gravity_compensation_forces()
        self._asset.set_joint_effort_target(
            gravity[:, self._joint_ids], joint_ids=self._joint_ids
        )


@configclass
class RightArmActionsCfg(XTrainerArmActionsCfg):
    left_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"J1_{i}" for i in range(1, 7)],
        use_default_offset=False,
    )
    left_gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["J1_7", "J1_8"],
        use_default_offset=False,
    )
    right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        class_type=GravityCompensatedIK,
        asset_name="robot",
        joint_names=[f"J2_{i}" for i in range(1, 7)],
        body_name="J2_6",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
    )
    right_gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["J2_7", "J2_8"],
        use_default_offset=False,
    )


@configclass
class RecordingObservationsCfg(ObservationGroupCfg):
    """Upstream recorder input only; never forwarded to the model."""

    joint_pos = ObservationTermCfg(func=mdp.joint_pos)
    joint_vel = ObservationTermCfg(func=mdp.joint_vel)

    def __post_init__(self):
        self.concatenate_terms = False
        self.enable_corruption = False


def configuration(
    output: str | None = None, *, original: bool = False
) -> PickCubeEnvCfg:
    cfg = PickCubeEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 42
    cfg.sim.device = "cuda:0"
    # Evaluate the original task explicitly; never auto-reset before evidence is captured.
    cfg.terminations.success = None
    cfg.terminations.time_out = None
    if original:
        cfg.use_teleop_device("bi_keyboard")
    else:
        cfg.actions = RightArmActionsCfg()
        for name in ("left_wrist", "right_wrist", "top", "stereo_left", "stereo_right"):
            setattr(cfg.scene, name, None)
        for name, (eye, _) in FIXED_VIEWS.items():
            setattr(cfg.scene, name, camera("{ENV_REGEX_NS}/" + name, eye, 640, 480))
        cfg.scene.wrist_rgb = camera(
            "{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6/wrist_rgb",
            (0.0, -0.065, 0.03),
            1280,
            720,
        )
        cfg.scene.wrist_rgb.offset = TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.065, 0.03), rot=euler_to_quat(-15, 0, 0), convention="ros"
        )
        # Public policy observations are assembled from sensors and measured
        # robot state by the embodiment, never from upstream privileged terms.
        cfg.observations.policy = RecordingObservationsCfg()
        cfg.observations.subtask_terms = None
    cfg.viewer.eye = (1.8, -1.8, 1.4)
    cfg.viewer.lookat = (0.6, 0.0, 0.3)
    if output:
        cfg.recorders = ActionStateRecorderManagerCfg()
        cfg.recorders.dataset_export_dir_path = output
        cfg.recorders.dataset_filename = "episode"
        cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    else:
        cfg.recorders = None
    return cfg
