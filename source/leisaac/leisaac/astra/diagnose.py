"""Bounded, privileged offline diagnosis. No provider or credential imports."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import LABELS


def spawn_with_contact_reporting(prim_path: str, cfg: Any, **kwargs: Any) -> Any:
    """Add reporting only; Isaac's generic helper also writes sleep thresholds."""
    from isaaclab.sim import spawn_from_usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    prim = spawn_from_usd(prim_path, cfg, **kwargs)
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            report = PhysxSchema.PhysxContactReportAPI.Apply(child)
            report.CreateThresholdAttr().Set(0.0)
    return prim


def validate_replay_commands(actions: np.ndarray) -> None:
    if actions.ndim != 2 or actions.shape[1] != 17 or not len(actions):
        raise ValueError("replay_requires_Nx17_controls")
    if not np.isfinite(actions).all():
        raise ValueError("replay_nonfinite_controls")


def load_replay(env: Any, source: Path) -> Any:
    """Upstream dataset reader and one initial reset_to, never state playback."""
    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    reader = HDF5DatasetFileHandler()
    reader.open(str(source / "episode.hdf5"))
    try:
        episode = reader.load_episode("demo_0", env.device)
    finally:
        reader.close()
    if episode is None:
        raise ValueError("missing_demo_0")
    validate_replay_commands(episode.data["actions"].cpu().numpy())
    env.seed(int(episode.seed) if episode.seed is not None else env.cfg.seed)
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset_to(episode.data["initial_state"], None, is_relative=True)
    return episode


def transformed_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """USD/Gf row-vector homogeneous transform (not a column-vector matrix)."""
    return (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]


def bounds(points: np.ndarray) -> list:
    return [points.min(axis=0).tolist(), points.max(axis=0).tolist()]


def surface_height(
    points: np.ndarray, triangles: np.ndarray, xy: np.ndarray, ceiling: float
) -> float:
    """Highest collision-mesh triangle below a vertical diagnostic ray."""
    tri = points[triangles]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v, w, d = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2], xy - a[:, :2]
    det = v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]
    valid = np.abs(det) > 1e-12
    u = np.divide(
        d[:, 0] * w[:, 1] - d[:, 1] * w[:, 0], det, out=np.zeros(len(det)), where=valid
    )
    t = np.divide(
        v[:, 0] * d[:, 1] - v[:, 1] * d[:, 0], det, out=np.zeros(len(det)), where=valid
    )
    z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + t * (c[:, 2] - a[:, 2])
    inside = valid & (u >= -1e-7) & (t >= -1e-7) & (u + t <= 1 + 1e-7) & (z < ceiling)
    if not inside.any():
        raise ValueError("collision_support_surface_unavailable")
    return float(z[inside].max())


class Diagnostics:
    """Read-only geometry/physics samples, strictly outside policy observations."""

    def __init__(self, emb: Any):
        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd

        self.emb = emb
        self.phase = "initial"
        self.samples: list[dict] = []
        self.deadline = time.monotonic() + 165.0  # reserve shutdown within 180 s
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache()
        self.geometry: dict[str, Any] = {
            "source": "composed USD collision mesh vertices, not visual bounds",
            "contacts": "unavailable: contact-pair/force reporting not enabled in original asset",
        }
        self.body_points: dict[str, np.ndarray] = {}
        self.body_ids = {}
        self.wrist_points: dict[str, np.ndarray] = {}
        root = emb.robot.cfg.prim_path.replace(".*", "0")
        # Resolve the actual rigid-body prim rather than assuming root nesting.
        for name in ("J2_6", "J2_7", "J2_8", "base_link"):
            body = next(
                p
                for p in stage.Traverse()
                if p.GetName() == name
                and str(p.GetPath()).startswith(root)
                and p.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            inverse = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)))
            points = []
            meshes = []
            for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if p.IsA(UsdGeom.Mesh) and "collisions" in str(p.GetPath()).lower():
                    pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
                    local = transformed_points(
                        pts, np.array(cache.GetLocalToWorldTransform(p)) @ inverse
                    )
                    points.append(local)
                    meshes.append({"path": str(p.GetPath()), "vertices": len(pts)})
            if not points:
                raise RuntimeError("collision_mesh_unavailable:" + name)
            self.body_points[name] = np.vstack(points)
            self.body_ids[name] = emb.robot.find_bodies(name)[0][0]
            self.geometry[name] = {
                "meshes": meshes,
                "local_bounds": bounds(self.body_points[name]),
            }
            if name == "J2_6":
                wrist_inverse = inverse
            elif name in ("J2_7", "J2_8"):
                wrist_points = transformed_points(
                    self.body_points[name],
                    np.array(cache.GetLocalToWorldTransform(body)) @ wrist_inverse,
                )
                self.wrist_points[name] = wrist_points
                self.geometry[name]["wrist_bounds_at_authored_pose"] = bounds(
                    wrist_points
                )
                # Distal narrow gripping stem starts after the last shoulder
                # vertices wider than the tip (3 mm half-width in this asset).
                shoulder_z = float(
                    wrist_points[np.abs(wrist_points[:, 1]) > 0.003, 2].max()
                )
                distal = wrist_points[wrist_points[:, 2] > shoulder_z + 1e-5]
                self.geometry[name]["distal_region_wrist_bounds"] = bounds(distal)
        cube_pos = emb.env.scene["cube"].data.root_pos_w[0].cpu().numpy()
        table_mesh = next(
            p
            for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
            if p.IsA(UsdGeom.Mesh) and "collisions" in str(p.GetPath()).lower()
        )
        mesh = UsdGeom.Mesh(table_mesh)
        if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
            raise RuntimeError("nontriangular_table_collision")
        table_points = transformed_points(
            np.asarray(mesh.GetPointsAttr().Get()),
            np.array(cache.GetLocalToWorldTransform(table_mesh)),
        )
        self.table_points = table_points
        self.table_faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(
            -1, 3
        )
        self.geometry["table_support_z_w"] = surface_height(
            table_points, self.table_faces, cube_pos[:2], cube_pos[2]
        )
        self.geometry["collision_warnings"] = [
            "J2_8 mesh is authored as both merged and standalone collision; unchanged"
        ]
        self.object_points = {}
        for name in ("cube", "Plate"):
            prefix = emb.env.scene[name].cfg.prim_path.replace(".*", "0")
            body_prim = next(
                p
                for p in stage.Traverse()
                if str(p.GetPath()).startswith(prefix)
                and p.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            inverse = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body_prim)))
            vertices = []
            for prim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()):
                if (
                    prim.IsA(UsdGeom.Mesh)
                    and "collisions" in str(prim.GetPath()).lower()
                ):
                    vertices.append(
                        transformed_points(
                            np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()),
                            np.array(cache.GetLocalToWorldTransform(prim)) @ inverse,
                        )
                    )
            if not vertices:
                raise RuntimeError("object_collision_mesh_unavailable:" + name)
            self.object_points[name] = np.vstack(vertices)
            self.geometry[name] = {
                "collision_local_bounds": bounds(self.object_points[name]),
                "prim": str(body_prim.GetPath()),
            }
        self.local_corners = {}
        for name, pts in self.body_points.items():
            lo, hi = pts.min(0), pts.max(0)
            self.local_corners[name] = emb.torch.tensor(
                [
                    [x, y, z]
                    for x in (lo[0], hi[0])
                    for y in (lo[1], hi[1])
                    for z in (lo[2], hi[2])
                ],
                device=emb.device,
                dtype=emb.torch.float32,
            )
        self.geometry["root_pose_w"] = emb.robot.data.root_state_w[0, :7].cpu().tolist()
        self.geometry["reference_offset_wrist"] = emb.offset[0].cpu().tolist()
        self.geometry["right_base_in_root"] = emb.base[0].cpu().tolist()
        (emb.output / "geometry.json").write_text(json.dumps(self.geometry, indent=2))

    def world_to_base(self, point: Any) -> np.ndarray:
        emb = self.emb
        point = emb.torch.as_tensor(point, device=emb.device).reshape(1, 3)
        local = emb.math.quat_apply(
            emb.math.quat_inv(emb.robot.data.root_quat_w),
            point - emb.robot.data.root_pos_w,
        )
        return (local - emb.base)[0].cpu().numpy()

    def candidate(
        self, shift: float, *, center_cube: bool = False
    ) -> tuple[np.ndarray, dict]:
        if shift not in (-0.005, 0.0, 0.005):
            raise ValueError("height_candidate_not_in_approved_set")
        emb = self.emb
        intervals = [
            self.geometry[n]["distal_region_wrist_bounds"] for n in ("J2_7", "J2_8")
        ]
        lower = max(b[0][2] for b in intervals)
        upper = min(b[1][2] for b in intervals)
        pad_z = (lower + upper) / 2
        cube = emb.env.scene["cube"].data.root_pos_w[0]
        offset = emb.torch.tensor(
            [[0, 0, pad_z - float(emb.offset[0, 2])]], device=emb.device
        )
        delta = emb.math.quat_apply(emb.reset_quat, offset)[0].cpu().numpy()
        target = np.array(
            [-0.55, 0, self.world_to_base(cube)[2] - delta[2] + shift, 0, 0, 0, 1]
        )
        if center_cube:
            target[:2] = self.world_to_base(cube)[:2]
        # Predict open collision vertices at the commanded height. Geometry only,
        # before any candidate motion; reject even nominal table penetration.
        wrist = (
            emb.torch.tensor(
                target[:3], device=emb.device, dtype=emb.torch.float32
            ).reshape(1, 3)
            + emb.base
            - emb.math.quat_apply(emb.reset_quat, emb.offset)
        )
        world_p, world_q = emb.math.combine_frame_transforms(
            emb.robot.data.root_pos_w, emb.robot.data.root_quat_w, wrist, emb.reset_quat
        )
        points = emb.torch.tensor(
            np.vstack(list(self.wrist_points.values())),
            device=emb.device,
            dtype=emb.torch.float32,
        )
        world = emb.math.quat_apply(world_q.expand(len(points), -1), points) + world_p
        # A ray can land in a table groove. Use the highest collision triangle
        # overlapping the entire open-finger footprint for penetration rejection.
        world_np = world.cpu().numpy()
        tri = self.table_points[self.table_faces]
        overlap = np.all(
            tri[:, :, :2].max(1) >= world_np[:, :2].min(0), axis=1
        ) & np.all(tri[:, :, :2].min(1) <= world_np[:, :2].max(0), axis=1)
        if not overlap.any():
            raise RuntimeError("candidate_support_geometry_unavailable")
        table_top = float(tri[overlap, :, 2].max())
        clearance = float(world[:, 2].min()) - table_top
        if clearance < 0:
            raise RuntimeError("candidate_table_intersection")
        if np.any(target[:3] < emb.limits.low) or np.any(target[:3] > emb.limits.high):
            raise RuntimeError("candidate_workspace_reject")
        return target, {
            "pad_interval_wrist_z": [lower, upper],
            "pad_center_wrist_z": pad_z,
            "height_shift_m": shift,
            "target": target.tolist(),
            "predicted_tip_table_clearance_m": clearance,
            "table_top_over_finger_footprint_w": table_top,
        }

    def __call__(self, command: Any) -> None:
        if time.monotonic() > self.deadline:
            raise RuntimeError("diagnostic_wall_limit")
        emb = self.emb
        poses = {}
        for name, index in self.body_ids.items():
            p = emb.robot.data.body_pos_w[0, index]
            q = emb.robot.data.body_quat_w[0, index]
            pts = emb.math.quat_apply(q.expand(8, -1), self.local_corners[name]) + p
            poses[name] = {
                "pose_w": [*p.cpu().tolist(), *q.cpu().tolist()],
                "collision_aabb_w": bounds(pts.cpu().numpy()),
            }
        cube = emb.env.scene["cube"].data
        self.samples.append(
            {
                "tick": emb.ticks,
                "sim_time": float(emb.env.sim.current_time),
                "phase": self.phase,
                "command_17": command[0].cpu().tolist(),
                "eef_state": emb.state().tolist(),
                "bodies": poses,
                "joint_pos": emb.robot.data.joint_pos[0, emb.right_ids + emb.finger_ids]
                .cpu()
                .tolist(),
                "joint_vel": emb.robot.data.joint_vel[0, emb.right_ids + emb.finger_ids]
                .cpu()
                .tolist(),
                "cube_state_w": cube.root_state_w[0].cpu().tolist(),
                "plate_state_w": emb.env.scene["Plate"]
                .data.root_state_w[0]
                .cpu()
                .tolist(),
                "contact_pairs": "unavailable",
                "contact_forces_N": "unavailable",
            }
        )

    def save(self) -> None:
        (self.emb.output / "physics.jsonl").write_text(
            "".join(json.dumps(sample) + "\n" for sample in self.samples)
        )


def move(emb: Any, target: np.ndarray, *, minimum_ticks: int = 1) -> None:
    """Use the pinned official move_to interpolation without building any client."""
    from inspect_robots_agent._tools import build_toolset
    from inspect_robots_agent._llm import ToolCall

    tools = build_toolset(
        emb.action_space(),
        emb.info.observation_space,
        emb.info.control_hz,
        pre_check=emb.pre_check,
    )
    call = ToolCall(
        id="offline",
        name="move_to",
        arguments=json.dumps(
            {
                "targets": dict(zip(LABELS, target.tolist())),
                "note": "Privileged scripted offline diagnosis, not an autonomous policy.",
            }
        ),
    )
    result = tools.execute(call, emb.current)
    if result.chunk is None:
        raise RuntimeError("diagnostic_move_rejected:" + str(result.error))
    emb.execute(result.chunk, emb.current, minimum_ticks=minimum_ticks)


class ContactDiagnostics:
    """Read-only PhysX contact and substep sampling, outside policy observations."""

    def __init__(self, diagnostics: Diagnostics):
        from isaacsim.core.simulation_manager import SimulationManager

        self.diagnostics = diagnostics
        self.emb = emb = diagnostics.emb
        self.samples: deque[dict] = deque(maxlen=256)
        view = SimulationManager.get_physics_sim_view()
        root = emb.robot.cfg.prim_path.replace(".*", "0")
        cube = diagnostics.geometry["cube"]["prim"]
        self.contact = view.create_rigid_contact_view(
            cube,
            [
                root + "/x_trainer_asm_0226_SLDASM/" + n
                for n in ("J2_7", "J2_8", "base_link")
            ],
            max_contact_data_count=256,
        )
        properties = {}
        for name, rigid in (("robot", emb.robot), ("cube", emb.env.scene["cube"])):
            physx = rigid.root_physx_view
            properties[name] = {
                "contact_offsets": physx.get_contact_offsets().cpu().tolist(),
                "rest_offsets": physx.get_rest_offsets().cpu().tolist(),
                "materials": physx.get_material_properties().cpu().tolist(),
            }
        properties["sensor_paths"] = self.contact.sensor_paths
        properties["filter_paths"] = self.contact.filter_paths
        (emb.output / "contact_properties.json").write_text(
            json.dumps(properties, indent=2)
        )
        self.original_update = emb.env.scene.update
        emb.env.scene.update = self.update

    def update(self, dt: float) -> None:
        self.original_update(dt)
        if self.diagnostics.phase != "close":
            return
        emb = self.emb
        force, points, normals, separation, counts, starts = (
            self.contact.get_contact_data(dt)
        )
        contacts = []
        for j in range(self.contact.filter_count):
            start, count = int(starts[0, j]), int(counts[0, j])
            sl = slice(start, start + count)
            contacts.append(
                {
                    "filter": j,
                    "count": count,
                    "normal_force_N": force[sl].cpu().tolist(),
                    "points_w": points[sl].cpu().tolist(),
                    "normals_w": normals[sl].cpu().tolist(),
                    "separation_m": separation[sl].cpu().tolist(),
                }
            )
        self.samples.append(
            {
                "sim_time": float(emb.env.sim.current_time),
                "cube_state": emb.env.scene["cube"].data.root_state_w[0].cpu().tolist(),
                "joint_pos": emb.robot.data.joint_pos[0, emb.right_ids + emb.finger_ids]
                .cpu()
                .tolist(),
                "joint_vel": emb.robot.data.joint_vel[0, emb.right_ids + emb.finger_ids]
                .cpu()
                .tolist(),
                "net_contact_force_N": self.contact.get_net_contact_forces(dt)
                .cpu()
                .tolist(),
                "contacts": contacts,
            }
        )

    def close(self) -> None:
        self.emb.env.scene.update = self.original_update
        (self.emb.output / "contact_substeps.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.samples)
        )


def run_diagnosis(
    emb: Any,
    episode: Any,
    source: Path,
    case: str,
    *,
    height_shift: float = 0.0,
    slow_close: bool = False,
    aperture: float = 0.0,
    center_cube: bool = False,
) -> dict:
    if not np.isfinite(aperture) or not 0 <= aperture <= 1:
        raise ValueError("invalid_diagnostic_aperture")
    diagnostics = Diagnostics(emb)
    contacts = ContactDiagnostics(diagnostics) if case == "contact" else None
    emb.diagnostic_hook = diagnostics
    result: dict[str, Any] = {
        "case": case,
        "privileged_scripted_baseline": True,
        "provider_requests": 0,
        "commanded_aperture": aperture,
        "center_cube": center_cube,
        "raw_replay_only_initial_state_restored": True,
        "source_sha256": hashlib.sha256(
            (source / "episode.hdf5").read_bytes()
        ).hexdigest(),
    }
    try:
        if case == "raw":
            emb.observe("initial")
            actions = episode.data["actions"]
            for index, command in enumerate(actions):
                diagnostics.phase = (
                    "recorded_close"
                    if float(command[-1]) > 0.001
                    else "recorded_approach"
                )
                emb.step_command(command.reshape(1, 17))
                if index == len(actions) - 121:
                    emb.diagnostic_capture("before_terminal_settle")
            tail = diagnostics.samples[-120:]
            stable = [
                max(abs(v) for v in sample["joint_vel"]) < 0.02
                and np.linalg.norm(sample["cube_state_w"][7:10]) < 0.01
                and np.linalg.norm(sample["cube_state_w"][10:13]) < 0.1
                for sample in tail
            ]
            result["terminal_stable_ticks"] = int(sum(stable))
            result["outcome"] = (
                "recorded_unsettled_close"
                if not any(
                    all(stable[i : i + 10]) for i in range(max(len(stable) - 9, 0))
                )
                else "recorded_settled"
            )
        else:
            emb.initialize_settled_scene()
            emb.observe("initial")
            candidate = None
            if case in ("height", "contact"):
                candidate, result["candidate"] = diagnostics.candidate(
                    height_shift, center_cube=center_cube
                )
            targets: list[tuple[float, float, float, float]] = [
                (-0.345, 0.0425, 0.2557, 1),
                (-0.55, -0.1, 0.2, 1),
                (-0.55, -0.16, 0.12, 1),
                (-0.55, -0.045, 0.12, 1),
                (-0.55, 0, 0.075, 1),
                (-0.55, 0, 0.075, 0),
            ]
            if candidate is not None:
                x, y, z = map(float, candidate[:3])
                targets[-2] = (x, y, z, 1)
                targets[-1] = (x, y, z, 0)
            for index, (x, y, z, grip) in enumerate(targets):
                closing = grip == 0
                diagnostics.phase = "close" if closing else f"approach_{index + 1}"
                if closing:
                    grip = aperture
                move(
                    emb,
                    np.array([x, y, z, 0, 0, 0, grip]),
                    minimum_ticks=2 if slow_close and closing else 1,
                )
            result["outcome"] = (
                "cartesian_settled" if case == "cartesian" else "candidate_settled"
            )
    except RuntimeError as exc:
        result.update(
            outcome="execution_failure",
            failure=str(exc),
            failure_phase=diagnostics.phase,
        )
    finally:
        if contacts is not None:
            contacts.close()
        diagnostics.save()
        emb.diagnostic_hook = None
        result["terminal_capture"] = emb.diagnostic_capture("terminal_diagnostic")
    return result
