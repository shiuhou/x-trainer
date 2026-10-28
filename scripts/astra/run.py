"""Run upstream X-Trainer or one bounded official-Astra episode; simulation only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("original", "smoke", "connect", "mission", "diagnose"),
        default="smoke",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-source", type=Path)
    parser.add_argument(
        "--diagnostic-case", choices=("raw", "cartesian", "height", "contact", "full")
    )
    parser.add_argument("--diagnostic-aperture", type=float, default=0.0)
    parser.add_argument("--center-cube", action="store_true")
    parser.add_argument("--vertical-approach", action="store_true")
    parser.add_argument(
        "--height-shift", type=float, choices=(-0.005, 0.0, 0.005), default=0.0
    )
    parser.add_argument("--slow-close", action="store_true")
    parser.add_argument("--provider-runtime", type=Path, default=Path("/opt/shared-s5"))
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument(
        "--provider-proxy",
        default="auto",
        help="auto, direct, or a local HTTP(S) proxy URL",
    )
    parser.add_argument("--connect-result", type=Path)
    parser.add_argument("--model", choices=("gpt-6-astra",), default="gpt-6-astra")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher

    args.enable_cameras = True
    app = AppLauncher(args).app
    env = None
    policy = None
    result: dict = {
        "status": "FAIL",
        "mode": args.mode,
        "provider_requests": 0,
        "decisions": 0,
    }
    started = time.monotonic()
    try:
        import torch
        import numpy as np
        from isaaclab.envs import ManagerBasedRLEnv
        from leisaac.astra.task import configuration
        from leisaac.astra.embodiment import XTrainerEmbodiment
        from inspect_robots import Action, ActionChunk
        from inspect_robots.scene import Scene

        cfg = configuration(
            None if args.mode == "original" else str(args.output),
            original=args.mode == "original",
        )
        if args.mode == "diagnose" and args.diagnostic_case == "contact":
            from leisaac.astra.diagnose import spawn_with_contact_reporting

            # Enable reporting only: preserve all collider and dynamics properties.
            cfg.scene.robot.spawn.func = spawn_with_contact_reporting
            cfg.scene.scene.spawn.func = spawn_with_contact_reporting
        result["simulation_configuration"] = {
            "physics_dt": cfg.sim.dt,
            "decimation": cfg.decimation,
            "render_interval": cfg.sim.render_interval,
            "solver_type": cfg.sim.physx.solver_type,
        }
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset(seed=42)
        if args.mode == "original":
            for _ in range(60):
                env.step(
                    torch.zeros(
                        (1, env.action_manager.total_action_dim), device=env.device
                    )
                )
            from PIL import Image

            for name in ("left_wrist", "right_wrist", "top"):
                im = env.scene[name].data.output["rgb"][0, ..., :3].cpu().numpy()
                Image.fromarray(im).save(args.output / f"{name}.png")
            result["status"] = "ORIGINAL_BOOT_PASS"
            return 0

        if args.mode == "diagnose":
            if not args.diagnostic_source or not args.diagnostic_case:
                raise ValueError("diagnostic_source_and_case_required")
            from leisaac.astra.diagnose import load_replay, run_diagnosis

            episode = load_replay(env, args.diagnostic_source)
            emb = XTrainerEmbodiment(env, args.output)
            result["diagnosis"] = run_diagnosis(
                emb,
                episode,
                args.diagnostic_source,
                args.diagnostic_case,
                height_shift=args.height_shift,
                slow_close=args.slow_close,
                aperture=args.diagnostic_aperture,
                center_cube=args.center_cube,
                vertical_approach=args.vertical_approach,
            )
            result["status"] = "DIAGNOSE_COMPLETE"
            return 0

        emb = XTrainerEmbodiment(env, args.output, open_only=args.mode == "connect")
        emb.initialize_settled_scene()
        if args.mode == "smoke":
            emb.instruction = "Offline simulator smoke test."
            obs = emb.observe("initial")
            initial = emb.state().copy()
            initial[3:6], initial[6] = 0, 1
            measurements = []
            for delta, grip in ((0.005, 1), (0.0, 0), (0.0, 1)):
                target = initial.copy()
                target[2] += delta
                target[6] = grip
                obs = emb.execute(ActionChunk([Action(target)]), obs)
                measurements.append(emb.state().copy())
            if not (
                measurements[0][2] - initial[2] > 0.003
                # Empty fingers stop before the nominal joint limit in this
                # authored assembly. Prove measured closure, not a fake
                # zero reading substituted for the command.
                and measurements[0][6] - measurements[1][6] > 0.8
                and measurements[2][6] > 0.95
            ):
                raise RuntimeError("smoke_motion_or_gripper_failure")
            result.update(
                status="SMOKE_PASS",
                eef_displacement_m=float(np.linalg.norm(emb.state()[:3] - initial[:3])),
                measured_smoke_states=[state.tolist() for state in measurements],
            )
            # Exercise official package metadata, observation serialization,
            # move_to and interpolation without any network or credential.
            sys.path.insert(0, str(args.provider_runtime.resolve()))
            from dataclasses import replace
            from leisaac.astra.contracts import Limits
            from s5_agent import make_replay_policy
            from s5_common import Budget, RunLimits

            # Exercise the exact open-only connectivity contract after the
            # independent manual gripper smoke, with measured aperture intact.
            emb.open_only = True
            emb.limits = Limits(chunk_path_m=0.01, max_waypoints=32)
            emb.info = replace(emb.info, action_space=emb.action_space())
            replay_budget = Budget(RunLimits())
            replay_policy, replay_transport = make_replay_policy(emb, replay_budget)
            try:
                replay_policy.reset(Scene(id="offline", instruction=emb.instruction))
                for _ in range(2):
                    chunk = replay_policy.act(obs)
                    replay_budget.check_steps(len(chunk.actions))
                    obs = emb.execute(chunk, obs)
                    replay_budget.steps += len(chunk.actions)
                result["offline_mock_requests"] = replay_budget.requests
            finally:
                replay_transport.close()
        else:
            if args.mode == "mission":
                if (
                    not args.connect_result
                    or json.loads(args.connect_result.read_text()).get("status")
                    != "CONNECT_PASS"
                ):
                    raise RuntimeError("mission_requires_connect_pass")
            if not args.provider_runtime or not args.credential_file:
                raise RuntimeError(
                    "explicit_provider_runtime_and_credential_file_required"
                )
            sys.path.insert(0, str(args.provider_runtime.resolve()))
            # Reuse reviewed S5 provider/credential/budget code directly. These
            # two modules do not import or construct a MuJoCo or hardware backend.
            from s5_common import (
                APPROVED_RELAY,
                Budget,
                RunLimits,
                BenchmarkRunLimits,
                parse_env_file,
                register_credentials,
                install_log_redaction,
                wall_deadline,
                select_provider_proxy,
                S5Error,
            )
            from s5_agent import SimulationIOGuard, make_live_policy, observable_tools

            credentials = parse_env_file(args.credential_file)
            register_credentials(credentials)
            install_log_redaction()
            proxy = select_provider_proxy(
                APPROVED_RELAY, {**credentials, **os.environ}, args.provider_proxy
            )
            credentials["S5_PROVIDER_HTTPS_PROXY"] = proxy or ""
            connect = args.mode == "connect"
            limits = (
                RunLimits(
                    max_llm_calls=2,
                    max_decisions=2,
                    max_episode_steps=32,
                    max_wall_time=180,
                )
                if connect
                else BenchmarkRunLimits(max_episode_steps=384)
            )
            budget = Budget(limits)
            guard = SimulationIOGuard(APPROVED_RELAY, proxy=proxy)
            instruction = (
                "Simulation connectivity check: use move_to to move the observed grasp center 0.005 m up in base Z with gripper=1; "
                "after the new observation, move 0.005 m down with gripper=1. Keep X/Y and orientation unchanged."
                if connect
                else "Pick up the cube and place it on the plate using the right arm. Release it, then return the right arm "
                "to its initial rest pose with the gripper open. Use the camera views to locate objects."
            )
            emb.instruction = instruction
            obs = emb.observe("initial")
            initial_pos = obs.state["eef_state"][:3].copy()
            policy, transport = make_live_policy(
                emb,
                budget,
                args.model,
                credentials,
                guard=guard,
                base_url=APPROVED_RELAY,
            )
            from provider_trace import PhaseTraceTransport

            phases = PhaseTraceTransport(transport.inner)
            transport.inner = phases
            scene = Scene(id="xtrainer-isaac-" + args.mode, instruction=instruction)
            policy.reset(scene)
            guard.install()
            # The shared bounded transport raises non-httpx errors on failures,
            # preventing the official Responses retry loop from re-transmitting.
            decisions = []
            moved = False
            try:
                with wall_deadline(limits.max_wall_time):
                    for index in range(limits.max_decisions):
                        budget.remaining()
                        chunk = policy.act(obs)
                        result["decisions"] += 1
                        if any(
                            action.meta.get("request_stop") for action in chunk.actions
                        ):
                            result["policy_stop"] = {
                                "request_budget_exhausted": budget.requests
                                >= limits.max_llm_calls,
                                "pre_check_rejections": emb.pre_check_rejections,
                            }
                            raise S5Error(
                                "policy_output",
                                "policy_stopped_before_evaluator_success",
                            )
                        budget.check_steps(len(chunk.actions))
                        before = obs.state["eef_state"][:3].copy()
                        obs = emb.execute(chunk, obs)
                        budget.steps += len(chunk.actions)
                        moved |= bool(
                            np.linalg.norm(obs.state["eef_state"][:3] - before) > 0.001
                        )
                        metrics = emb.metrics()
                        decisions.append(
                            {
                                "decision": index + 1,
                                "generation": emb.generation,
                                "evaluator_only": metrics,
                            }
                        )
                        if not connect and metrics["success"]:
                            break
                    result["status"] = (
                        "CONNECT_PASS"
                        if connect
                        and moved
                        and emb.generation >= 2
                        and transport.transmitted_requests > 0
                        else "MISSION_PASS"
                        if not connect and emb.metrics()["success"]
                        else "PARTIAL"
                    )
            finally:
                result.update(
                    provider_requests=transport.transmitted_requests,
                    request_attempts=budget.requests,
                    usage=transport.events,
                    transport_phases=phases.events,
                    decisions_trace=decisions,
                    model=args.model,
                    model_seconds=transport.seconds,
                    waypoint_count=emb.step_count,
                    initial_position=initial_pos.tolist(),
                    tools=observable_tools(policy),
                    blocked_io=guard.blocked,
                )
        result.update(
            evaluator=emb.metrics(),
            timing=emb.timing,
            waypoint_count=emb.step_count,
            frame_generations=[x["generation"] for x in emb.observation_trace],
        )
        # Capture failure/success state without advancing physics or inventing a generation.
        for filename, records in (
            ("observations.jsonl", emb.observation_trace),
            ("actions.jsonl", emb.action_trace),
        ):
            (args.output / filename).write_text(
                "".join(json.dumps(row) + "\n" for row in records)
            )
        return 0 if result["status"].endswith("PASS") else 1
    except Exception as exc:
        # Fixed/type-only diagnostics on live paths; provider errors are already
        # sanitized by the reused runtime. No arbitrary exception chain or env dump.
        if args.mode in ("connect", "mission"):
            from s5_common import redact_text

            result["error"] = redact_text(getattr(exc, "detail", str(exc)))
            result["category"] = getattr(exc, "category", "adapter_or_execution")
        else:
            import traceback

            traceback.print_exc()
            result["error"] = str(exc)
        return 1
    finally:
        if "emb" in locals():
            result.update(
                timing=emb.timing,
                waypoint_count=emb.step_count,
                frame_generations=[x["generation"] for x in emb.observation_trace],
            )
            try:
                result["evaluator"] = emb.metrics()
                if "terminal_capture" in result.get("diagnosis", {}):
                    result["final_capture"] = result["diagnosis"]["terminal_capture"]
                else:
                    result["final_capture"] = emb.diagnostic_capture(
                        "failure_diagnostic"
                        if result["status"] == "FAIL"
                        else "final_diagnostic"
                    )
            except Exception:
                result["final_capture"] = "unavailable"
            for filename, records in (
                ("observations.jsonl", emb.observation_trace),
                ("actions.jsonl", emb.action_trace),
            ):
                (args.output / filename).write_text(
                    "".join(json.dumps(row) + "\n" for row in records)
                )
        if "transport" in locals():
            transport.close()
        if env is not None:
            try:
                env.recorder_manager.export_episodes()
                import h5py

                dataset = args.output / "episode.hdf5"
                if dataset.exists():
                    with h5py.File(dataset, "r") as record:
                        result["recording_episodes"] = list(
                            record.get("data", {}).keys()
                        )
                        result["recording_steps"] = sum(
                            len(episode["actions"])
                            for episode in record["data"].values()
                            if "actions" in episode
                        )
                        if result["recording_steps"] < 1:
                            raise RuntimeError("recording_empty")
                elif args.mode != "original":
                    raise RuntimeError("recording_missing")
            except Exception as exc:
                result["status"] = "FAIL"
                result["recording_error"] = type(exc).__name__
            finally:
                env.close()
        result["total_seconds"] = time.monotonic() - started
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
