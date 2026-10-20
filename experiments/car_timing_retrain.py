"""Retrain the round-3 camera policy (final_d1) at the DeepRacer Evo's timing.

Watch Policy/mean_std: it sat at its 1.0 cap in all prior camera runs; a separate
investigation follows.
"""

# Help, dry run and CPU smoke: uv run python -m experiments.car_timing_retrain -h
# Everything not named below is final_d1's (best_camera_completion.ipynb): network,
# PPO settings, reward, tracks, DR stack, 35 M env-steps, takeoff-guarded draws.
# Draw N trains with seed SEED + N (round 3 redrew with the same seed). Every draw,
# failed or not, leaves logs/<variant>_d<N>.result.json and a _launch/ snapshot of
# this module and the studies git state in its run dir. What changes, per the
# 2026-10-12 car-timing eval:
#   - decisions every 70 ms with the ADR 0005 actuation-timing DR (placeholders);
#   - motion information, variants trained one after the other:
#     car_timing_fs2 stacks 2 frames 70 ms apart; car_timing_proprio keeps one
#     frame and routes the last issued action and the speed to the actor (the
#     critic keeps final_d1's full state); car_timing_lastact is proprio with
#     the speed dropped from the actor (last action only) — deployable on the
#     Evo, which has no speed sensor. Both proprio actors train but cannot be
#     exported yet: the second ONNX input (and, for proprio, the car's speed
#     source) comes later;
#   - camera geometry DR, per env: vertical FOV 50-90 deg, tilt 5-25 deg down;
#   - a quadratic steering penalty against near-full-lock steering;
#   - gamma and GAE lambda from final_d1's horizons in seconds (3.99 s, 0.39 s).
# FOV caveat: every renderer takes one FOV per camera, so each env's view is a
# centre crop of the 90 deg render, resampled to 160x120. A 50 deg view comes from
# ~75x56 rendered pixels and is softer than the car's (which downsamples
# 640x480). Accepted for this run; revisit if the car eval shows a FOV gap.

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from deepracer_genesis.envs.rewards import deepracer, reads, reward_reads
from deepracer_genesis.experiment import (
    PPO,
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationPhysics,
    DomainRandomizationTrackAppearance,
    Evaluation,
    ExperimentSpec,
    RewardShaping,
    run,
)
from deepracer_genesis.experiment.rsl_backend import spec_to_train_cfg
from deepracer_genesis.experiment.spec import control_decimation

SEED = 0
ROOT = "runs/car_timing_retrain"
GROUP = "car_timing_retrain"

TRAIN_TRACKS = (
    "reinvent_base",
    "Oval_track",
    "Mexico_track",
    "China_track",
    "reInvent2019_track",
    "2022_summit_speedway",
    "caecer_loop",
)
HOLDOUT_TRACKS = ("Bowtie_track", "New_York_Track", "thunder_hill_open", "arctic_open")
NUM_ENVS = 128
RESOLUTION = (160, 120)
# Genesis fov is vertical. final_d1 rendered at 90 deg; the FOV DR crops down from it.
FOV_DEG = 90.0

# final_d1's PPO (round-1 HPO winner) and its 50 Hz discounting
PPO_BEST = {
    "lr": 1.5529021289627495e-4,
    "entropy_coef": 3.68356162276608e-3,
    "epochs": 5,
    "minibatches": 4,
}
DT_50HZ_S = 0.02
GAMMA_50HZ = 0.995
GAE_LAMBDA_50HZ = 0.95
# ADR 0005: gamma = exp(-T / H), so these keep final_d1's horizons in seconds
HORIZON_S = -DT_50HZ_S / math.log(GAMMA_50HZ)  # 3.99 s
LAM_HORIZON_S = -DT_50HZ_S / math.log(GAE_LAMBDA_50HZ)  # 0.39 s

# Car timing. PLACEHOLDERS until the Evo is measured (ADR 0005 catalog values).
CONTROL_DT_S = 0.07
LATENCY_MS = (30.0, 90.0)  # per-episode base command latency
LATENCY_JITTER_MS = 10.0  # +/- per decision
DECISION_DROP = 0.05  # one probability per run: the mean of the catalog's 0-0.1

# Camera geometry DR, per env. PLACEHOLDERS until the Evo camera is measured.
# final_d1's camera looked 5 deg below horizontal (cfg pitch 10 tilts the view UP
# from the URDF's 15 deg-down link), not 25 deg as the car-timing eval assumed.
# The range spans final_d1's 5 deg, the URDF/Gazebo 15 deg and steeper mounts.
CAMERA_TILT_DEG = 15.0  # nominal tilt below horizontal (the URDF mount)
PITCH_JITTER_DEG = 10.0  # +/- per env: 5-25 deg down
POS_JITTER_M = 0.005  # final_d1's mount position jitter
FOV_RANGE_DEG = (50.0, 90.0)  # per-env vertical FOV (65 deg H on the car is ~51 V)

# final_d1's reward: the built-in terms with speed pulled toward 1.6 m/s
TARGET_SPEED = 1.6
REWARD_SCALES_FINAL_D1 = {
    "progress": 10.0,
    "speed": 0.6,
    "centered": 1.0,
    "heading": 0.5,
    "steering": 0.3,
    "action_rate": 0.05,
    "off_track": 2.0,
}
# -steer^2 * dt. Full lock costs 1.0/s, 6 % of the 16/s progress reward at the
# target pace; a 0.5 command costs 0.25/s. A starting point, not a tuned value.
STEERING_SQ_SCALE = 1.0
REWARD_SCALES = {**REWARD_SCALES_FINAL_D1, "steering_sq": STEERING_SQ_SCALE}

# final_d1's budget and its takeoff-guarded draws (round 3's final cell)
FINAL_STEPS = 35_000_000
FINAL_EVALS = 8
EVAL_EVERY = FINAL_STEPS // FINAL_EVALS
FINAL_DRAWS_MAX = 3
FINAL_TAKEOFF_MIN = 4.0  # first-eval mean progress (m) below this = redraw

# Variant B's actor vector: speed / max_speed and the last issued [steer, throttle].
# The base is ClassicFeatures' blocks in order, so the critic's vector is
# numerically final_d1's "state".
CLASSIC_BLOCKS = (
    "v_forward",
    "v_lateral",
    "yaw_rate",
    "lateral",
    "heading",
    "last_action",
    "lookahead_xy",
)
PROPRIO_ROUTING = {
    "base": CLASSIC_BLOCKS,
    "actor": ("v_forward", "last_action"),
    "critic": ("*",),
}
# Variant C: does B's gain come from the last action alone, with no speed
# estimate? The Evo has no speed sensor, so this is deployable where B is not.
# Same base/critic as B (the critic still gets final_d1's full state); the
# actor drops "v_forward" and keeps only the last issued [steer, throttle].
# No genesis change is needed: Part K.3 routing already resolves any subset
# of the declared base blocks (tests/test_camera_proprio_routing.py covers
# the general mechanism), so this is a studies-only config.
LASTACT_ROUTING = {
    "base": CLASSIC_BLOCKS,
    "actor": ("last_action",),
    "critic": ("*",),
}

#: launchable variants (the spec family next to final_d1) -> what sets them apart
VARIANTS = {
    "car_timing_fs2": {"frame_stack": 2, "proprio": False, "routing": None},
    "car_timing_proprio": {"frame_stack": 1, "proprio": True, "routing": PROPRIO_ROUTING},
    "car_timing_lastact": {"frame_stack": 1, "proprio": True, "routing": LASTACT_ROUTING},
}


# reads exactly the built-in's signals, so genesis's K.5 check can verify it
@reads(*reward_reads(deepracer))
def reward_speed_target(env):
    """Built-in shaping with ``speed`` replaced by ``-|v - target_speed| * dt``.

    Args:
        env: The live DeepRacerEnv; ``target_speed`` comes from its reward params.

    Returns:
        The built-in term dict with the ``speed`` term rewritten.
    """
    terms = deepracer(env)
    target = env.reward_params["target_speed"]
    terms["speed"] = -(env.v_forward - target).abs() * env.dt
    return terms


def car_timing_spec(
    variant: str,
    *,
    draw: str | None = None,
    seed: int = SEED,
    backend: str = "gpu",
    num_envs: int = NUM_ENVS,
    tracks: Sequence[str] = TRAIN_TRACKS,
    holdout: Sequence[str] = HOLDOUT_TRACKS,
    steps: int = FINAL_STEPS,
    eval_every: int = EVAL_EVERY,
) -> ExperimentSpec:
    """Build one variant's validated spec at the car's timing.

    Per-env FOV is a centre crop of the 90 deg render, softer than the car's view.

    Args:
        variant: A :data:`VARIANTS` key.
        draw: Run-dir variant name of a takeoff draw (defaults to ``variant``).
        seed: The spec seed (draw N of the launcher uses ``SEED + N``).
        backend: ``"gpu"`` (Madrona) or ``"cpu"`` (rasterizer, smoke only).
        num_envs: Parallel envs.
        tracks: Training tracks.
        holdout: Tracks for the post-training no-DR holdout eval.
        steps: Total env-steps (policy decisions x envs).
        eval_every: Periodic-eval cadence in env-steps.

    Returns:
        The validated frozen spec.
    """
    frame_stack = VARIANTS[variant]["frame_stack"]
    proprio = VARIANTS[variant]["proprio"]
    routing = VARIANTS[variant]["routing"]
    pipe = (
        CameraEnvironment(
            render="madrona",
            backend=backend,
            resolution=RESOLUTION,
            fov=FOV_DEG,
            camera_tilt_deg=CAMERA_TILT_DEG,
            num_envs=num_envs,
            tracks=tuple(tracks),
            frame_stack=frame_stack,
            random_direction=True,
            control_dt_s=CONTROL_DT_S,
            # one stacked frame per decision; a single frame has no spacing
            frame_interval_s=CONTROL_DT_S if frame_stack >= 2 else None,
            obs_routing=routing if proprio else None,
        )
        >> DomainRandomizationTrackAppearance(strength=0.6)
        >> DomainRandomizationCamera(
            brightness=(0.7, 1.3),
            hue=0.05,
            blur=0.3,
            camera_jitter={
                "pitch_deg": PITCH_JITTER_DEG,
                "pos_m": POS_JITTER_M,
                "fov_deg": FOV_RANGE_DEG,
            },
        )
        >> DomainRandomizationPhysics()
        >> AsymmetricCameraPolicy(
            actor_keys=("camera", "obs_actor") if proprio else ("camera",),
            critic_keys=("camera", "obs_critic") if proprio else ("camera", "state"),
            distribution={"std_range": (0.1, 1.0)},
        )
        # latency_ms replaces final_d1's delay_steps=1 (genesis refuses both)
        >> DomainRandomizationActions(
            steer_noise=0.02,
            speed_noise=0.05,
            latency_ms=LATENCY_MS,
            latency_jitter_ms=LATENCY_JITTER_MS,
            decision_drop=DECISION_DROP,
        )
        >> RewardShaping(
            fn=reward_speed_target,
            scales=REWARD_SCALES,
            params={"target_speed": TARGET_SPEED},
        )
        # rollout length stays final_d1's 24 decisions (1.68 s at 70 ms)
        >> PPO(**PPO_BEST, horizon_s=HORIZON_S, lam_horizon_s=LAM_HORIZON_S)
    )
    if holdout:
        pipe = pipe >> Evaluation(real_tracks=tuple(holdout), charts=True)
    return pipe.build(
        seed=seed,
        total_env_steps=steps,
        eval_every_steps=eval_every,
        group=GROUP,
        variant=draw or variant,
    )


def describe(spec: ExperimentSpec) -> dict:
    """Summarize the timing, discounting and camera DR a spec trains with.

    Args:
        spec: A built car-timing spec.

    Returns:
        A JSON-able dict (spec id, decimation, gamma, lambda, DR ranges).
    """
    ppo = spec.algorithm.ppo
    algorithm = spec_to_train_cfg(spec)["algorithm"]  # what the PPO runner gets
    jitter = spec.obs_dr.camera_jitter
    tilt = spec.env.camera_tilt_deg
    return {
        "variant": spec.variant,
        "spec_id": spec.id(),
        "run_dir": spec.run_dir(ROOT),
        "frame_stack": spec.env.frame_stack,
        "actor_keys": spec.policy.actor_keys,
        "critic_keys": spec.policy.critic_keys,
        "obs_routing": spec.env.obs_routing,
        "control_dt_s": spec.env.control_dt_s,
        "decimation": control_decimation(spec.env.control_dt_s),
        "frame_interval_s": spec.env.frame_interval_s,
        "gamma": algorithm["gamma"],
        "gae_lambda": algorithm["lam"],
        "horizon_s": ppo["horizon_s"],
        "lam_horizon_s": ppo["lam_horizon_s"],
        "rollout_s": ppo["horizon"] * spec.env.control_dt_s,
        "latency_ms": spec.action_dr.latency_ms,
        "latency_jitter_ms": spec.action_dr.latency_jitter_ms,
        "decision_drop": spec.action_dr.decision_drop,
        "fov_deg_range": jitter["fov_deg"],
        "tilt_deg_down_range": (tilt - jitter["pitch_deg"], tilt + jitter["pitch_deg"]),
        "reward_scales": spec.env.reward_scales,
        "total_env_steps": spec.total_env_steps,
        "eval_every_steps": spec.eval_every_steps,
    }


class TakeoffFailed(Exception):
    """A draw whose first periodic eval shows no takeoff.

    Attributes:
        frames: Env-steps trained at that eval.
        metrics: The eval's metrics.
    """

    def __init__(self, message: str, frames: int, metrics: dict) -> None:
        super().__init__(message)
        self.frames = frames
        self.metrics = metrics


STUDIES_ROOT = Path(__file__).resolve().parents[1]


def _record_path(root: str, name: str) -> Path:
    return Path(root) / "logs" / f"{name}.result.json"


def _git(repo: Path, *args: str) -> str:
    """Run a read-only git command (no optional locks, so the index is untouched)."""
    out = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout if out.returncode == 0 else f"git {args} failed: {out.stderr}"


def snapshot_launch(run_dir: str, repo: Path | None = None) -> Path:
    """Copy this module and the studies git state into ``<run_dir>/_launch``.

    rsl-rl snapshots only the genesis checkout, and this module may be uncommitted.

    Args:
        run_dir: The draw's run directory (created if missing).
        repo: The deepracer-studies checkout to describe (default: this one).

    Returns:
        The snapshot directory.
    """
    repo = repo or STUDIES_ROOT
    out = Path(run_dir) / "_launch"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), out / Path(__file__).name)
    (out / "studies_head.txt").write_text(_git(repo, "rev-parse", "HEAD"))
    (out / "studies_diff_head.patch").write_text(_git(repo, "diff", "HEAD"))
    (out / "studies_status.txt").write_text(_git(repo, "status", "--porcelain"))
    (out / "argv.json").write_text(json.dumps(sys.argv))
    return out


def _set_aside(run_dir: str) -> None:
    """Rename a crashed draw's partial run dir so its retrain cannot overwrite it."""
    path = Path(run_dir)
    if path.exists():
        path.rename(path.with_name(f"{path.name}.crashed-{int(time.time())}"))


def guarded_run(variant: str, *, root: str = ROOT) -> dict:
    """Train ``variant`` with takeoff-guarded draws, recording every draw.

    Resumes after the last recorded draw (failed or not) and never retrains one.

    Args:
        variant: A :data:`VARIANTS` key.
        root: Runs root; draw records go to ``<root>/logs``.

    Returns:
        The successful draw's record, or ``{"error", "failed_draws"}``.
    """
    failed: list[dict] = []
    start = 0
    for draw in range(FINAL_DRAWS_MAX):
        path = _record_path(root, f"{variant}_d{draw}")
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if not record.get("takeoff_failed"):
            return record
        failed.append(record)
        start = draw + 1

    for draw in range(start, FINAL_DRAWS_MAX):
        name = f"{variant}_d{draw}"
        seed = SEED + draw
        last = draw == FINAL_DRAWS_MAX - 1

        def takeoff_guard(
            frames: int, metrics: dict, draw: int = draw, last: bool = last
        ) -> None:
            progress = metrics["mean_progress_m"]
            if not last and frames <= EVAL_EVERY and progress < FINAL_TAKEOFF_MIN:
                raise TakeoffFailed(
                    f"draw {draw}: {progress:.2f} m at {frames:,} steps "
                    f"(< {FINAL_TAKEOFF_MIN} m)",
                    frames,
                    metrics,
                )

        spec = car_timing_spec(variant, draw=name, seed=seed)
        run_dir = spec.run_dir(root)
        _set_aside(run_dir)  # a crashed earlier attempt at this draw, if any
        snapshot_launch(run_dir)
        base = {"draw": draw, "seed": seed, "spec_id": spec.id(), "run_dir": run_dir}
        t0 = time.time()
        try:
            rec = run(spec, root=root, on_eval=takeoff_guard)
        except TakeoffFailed as err:
            print(f"[car_timing] {err}; redrawing", flush=True)
            record = {
                **base,
                "takeoff_failed": True,
                "reason": str(err),
                "first_eval": {"frames": err.frames, **err.metrics},
                "wall_h": round((time.time() - t0) / 3600.0, 2),
            }
            _write_record(root, name, record)
            failed.append(record)
            continue
        result = {
            **base,
            "failed_draws": [_failure_summary(f) for f in failed],
            "metrics": rec.metrics,
            "history": rec.eval_history,
            "holdout": rec.holdout,
            "train": rec.train,
            "wall_h": round((time.time() - t0) / 3600.0, 2),
        }
        _write_record(root, name, result)
        return result
    return {
        "error": "every draw is recorded as a takeoff failure",
        "failed_draws": [_failure_summary(f) for f in failed],
    }


def _write_record(root: str, name: str, record: dict) -> None:
    path = _record_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _failure_summary(record: dict) -> dict:
    """The fields of a failed draw's record that the success record repeats."""
    first = record["first_eval"]
    return {
        "draw": record["draw"],
        "seed": record["seed"],
        "run_dir": record["run_dir"],
        "frames": first["frames"],
        "mean_progress_m": first["mean_progress_m"],
    }


def smoke(variant: str, *, steps: int, num_envs: int, root: str) -> dict:
    """Build the variant on the CPU rasterizer, step it, then train two PPO iterations.

    Args:
        variant: A :data:`VARIANTS` key.
        steps: Control steps to roll with random actions.
        num_envs: Parallel envs (small: the rasterizer walks every track tile).
        root: Scratch runs root for the two-iteration training.

    Returns:
        A JSON-able dict of the checks and their numbers.
    """
    import torch
    from deepracer_genesis.configs.cfgs import CAMERA_LINK_TILT_DEG
    from deepracer_genesis.experiment.builder import Builder
    from deepracer_genesis.seeding import seed_everything

    spec = car_timing_spec(
        variant, backend="cpu", num_envs=num_envs, holdout=(), eval_every=0
    )
    seed_everything(spec.seed)
    sim = Builder(spec).sim()
    obs = sim.get_observations()
    generator = torch.Generator().manual_seed(0)
    rewards, issued_vs_applied = [], 0
    t0 = time.perf_counter()
    for _ in range(steps):
        action = torch.rand(sim.num_envs, 2, generator=generator) * 2 - 1
        obs, reward, _done, _extras = sim.step(action.to(sim.device))
        rewards.append(reward.clone())
        issued_vs_applied += int((sim.actions != sim.applied_actions).any())
    rollout_s = time.perf_counter() - t0
    reward_t = torch.stack(rewards)
    renderer = sim.renderer
    offsets = renderer.cam._attached_offset_T
    # genesis renderers.camera_offset_T(p) has R[0, 1] = -sin(p), R[2, 1] = cos(p)
    pitch = torch.rad2deg(torch.atan2(-offsets[:, 0, 1], offsets[:, 2, 1]))
    out = {
        "spec_id": spec.id(),
        "obs_shapes": {key: list(obs[key].shape) for key in sorted(obs.keys())},
        "steps": steps,
        "env_steps_per_s": round(steps * sim.num_envs / rollout_s, 1),
        "reward_finite": bool(torch.isfinite(reward_t).all()),
        "reward_mean": float(reward_t.mean()),
        "steps_with_issued_ne_applied": issued_vs_applied,
        "decimation": sim.cfg["sim"]["decimation"],
        "dt": sim.dt,
        "frame_stride": sim.frame_stride,
        "fov_deg_per_env": [round(float(v), 2) for v in renderer.fov_deg],
        "tilt_deg_down_per_env": [
            round(CAMERA_LINK_TILT_DEG - float(v), 2) for v in pitch
        ],
        "reward_terms_logged": sorted(sim.episode_sums),
    }
    if "obs_actor" in obs.keys():  # noqa: SIM118 (TensorDict refuses `in`)
        actor_blocks = VARIANTS[variant]["routing"]["actor"]
        if "v_forward" in actor_blocks:
            # the routed actor vector is [v_forward / max_speed, last issued command]
            speed = sim.v_forward / sim.cfg["action"]["max_speed"]
            expected = torch.cat([speed[:, None], sim.actions], dim=1)
            out["obs_actor_is_speed_and_last_action"] = bool(
                torch.allclose(obs["obs_actor"], expected)
            )
        else:
            # variant C: the routed actor vector is the last issued command only
            out["obs_actor_is_last_action_only"] = bool(
                torch.allclose(obs["obs_actor"], sim.actions)
            )
    t0 = time.perf_counter()
    record = run(
        car_timing_spec(
            variant,
            backend="cpu",
            num_envs=num_envs,
            holdout=(),
            steps=num_envs * 24 * 2,
            eval_every=0,
        ),
        root=root,
    )
    out["train_2_iters_s"] = round(time.perf_counter() - t0, 1)
    out["train_metrics_finite"] = all(
        math.isfinite(v) or math.isnan(v)
        for v in record.metrics.values()
        if isinstance(v, float)
    )
    out["train_record_gamma_lam"] = [
        record.spec["algorithm"]["ppo"]["horizon_s"],
        record.spec["algorithm"]["ppo"]["lam_horizon_s"],
    ]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI: launch a variant, print its specs (dry run), or CPU-smoke it.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m experiments.car_timing_retrain", description=__doc__
    )
    parser.add_argument("variants", nargs="+", choices=[*VARIANTS, "all"])
    parser.add_argument("--root", default=ROOT, help="runs root (repo-relative)")
    parser.add_argument("--dry-run", action="store_true", help="build and print specs")
    parser.add_argument("--smoke", action="store_true", help="CPU build + step + train")
    parser.add_argument("--smoke-steps", type=int, default=300)
    parser.add_argument("--smoke-envs", type=int, default=4)
    args = parser.parse_args(argv)
    variants = list(VARIANTS) if "all" in args.variants else args.variants

    if args.dry_run:
        for variant in variants:
            print(json.dumps(describe(car_timing_spec(variant)), indent=2))
        return 0
    if args.smoke:
        for variant in variants:
            summary = smoke(
                variant,
                steps=args.smoke_steps,
                num_envs=args.smoke_envs,
                root=os.path.join(args.root, "smoke"),
            )
            print(json.dumps(summary, indent=2))
        return 0

    # survive a shared GPU, as the camera notebooks do
    os.environ.setdefault("MADRONA_MWGPU_DEVICE_HEAP_SIZE", str(1 << 30))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for variant in variants:
        result = guarded_run(variant, root=args.root)
        print(json.dumps({k: v for k, v in result.items() if k != "history"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
