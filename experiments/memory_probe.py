"""Measure one training config's peak VRAM in a fresh process, for the memory model.

Model under test: ``peak ≈ a * (T * N_env * |o| * 4 bytes) + b`` (memory_model.ipynb).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

#: rollout horizon (num_steps_per_env) in both train cfgs; the probe verifies it
HORIZON = 24
BASE_RES = (160, 120)                       # (w, h) the camera env defaults to
STATE_DIM = 28                              # ClassicFeatures, lookahead_k=10

#: guard coefficients used ONLY to skip configs that cannot fit in the
#: currently free VRAM. Originally blind guesses (2.5 / 3.5 GiB); tightened
#: to the measured fit (a=2.32, b=2.17 GiB, 2026-09-30) plus a small margin
#: so the >10 GiB points are reachable on an idle 16 GiB card.
A_GUESS = 2.4
B_GUESS = 2.4 * 2**30

#: the measurement grid. Camera points span each axis (N_env, frame stack,
#: resolution) with three DISTINCT configs sharing x ≈ 1.42 GB — if x alone
#: predicts memory, their peaks must agree. Feature points pin the
#: no-renderer intercept.
GRID = (
    [{"modality": "camera", "num_envs": n, "frame_stack": 1, "res_scale": 1}
     for n in (32, 64, 128, 256)]
    + [{"modality": "camera", "num_envs": 64, "frame_stack": k, "res_scale": 1}
       for k in (2, 4)]
    + [{"modality": "camera", "num_envs": 32, "frame_stack": 4, "res_scale": 1},
       {"modality": "camera", "num_envs": 64, "frame_stack": 2, "res_scale": 2}]
    + [{"modality": "camera", "num_envs": 128, "frame_stack": 2, "res_scale": 1},
       # the slope anchor: double the largest equal-x cluster
       {"modality": "camera", "num_envs": 256, "frame_stack": 2, "res_scale": 1},
       # the >10 GiB regime: x = 3.96 / 4.62 GiB, predicted ~11.4 / ~12.9 GiB
       # peak — near the 16 GiB card's practical ceiling (guard-protected)
       {"modality": "camera", "num_envs": 384, "frame_stack": 2, "res_scale": 1},
       {"modality": "camera", "num_envs": 448, "frame_stack": 2, "res_scale": 1},
       {"modality": "camera", "num_envs": 32, "frame_stack": 1, "res_scale": 2},
       {"modality": "camera", "num_envs": 64, "frame_stack": 1, "res_scale": 2},
       {"modality": "camera", "num_envs": 32, "frame_stack": 2, "res_scale": 2}]
    + [{"modality": "feature", "num_envs": n} for n in (1024, 4096, 8192)]
    # CNN-dataset collection (datasets.rollout): camera sim + renderer but NO
    # training allocations — the per-env cost of gathering perception data
    + [{"modality": "collect", "num_envs": n, "frame_stack": 1, "res_scale": 1}
       for n in (32, 64, 128, 256, 512, 1024, 2048, 4096)]
    # collection at 2x resolution: k never changes what is SAVED (single
    # newest frame per row; stacks are rebuilt at train time), so resolution
    # is the only collect axis that matters
    + [{"modality": "collect", "num_envs": n, "frame_stack": 1, "res_scale": 2}
       for n in (256, 512, 1024)]
)


def _camera_family(frame_stack: int, res_scale: int, envs: tuple) -> list:
    """One camera line: the same (k, resolution) across env counts."""
    return [{"modality": "camera", "num_envs": n, "frame_stack": frame_stack,
             "res_scale": res_scale} for n in envs]


#: the 96 GiB grid (Colab G4, RTX PRO 6000 Blackwell). Every line gets >=5
#: points and crosses 32 GiB: camera lines double N until the 4060 Ti fit
#: predicts ~76 GiB (x ~= 31 GiB); feature/collect lines extend along their
#: measured G4 slopes (~0.17 / ~2.4 / ~9 MiB per env) to ~60-75 GiB.
GRID_G4 = (
    _camera_family(1, 1, (256, 512, 1024, 2048, 4096, 6144))
    + _camera_family(2, 1, (128, 256, 512, 1024, 2048, 3072))
    + _camera_family(4, 1, (64, 128, 256, 512, 1024, 1536))
    + _camera_family(1, 2, (64, 128, 256, 512, 1024, 1536))
    + _camera_family(2, 2, (32, 64, 128, 256, 512, 768))
    + [{"modality": "feature", "num_envs": n}
       for n in (1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
                 393216)]
    + [{"modality": "collect", "num_envs": n, "frame_stack": 1, "res_scale": 1}
       for n in (256, 1024, 2048, 4096, 8192, 16384, 24576, 32768)]
    + [{"modality": "collect", "num_envs": n, "frame_stack": 1, "res_scale": 2}
       for n in (256, 512, 1024, 2048, 4096, 6144, 8192)]
)

#: card profile -> grid; the notebook picks one with its CARD variable
GRIDS = {"4060ti": GRID, "g4": GRID_G4}


def config_name(cfg: dict) -> str:
    """Stable result-file stem for one grid config."""
    if cfg["modality"] == "feature":
        return f"feat_N{cfg['num_envs']}"
    if cfg["modality"] == "collect":
        return f"collect_N{cfg['num_envs']}_r{cfg['res_scale']}" \
            if cfg["res_scale"] != 1 else f"collect_N{cfg['num_envs']}"
    return f"cam_N{cfg['num_envs']}_k{cfg['frame_stack']}_r{cfg['res_scale']}"


def obs_floats(cfg: dict) -> int:
    """Per-env, per-step observation floats the rollout buffer stores."""
    if cfg["modality"] == "feature":
        return STATE_DIM
    w, h = (d * cfg["res_scale"] for d in BASE_RES)
    return 3 * cfg["frame_stack"] * w * h + STATE_DIM   # camera + critic state


def predicted_bytes(cfg: dict) -> int:
    """The model's x term: rollout-buffer bytes ``T * N * |o| * 4``.

    Collection keeps no T-step rollout buffer, so its x uses T = 1.
    """
    horizon = 1 if cfg["modality"] == "collect" else HORIZON
    return horizon * cfg["num_envs"] * obs_floats(cfg) * 4


def free_vram_bytes() -> int:
    """Currently free device memory, via nvidia-smi (no pynvml dependency)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0]) * 2**20


class _ProcessPeakSampler:
    """Poll this process's device memory via nvidia-smi and keep the maximum.

    Per-PID, so other residents (e.g. ollama) never contaminate the reading.

    Attributes:
        peak_bytes: Largest sampled value so far.
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        self.peak_bytes = 0
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        pid = str(os.getpid())
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == pid and parts[1].isdigit():
                    self.peak_bytes = max(self.peak_bytes, int(parts[1]) * 2**20)
            self._stop.wait(self._interval)

    def __enter__(self) -> "_ProcessPeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def measure(cfg: dict, out_path: Path, iterations: int = 3) -> dict:
    """Build the config's sim, train a few PPO updates, record both peaks.

    Args:
        cfg: One GRID entry.
        out_path: JSON result destination (written even on failure).
        iterations: PPO updates to run; peak lands after the first.

    Returns:
        The result record that was written.
    """
    record: dict = {"config": cfg, "name": config_name(cfg),
                    "x_bytes": predicted_bytes(cfg), "ok": False}
    t0 = time.perf_counter()
    try:
        with _ProcessPeakSampler() as sampler:
            import torch
            from rsl_rl.runners import OnPolicyRunner

            from deepracer_genesis.configs.cfgs import get_train_cfg
            from deepracer_genesis.experiment import (
                PPO, AsymmetricCameraPolicy, CameraEnvironment,
                FeatureEnvironment, VectorPolicy)
            from deepracer_genesis.experiment.builder import Builder
            from deepracer_genesis.experiment.rsl_backend import spec_to_train_cfg
            from deepracer_genesis.seeding import seed_everything

            if cfg["modality"] == "collect":
                # the real perception-dataset pipeline: sim + renderer + the
                # privileged expert, PNG-encoded to parquet — no PPO anywhere
                import shutil

                from deepracer_genesis.datasets.rollout import (
                    collect_rollout_dataset)
                data_dir = str(out_path.with_suffix("")) + "_data"
                w, h = (d * cfg["res_scale"] for d in BASE_RES)
                collect_rollout_dataset(
                    CameraEnvironment(render="madrona",
                                      num_envs=cfg["num_envs"],
                                      frame_stack=cfg["frame_stack"],
                                      resolution=(w, h)),
                    out=data_dir,
                    # small shards: the collector holds a whole shard in HOST
                    # RAM — one 72-step shard OOM-killed 24k envs on a 176 GB
                    # VM; sharding is host-side, the GPU peak is unaffected
                    steps=3 * HORIZON, shard_steps=HORIZON // 3, seed=0)
                # the parquet is a measurement side-effect (GiBs at large N),
                # not a result — the JSON keeps the numbers
                shutil.rmtree(data_dir, ignore_errors=True)
            else:
                vision = cfg["modality"] == "camera"
                assert (get_train_cfg(vision=vision)["num_steps_per_env"]
                        == HORIZON)
                seed_everything(0)
                if vision:
                    w, h = (d * cfg["res_scale"] for d in BASE_RES)
                    pipe = (CameraEnvironment(render="madrona",
                                              num_envs=cfg["num_envs"],
                                              frame_stack=cfg["frame_stack"],
                                              resolution=(w, h))
                            >> AsymmetricCameraPolicy(
                                actor_keys=("camera",),
                                critic_keys=("camera", "state")))
                else:
                    pipe = (FeatureEnvironment(num_envs=cfg["num_envs"])
                            >> VectorPolicy())
                spec = (pipe >> PPO()).build()

                sim = Builder(spec).sim()
                # exact |o| from the live sim, not the module's assumption
                cam_floats = (3 * cfg["frame_stack"]
                              * spec.env.resolution[0] * spec.env.resolution[1]
                              if vision else 0)
                record["x_bytes"] = (HORIZON * cfg["num_envs"]
                                     * (cam_floats + sim.num_state_obs) * 4)

                run_dir = str(out_path.with_suffix("")) + "_logs"
                runner = OnPolicyRunner(sim, spec_to_train_cfg(spec), run_dir,
                                        device=str(sim.device))
                runner.learn(num_learning_iterations=iterations,
                             init_at_random_ep_len=True)
            torch.cuda.synchronize()
            record["torch_peak_bytes"] = torch.cuda.max_memory_allocated()
        # the sampler's last poll may have run before the true peak decayed;
        # it sampled throughout training, which is where the peak lives
        record["proc_peak_bytes"] = sampler.peak_bytes
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - an OOM config is a data point
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_s"] = round(time.perf_counter() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    return record


def sweep(out_dir: Path, grid: list | tuple = GRID,
          timeout_s: float = 1800.0) -> None:
    """Run every grid config missing a result, each in a fresh subprocess.

    Skips configs whose conservative prediction exceeds 85% of free VRAM
    (they stay pending, so a later sweep with more headroom picks them up).
    Configs run smallest-x first so a sweep cut short still yields full lines.
    """
    for cfg in sorted(grid, key=predicted_bytes):
        out = out_dir / f"{config_name(cfg)}.json"
        if out.exists():
            print(f"[sweep] {config_name(cfg)}: cached")
            continue
        need = A_GUESS * predicted_bytes(cfg) + B_GUESS
        free = free_vram_bytes()
        if need > 0.85 * free:
            print(f"[sweep] {config_name(cfg)}: SKIPPED — needs ~"
                  f"{need / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB free")
            continue
        print(f"[sweep] {config_name(cfg)}: running "
              f"(x = {predicted_bytes(cfg) / 2**30:.2f} GiB)", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__),
                 "--config", json.dumps(cfg), "--out", str(out)],
                timeout=timeout_s)
            reason = (f"process killed (exit {proc.returncode}) — "
                      "likely OOM / Madrona heap")
        except subprocess.TimeoutExpired:    # a hung build must not stall the grid
            reason = f"timed out after {timeout_s} s"
        if not out.exists():                 # process died before writing
            out.write_text(json.dumps({
                "config": cfg, "name": config_name(cfg),
                "x_bytes": predicted_bytes(cfg), "ok": False,
                "error": reason}))
        status = json.loads(out.read_text())
        peak = status.get("proc_peak_bytes", 0) / 2**30
        print(f"[sweep] {config_name(cfg)}: "
              + (f"peak {peak:.2f} GiB in {status.get('wall_s', '?')} s"
                 if status["ok"] else f"FAILED — {status.get('error')}"))


def main(argv=None) -> None:
    """CLI: ``--sweep DIR [--grid g4]``, or ``--config JSON --out FILE``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=None)
    parser.add_argument("--grid", choices=sorted(GRIDS), default="4060ti")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.sweep is not None:
        sweep(args.sweep, GRIDS[args.grid])
    elif args.config and args.out:
        measure(json.loads(args.config), args.out)
    else:
        parser.error("pass --sweep DIR, or --config JSON with --out FILE")


if __name__ == "__main__":
    main()
