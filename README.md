# deepracer-studies

Studies run against [deepracer-genesis](https://github.com/Luna-v0/deepracer-genesis),
split out so the simulator repo stays lean. `Experiment` subclasses and notebooks that
compose the simulator's stage DSL, train policies, and analyze results — no simulator,
environment, renderer, or training-backend code. That lives upstream and is consumed as a
git dependency.

## Install

Needs `uv`, Python `>=3.10,<3.13`, and an NVIDIA GPU — every camera study renders through
Madrona or Nyx.

```bash
uv sync
```

The genesis dependency is pinned to its `dev` branch with five extras: `vision` and
`nyx` (the two GPU renderers, which `best_camera.ipynb` benchmarks against each other),
`hpo` (optuna), `analysis` (the telemetry and trackplot modules), and `export` (the
`python -m deepracer_genesis.deploy.onnx` subprocess the camera notebooks end on, which
runs in this venv).

> The pin is `dev` because `main` does not yet carry the `analysis` package. Repoint the
> `[tool.uv.sources]` rev to `main` (or to a commit SHA) once `dev` is merged.
>
> Separately, `from deepracer_genesis.experiment import rollout_video` / `view_zoo` — used
> by `best_camera*.ipynb` — fails on `dev` as well: both functions exist, but at
> `deepracer_genesis.experiment.visualize` and `deepracer_genesis.tools.zoo`. They need
> re-exporting upstream, or those cells need the fully qualified paths.

Use `uv` for everything: never `pip`, never bare `python`, never activate a venv by hand.

## Layout

```
experiments/
├── camera_std_sweep.py       feature_vector_proof.py    tight_oval.py
├── baselines/camera.py       copied from genesis examples/ (see below)
└── best_camera.ipynb         best_camera_diverse.ipynb  best_camera_completion.ipynb
    ablation_individual.ipynb proof_feature.ipynb        proof_camera.ipynb
    proof_cnn.ipynb           analysis_feature.ipynb     analysis_camera.ipynb
runs/                         generated; almost entirely gitignored
```

## Running

Run from the repo root — `ROOT = "runs/<study>"` and every artifact path is relative to
the CWD. An experiment is referenced by its class; there is no name registry.

```bash
uv run python experiments/camera_std_sweep.py entropy_0 entropy_1e3
uv run python -m deepracer_genesis.experiment experiments.my_run:MyRun
uv run papermill experiments/best_camera.ipynb /tmp/out.ipynb -p REUSE True
```

Or `MyRun().run()` / `run(MyRun)` from a notebook. `uv sync` installs this project
editable, so `experiments.*` resolves however a file is launched.

Opening a notebook in JupyterLab with a per-notebook working directory resolves `runs/`
to `experiments/runs/`, misses every cached record, and silently starts training from
scratch.

## The studies

| Study | What it asks |
| --- | --- |
| `camera_std_sweep.py` | Why the camera policy's action-std drifts from ~1.3 to 18 while eval stays at 0%. Five short variants isolating entropy bonus, std ceiling, frame stack, DR. |
| `feature_vector_proof.py` | Can a policy race on the feature vector alone, no camera? Needs a display (`view="gui"`). |
| `tight_oval.py` | Is the 1.5 x 1.6 m home-room oval drivable at all? `TightOvalLive` renders through Nyx with the viewer open; needs a display. |
| `best_camera.ipynb` | Round 1 of the vision study, and the first camera policy off 0%. Renderer bench, 16-trial HPO on one track, scaled final run, ONNX export. |
| `best_camera_diverse.ipynb` | Round 2, generalization: trains on seven circuits, holds out four. Where the "takeoff lottery" was characterized — same config and seed can park or fly. |
| `best_camera_completion.ipynb` | Round 3, completion over speed, after the `off_track` reward sign bug. Searches reward *families*; `speed_target` won, `safe_progress` hacked itself into parking. |
| `ablation_individual.ipynb` | Scaffold only — tabulates the 31 DR knobs by layer, runs nothing. |
| `proof_feature.ipynb` | Can a policy lap one track (`reinvent_base`, no DR) from the state vector alone? 200-trial HPO over PPO knobs, MLP and reward, then a long final run. The recorded HPO answers yes: best trial 100% completion, 0% off-track. |
| `proof_camera.ipynb` | The same question from camera pixels, adding resolution, frame stack and CNN trunk to the search, with a VRAM preflight. Unanswered: the recorded study has no completed trial and no final run. |
| `proof_cnn.ipynb` | Can a CNN read the driving state out of the image? Trains on expert rollouts from six tracks, HPO over trunk, resolution, frame stack and optimizer, ranked on two held-out tracks. Best of 41 trials scored 1.34 sigma-normalized MSE; below 1.0 would beat the stock network. |
| `analysis_feature.ipynb` | Loader only — reads `proof_feature`'s `study.db` and every `eval_record.json` into `trials`, `curves`, `evals` and `finals` dataframes, analyzes nothing. |
| `analysis_camera.ipynb` | The same loader for `proof_camera`. |

## runs/

87 GB of checkpoints, telemetry and video stayed behind. What is committed is each
study's optuna `study.db` and its `logs/*.result.json` step cache (~700 KB total), which
is what makes the notebooks replay in seconds. **A missing record does not raise** —
`recorded()` returns `None` and the step retrains, turning a 20-second replay into hours
of GPU. `study.db` is also the only place each study's winning hyperparameters exist.

Not committed: the final run directory per study. Two cells need one — the telemetry read
(`load_telemetry(<RUN_DIR>/telemetry/final.parquet)`) and the ONNX export
(`subprocess.run(..., check=True)`) are both unguarded and fail without it. Copy the dir
out of the genesis working tree if you need those cells:

- `best_camera/best_camera/final-0-fc7ff6377a6c/`
- `best_camera_diverse/best_camera_diverse/final-0-f25d6d245c08/`
- `best_camera_completion/best_camera_completion/final_d1-0-75f935de3139/` (model,
  telemetry) and `final_d1-0-62fd8667fee0/` (video, export)

Only `model.pt`, `eval_record.json` and `telemetry/{final,holdout_*}.parquet` are read.
Runs written before genesis ADR 0004 also hold periodic `model_NNNN.pt` checkpoints,
which is what made those dirs multi-gigabyte. Newer runs keep only `model.pt` +
`model_best.pt` and record periodic telemetry for a 64-env subset
(`Evaluation(keep_checkpoints=..., telemetry_envs=...)`).

## experiments/baselines/camera.py

Copied from genesis `examples/camera.py`, because the genesis wheel packages only
`deepracer_genesis*` — the repo-root `examples/` is not importable from a git install.
`camera_std_sweep.py` subclasses `CameraMadronaDr` from it. Two edits: the docstring
records provenance, and `group = "examples"` became `"baselines"`. No sync mechanism —
re-copy by hand if upstream changes.

`CONVENTIONS.md` is binding for Python written here.
