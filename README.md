# deepracer-studies

The experiment layer for [deepracer-genesis](https://github.com/Luna-v0/deepracer-genesis),
split out of the simulator repo so that repo stays lean. Everything here is a *study*:
`Experiment` subclasses and notebooks that compose the simulator's stage DSL, train
policies, and analyze the results. It holds no simulator, environment, renderer, or
training-backend code — `deepracer-genesis` is consumed as a git dependency and is the
only place that code lives. Changes to the simulator belong upstream; changes to what is
being measured belong here.

## Install

Requires `uv`, Python `>=3.10,<3.13` (matching deepracer-genesis), and an NVIDIA GPU —
every camera study renders through Madrona or Nyx and cannot run on CPU.

```bash
uv sync
```

That is all — `pyproject.toml` and `uv.lock` are committed. The dependency set was built
with:

```bash
uv add "deepracer-genesis[analysis,export,hpo,nyx,vision] @ git+https://github.com/Luna-v0/deepracer-genesis.git@dev"
uv add "torch>=2.5" numpy "pandas>=2.0" matplotlib "seaborn>=0.13.2" "optuna>=4.9.0"
uv add --dev jupyterlab ipykernel ipywidgets ipython papermill jupytext pytest ruff
```

Use `uv` for everything — never `pip`, never a bare `python`, never activate a venv by
hand.

### Why those five extras

All are load-bearing: `vision` (gs-madrona) and `nyx` (gs-nyx, gs-nyx-plugin) for the two
GPU renderers, both of which `best_camera.ipynb` benchmarks against each other; `hpo`
(optuna) for every search; `analysis` (pandas, matplotlib) for
`deepracer_genesis.analysis.telemetry` and `.trackplots`; `export` (onnx, onnxruntime)
for the `python -m deepracer_genesis.deploy.onnx` subprocess the camera notebooks end on,
which runs in *this* venv. `tracking` (mlflow) and `docs` are not used here.

### The `@dev` pin

The dependency points at the **`dev` branch**, not the default branch. `main` is several
commits behind and has no `deepracer_genesis/analysis/` package and no `analysis` extra
at all, so nothing here resolves against it. Once `dev` lands on `main`, drop the `@dev`
suffix — or better, pin a commit SHA: every recorded run carries its exact provenance in
`runs/<study>/<group>/<run>/git/deepracer-genesis.diff`.

## Layout

```
deepracer-studies/
├── experiments/
│   ├── __init__.py                     # package marker; author studies here
│   ├── camera_std_sweep.py             # Experiment subclasses
│   ├── feature_vector_proof.py
│   ├── tight_oval.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── camera.py                   # copied from genesis examples/ (see below)
│   ├── best_camera.ipynb               # notebook studies
│   ├── best_camera_diverse.ipynb
│   ├── best_camera_completion.ipynb
│   ├── ablation_individual.ipynb
│   └── proof_of_concepts.ipynb
├── runs/                               # generated; almost entirely gitignored
│   ├── report.csv, report.md
│   ├── best_camera/{study.db, logs/*.result.json}
│   ├── best_camera_diverse/{study.db, logs/*.result.json}
│   └── best_camera_completion/{study.db, report.*, logs/*.result.json}
└── CONVENTIONS.md
```

`runs/` is where training writes. The heavy artifacts from the original repo were **not**
migrated — checkpoints (`*.pt`), telemetry parquet, videos, tensorboard event files and
per-run export bundles totalled ~87 GB. What was carried over is the small research
record that keeps the notebooks replayable: each study's optuna `study.db` and its
`logs/*.result.json` step cache, ~700 KB in total. `.gitignore` ignores `runs/**` and
re-includes exactly those.

To get a full run directory back, either re-run the study (hours of GPU time; see the
budget constants at the top of each notebook) or copy the specific
`runs/<study>/<group>/<variant>-<seed>-<hash>/` directory out of the original
deepracer-genesis working tree.

## Running an experiment

Run everything **from the repository root** — `ROOT = "runs/<study>"` and every other
artifact path is relative to the process CWD.

An experiment is referenced by its class; there is no name registry. Equivalent ways to
launch one:

```python
MyRun().run()          # from a notebook or a __main__ block
run(MyRun)             # from deepracer_genesis.experiment import run
```

```bash
uv run python -m deepracer_genesis.experiment experiments.my_run:MyRun
uv run python experiments/camera_std_sweep.py entropy_0 entropy_1e3
```

`uv sync` installs this project itself in editable mode, so
`from experiments.baselines.camera import CameraMadronaDr` resolves regardless of how the
file is launched — by path or with `-m`.

Notebooks are executed headless with papermill, again from the repo root:

```bash
uv run papermill experiments/best_camera.ipynb /tmp/best_camera.out.ipynb -p REUSE True
```

Opening a notebook in JupyterLab with a per-notebook working directory resolves `runs/`
to `experiments/runs/`, misses every cached record, and silently starts training from
scratch.

## The studies

### Python

- **`experiments/camera_std_sweep.py`** — Diagnostic sweep chasing the camera policy's
  action-std explosion (std drifting from ~1.3 to 18 while deterministic eval stayed at
  0% completion). Five short 2.5M-step variants isolate the cause: no entropy bonus, a
  10x smaller bonus, a `std_range=(0.1, 1.0)` ceiling, ceiling + 4-frame stack, and
  ceiling + stack with domain randomization removed entirely.
- **`experiments/feature_vector_proof.py`** — Asks whether a policy can race using only
  the feature vector, with no camera at all: 256 GPU envs over six training tracks,
  evaluated on `reinvent_base` and `Oval_track`. Configured with `view="gui"`, so it
  needs a display.
- **`experiments/tight_oval.py`** — Feasibility probe for the 1.5 x 1.6 m home-room oval
  (`donut_track`), whose ~0.40 m centerline turn radius sits above the car's full-lock
  minimum. `TightOval` is the headless feature-vector run; `TightOvalLive` is the same
  study rendered through Nyx with the interactive viewer open. `TightOvalLive` is the
  `__main__` default and needs both a display and the `nyx` extra.

### Notebooks

- **`experiments/best_camera.ipynb`** — Round 1 of the end-to-end vision study, and the
  first camera policy to leave 0% completion. Benches Madrona against Nyx at 64 and 128
  envs and picks a renderer, runs a 16-trial optuna search with no DR on a single track
  (so trials measure pure learning ability, with the std ceiling held fixed from the
  sweep above), trains the winner at scale with the full DR stack and the track zoo,
  plots telemetry, and exports an ONNX car bundle.
- **`experiments/best_camera_diverse.ipynb`** — Round 2, on held-out generalization.
  Round 1 evaluated only on reinvent-route variants; this one trains on seven genuinely
  different circuits and holds out four the policy never sees, spanning 18–42 m and
  including the Bowtie crossover topology. It re-runs HPO on the diverse task itself with
  `entropy_coef` and `dr` searchable, and is where the "takeoff lottery" was
  characterized: identical config and seed can park or fly on GPU nondeterminism alone.
- **`experiments/best_camera_completion.ipynb`** — Round 3, targeting lap completion
  rather than speed, after a reward sign bug (the `off_track` term paying for riding the
  edge) was fixed upstream. Searches over *reward-function families* defined inline and
  injected through `RewardShaping`, scored on the un-hackable deterministic-eval
  `completion_rate`, with takeoff-guarded final training that redraws a flat run. The
  recorded outcome: `speed_target` won decisively, `safe_progress` reward-hacked itself
  into parking, and `lap_bonus` produced never-crashing crawlers.
- **`experiments/ablation_individual.ipynb`** — Scaffold for the per-knob domain
  randomization ablation. As committed it only tabulates the 31 catalog knobs by layer
  (image, physics, geometry, visual, actuation); the search space and imports below that
  are declared but never invoked, and no training runs.
- **`experiments/proof_of_concepts.ipynb`** — Early scratch notebook sketching the HPO
  search space and a minimal camera-training setup. It does not execute end to end: a
  cell references `np` without importing numpy. That defect predates the split.

### What the notebooks read, and what is missing

All three camera notebooks cache each named step to `runs/<study>/logs/<step>.result.json`
and replay it when `REUSE = True`. **With a record missing, `recorded()` returns `None`
and the step retrains silently rather than raising** — a 20-second replay becomes a
multi-hour GPU run. The step cache and `study.db` are committed precisely to prevent
that; `study.db` in particular is the only place each study's winning hyperparameters
exist.

What is *not* committed is each study's final run directory, so the cells that read a
checkpoint or telemetry parquet still need artifacts from the genesis working tree:

| Notebook | Committed | Needs from genesis `runs/` |
| --- | --- | --- |
| `best_camera.ipynb` | 23 step records, `study.db` | `best_camera/best_camera/final-0-fc7ff6377a6c/` |
| `best_camera_diverse.ipynb` | 18 step records, `study.db` | `best_camera_diverse/best_camera_diverse/final-0-f25d6d245c08/` |
| `best_camera_completion.ipynb` | 33 step records, `study.db`, `report.*` | `best_camera_completion/best_camera_completion/final_d1-0-75f935de3139/` (model, telemetry) and `final_d1-0-62fd8667fee0/` (recorded video, export) |

Within those directories only `model.pt`, `eval_record.json` and
`telemetry/{final,holdout_*}.parquet` are read; the periodic `model_NNNN.pt` checkpoints
are what make the directories multi-gigabyte. Two cells are unguarded and will fail
outright without them: the telemetry read (`load_telemetry(<RUN_DIR>/telemetry/final.parquet)`)
and the ONNX export (`subprocess.run(..., check=True)`). `best_camera.ipynb`'s `view_zoo`
cell is unguarded by design and builds a real sim on every execution, so a GPU is needed
even for a pure replay.

`ablation_individual.ipynb` and `proof_of_concepts.ipynb` read and write nothing — both
assign `ROOT = "runs/best_camera"` and never dereference it.

## Copied from deepracer-genesis

`experiments/baselines/camera.py` is a copy of the simulator repo's `examples/camera.py`.
It is copied because the deepracer-genesis wheel packages only `deepracer_genesis*` — the
repo-root `examples/` package is not shipped, so a git install of the simulator cannot
import it. `experiments/camera_std_sweep.py` subclasses `CameraMadronaDr` from it;
upstream, that import was `sys.path.insert(0, ".")` plus
`from examples.camera import CameraMadronaDr`, which only worked from the genesis repo
root.

The copy imports nothing outside `deepracer_genesis` and the stdlib, so it is
self-contained. Two deliberate edits: the module docstring records the provenance, and
`group = "examples"` became `group = "baselines"` so studies here do not write into
`runs/examples/`. There is no sync mechanism — if `examples/camera.py` changes upstream,
re-copy it by hand. `experiments/baselines/__init__.py` is written fresh rather than
copied, because the upstream `examples/__init__.py` also imports `feature_vector` and
`watch_live`, which were not needed here.

## Conventions

`CONVENTIONS.md` is binding for Python written here: uv only, absolute imports,
`from x import y`, Google docstrings with at most two lines of prose but keeping the
`Args:` / `Returns:` / `Raises:` / `Attributes:` sections, 88-column lines, and `X | None`
rather than `Optional[X]`.
