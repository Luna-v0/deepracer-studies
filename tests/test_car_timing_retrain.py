"""Takeoff-guarded draws of the car-timing retrain (no GPU, no sim: run is faked)."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import car_timing_retrain

VARIANT = "car_timing_fs2"


class _Spec:
    def __init__(self, variant, seed):
        self.variant = variant
        self.seed = seed

    def id(self):
        return f"id{self.seed}"

    def run_dir(self, root):
        group = car_timing_retrain.GROUP
        return f"{root}/{group}/{self.variant}-{self.seed}-id{self.seed}"


def _fake_run(outcomes, calls):
    """A ``run`` whose first eval reports each draw's progress, or raises."""

    def run(spec, root, on_eval):
        calls.append((spec.variant, spec.seed))
        Path(spec.run_dir(root)).mkdir(parents=True, exist_ok=True)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        on_eval(car_timing_retrain.EVAL_EVERY, {"mean_progress_m": outcome})
        return SimpleNamespace(
            metrics={"completion_rate": 0.5},
            eval_history=[],
            holdout={},
            train={"steps_per_s": 1.0},
        )

    return run


@pytest.fixture()
def git_repo(tmp_path):
    """A throwaway git checkout standing in for deepracer-studies."""
    repo = tmp_path / "studies"
    repo.mkdir()

    def git(*args):
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "-C",
                str(repo),
                *args,
            ],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (repo / "tracked.py").write_text("x = 1\n")
    git("add", "tracked.py")
    git("commit", "-q", "-m", "init")
    (repo / "tracked.py").write_text("x = 2\n")
    (repo / "untracked.py").write_text("y = 1\n")
    return repo


def _patch(monkeypatch, git_repo, outcomes):
    calls = []
    monkeypatch.setattr(car_timing_retrain, "run", _fake_run(outcomes, calls))
    monkeypatch.setattr(
        car_timing_retrain,
        "car_timing_spec",
        lambda variant, draw, seed: _Spec(draw, seed),
    )
    monkeypatch.setattr(car_timing_retrain, "STUDIES_ROOT", git_repo)
    return calls


def _record(root, draw):
    path = Path(root) / "logs" / f"{VARIANT}_d{draw}.result.json"
    return json.loads(path.read_text())


def test_a_flat_first_eval_is_recorded_as_a_failure_and_redrawn(
    tmp_path, monkeypatch, git_repo
):
    """Deliberately inverted 2026-10-13 (review): a failed draw now leaves a record."""
    root = tmp_path / "runs"
    calls = _patch(monkeypatch, git_repo, [1.0, 5.0, 5.0])
    result = car_timing_retrain.guarded_run(VARIANT, root=str(root))
    assert calls == [(f"{VARIANT}_d0", 0), (f"{VARIANT}_d1", 1)]
    failed = _record(root, 0)
    assert failed["takeoff_failed"] is True
    assert failed["seed"] == 0 and failed["draw"] == 0
    assert "1.00 m" in failed["reason"]
    assert failed["first_eval"]["mean_progress_m"] == 1.0
    assert failed["first_eval"]["frames"] == car_timing_retrain.EVAL_EVERY
    assert failed["run_dir"].endswith(f"{VARIANT}_d0-0-id0")
    assert _record(root, 1) == result
    assert result["draw"] == 1 and result["seed"] == 1
    assert "takeoff_failed" not in result
    assert [f["draw"] for f in result["failed_draws"]] == [0]
    assert result["failed_draws"][0]["mean_progress_m"] == 1.0


def test_each_draw_trains_with_its_own_seed(tmp_path, monkeypatch, git_repo):
    calls = _patch(monkeypatch, git_repo, [0.0, 0.0, 0.0])
    result = car_timing_retrain.guarded_run(VARIANT, root=str(tmp_path))
    seeds = [seed for _, seed in calls]
    assert seeds == [car_timing_retrain.SEED + n for n in range(3)]
    run_dirs = {
        json.loads(path.read_text())["run_dir"]
        for path in (tmp_path / "logs").glob("*.result.json")
    }
    assert len(run_dirs) == 3  # a distinct run dir per draw: nothing is overwritten
    assert result["draw"] == car_timing_retrain.FINAL_DRAWS_MAX - 1  # last always runs


def test_a_relaunch_resumes_after_the_last_failed_draw(tmp_path, monkeypatch, git_repo):
    root = tmp_path / "runs"
    calls = _patch(monkeypatch, git_repo, [1.0, RuntimeError("GPU lost")])
    with pytest.raises(RuntimeError, match="GPU lost"):
        car_timing_retrain.guarded_run(VARIANT, root=str(root))
    d0_before = (root / "logs" / f"{VARIANT}_d0.result.json").read_text()
    crashed_d1 = Path(_Spec(f"{VARIANT}_d1", 1).run_dir(str(root)))
    assert crashed_d1.is_dir()

    calls2 = _patch(monkeypatch, git_repo, [5.0])
    result = car_timing_retrain.guarded_run(VARIANT, root=str(root))
    assert calls2 == [(f"{VARIANT}_d1", 1)]  # d0 is never re-run
    assert (root / "logs" / f"{VARIANT}_d0.result.json").read_text() == d0_before
    assert result["draw"] == 1 and [f["draw"] for f in result["failed_draws"]] == [0]
    # the crashed attempt was set aside, not overwritten
    aside = list(crashed_d1.parent.glob(f"{crashed_d1.name}.crashed-*"))
    assert len(aside) == 1 and calls[0][0] == f"{VARIANT}_d0"


def test_a_recorded_success_is_reused_and_never_retrained(
    tmp_path, monkeypatch, git_repo
):
    calls = _patch(monkeypatch, git_repo, [5.0, 5.0])
    first = car_timing_retrain.guarded_run(VARIANT, root=str(tmp_path))
    again = car_timing_retrain.guarded_run(VARIANT, root=str(tmp_path))
    assert calls == [(f"{VARIANT}_d0", 0)] and again == first


def test_every_draw_snapshots_the_launcher_and_the_studies_git_state(
    tmp_path, monkeypatch, git_repo
):
    root = tmp_path / "runs"
    _patch(monkeypatch, git_repo, [1.0, 5.0])
    car_timing_retrain.guarded_run(VARIANT, root=str(root))
    module = Path(car_timing_retrain.__file__)
    for draw in (0, 1):
        launch = Path(_record(root, draw)["run_dir"]) / "_launch"
        assert (launch / module.name).read_bytes() == module.read_bytes()
        assert "+x = 2" in (launch / "studies_diff_head.patch").read_text()
        status = (launch / "studies_status.txt").read_text()
        assert " M tracked.py" in status and "?? untracked.py" in status
        assert len((launch / "studies_head.txt").read_text().strip()) == 40


def test_horizons_keep_final_d1s_seconds():
    assert round(car_timing_retrain.HORIZON_S, 2) == 3.99
    assert round(car_timing_retrain.LAM_HORIZON_S, 2) == 0.39
    assert car_timing_retrain.EVAL_EVERY == 4_375_000


def test_car_timing_lastact_routes_only_the_last_action_to_the_actor():
    """Variant C: B's routing minus v_forward (no speed sensor on the Evo)."""
    spec = car_timing_retrain.car_timing_spec("car_timing_lastact")
    assert spec.env.obs_routing["base"] == car_timing_retrain.CLASSIC_BLOCKS
    assert spec.env.obs_routing["actor"] == ("last_action",)
    assert spec.env.obs_routing["critic"] == ("*",)  # same privileged critic as B
    assert spec.policy.actor_keys == ("camera", "obs_actor")
    assert spec.policy.critic_keys == ("camera", "obs_critic")


def test_car_timing_lastact_spec_id_is_stable_and_differs_from_proprios():
    first = car_timing_retrain.car_timing_spec("car_timing_lastact").id()
    second = car_timing_retrain.car_timing_spec("car_timing_lastact").id()
    proprio = car_timing_retrain.car_timing_spec("car_timing_proprio").id()
    assert first == second
    assert first != proprio
