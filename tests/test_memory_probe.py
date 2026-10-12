"""Contracts of the VRAM probe's grids and sweep loop (no GPU, no genesis)."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import memory_probe  # noqa: E402
from experiments.memory_probe import GRIDS, config_name, predicted_bytes  # noqa: E402


def _line(cfg: dict) -> tuple:
    """The plotted line a config belongs to."""
    return (cfg["modality"], cfg.get("frame_stack"), cfg.get("res_scale"))


def test_g4_grid_has_at_least_four_points_per_line():
    counts = Counter(_line(c) for c in GRIDS["g4"])
    assert len(counts) == 8
    assert min(counts.values()) >= 4


def test_grid_names_are_unique():
    for grid in GRIDS.values():
        names = [config_name(c) for c in grid]
        assert len(names) == len(set(names))


def test_sweep_runs_smallest_first_and_records_timeouts(tmp_path, monkeypatch):
    grid = list(reversed(GRIDS["g4"][:3]))
    launched = []

    def fake_run(cmd, timeout):
        launched.append(json.loads(cmd[cmd.index("--config") + 1]))
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(memory_probe, "free_vram_bytes", lambda: 2**40)
    monkeypatch.setattr(memory_probe.subprocess, "run", fake_run)
    memory_probe.sweep(tmp_path, grid, timeout_s=7)

    assert launched == sorted(grid, key=predicted_bytes)
    record = json.loads((tmp_path / f"{config_name(grid[0])}.json").read_text())
    assert record["ok"] is False
    assert record["error"] == "timed out after 7 s"
