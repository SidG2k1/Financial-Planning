import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _repository_environment(repository: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    paths = [str(repository / "src")]
    if existing:
        paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def _load_simulation_benchmark(repository: Path) -> object:
    path = repository / "benchmarks" / "simulation.py"
    spec = importlib.util.spec_from_file_location("simulation_benchmark", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_return_model_benchmark_has_lightweight_smoke_options() -> None:
    repository = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/return_models.py",
            "--moment-paths",
            "200",
            "--moment-warmup-paths",
            "20",
            "--benchmark-paths",
            "20",
            "--benchmark-repeats",
            "1",
            "--horizon",
            "2",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        env=_repository_environment(repository),
    )

    assert completed.returncode == 0, completed.stderr
    assert "Moment comparison" in completed.stdout
    assert "Performance benchmark" in completed.stdout


def test_simulation_benchmark_reports_timing_and_peak_rss() -> None:
    repository = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/simulation.py",
            "--paths",
            "20",
            "--repeats",
            "1",
            "--chunk-size",
            "2",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=_repository_environment(repository),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == 1
    assert payload["config"] == {
        "paths": 20,
        "repeats": 1,
        "chunk_size": 2,
        "seed": 20_260_726,
        "horizon": 73,
        "memory_only": False,
    }
    timing_names = (
        "solver_seconds",
        "full_seconds",
        "summary_seconds",
    )
    for name in timing_names:
        assert isinstance(payload[name], float)
        assert math.isfinite(payload[name])
        assert payload[name] >= 0.0
        assert len(payload["raw_runs"][name]) == 1
        raw_value = payload["raw_runs"][name][0]
        assert isinstance(raw_value, float)
        assert math.isfinite(raw_value)
        assert raw_value >= 0.0

    assert payload["rss_unit"] == "bytes"
    assert isinstance(payload["summary_peak_rss"], int)
    assert payload["summary_peak_rss"] > 0
    assert isinstance(payload["summary_baseline_rss"], int)
    assert payload["summary_baseline_rss"] > 0
    assert isinstance(payload["summary_increment_rss"], int)
    assert payload["summary_increment_rss"] >= 0


def test_simulation_benchmark_prefers_repository_source(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    stale_package = tmp_path / "lifecycle_finance"
    stale_package.mkdir()
    (stale_package / "__init__.py").write_text(
        "raise RuntimeError('stale lifecycle-finance import')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/simulation.py",
            "--paths",
            "2",
            "--repeats",
            "1",
            "--chunk-size",
            "2",
            "--memory-only",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["config"]["horizon"] == 73


def test_simulation_benchmark_rejects_negative_seed() -> None:
    repository = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/simulation.py",
            "--paths",
            "2",
            "--chunk-size",
            "2",
            "--seed",
            "-1",
            "--memory-only",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_repository_environment(repository),
    )

    assert completed.returncode == 2
    assert "argument --seed: must be nonnegative" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_simulation_benchmark_reports_rss_child_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).parents[1]
    benchmark = _load_simulation_benchmark(repository)
    observed: dict[str, object] = {}

    def failed_child(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        observed["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(
            command,
            returncode=17,
            stdout="",
            stderr="rss child detail",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", failed_child)
    arguments = argparse.Namespace(
        paths=20,
        repeats=1,
        chunk_size=2,
        seed=0,
    )

    with pytest.raises(
        RuntimeError,
        match=r"RSS child failed with exit code 17.*rss child detail",
    ):
        benchmark._fresh_rss_payload(arguments)

    assert observed["timeout"] == 300
