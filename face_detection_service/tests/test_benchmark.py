"""Smoke test: ensure the benchmark script runs without errors."""

from __future__ import annotations

from face_detection_service.benchmarks.run_benchmark import run


def test_benchmark_runs():
    """Benchmark should complete and return a non-empty result list."""
    results = run(iterations=1)
    assert isinstance(results, list)
    assert len(results) > 0

    for row in results:
        assert "Detector" in row
        assert "Median ms" in row
        assert row["Median ms"] >= 0
