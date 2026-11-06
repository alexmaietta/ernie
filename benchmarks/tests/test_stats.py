# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from ernie_bench.stats import describe


def test_describe_reports_median_spread_ci_and_outliers():
    values = np.concatenate([np.linspace(9, 11, 99), [50.0]])
    summary = describe(values)
    assert summary.count == 100
    assert summary.median == pytest.approx(10, abs=0.05)
    assert summary.ci_low <= summary.median <= summary.ci_high
    assert summary.relative_ci < 0.05
    assert summary.outliers == 1
    assert describe(values) == summary


def test_describe_rejects_empty():
    with pytest.raises(ValueError):
        describe([])


def test_median_and_quartiles_follow_numpy():
    values = np.random.default_rng(1).exponential(size=64)
    summary = describe(values)
    q1, median, q3, p95, p99 = np.percentile(values, [25, 50, 75, 95, 99])
    assert (summary.q1, summary.median, summary.q3, summary.p95, summary.p99) == (q1, median, q3, p95, p99)


def test_constant_samples_have_a_zero_width_interval():
    summary = describe([0.004] * 50)
    assert summary.ci_low == summary.ci_high == 0.004
    assert summary.relative_ci == 0
    assert summary.outliers == 0


def test_zero_median_has_an_unbounded_relative_interval():
    assert describe([0.0, 0.0, 0.0]).relative_ci == float("inf")


def test_interval_narrows_as_samples_grow():
    rng = np.random.default_rng(2)
    small = describe(rng.lognormal(0, 0.3, 50)).relative_ci
    large = describe(rng.lognormal(0, 0.3, 5000)).relative_ci
    assert large < small / 5


def test_interval_covers_the_true_median_at_its_confidence():
    rng = np.random.default_rng(3)
    trials = 300
    covered = 0
    for trial in range(trials):
        sample = rng.lognormal(0, 0.5, 101)
        summary = describe(sample, confidence=0.95, seed=trial)
        covered += summary.ci_low <= 1.0 <= summary.ci_high
    assert 0.90 <= covered / trials <= 0.99
