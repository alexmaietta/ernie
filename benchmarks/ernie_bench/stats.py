# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Summary:
    count: int
    median: float
    q1: float
    q3: float
    p95: float
    p99: float
    ci_low: float
    ci_high: float
    outliers: int

    @property
    def relative_ci(self) -> float:
        return (self.ci_high - self.ci_low) / (2 * self.median) if self.median > 0 else math.inf


def describe(values, confidence: float = 0.95, resamples: int = 1000, seed: int = 0) -> Summary:
    values = np.asarray(values, np.float64)
    if not len(values):
        raise ValueError("cannot describe an empty sample")
    q1, median, q3, p95, p99 = np.percentile(values, [25, 50, 75, 95, 99])
    rng = np.random.default_rng(seed)
    medians = np.median(values[rng.integers(0, len(values), (resamples, len(values)))], axis=1)
    tail = (1 - confidence) / 2
    ci_low, ci_high = np.quantile(medians, [tail, 1 - tail])
    fence = 1.5 * (q3 - q1)
    return Summary(
        count=len(values),
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        p95=float(p95),
        p99=float(p99),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        outliers=int(np.sum((values < q1 - fence) | (values > q3 + fence))),
    )
