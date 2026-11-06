# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

from ernie_bench.results import read_run
from ernie_bench.stats import describe

POINT = ("backend", "scene", "worlds", "cameras_per_world", "width", "height", "outputs", "tier", "spp", "sm_percent")
MIN_ENERGY_SECONDS = 1.0
COUNTER_PREFIXES = ("process.", "perf.", "system.", "gpu")
UNAGGREGATED = ("samples", "disturbed", "energy_joules")
STARTUP = ("interpreter_seconds", "harness_import_seconds", "create_seconds")


def _aggregation(metric: str) -> str:
    leaf = metric.rsplit(".", 1)[-1]
    if leaf in ("rss_bytes", "peak_rss_bytes", "threads", "load_average") or leaf.endswith("_max"):
        return "max"
    if leaf.endswith("_min"):
        return "min"
    if leaf.endswith("_mean"):
        return "mean"
    if leaf.endswith("utilization") or leaf.endswith("_growth_bytes"):
        return "median"
    return "per_frame"


def _ratio(counters: dict, numerator: str, denominator: str) -> float | None:
    if counters.get(denominator):
        return counters[numerator] / counters[denominator]
    return None


def summarize_run(directory: Path) -> tuple[dict, list[dict]]:
    manifest, table = read_run(directory)
    frames, disturbed, energy, preempted = defaultdict(float), set(), defaultdict(float), set()
    device = {}
    counters = defaultdict(lambda: defaultdict(dict))
    once = {phase: defaultdict(lambda: defaultdict(list)) for phase in ("startup", "load", "first_frame")}
    for phase, collected in once.items():
        for row in table.filter(pc.equal(table["phase"], phase)).to_pylist():
            if ".throttle." not in row["metric"]:
                collected[tuple(row[column] for column in POINT)][row["metric"]].append(row["value"])
    for row in table.filter(pc.equal(table["phase"], "steady")).to_pylist():
        point = tuple(row[column] for column in POINT)
        metric = row["metric"]
        repeat = (row["process"], row["repeat"])
        if metric.startswith(COUNTER_PREFIXES) and ".throttle." not in metric and not metric.endswith(UNAGGREGATED):
            counters[point][metric][repeat] = row["value"]
        if metric in ("update_seconds", "render_seconds"):
            frames[point, repeat, row["iteration"]] += row["value"]
        elif metric == "render_device_seconds":
            device[point, repeat, row["iteration"]] = row["value"]
        elif metric == "preempted" and row["value"]:
            preempted.add((point, repeat, row["iteration"]))
        elif metric == "disturbed" and row["value"]:
            disturbed.add((point, repeat))
        elif metric.endswith(".energy_joules"):
            energy[point, repeat] += row["value"]

    clean_frames, quiet_frames, device_frames, repeats = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(set)
    by_process = defaultdict(lambda: defaultdict(list))
    for (point, repeat, iteration), seconds in sorted(frames.items()):
        repeats[point].add(repeat)
        if (point, repeat) not in disturbed:
            clean_frames[point].append(seconds)
            by_process[point][repeat[0]].append(seconds)
            if (point, repeat, iteration) in device:
                device_frames[point].append(device[point, repeat, iteration])
            if (point, repeat, iteration) not in preempted:
                quiet_frames[point].append(seconds)

    records = []
    for point in sorted(repeats):
        values = clean_frames[point]
        clean = [repeat for repeat in sorted(repeats[point]) if (point, repeat) not in disturbed]
        startup = {metric: float(np.median(found)) for metric, found in once["startup"][point].items()}
        record = dict(zip(POINT, point)) | {
            "repeats": len(repeats[point]),
            "clean_repeats": len(clean),
            "frames": len(values),
            "processes": len({process for process, _ in repeats[point]}),
            "startup": startup,
            "startup_seconds": sum(startup.get(metric, 0.0) for metric in STARTUP),
            "load": {metric: float(np.median(found)) for metric, found in once["load"][point].items()},
            "first_frame_seconds": sum(float(np.median(once["first_frame"][point][metric])) for metric in ("update_seconds", "render_seconds") if metric in once["first_frame"][point]),
        }
        if values:
            stats = describe(values)
            quiet = quiet_frames[point]
            trend = quiet or values
            half = len(trend) // 2
            process_medians = [np.median(found) for found in by_process[point].values()]
            images = record["worlds"] * record["cameras_per_world"]
            pixels = images * record["width"] * record["height"]
            joules = sum(energy[point, repeat] for repeat in clean)
            record |= {
                "median_seconds": stats.median,
                "min_seconds": min(values),
                "p95_seconds": stats.p95,
                "p99_seconds": stats.p99,
                "ci_low_seconds": stats.ci_low,
                "ci_high_seconds": stats.ci_high,
                "relative_ci": stats.relative_ci,
                "iqr_seconds": stats.q3 - stats.q1,
                "outliers": stats.outliers,
                "quiet_median_seconds": float(np.median(quiet)) if quiet else None,
                "preempted_fraction": 1 - len(quiet) / len(values),
                "process_spread": (max(process_medians) - min(process_medians)) / stats.median if stats.median > 0 else 0.0,
                "drift": (np.median(trend[:half]) - np.median(trend[half:])) / np.median(trend) if half and np.median(trend) > 0 else 0.0,
                "images_per_second": images / stats.median if stats.median > 0 else float("inf"),
                "pixels_per_second": pixels / stats.median if stats.median > 0 else float("inf"),
                "primary_rays_per_second": pixels * record["spp"] / stats.median if stats.median > 0 else float("inf"),
                "images_per_joule": images * len(values) / joules if joules > 0 and sum(values) >= MIN_ENERGY_SECONDS else None,
            }
            if device_frames[point]:
                timed = describe(device_frames[point])
                record |= {
                    "device_median_seconds": timed.median,
                    "device_p95_seconds": timed.p95,
                    "device_relative_ci": timed.relative_ci,
                    "device_frames": len(device_frames[point]),
                }
            aggregated = {}
            for metric, by_repeat in counters[point].items():
                kept = [by_repeat[repeat] for repeat in clean if repeat in by_repeat]
                if not kept:
                    continue
                aggregation = _aggregation(metric)
                if aggregation == "per_frame":
                    aggregated[metric] = sum(kept) / len(values)
                else:
                    aggregated[metric] = float(getattr(np, aggregation)(kept))
            aggregated["perf.ipc"] = _ratio(aggregated, "perf.instructions", "perf.cycles")
            aggregated["perf.cache_miss_rate"] = _ratio(aggregated, "perf.cache_misses", "perf.cache_references")
            aggregated["perf.llc_miss_rate"] = _ratio(aggregated, "perf.llc_misses", "perf.llc_loads")
            record["counters"] = aggregated
        records.append(record)
    return manifest, records
