# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc

from ernie_bench.results import Sample, read_run

NCU_METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__cycles_active.avg",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "launch__registers_per_thread",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)
NSYS_COLUMNS = {
    "Total Time (ns)": ("nsys.total_seconds", 1e-9),
    "Instances": ("nsys.instances", 1.0),
    "Med (ns)": ("nsys.median_seconds", 1e-9),
    "Max (ns)": ("nsys.max_seconds", 1e-9),
    "Time (%)": ("nsys.time_fraction", 0.01),
}
SUFFIXES = {"ncu": ".ncu-rep", "nsys": ".nsys-rep"}


def executable(tool: str) -> str:
    bundled = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / tool
    found = str(bundled) if bundled.exists() else shutil.which(tool)
    if found is None:
        raise FileNotFoundError(f"{tool} is not installed, set CUDA_HOME or put it on PATH")
    return found


def command(tool: str, report: Path) -> list[str]:
    if tool == "ncu":
        return [
            executable("ncu"),
            "--profile-from-start", "off",
            "--target-processes", "all",
            "--metrics", ",".join(NCU_METRICS),
            "--export", str(report),
            "--force-overwrite",
        ]
    if tool == "nsys":
        return [
            executable("nsys"), "profile",
            "--capture-range", "cudaProfilerApi",
            "--capture-range-end", "stop",
            "--trace", "cuda,nvtx,osrt,vulkan",
            "--cuda-graph-trace", "node",
            "--output", str(report),
            "--force-overwrite", "true",
        ]
    raise ValueError(f"unknown profiler {tool!r}, expected ncu or nsys")


def number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def parse_ncu(text: str) -> list[Sample]:
    rows = list(csv.reader(io.StringIO(text)))
    start = next((index for index, row in enumerate(rows) if "Kernel Name" in row), None)
    if start is None:
        return []
    header = rows[start]
    name = header.index("Kernel Name")
    samples = []
    for launch, row in enumerate(rows[start + 2 :]):
        for column, metric in enumerate(header):
            if metric in NCU_METRICS and (value := number(row[column])) is not None:
                samples.append(Sample(f"kernel/{row[name]}", f"ncu.{metric}", value, 0, launch))
    return samples


def parse_nsys(text: str) -> list[Sample]:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if line.lstrip('"').startswith("Time (%)")), len(lines))
    samples = []
    for row in csv.DictReader(lines[start:]):
        for column, (metric, scale) in NSYS_COLUMNS.items():
            if (value := number(row.get(column, ""))) is not None:
                samples.append(Sample(f"kernel/{row['Name']}", metric, value * scale, 0))
    return samples


def collect(tool: str, report: Path) -> list[Sample]:
    if tool == "ncu":
        arguments = [executable("ncu"), "--import", str(report), "--csv", "--page", "raw", "--print-kernel-base", "function", "--print-units", "base", "--metrics", ",".join(NCU_METRICS)]
        return parse_ncu(subprocess.run(arguments, capture_output=True, text=True, check=True).stdout)
    arguments = [executable("nsys"), "stats", "--report", "cuda_gpu_kern_sum:base", "--format", "csv", "--output", "-", "--force-export", "true", str(report)]
    return parse_nsys(subprocess.run(arguments, capture_output=True, text=True, check=True).stdout)


def summarize(directory: Path) -> list[dict]:
    manifest, table = read_run(directory)
    frames = manifest["suite"]["profile"]["frames"]
    table = table.filter(pc.starts_with(table["phase"], "kernel/"))
    kernels = defaultdict(lambda: defaultdict(list))
    for row in table.to_pylist():
        point = (row["backend"], row["scene"], row["worlds"], row["width"], row["height"], row["tier"], row["sm_percent"])
        kernels[point, row["phase"].removeprefix("kernel/")][row["metric"]].append(row["value"])
    records = []
    for (point, kernel), metrics in kernels.items():
        if "ncu.gpu__time_duration.sum" in metrics:
            durations = metrics["ncu.gpu__time_duration.sum"]
            total, launches = sum(durations) * 1e-9, len(durations)
            weighted = {
                metric.removeprefix("ncu."): sum(value * duration for value, duration in zip(values, durations)) / sum(durations)
                for metric, values in metrics.items()
                if metric != "ncu.gpu__time_duration.sum" and len(values) == launches and sum(durations) > 0
            }
        else:
            total, launches, weighted = sum(metrics["nsys.total_seconds"]), int(sum(metrics["nsys.instances"])), {}
        records.append(
            dict(zip(("backend", "scene", "worlds", "width", "height", "tier", "sm_percent"), point))
            | {"kernel": kernel, "seconds_per_frame": total / frames, "launches_per_frame": launches / frames} | weighted
        )
    return sorted(records, key=lambda record: (record["backend"], record["worlds"], -record["seconds_per_frame"]))
