# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter, sleep

TICKS = os.sysconf("SC_CLK_TCK")


def cpu_times() -> dict[int, tuple[int, int]]:
    times = {}
    with open("/proc/stat") as stat:
        for line in stat:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            name, *fields = line.split()
            user, nice, system, idle, iowait, irq, softirq, steal = map(int, fields[:8])
            total = user + nice + system + idle + iowait + irq + softirq + steal
            times[int(name[3:])] = (total - idle - iowait, total)
    return times


def difference(start: dict[int, tuple[int, int]], end: dict[int, tuple[int, int]], cpus: set[int], own_cpu_seconds: float) -> dict[str, float]:
    def busy_and_total(selected):
        busy = sum(end[cpu][0] - start[cpu][0] for cpu in selected)
        total = sum(end[cpu][1] - start[cpu][1] for cpu in selected)
        return busy, total

    busy, total = busy_and_total(end.keys())
    local_busy, local_total = busy_and_total(cpus & end.keys())
    return {
        "cpu_utilization": busy / total if total else 0.0,
        "benchmark_cpus_utilization": local_busy / local_total if local_total else 0.0,
        "foreign_cpu_seconds": max(0.0, local_busy / TICKS - own_cpu_seconds),
        "load_average": os.getloadavg()[0],
    }


def _process_ticks() -> dict[int, tuple[str, int]]:
    ticks = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        name = stat[stat.index("(") + 1 : stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2 :].split()
        ticks[int(entry.name)] = (name, int(fields[11]) + int(fields[12]))
    return ticks


def top_processes(interval: float = 0.25, count: int = 10) -> list[dict]:
    opened = perf_counter()
    first = _process_ticks()
    sleep(interval)
    closed = perf_counter()
    second = _process_ticks()
    elapsed = closed - opened
    own = os.getpid()
    usage = [
        {"pid": pid, "name": name, "cpu_percent": 100 * (ticks - first[pid][1]) / TICKS / elapsed}
        for pid, (name, ticks) in second.items()
        if pid in first and pid != own
    ]
    return sorted((u for u in usage if u["cpu_percent"] > 0), key=lambda u: -u["cpu_percent"])[:count]
