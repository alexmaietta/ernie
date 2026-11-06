# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import resource
from time import perf_counter, process_time
from typing import NamedTuple

PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")


class Usage(NamedTuple):
    wall: float
    cpu: float
    user: float
    system: float
    voluntary_switches: int
    involuntary_switches: int
    minor_faults: int
    major_faults: int
    rss_bytes: int
    peak_rss_bytes: int
    threads: int


def usage() -> Usage:
    r = resource.getrusage(resource.RUSAGE_SELF)
    with open("/proc/self/statm") as statm:
        resident_pages = int(statm.read().split()[1])
    return Usage(
        perf_counter(),
        process_time(),
        r.ru_utime,
        r.ru_stime,
        r.ru_nvcsw,
        r.ru_nivcsw,
        r.ru_minflt,
        r.ru_majflt,
        resident_pages * PAGE_BYTES,
        max(r.ru_maxrss * 1024, resident_pages * PAGE_BYTES),
        len(os.listdir("/proc/self/task")),
    )


def difference(start: Usage, end: Usage) -> dict[str, float]:
    cpu = end.cpu - start.cpu
    wall = end.wall - start.wall
    return {
        "cpu_seconds": cpu,
        "user_seconds": end.user - start.user,
        "system_seconds": end.system - start.system,
        "cpu_utilization": cpu / wall if wall > 0 else 0.0,
        "voluntary_switches": end.voluntary_switches - start.voluntary_switches,
        "involuntary_switches": end.involuntary_switches - start.involuntary_switches,
        "minor_faults": end.minor_faults - start.minor_faults,
        "major_faults": end.major_faults - start.major_faults,
        "rss_bytes": end.rss_bytes,
        "rss_growth_bytes": end.rss_bytes - start.rss_bytes,
        "peak_rss_bytes": end.peak_rss_bytes,
        "threads": end.threads,
    }
