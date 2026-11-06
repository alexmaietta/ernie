# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import ctypes
import errno
import os
import platform
import struct
from pathlib import Path

from ernie_bench.system.machine import parse_cpu_list, read

HARDWARE, SOFTWARE, HW_CACHE = 0, 1, 3
LLC, L1D = 2, 0
READ, ACCESS, MISS = 0, 0, 1

EVENTS = {
    "cycles": (HARDWARE, 0),
    "instructions": (HARDWARE, 1),
    "cache_references": (HARDWARE, 2),
    "cache_misses": (HARDWARE, 3),
    "branch_misses": (HARDWARE, 5),
    "llc_loads": (HW_CACHE, LLC | READ << 8 | ACCESS << 16),
    "llc_misses": (HW_CACHE, LLC | READ << 8 | MISS << 16),
    "l1d_misses": (HW_CACHE, L1D | READ << 8 | MISS << 16),
    "context_switches": (SOFTWARE, 3),
    "cpu_migrations": (SOFTWARE, 4),
}

SYSCALL = {"x86_64": 298, "aarch64": 241}
INHERIT, EXCLUDE_KERNEL, EXCLUDE_HV = 1 << 1, 1 << 5, 1 << 6
TIME_ENABLED, TIME_RUNNING = 1 << 0, 1 << 1
FD_CLOEXEC = 1 << 3
PARANOID = Path("/proc/sys/kernel/perf_event_paranoid")
DEVICES = Path("/sys/bus/event_source/devices")
HYBRID_PMUS = ("cpu_core", "cpu_atom")


class Attributes(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
    ]


class PerfUnavailable(Exception):
    pass


_libc = ctypes.CDLL(None, use_errno=True)


def _pmus() -> list[tuple[int, bool]]:
    affinity = os.sched_getaffinity(0)
    found = []
    for name in HYBRID_PMUS:
        directory = DEVICES / name
        if directory.exists():
            cpus = set(parse_cpu_list(read(directory / "cpus")) or [])
            if cpus & affinity:
                found.append((int(read(directory / "type")), affinity <= cpus))
    return found or [(0, True)]


def _open(kind: int, config: int, pmu: int, kernel: bool) -> int:
    attributes = Attributes(
        type=kind,
        size=ctypes.sizeof(Attributes),
        config=config | pmu << 32,
        read_format=TIME_ENABLED | TIME_RUNNING,
        flags=INHERIT | EXCLUDE_HV | (0 if kernel else EXCLUDE_KERNEL),
    )
    fd = _libc.syscall(SYSCALL[platform.machine()], ctypes.byref(attributes), 0, -1, -1, FD_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    return fd


class Counters:
    def __init__(self):
        if platform.machine() not in SYSCALL:
            raise PerfUnavailable(f"perf_event_open is not wired up for {platform.machine()}")
        self.fds: dict[str, list[tuple[int, bool]]] = {}
        self.kernel = True
        pmus = _pmus()
        try:
            for name, (kind, config) in EVENTS.items():
                for pmu, scalable in pmus if kind != SOFTWARE else [(0, True)]:
                    try:
                        fd = _open(kind, config, pmu, self.kernel)
                    except OSError as error:
                        if error.errno in (errno.EACCES, errno.EPERM) and self.kernel and not self.fds:
                            self.kernel = False
                            fd = _open(kind, config, pmu, self.kernel)
                        elif error.errno in (errno.EACCES, errno.EPERM):
                            raise
                        else:
                            continue
                    self.fds.setdefault(name, []).append((fd, scalable))
        except OSError as error:
            self.close()
            paranoid = PARANOID.read_text().strip() if PARANOID.exists() else "unknown"
            raise PerfUnavailable(f"perf_event_open failed: {error.strerror} (perf_event_paranoid={paranoid})") from None

    def read(self) -> dict[str, list[tuple[bool, int, int, int]]]:
        return {
            name: [(scalable, *struct.unpack("QQQ", os.read(fd, 24))) for fd, scalable in opened]
            for name, opened in self.fds.items()
        }

    def close(self) -> None:
        for opened in self.fds.values():
            for fd, _ in opened:
                os.close(fd)
        self.fds = {}


def difference(start: dict[str, list[tuple[bool, int, int, int]]], end: dict[str, list[tuple[bool, int, int, int]]]) -> dict[str, float]:
    counts = {}
    for name, readings in end.items():
        total = 0.0
        for (scalable, value, enabled, running), (_, first_value, first_enabled, first_running) in zip(readings, start[name]):
            counted = value - first_value
            ran = running - first_running
            total += counted * (enabled - first_enabled) / ran if scalable and ran > 0 else counted
        counts[name] = total
    return counts
