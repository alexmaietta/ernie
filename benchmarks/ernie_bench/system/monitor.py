# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
import multiprocessing
import os
from time import perf_counter

import numpy as np
import pynvml

from ernie_bench.system import sampler
from ernie_bench.system.sampler import GpuSample

THROTTLE_REASONS = {
    "sw_power_cap": pynvml.nvmlClocksEventReasonSwPowerCap,
    "sw_thermal": pynvml.nvmlClocksEventReasonSwThermalSlowdown,
    "hw_slowdown": pynvml.nvmlClocksEventReasonHwSlowdown,
    "hw_thermal": pynvml.nvmlClocksEventReasonHwThermalSlowdown,
    "hw_power_brake": pynvml.nvmlClocksEventReasonHwPowerBrakeSlowdown,
}

DISTURBING = {"sw_thermal", "hw_slowdown", "hw_thermal", "hw_power_brake", "below_locked_clock"}


def summarize(samples: list[GpuSample], locked_clock_mhz: float | None = None) -> dict:
    data = np.array([s[2:] for s in samples], np.float64)
    sm, memory_clock, temperature, power, reasons, utilization, memory, process_memory, energy = data.T
    names = {name for name, bit in THROTTLE_REASONS.items() if np.any(reasons.astype(np.int64) & bit)}
    if locked_clock_mhz and sm.min() < 0.97 * locked_clock_mhz:
        names.add("below_locked_clock")
    metered = energy[~np.isnan(energy)]
    return {
        "samples": len(samples),
        "sm_clock_mhz_min": sm.min(),
        "sm_clock_mhz_max": sm.max(),
        "sm_clock_mhz_mean": sm.mean(),
        "memory_clock_mhz_min": memory_clock.min(),
        "temperature_c_max": temperature.max(),
        "power_watts_mean": power.mean(),
        "utilization_mean": utilization.mean(),
        "memory_used_bytes_max": memory.max(),
        "memory_growth_bytes": memory[-1] - memory[0],
        "process_memory_bytes_max": process_memory.max(),
        "energy_joules": metered[-1] - metered[0] if len(metered) >= 2 else math.nan,
        "throttle_reasons": sorted(names),
        "disturbed": bool(names & DISTURBING),
    }


class Monitor:
    def __init__(self, interval: float = 0.05, locked_clocks_mhz: dict[int, float] | None = None):
        self.interval = interval
        self.locked_clocks_mhz = locked_clocks_mhz or {}
        self.gpu_count = 0
        self._samples: list[GpuSample] = []
        self._connection = None
        self._process = None

    def __enter__(self) -> Monitor:
        context = multiprocessing.get_context("forkserver")
        context.set_forkserver_preload(["ernie_bench.system.sampler"])
        self._connection, child = context.Pipe()
        cpus = sorted(set(range(os.cpu_count())) - os.sched_getaffinity(0))
        self._process = context.Process(
            target=sampler.run, args=(child, self.interval, cpus or None, os.getpid()), name="gpu-monitor", daemon=True
        )
        self._process.start()
        child.close()
        self.gpu_count = self._connection.recv()
        return self

    def __exit__(self, *exc) -> None:
        self._connection.send("stop")
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.kill()
        self._connection.close()

    def mark(self) -> float:
        if not self.gpu_count:
            return perf_counter()
        self._connection.send("mark")
        return self._connection.recv()

    def window(self, start: float, end: float) -> list[dict]:
        if self._connection is not None and self.gpu_count:
            self._connection.send("drain")
            self._samples += self._connection.recv()
        inside = [s for s in self._samples if start <= s.time <= end]
        self._samples = [s for s in self._samples if s.time > end]
        return [
            {"gpu": index} | summarize([s for s in inside if s.gpu == index], self.locked_clocks_mhz.get(index))
            for index in sorted({s.gpu for s in inside})
        ]
