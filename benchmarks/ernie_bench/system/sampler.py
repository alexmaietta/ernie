# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
from time import perf_counter
from typing import NamedTuple

import pynvml

from ernie_bench.system.nvml import devices, query


class GpuSample(NamedTuple):
    time: float
    gpu: int
    sm_clock_mhz: float
    memory_clock_mhz: float
    temperature_c: float
    power_watts: float
    reasons: int
    utilization: float
    memory_used_bytes: float
    process_memory_bytes: float
    energy_joules: float


def _process_memory(handle, pid: int) -> float:
    used = 0
    for function in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
        for process in query(function, handle) or []:
            if process.pid == pid and process.usedGpuMemory:
                used += process.usedGpuMemory
    return float(used)


def read(handles, pid: int, energy: bool) -> list[GpuSample]:
    now = perf_counter()
    samples = []
    for index, handle in enumerate(handles):
        memory = query(pynvml.nvmlDeviceGetMemoryInfo, handle)
        utilization = query(pynvml.nvmlDeviceGetUtilizationRates, handle)
        joules = query(pynvml.nvmlDeviceGetTotalEnergyConsumption, handle) if energy else None
        samples.append(
            GpuSample(
                now,
                index,
                query(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM) or 0,
                query(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM) or 0,
                query(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU) or 0,
                (query(pynvml.nvmlDeviceGetPowerUsage, handle) or 0) / 1000,
                query(pynvml.nvmlDeviceGetCurrentClocksEventReasons, handle) or 0,
                utilization.gpu if utilization else 0,
                memory.used if memory else 0,
                _process_memory(handle, pid),
                joules / 1000 if joules is not None else float("nan"),
            )
        )
    return samples


def run(connection, interval: float, cpus: list[int] | None, pid: int) -> None:
    if cpus:
        os.sched_setaffinity(0, cpus)
    with devices() as handles:
        connection.send(len(handles))
        samples = []
        while True:
            if not connection.poll(interval):
                samples += read(handles, pid, energy=False)
                continue
            message = connection.recv()
            if message == "mark":
                marked = read(handles, pid, energy=True)
                samples += marked
                connection.send(marked[0].time)
            elif message == "drain":
                connection.send(samples)
                samples = []
            else:
                break
