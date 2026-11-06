# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import resource
from glob import glob
from pathlib import Path

import pynvml

from ernie_bench.system.machine import parse_cpu_list, performance_cpus, read
from ernie_bench.system.nvml import devices, query

SYSTEM_PREP_STATE = Path("/run/ernie-bench/system-prep.env")
BUSY_LOAD = 1.5
SCHEDULERS = {os.SCHED_OTHER: "other", os.SCHED_FIFO: "fifo", os.SCHED_RR: "rr", os.SCHED_BATCH: "batch", os.SCHED_IDLE: "idle"}


def mps_running() -> bool:
    return (Path(os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")) / "control").exists()


def system_prep_state(path: Path | None = None) -> dict | None:
    text = read(path or SYSTEM_PREP_STATE)
    if text is None:
        return None
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def _values(pattern: str) -> list[str]:
    return sorted({value for path in glob(pattern) if (value := read(path)) is not None})


def current() -> dict:
    mains = [
        read(Path(supply) / "online") == "1"
        for supply in glob("/sys/class/power_supply/*")
        if read(Path(supply) / "type") == "Mains"
    ]
    with devices() as handles:
        gpu_state = [
            {
                "index": index,
                "persistence_mode": query(pynvml.nvmlDeviceGetPersistenceMode, handle),
                "enforced_power_limit_watts": (limit := query(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)) and limit / 1000,
                "pcie_generation": query(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, handle),
                "pcie_width": query(pynvml.nvmlDeviceGetCurrPcieLinkWidth, handle),
                "temperature_c": query(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU),
            }
            for index, handle in enumerate(handles)
        ]
    return {
        "system_prep": system_prep_state(),
        "cpu": {
            "governors": _values("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor"),
            "energy_performance_preferences": _values("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference"),
            "no_turbo": read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        },
        "mains_power": any(mains) if mains else None,
        "load_average": list(os.getloadavg()),
        "perf_event_paranoid": int(paranoid) if (paranoid := read("/proc/sys/kernel/perf_event_paranoid")) else None,
        "gpus": gpu_state,
        "isolation": {
            "kernel_isolated": parse_cpu_list(read("/sys/devices/system/cpu/isolated")) or [],
            "nohz_full": parse_cpu_list(read("/sys/devices/system/cpu/nohz_full")) or [],
            "services": parse_cpu_list(read("/sys/fs/cgroup/system.slice/cpuset.cpus.effective")),
            "default_irq_affinity": read("/proc/irq/default_smp_affinity"),
        },
        "mps": mps_running(),
        "process": {
            "affinity": sorted(os.sched_getaffinity(0)),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "scheduler": SCHEDULERS.get(os.sched_getscheduler(0), "unknown"),
            "realtime_priority": os.sched_getparam(0).sched_priority,
            "nice_limit": resource.getrlimit(resource.RLIMIT_NICE)[0],
            "realtime_limit": resource.getrlimit(resource.RLIMIT_RTPRIO)[0],
        },
    }


def check(state: dict) -> list[str]:
    warnings = []
    if state["system_prep"] is None:
        warnings.append("system-prep has not been applied, run: sudo scripts/system-prep apply")
    if state["cpu"]["governors"] and state["cpu"]["governors"] != ["performance"]:
        warnings.append(f"cpu governors are {state['cpu']['governors']}, not performance")
    if state.get("load_average") and state["load_average"][0] > BUSY_LOAD:
        warnings.append(f"machine is busy, 1 minute load average is {state['load_average'][0]:.1f}")
    if state["mains_power"] is False:
        warnings.append("running on battery")
    for gpu in state["gpus"]:
        if gpu["persistence_mode"] == 0:
            warnings.append(f"gpu {gpu['index']} persistence mode is off")
    isolation = state.get("isolation", {})
    isolated = set(isolated_cpus(state["system_prep"], isolation.get("kernel_isolated", [])))
    affinity = set(state.get("process", {}).get("affinity", []))
    if isolated and not affinity <= isolated:
        warnings.append(f"cpus {sorted(affinity - isolated)} are not isolated, pass --cpus isolated")
    services = set(isolation.get("services") or [])
    if isolated & services:
        warnings.append(f"system services can still run on isolated cpus {sorted(isolated & services)}")
    if state.get("mps"):
        warnings.append("MPS is running, so every CUDA backend goes through the MPS server")
    return warnings


def isolated_cpus(system_prep: dict | None, kernel_isolated: list[int]) -> list[int]:
    return parse_cpu_list((system_prep or {}).get("APPLIED_ISOLATE")) or kernel_isolated


def pin(selection: str) -> list[int]:
    available = sorted(os.sched_getaffinity(0))
    if selection == "all":
        cpus = available
    elif selection == "performance":
        cpus = performance_cpus() or available
    elif selection == "isolated":
        cpus = isolated_cpus(system_prep_state(), parse_cpu_list(read("/sys/devices/system/cpu/isolated")))
        if not cpus:
            raise ValueError("no isolated cpus, run: sudo scripts/system-prep apply --isolate <cpus>")
    else:
        cpus = parse_cpu_list(selection)
    os.sched_setaffinity(0, cpus)
    return sorted(os.sched_getaffinity(0))


def prioritise(nice: int) -> str | None:
    try:
        os.setpriority(os.PRIO_PROCESS, 0, nice)
    except PermissionError:
        return f"could not set nice {nice} without CAP_SYS_NICE, staying at {os.getpriority(os.PRIO_PROCESS, 0)}; sudo scripts/system-prep apply --realtime allows it"
    return None


def schedule_realtime(priority: int) -> str | None:
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except PermissionError:
        return f"could not use SCHED_FIFO {priority} without CAP_SYS_NICE; sudo scripts/system-prep apply --realtime allows it"
    return None
