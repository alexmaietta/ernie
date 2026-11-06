# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import platform
import sys
from glob import glob
from importlib import metadata
from pathlib import Path

import pynvml

from ernie_bench.system.nvml import devices, query

CORES_PER_SM = {
    (7, 0): 64,
    (7, 5): 64,
    (8, 0): 64,
    (8, 6): 128,
    (8, 7): 128,
    (8, 9): 128,
    (9, 0): 128,
    (10, 0): 128,
    (10, 3): 128,
    (12, 0): 128,
    (12, 1): 128,
}

PACKAGES = ("ernie-bench", "ernie-renderer", "numpy", "pyarrow", "nvidia-ml-py")


def read(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def parse_cpu_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    cpus = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def _cpuinfo(field: str) -> str | None:
    for line in (read("/proc/cpuinfo") or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == field:
            return value.strip()
    return None


def _megahertz(path: str) -> float | None:
    khz = read(path)
    return int(khz) / 1000 if khz else None


def performance_cpus() -> list[int]:
    return parse_cpu_list(read("/sys/devices/cpu_core/cpus"))


def cpu() -> dict:
    return {
        "model": _cpuinfo("model name") or platform.processor(),
        "microcode": _cpuinfo("microcode"),
        "logical_cpus": os.cpu_count(),
        "performance_cpus": performance_cpus(),
        "efficiency_cpus": parse_cpu_list(read("/sys/devices/cpu_atom/cpus")),
        "min_mhz": _megahertz("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq"),
        "max_mhz": _megahertz("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"),
        "smt": read("/sys/devices/system/cpu/smt/control"),
        "scaling_driver": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"),
    }


def memory() -> dict:
    total = None
    for line in (read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) * 1024
    return {"total_bytes": total, "numa_nodes": len(glob("/sys/devices/system/node/node[0-9]*")) or None}


def operating_system() -> dict:
    release = dict(
        line.split("=", 1) for line in (read("/etc/os-release") or "").splitlines() if "=" in line
    )
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "distribution": release.get("PRETTY_NAME", "").strip('"') or None,
        "hostname": platform.node(),
        "python": sys.version,
    }


def gpus() -> list[dict]:
    with devices() as handles:
        driver = query(pynvml.nvmlSystemGetDriverVersion)
        cuda = query(pynvml.nvmlSystemGetCudaDriverVersion_v2)
        described = []
        for index, handle in enumerate(handles):
            capability = query(pynvml.nvmlDeviceGetCudaComputeCapability, handle)
            cores = query(pynvml.nvmlDeviceGetNumGpuCores, handle)
            memory_info = query(pynvml.nvmlDeviceGetMemoryInfo, handle)
            enforced = query(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
            default = query(pynvml.nvmlDeviceGetPowerManagementDefaultLimit, handle)
            described.append(
                {
                    "index": index,
                    "name": query(pynvml.nvmlDeviceGetName, handle),
                    "uuid": query(pynvml.nvmlDeviceGetUUID, handle),
                    "pci_bus_id": getattr(query(pynvml.nvmlDeviceGetPciInfo, handle), "busId", None),
                    "driver": driver,
                    "cuda_driver": f"{cuda // 1000}.{cuda % 1000 // 10}" if cuda else None,
                    "vbios": query(pynvml.nvmlDeviceGetVbiosVersion, handle),
                    "compute_capability": ".".join(map(str, capability)) if capability else None,
                    "cuda_cores": cores,
                    "sm_count": cores // CORES_PER_SM[capability] if cores and capability in CORES_PER_SM else None,
                    "memory_bytes": memory_info.total if memory_info else None,
                    "max_sm_clock_mhz": query(pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_SM),
                    "max_memory_clock_mhz": query(pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_MEM),
                    "default_power_limit_watts": default / 1000 if default else None,
                    "enforced_power_limit_watts": enforced / 1000 if enforced else None,
                    "max_pcie_generation": query(pynvml.nvmlDeviceGetMaxPcieLinkGeneration, handle),
                    "max_pcie_width": query(pynvml.nvmlDeviceGetMaxPcieLinkWidth, handle),
                    "ecc": query(pynvml.nvmlDeviceGetEccMode, handle),
                }
            )
        return described


def packages() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def snapshot() -> dict:
    return {
        "cpu": cpu(),
        "memory": memory(),
        "os": operating_system(),
        "gpus": gpus(),
        "packages": packages(),
    }
