# SPDX-License-Identifier: BSD-3-Clause
import os

import numpy as np
import pynvml
import pytest

from ernie_bench.system import machine, tuning
from ernie_bench.system.monitor import GpuSample, Monitor, summarize


@pytest.mark.parametrize(
    "text, cpus",
    [("0-5", [0, 1, 2, 3, 4, 5]), ("0,2,4-6", [0, 2, 4, 5, 6]), ("", None), (None, None)],
)
def test_parse_cpu_list(text, cpus):
    assert machine.parse_cpu_list(text) == cpus


def gpu_sample(time, sm=2000, reasons=0, energy=float("nan"), gpu=0, memory=1e9, process=0.0, utilization=50.0):
    return GpuSample(time, gpu, sm, 12000, 60, 80.0, reasons, utilization, memory, process, energy)


def test_summarize_flags_disturbing_reasons_only():
    calm = summarize([gpu_sample(0, 2000, pynvml.nvmlClocksEventReasonSwPowerCap, 10), gpu_sample(1, 2100, 0, 70)])
    assert calm["throttle_reasons"] == ["sw_power_cap"]
    assert not calm["disturbed"]
    assert calm["energy_joules"] == 60
    assert calm["sm_clock_mhz_min"] == 2000

    hot = summarize([gpu_sample(0, 2000, pynvml.nvmlClocksEventReasonHwThermalSlowdown)])
    assert hot["disturbed"]


def test_summarize_flags_clocks_below_the_lock():
    assert summarize([gpu_sample(0, 2100), gpu_sample(1, 1900)], locked_clock_mhz=2100)["throttle_reasons"] == ["below_locked_clock"]
    assert not summarize([gpu_sample(0, 2095)], locked_clock_mhz=2100)["disturbed"]


def test_summarize_uses_only_metered_samples_for_energy():
    samples = [gpu_sample(0, energy=10.0), gpu_sample(1), gpu_sample(2), gpu_sample(3, energy=25.0)]
    assert summarize(samples)["energy_joules"] == 15
    assert np.isnan(summarize([gpu_sample(0, energy=10.0), gpu_sample(1)])["energy_joules"])


def test_summarize_tracks_memory_and_utilization():
    window = summarize(
        [
            gpu_sample(0, memory=1.0e9, process=0.2e9, utilization=20),
            gpu_sample(1, memory=1.5e9, process=0.6e9, utilization=80),
            gpu_sample(2, memory=1.25e9, process=0.4e9, utilization=50),
        ]
    )
    assert window["memory_used_bytes_max"] == 1.5e9
    assert window["memory_growth_bytes"] == 0.25e9
    assert window["process_memory_bytes_max"] == 0.6e9
    assert window["utilization_mean"] == 50


def test_monitor_windows_are_bounded_by_marks():
    with Monitor(interval=0.01) as monitor:
        start = monitor.mark()
        end = monitor.mark()
        windows = monitor.window(start, end)
    for window in windows:
        assert window["samples"] >= 2


def test_snapshot_and_tuning_describe_this_machine():
    snapshot = machine.snapshot()
    assert snapshot["cpu"]["logical_cpus"] == os.cpu_count()
    assert snapshot["packages"]["ernie-bench"]
    state = tuning.current()
    assert state["process"]["affinity"] == sorted(os.sched_getaffinity(0))
    assert isinstance(tuning.check(state), list)


def test_check_reports_untuned_settings():
    state = {
        "system_prep": None,
        "cpu": {"governors": ["powersave"]},
        "mains_power": False,
        "gpus": [{"index": 0, "persistence_mode": 0}],
    }
    warnings = tuning.check(state)
    assert len(warnings) == 4


def test_system_prep_state_parses_env_file(tmp_path):
    path = tmp_path / "system-prep.env"
    path.write_text("GPU=0\nAPPLIED_GPU_CLOCK=2100\nAPPLIED_NO_TURBO=\n")
    assert tuning.system_prep_state(path) == {"GPU": "0", "APPLIED_GPU_CLOCK": "2100", "APPLIED_NO_TURBO": ""}
    assert tuning.system_prep_state(tmp_path / "missing") is None


def test_pin_restricts_affinity():
    original = os.sched_getaffinity(0)
    try:
        first = min(original)
        assert tuning.pin(str(first)) == [first]
        assert tuning.pin("all") == [first]
    finally:
        os.sched_setaffinity(0, original)


def test_monitor_windows_only_include_samples_inside_their_bounds():
    monitor = Monitor()
    monitor._samples = [
        gpu_sample(0.5, 9999, pynvml.nvmlClocksEventReasonHwThermalSlowdown, 0.0),
        gpu_sample(1.0, 2000, energy=100.0),
        gpu_sample(1.5, 1500, gpu=1, energy=7.0),
        gpu_sample(2.0, 2100, energy=130.0),
        gpu_sample(2.5, 1600, gpu=1, energy=9.0),
        gpu_sample(3.0, 9999, pynvml.nvmlClocksEventReasonHwThermalSlowdown, 500.0),
    ]
    gpu0, gpu1 = monitor.window(1.0, 2.5)
    assert gpu0["gpu"] == 0 and gpu0["samples"] == 2
    assert gpu0["sm_clock_mhz_max"] == 2100
    assert gpu0["energy_joules"] == 30
    assert not gpu0["disturbed"]
    assert gpu1["gpu"] == 1 and gpu1["energy_joules"] == 2
    assert [s.time for s in monitor._samples] == [3.0]


def test_monitor_sampler_runs_off_the_benchmark_cpus():
    original = os.sched_getaffinity(0)
    benchmark = {min(original)}
    try:
        os.sched_setaffinity(0, benchmark)
        with Monitor(interval=0.01) as monitor:
            sampler = os.sched_getaffinity(monitor._process.pid)
    finally:
        os.sched_setaffinity(0, original)
    assert sampler == set(range(os.cpu_count())) - benchmark
