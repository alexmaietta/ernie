# SPDX-License-Identifier: BSD-3-Clause
import subprocess
from pathlib import Path

import pytest

from ernie_bench import profile
from ernie_bench.config import Profile

NCU = '''==PROF== Connected to process 1
"ID","Process ID","Kernel Name","gpu__time_duration.sum","sm__warps_active.avg.pct_of_peak_sustained_active","launch__registers_per_thread"
"","","","ns","%","register/thread"
"0","1","trace","1,500","40.5","96"
"1","1","shade","500","n/a","32"
'''

NSYS = '''Generating SQLite file report.sqlite from report.nsys-rep
Processing [report.sqlite] with [cuda_gpu_kern_sum.py](base)...
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
75.0,3000000,2,1500000.0,1500000.0,1400000,1600000,1000.0,trace
25.0,1000000,2,500000.0,500000.0,400000,600000,1000.0,shade
'''


def test_ncu_rows_become_samples_per_launch():
    samples = profile.parse_ncu(NCU)
    assert {(s.phase, s.metric, s.value, s.iteration) for s in samples} == {
        ("kernel/trace", "ncu.gpu__time_duration.sum", 1500.0, 0),
        ("kernel/trace", "ncu.sm__warps_active.avg.pct_of_peak_sustained_active", 40.5, 0),
        ("kernel/trace", "ncu.launch__registers_per_thread", 96.0, 0),
        ("kernel/shade", "ncu.gpu__time_duration.sum", 500.0, 1),
        ("kernel/shade", "ncu.launch__registers_per_thread", 32.0, 1),
    }


def test_nsys_summary_skips_the_preamble_and_converts_to_seconds():
    samples = {(s.phase, s.metric): s.value for s in profile.parse_nsys(NSYS)}
    assert samples[("kernel/trace", "nsys.total_seconds")] == pytest.approx(0.003)
    assert samples[("kernel/shade", "nsys.instances")] == 2
    assert samples[("kernel/trace", "nsys.time_fraction")] == pytest.approx(0.75)


def test_profilers_wrap_the_worker_and_stay_off_until_the_harness_starts_them():
    ncu = profile.command("ncu", Path("out.ncu-rep"))
    assert ncu[ncu.index("--profile-from-start") + 1] == "off"
    nsys = profile.command("nsys", Path("out.nsys-rep"))
    assert nsys[nsys.index("--capture-range") + 1] == "cudaProfilerApi"
    with pytest.raises(ValueError):
        Profile("perf")


def test_system_prep_splits_cpus_between_benchmark_and_housekeeping():
    script = Path(__file__).parents[1] / "scripts" / "system-prep"
    functions = subprocess.run(["sed", "-n", "/^expand_cpus()/,/^}/p;/^cpu_mask()/,/^}/p", str(script)], capture_output=True, text=True).stdout
    result = subprocess.run(["bash", "-c", functions + "expand_cpus 2-4,9; cpu_mask 0,1,8-9"], capture_output=True, text=True, check=True)
    assert result.stdout.split("\n")[:2] == ["2 3 4 9", "303"]
