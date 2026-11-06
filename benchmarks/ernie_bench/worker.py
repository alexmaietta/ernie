# SPDX-License-Identifier: BSD-3-Clause
from time import perf_counter

ENTERED = perf_counter()

import json
import sys
from dataclasses import astuple
from pathlib import Path

from ernie_bench import __version__
from ernie_bench.datasets import build_scene
from ernie_bench.harness import PROTOCOL, attempt
from ernie_bench.results import Sample
from ernie_bench.config import Capture, Measurement, SceneSpec, Workload
from ernie_bench.system.monitor import Monitor

IMPORTED = perf_counter()


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    if job["protocol"] != PROTOCOL:
        result = {"status": "error", "reason": f"orchestrator speaks protocol {job['protocol']}, worker expects {PROTOCOL}"}
    else:
        workload = Workload(**job["workload"])
        measurement = Measurement(**job["measurement"])
        scene = build_scene(SceneSpec(**job["scene"]), workload.worlds, workload.cameras_per_world)
        locked = {int(gpu): clock for gpu, clock in job["locked"].items()}
        with Monitor(measurement.monitor_interval, locked) as monitor:
            capture = Capture(**job["capture"]) if job["capture"] else None
            captures = Path(job["captures"]) if job["captures"] else None
            result = attempt(job["backend"], scene, workload, measurement, monitor, capture, captures, job.get("profiled_frames"))
        startup = [
            Sample("startup", "interpreter_seconds", ENTERED - job["spawned"]),
            Sample("startup", "harness_import_seconds", IMPORTED - ENTERED),
        ]
        result["samples"] = [astuple(sample) for sample in startup + result.get("samples", [])]
    result["protocol"] = PROTOCOL
    result["worker"] = {"python": sys.version.split()[0], "executable": sys.executable, "ernie_bench": __version__}
    Path(job["result"]).write_text(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])
