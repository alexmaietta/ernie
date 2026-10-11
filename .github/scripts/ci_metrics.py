# SPDX-License-Identifier: BSD-3-Clause
"""One CI run's metrics for github-action-benchmark (customSmallerIsBetter): the run's wall time,
each successful job's time and its build and test steps' from the Actions API, and from the jobs'
Ninja logs their compile time, compile count and ccache miss rate.

    GH_TOKEN=... GITHUB_REPOSITORY=owner/repo python ci_metrics.py RUN_ID LOG_DIR OUTPUT

LOG_DIR holds one directory per job's artifact (ninja-log-<job>), each with the build's .ninja_log
and, where the job uses ccache, ccache-stats.tsv (`ccache --print-stats`).
"""

import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# The steps whose times are worth tracking on their own: building, testing, wheels and packages.
STEPS = ("Build", "Test on lavapipe", "Build the packages", "Run pypa/cibuildwheel")


def api(path):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def seconds(start, end):
    parse = lambda stamp: datetime.fromisoformat(stamp.replace("Z", "+00:00"))  # noqa: E731
    return (parse(end) - parse(start)).total_seconds()


def metric(name, unit, value):
    return {"name": name, "unit": unit, "value": round(value, 2)}


def run_metrics(run_id):
    run = api(f"actions/runs/{run_id}")
    metrics = [metric("CI: wall time", "min", seconds(run["run_started_at"], run["updated_at"]) / 60)]
    jobs = api(f"actions/runs/{run_id}/jobs?per_page=100")["jobs"]
    for job in jobs:
        if job["conclusion"] != "success" or not job["completed_at"]:
            continue
        name = job["name"]
        metrics.append(metric(f"{name}: job", "min", seconds(job["started_at"], job["completed_at"]) / 60))
        for step in job.get("steps", []):
            if step["conclusion"] == "success" and step["name"].startswith(STEPS):
                label = "wheel build" if step["name"].startswith("Run pypa/cibuildwheel") else step["name"]
                duration = seconds(step["started_at"], step["completed_at"]) / 60
                metrics.append(metric(f"{name}: {label}", "min", duration))
    return metrics


def ninja_metrics(job, log):
    """Compile time and count from a .ninja_log: a "# ninja log vN" header, then a line per
    output (start and end in ms, mtime, output, hash), the last of an output rebuilt counting."""
    outputs = {}
    for line in log.read_text().splitlines():
        fields = line.split("\t")
        if line.startswith("#") or len(fields) < 4:
            continue
        outputs[fields[3]] = int(fields[1]) - int(fields[0])
    compiles = [ms for output, ms in outputs.items() if output.endswith((".o", ".obj"))]
    if not compiles:
        return []
    return [
        metric(f"{job}: compile time", "min", sum(compiles) / 60000),
        metric(f"{job}: compiles", "files", len(compiles)),
    ]


def ccache_metrics(job, stats):
    """The compiler cache's miss rate, from `ccache --print-stats` (key, tab, value per line)."""
    values = dict(line.split("\t", 1) for line in stats.read_text().splitlines() if "\t" in line)
    hits = int(values.get("direct_cache_hit", 0)) + int(values.get("preprocessed_cache_hit", 0))
    misses = int(values.get("cache_miss", 0))
    if hits + misses == 0:
        return []
    return [metric(f"{job}: ccache miss rate", "%", 100 * misses / (hits + misses))]


def main():
    run_id, log_dir, output = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    metrics = run_metrics(run_id)
    for artifact in sorted(log_dir.glob("ninja-log-*")):
        job = artifact.name.removeprefix("ninja-log-")
        if (artifact / ".ninja_log").exists():
            metrics += ninja_metrics(job, artifact / ".ninja_log")
        if (artifact / "ccache-stats.tsv").exists():
            metrics += ccache_metrics(job, artifact / "ccache-stats.tsv")
    output.write_text(json.dumps(metrics, indent=2))
    print(f"{len(metrics)} metrics written to {output}")


if __name__ == "__main__":
    main()
