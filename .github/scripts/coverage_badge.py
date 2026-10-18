# SPDX-License-Identifier: BSD-3-Clause
"""Writes a shields.io endpoint badge and a job summary from gcovr's JSON summary."""

import json
import os
import sys
from pathlib import Path


def badge_colour(percent: float) -> str:
    if percent >= 80.0:
        return "brightgreen"
    if percent >= 70.0:
        return "green"
    if percent >= 60.0:
        return "yellow"
    return "red"


def main(report: Path) -> None:
    summary = json.loads((report / "summary.json").read_text())
    lines = summary["line_percent"]
    branches = summary["branch_percent"]
    badge = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{lines:.1f}%",
        "color": badge_colour(lines),
    }
    (report / "badge.json").write_text(json.dumps(badge))
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a") as out:
            out.write(f"### Coverage\n\nLines {lines:.1f}%, branches {branches:.1f}%\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
