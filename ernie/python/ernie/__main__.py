# SPDX-License-Identifier: BSD-3-Clause
"""The `ernie` command a pip install puts on the path, and `python -m
ernie`: the native `ernie` program the package ships beside this file
(the same program as the C++ build's), which this replaces itself
with.

    ernie
        the viewer on the default scene, with the Demos panel
    ernie scene.json
        the viewer on a scene file (ernie.save_scene's JSON)
    ernie scene.json --output out.png
        renders the scene file to a PNG
    ernie --demo lighting/illuminants
        the viewer with one of the C++ demos running
    ernie --list
        the C++ demos' names

The Python demos run with `python -m ernie.demos`.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Runs the native `ernie` program with this command's arguments:
    in place of this process where the platform can (exec), else as a
    child whose exit status is returned."""
    program = Path(__file__).with_name(
        "ernie.exe" if os.name == "nt" else "ernie"
    )
    if not program.is_file():
        print(
            f"ernie: the native program is not installed ({program})",
            file=sys.stderr,
        )
        return 2
    arguments = [str(program), *sys.argv[1:]]
    if os.name == "nt":
        return subprocess.call(arguments)
    os.execv(program, arguments)


if __name__ == "__main__":
    sys.exit(main())
