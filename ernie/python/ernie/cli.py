# SPDX-License-Identifier: BSD-3-Clause
"""The `ernie` command a pip install puts on the path, and `python -m ernie`: the same program as
the C++ build's `ernie` (ernie/app/cli.h).

    ernie                               the viewer on the default scene, with the Demos panel
    ernie scene.json                    the viewer on a scene file (ernie.save_scene's JSON)
    ernie scene.json --output out.png   renders the scene file to a PNG
    ernie --demo lighting/illuminants   the viewer with one of the C++ demos running
    ernie --list                        the C++ demos' names

The Python demos run with `python -m ernie.demos`.
"""

import sys
from importlib import metadata

from . import _ernie


def version():
    """The installed package's version, or the build's when Ernie runs from a build tree."""
    try:
        return metadata.version("ernie-renderer")
    except metadata.PackageNotFoundError:
        return "dev"


def main(argv=None):
    """Runs the `ernie` program with `argv` (the command line without the program name); returns
    its exit status."""
    if not hasattr(_ernie, "app"):
        print("ernie: this build of Ernie has no viewer (ERNIE_BUILD_APP is off)", file=sys.stderr)
        return 2
    return _ernie.app.main(list(sys.argv[1:] if argv is None else argv), version())
