# SPDX-License-Identifier: BSD-3-Clause
"""Ernie's demos, shipped with the package: `python -m ernie.demos [name] [options]` runs one, as
`python -m newton.examples` runs Newton's examples.

    python -m ernie.demos                     # the sun panel demo, with the Demos panel
    python -m ernie.demos --list              # every demo's name
    python -m ernie.demos lighting_illuminants --viewer null --output illuminants.png
    python -m ernie.demos lighting_illuminants --help

A demo's name is its file's without `demo_` and `.py` (`lighting_illuminants`). The options after
the name are the demo's: those every demo takes (`ernie.demo.create_parser`) and its own. In a
window, the Demos panel switches between them all.

To write demos of your own, subclass `ernie.demo.Demo` in Python or `ernie::app::Demo` in C++
(ernie/app/demo.h); they run the same way, from your own files.
"""

import os
import runpy
import sys

from ..demo import discover_demos

#: Where the demos are: this package's directory.
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

#: The demo `python -m ernie.demos` runs without a name.
DEFAULT = "viewer_sun_panel"

USAGE = """Usage: python -m ernie.demos [name] [options]
       python -m ernie.demos --list
       python -m ernie.demos name --help

Runs one of Ernie's demos (the sun panel demo without a name); --list lists them, and NAME
--help shows a demo's options."""


def demos():
    """The demos here as {name: path}, by category and name."""
    return {os.path.splitext(os.path.basename(entry.path))[0].removeprefix("demo_"): entry.path
            for entry in discover_demos(DIRECTORY)}


def main(argv=None):
    """Runs the demo named first in `argv` (the command line without the program name) with the
    rest as its options; returns the exit status."""
    argv = sys.argv[1:] if argv is None else list(argv)
    found = demos()
    if argv and argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv and argv[0] == "--list":
        print("\n".join(found))
        return 0
    name = argv.pop(0) if argv and not argv[0].startswith("-") else DEFAULT
    name = name.removeprefix("demo_").removesuffix(".py").replace(" ", "_")
    if name not in found:
        print(f"No demo named '{name}'; these are:\n  " + "\n  ".join(found), file=sys.stderr)
        return 2
    # The demos import the helpers beside them (demo_support).
    if DIRECTORY not in sys.path:
        sys.path.insert(0, DIRECTORY)
    sys.argv = [found[name], *argv]
    runpy.run_path(found[name], run_name="__main__")
    return 0
