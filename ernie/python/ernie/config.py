# SPDX-License-Identifier: BSD-3-Clause
"""What Ernie prints about itself, as Warp's ``warp.config``.

``ernie.config.quiet = True`` prints nothing; ``ernie.config.verbose =
True`` adds a line for each shader its cache answers to the banner
(Ernie's version, device and shader cache, once it first opens a
device) and the lines for shaders it compiles. Set them before the
first render or viewer to silence the banner too. They start from the
environment: ``ERNIE_QUIET=1`` or ``ERNIE_VERBOSE=1``.
"""

import sys
import types

from . import _ernie

_QUIET, _NORMAL, _VERBOSE = 0, 1, 2


class _Config(types.ModuleType):
    @property
    def quiet(self) -> bool:
        """Whether Ernie prints nothing about itself."""
        return _ernie._verbosity() == _QUIET

    @quiet.setter
    def quiet(self, on: bool) -> None:
        _ernie._set_verbosity(_QUIET if on else _NORMAL)

    @property
    def verbose(self) -> bool:
        """Whether Ernie also prints the shaders its cache answers."""
        return _ernie._verbosity() == _VERBOSE

    @verbose.setter
    def verbose(self, on: bool) -> None:
        _ernie._set_verbosity(_VERBOSE if on else _NORMAL)


sys.modules[__name__].__class__ = _Config
