# SPDX-License-Identifier: BSD-3-Clause
"""Ernie in Newton: ``ViewerErnie``, a Newton viewer drawn by Ernie,
and ``install``, which adds it to Newton's examples as ``--viewer
ernie``. ``python -m ernie.integrations.newton <example>`` runs one of
Newton's examples in it, taking the options ``python -m
newton.examples <example>`` takes."""

from __future__ import annotations

import argparse
import functools

from .viewer import ViewerErnie

__all__ = ["ViewerErnie", "install"]

_installed = False


def install(default: bool = False) -> None:
    """Adds ``ernie`` to the ``--viewer`` choices of Newton's examples
    (``newton.examples.create_parser`` and ``init``), as the default
    with ``default``. Benchmarks keep Newton's null viewer."""
    global _installed
    import newton.examples as examples

    if _installed:
        return
    _installed = True
    create_parser, init = examples.create_parser, examples.init

    @functools.wraps(create_parser)
    def create_parser_with_ernie(*args, **kwargs):
        parser = create_parser(*args, **kwargs)
        for action in parser._actions:
            if action.dest == "viewer":
                action.choices = [*action.choices, "ernie"]
                if default:
                    action.default = "ernie"
                action.help = "Viewer to use, or ernie for Ernie's."
        # Newton's init builds its null viewer for "ernie", which
        # init_with_ernie then replaces.
        for method in ("parse_args", "parse_known_args"):
            setattr(
                parser,
                method,
                _ernie_as_null(getattr(parser, method)),
            )
        return parser

    @functools.wraps(init)
    def init_with_ernie(parser=None):
        viewer, args = init(parser)
        if getattr(args, "ernie_viewer", False) and (
            args.benchmark is False
        ):
            viewer.close()
            viewer = ViewerErnie(
                headless=getattr(args, "headless", False),
                paused=getattr(args, "paused", False),
                num_frames=getattr(args, "num_frames", None),
            )
            args.viewer = "ernie"
        return viewer, args

    examples.create_parser = create_parser_with_ernie
    examples.init = init_with_ernie


def _ernie_as_null(parse):
    @functools.wraps(parse)
    def wrapper(*args, **kwargs):
        result = parse(*args, **kwargs)
        namespace = result[0] if isinstance(result, tuple) else result
        if isinstance(namespace, argparse.Namespace) and (
            getattr(namespace, "viewer", None) == "ernie"
        ):
            namespace.viewer = "null"
            namespace.ernie_viewer = True
        return result

    return wrapper
