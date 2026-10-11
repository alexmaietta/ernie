# SPDX-License-Identifier: BSD-3-Clause
"""`python -m ernie`: the `ernie` program (see ernie.cli)."""

import sys

from .cli import main

sys.exit(main())
