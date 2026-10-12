# SPDX-License-Identifier: BSD-3-Clause
"""Runs one of Newton's examples in Ernie's viewer::

python -m ernie.integrations.newton basic_pendulum
python -m ernie.integrations.newton robot_anymal_d --num-frames 600
python -m ernie.integrations.newton --list
"""

import newton.examples

from . import install

install(default=True)
newton.examples.main()
