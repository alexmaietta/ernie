# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from contextlib import contextmanager

import pynvml


def query(function, *args):
    try:
        return function(*args)
    except pynvml.NVMLError:
        return None


@contextmanager
def devices():
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        yield []
        return
    try:
        yield [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    finally:
        pynvml.nvmlShutdown()
