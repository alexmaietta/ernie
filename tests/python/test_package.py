# SPDX-License-Identifier: BSD-3-Clause
"""Tests of the Python package's basics."""

import re

import ernie


def test_version_is_semantic():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", ernie.__version__)


def test_devices_are_listed():
    devices = ernie.vulkan_devices()
    assert all(isinstance(device.name, str) for device in devices)
