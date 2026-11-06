# SPDX-License-Identifier: BSD-3-Clause
import re

import ernie


def test_exports_resolve():
    missing = [name for name in ernie.__all__ if not hasattr(ernie, name)]
    assert missing == []


def test_reports_its_version_and_build():
    assert re.match(r"^\d+\.\d+\.\d+", ernie.__version__)
    info = ernie.build_info()
    assert info.version
    assert ernie.backend_name(ernie.Backend.VULKAN) == "vulkan"


def test_lists_adapters():
    adapters = ernie.list_adapters()
    assert all(adapter.alias and adapter.name for adapter in adapters)
