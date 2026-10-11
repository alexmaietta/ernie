# SPDX-License-Identifier: BSD-3-Clause
"""Runs the README's Python examples as written, so they keep working
as the API changes."""

import pathlib
import re
import runpy

import pytest

import ernie

README = pathlib.Path(__file__).resolve().parents[3] / "README.md"


def python_blocks():
    return re.findall(
        r"```python\n(.*?)```", README.read_text(), re.DOTALL
    )


def test_the_readme_has_python_examples():
    assert python_blocks()


@pytest.mark.parametrize("index", range(len(python_blocks())))
def test_readme_python_example_runs(index, tmp_path, monkeypatch):
    pytest.importorskip("PIL.Image")
    try:
        ernie.Renderer()
    except RuntimeError as error:
        pytest.skip(f"No Vulkan device: {error}")
    script = tmp_path / f"readme_{index}.py"
    script.write_text(python_blocks()[index])
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")
