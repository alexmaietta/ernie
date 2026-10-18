# SPDX-License-Identifier: BSD-3-Clause
"""The generated JSON Schema, checked by an independent validator: it is a valid draft 2020-12
schema, accepts the scene fixtures and every reference page's example, and rejects typos."""

import json
import os
import pathlib
import re
import subprocess

import pytest

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("schema")
    subprocess.run(
        [
            os.environ["ERNIE_EXE"],
            "schema",
            "--json-schema",
            str(out / "scene.schema.json"),
            "--markdown",
            str(out / "reference"),
        ],
        check=True,
    )
    return out


@pytest.fixture(scope="module")
def validator(generated):
    schema = json.loads((generated / "scene.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def errors(validator, scene):
    return [f"{e.json_path}: {e.message}" for e in validator.iter_errors(scene)]


def test_fixtures_pass(validator):
    scenes = pathlib.Path(os.environ["ERNIE_TEST_FIXTURES"]) / "scenes"
    for name in ("rover.json", "rover.saved.json"):
        assert errors(validator, json.loads((scenes / name).read_text())) == [], name


def test_reference_examples_pass(validator, generated):
    examples = 0
    for page in sorted((generated / "reference").glob("*.md")):
        for block in re.findall(r"```json\n(.*?)```", page.read_text(), re.S):
            assert errors(validator, json.loads(block)) == [], page.name
            examples += 1
    assert examples > 40


def test_typos_and_bad_values_fail(validator):
    scene = {
        "prims": {
            "/Ball": {"typeName": "Sphere", "radiuss": 1},
            "/Lamp": {"typeName": "SphereLight", "inputs:radius": -1},
            "/Group": {"anything": 1},
        }
    }
    found = "\n".join(errors(validator, scene))
    assert "radiuss" in found
    assert "-1 is less than the minimum" in found
    assert "anything" not in found, "typeless prims take any attribute"
