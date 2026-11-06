# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ernie_bench.config import Workload

SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("backend", pa.string()),
        ("scene", pa.string()),
        ("workload", pa.string()),
        ("worlds", pa.int32()),
        ("cameras_per_world", pa.int32()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("outputs", pa.string()),
        ("tier", pa.string()),
        ("spp", pa.int32()),
        ("sm_percent", pa.int32()),
        ("process", pa.int32()),
        ("phase", pa.string()),
        ("metric", pa.string()),
        ("repeat", pa.int32()),
        ("iteration", pa.int32()),
        ("value", pa.float64()),
    ]
)


@dataclass
class Sample:
    phase: str
    metric: str
    value: float
    repeat: int = -1
    iteration: int = -1


class RunWriter:
    def __init__(self, root: Path, manifest: dict):
        self.manifest = manifest
        self.directory = Path(root) / manifest["run_id"]
        (self.directory / "rows").mkdir(parents=True)
        self._part = 0
        self.save_manifest()

    def save_manifest(self) -> None:
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))

    def write(self, backend: str, scene: str, workload: Workload, samples: list[Sample], process: int = 0) -> None:
        context = {
            "run_id": self.manifest["run_id"],
            "backend": backend,
            "scene": scene,
            "workload": json.dumps(asdict(workload)),
            "worlds": workload.worlds,
            "cameras_per_world": workload.cameras_per_world,
            "width": workload.width,
            "height": workload.height,
            "outputs": ",".join(workload.outputs),
            "tier": workload.tier,
            "spp": workload.spp,
            "sm_percent": workload.sm_percent,
            "process": process,
        }
        table = pa.Table.from_pylist([context | asdict(sample) for sample in samples], schema=SCHEMA)
        pq.write_table(table, self.directory / "rows" / f"part-{self._part:05d}.parquet")
        self._part += 1


def read_run(directory: Path) -> tuple[dict, pa.Table]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not any((directory / "rows").glob("*.parquet")):
        return manifest, SCHEMA.empty_table()
    table = pq.read_table(directory / "rows", schema=SCHEMA)
    return manifest, table.set_column(SCHEMA.get_field_index("sm_percent"), "sm_percent", pc.fill_null(table["sm_percent"], 100))
