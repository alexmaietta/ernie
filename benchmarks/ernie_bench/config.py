# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

TIERS = ("depth", "segmentation", "unlit", "direct", "pathtraced")
OUTPUTS = ("rgb", "depth", "segmentation")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@dataclass(frozen=True)
class SceneConfig:
    name: str
    dataset: str
    seed: int = 0
    params: dict = field(default_factory=dict)
    cameras_per_world: int | None = None


@dataclass(frozen=True)
class Workload:
    name: str = "default"
    worlds: int = 1
    cameras_per_world: int = 1
    width: int = 128
    height: int = 128
    outputs: tuple[str, ...] = ("rgb",)
    tier: str = "unlit"
    spp: int = 1
    sm_percent: int = 100
    dt: float = 1 / 30

    def __post_init__(self):
        object.__setattr__(self, "outputs", tuple(self.outputs))
        unknown = set(self.outputs) - set(OUTPUTS)
        if unknown:
            raise ValueError(f"unknown outputs {sorted(unknown)}, expected {OUTPUTS}")
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}, expected {TIERS}")
        if not 0 < self.sm_percent <= 100:
            raise ValueError(f"sm_percent must be in (0, 100], got {self.sm_percent}")

    @property
    def image_count(self) -> int:
        return self.worlds * self.cameras_per_world


@dataclass(frozen=True)
class Measurement:
    warmup_min_iterations: int = 10
    warmup_min_seconds: float = 1.0
    warmup_max_seconds: float = 30.0
    warmup_window: int = 10
    warmup_cv: float = 0.02
    iterations: int = 50
    repeats: int = 5
    max_repeats: int = 20
    target_ci: float = 0.01
    confidence: float = 0.95
    monitor_interval: float = 0.05
    processes: int = 1
    timeout_seconds: float = 900.0
    caches: str = "warm"

    def __post_init__(self):
        if self.max_repeats < self.repeats:
            raise ValueError(f"max_repeats {self.max_repeats} is below repeats {self.repeats}")
        if self.caches not in ("warm", "cold"):
            raise ValueError(f"caches must be warm or cold, got {self.caches!r}")


@dataclass(frozen=True)
class Capture:
    steps: tuple[int, ...] = (0,)
    frames: int = 0
    images: tuple[int, ...] | str = (0,)
    outputs: tuple[str, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))
        if self.images != "all":
            object.__setattr__(self, "images", tuple(self.images))
        if self.outputs is not None:
            object.__setattr__(self, "outputs", tuple(self.outputs))
            unknown = set(self.outputs) - set(OUTPUTS)
            if unknown:
                raise ValueError(f"unknown capture outputs {sorted(unknown)}, expected {OUTPUTS}")
        if self.frames < 0:
            raise ValueError("capture frames cannot be negative")

    @property
    def sequence(self) -> tuple[int, ...]:
        return tuple(range(self.frames)) if self.frames else self.steps


@dataclass(frozen=True)
class Profile:
    tool: str
    frames: int = 10

    def __post_init__(self):
        if self.tool not in ("ncu", "nsys"):
            raise ValueError(f"unknown profiler {self.tool!r}, expected ncu or nsys")
        if self.frames < 1:
            raise ValueError("profile at least one frame")


@dataclass(frozen=True)
class Suite:
    name: str
    backends: tuple[str, ...]
    scenes: tuple[SceneConfig, ...]
    workload: Workload
    measurement: Measurement = Measurement()
    sweep: dict = field(default_factory=dict)
    seed: int = 0
    capture: Capture | None = None

    def __post_init__(self):
        axes = {f.name for f in fields(Workload)} - {"name"} | {"resolution"}
        unknown = set(self.sweep) - axes
        if unknown:
            raise ValueError(f"unknown sweep axes {sorted(unknown)}, expected {sorted(axes)}")

    def points(self) -> list[Workload]:
        points = []
        for values in itertools.product(*self.sweep.values()):
            changes = dict(zip(self.sweep, values))
            if "resolution" in changes:
                changes["width"], changes["height"] = changes.pop("resolution")
            points.append(replace(self.workload, **changes))
        return sorted(points, key=lambda w: (w.worlds, w.cameras_per_world))

    def scene_points(self, spec: SceneConfig) -> list[Workload]:
        if spec.cameras_per_world is None:
            return self.points()
        return list(dict.fromkeys(replace(point, cameras_per_world=spec.cameras_per_world) for point in self.points()))


def _toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    return tomllib.loads(Path(path).read_text())


def load_scene(path: Path) -> SceneConfig:
    data = _toml(path)
    return SceneConfig(
        name=Path(path).stem,
        dataset=data["dataset"],
        seed=data.get("seed", 0),
        params=data.get("params", {}),
        cameras_per_world=data.get("cameras_per_world"),
    )


def scene_spec(name: str, configs: Path = CONFIGS) -> SceneConfig:
    path = configs / "scenes" / f"{name}.toml"
    return load_scene(path) if path.exists() else SceneConfig(name, name)


def load_suite(path: Path) -> Suite:
    path = Path(path)
    data = _toml(path)
    configs = path.resolve().parents[1]
    base = _toml(configs / "workloads" / f"{data['workload']}.toml")
    return Suite(
        name=data.get("name", path.stem),
        backends=tuple(data["backends"]),
        scenes=tuple(scene_spec(name, configs) for name in data["scenes"]),
        workload=Workload(name=data["workload"], **base.get("workload", {})),
        measurement=Measurement(**base.get("measurement", {}) | data.get("measurement", {})),
        sweep=data.get("sweep", {}),
        seed=data.get("seed", 0),
        capture=Capture(**data["capture"]) if "capture" in data else None,
    )
