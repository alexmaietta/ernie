# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, TypeVar

from ernie._ernie import Prim, Selection

_S = TypeVar("_S", bound="SchemaSelection")


def _given(**values: Any) -> dict[str, Any]:
    return {field: value for field, value in values.items() if value is not None}


def _targets(value: Targets | None) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, SchemaPrim, Prim)):
        value = [value]
    return [item.prim if isinstance(item, SchemaPrim) else item for item in value]


class SchemaPrim:
    schema: ClassVar[str] = ""
    api: ClassVar[bool] = False
    __slots__ = ("prim",)

    def __init__(self, prim: Prim) -> None:
        held = prim.has_api(self.schema) if self.api else prim.is_a(self.schema)
        if not held:
            raise TypeError(f"{prim.path} is a {prim.schema}, without {self.schema}")
        self.prim = prim

    @property
    def path(self) -> str:
        return self.prim.path

    @property
    def name(self) -> str:
        return self.prim.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SchemaPrim) and self.prim == other.prim

    def __hash__(self) -> int:
        return hash(self.prim)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.prim.path})"


class InstancePrim(SchemaPrim):
    __slots__ = ("instance",)

    def __init__(self, prim: Prim | SchemaPrim, instance: str) -> None:
        if isinstance(prim, SchemaPrim):
            prim = prim.prim
        if instance not in prim.instances(self.schema):
            raise TypeError(f"{prim.path} has no {self.schema} {instance}")
        self.prim = prim
        self.instance = instance

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, InstancePrim)
            and self.prim == other.prim
            and self.instance == other.instance
        )

    def __hash__(self) -> int:
        return hash((self.prim, self.instance))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.prim.path}, {self.instance})"


class SchemaSelection:
    schema: ClassVar[str] = ""
    __slots__ = ("selection",)

    def __init__(self, selection: Selection) -> None:
        self.selection = selection

    def __len__(self) -> int:
        return len(self.selection)

    @property
    def prims(self) -> list[Prim]:
        return self.selection.prims

    @property
    def paths(self) -> list[str]:
        return self.selection.paths

    def world(self: _S, world: int) -> _S:
        return type(self)(self.selection.world(world))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.selection)} prims)"


Targets = str | SchemaPrim | Prim | Iterable[str | SchemaPrim | Prim]
Prims = SchemaPrim | Prim | SchemaSelection | Selection | Iterable[SchemaPrim | Prim]


def _prims(prims: Prims) -> list[Prim]:
    if isinstance(prims, (SchemaSelection, Selection)):
        return prims.prims
    if isinstance(prims, (SchemaPrim, Prim)):
        prims = [prims]
    return [item.prim if isinstance(item, SchemaPrim) else item for item in prims]


def _apply(prims: Prims, api: str, values: dict[str, Any], instance: str | None = None) -> None:
    items = _prims(prims)
    if items:
        items[0].scene.apply(items, api, values, instance=instance)


def _unapply(prims: Prims, api: str, instance: str | None = None) -> None:
    items = _prims(prims)
    if items:
        items[0].scene.unapply(items, api, instance=instance)


def _fields(cls: type, values: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if values is None:
        return None
    unknown = set(values) - set(cls._fields)
    if unknown:
        raise TypeError(f"{cls.schema} has no {', '.join(sorted(unknown))}")
    return {cls._fields[key]: _targets(value) if _is_target(value) else value for key, value in values.items()}


def _instances(
    cls: type, values: Mapping[str, Mapping[str, Any]] | None
) -> dict[str, dict[str, Any]] | None:
    if values is None:
        return None
    return {instance: _fields(cls, fields) for instance, fields in values.items()}


def _is_target(value: Any) -> bool:
    if isinstance(value, (SchemaPrim, Prim)):
        return True
    return isinstance(value, (list, tuple)) and any(isinstance(item, (SchemaPrim, Prim)) for item in value)


def _apis(*applied: tuple[type, Mapping[str, Any] | None]) -> dict[str, dict[str, Any]]:
    return {cls.schema: _fields(cls, values) for cls, values in applied if values is not None}
