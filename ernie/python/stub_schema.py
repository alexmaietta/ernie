# SPDX-License-Identifier: BSD-3-Clause
"""Writes ernie/schema.pyi: the typed prim classes `ernie.schema` builds from the prim type
registry at import, as stubs IDEs, type checkers and the API docs can read.

    python stub_schema.py <output.pyi>    (with the built package on PYTHONPATH)
"""

import sys
import textwrap

import ernie
from ernie import AttributeKind

_TYPES = {
    AttributeKind.BOOL: "bool",
    AttributeKind.INT: "int",
    AttributeKind.DOUBLE: "float",
    AttributeKind.VEC2: "tuple[float, float]",
    AttributeKind.VEC3: "tuple[float, float, float]",
    AttributeKind.TOKEN: "str",
    AttributeKind.RELATIONSHIP: "Path | None",
    AttributeKind.FLOAT_ARRAY: "list[float]",
    AttributeKind.TENSOR: "Any",
}

# What setting accepts, where it is wider than what reading returns.
_ARGUMENTS = {
    AttributeKind.DOUBLE: "float",
    AttributeKind.VEC2: "Sequence[float]",
    AttributeKind.VEC3: "Sequence[float]",
    AttributeKind.RELATIONSHIP: "str | Path | Prim | Schema",
    AttributeKind.FLOAT_ARRAY: "Sequence[float]",
}

_XFORM = (
    "translate: Sequence[float] = ...",
    "rotate: Sequence[float] = ...",
    'rotation_order: str = "XYZ"',
    "orient: Quat = ...",
    "scale: float | Sequence[float] = ...",
)


def _doc(text, indent):
    text = text.replace('"""', "'''").strip()
    if not text:
        return []
    lines = textwrap.wrap(text, 96 - len(indent))
    return [f'{indent}"""{lines[0]}' + ('"""' if len(lines) == 1 else "")] + (
        [f"{indent}{line}" for line in lines[1:]] + [f'{indent}"""'] if len(lines) > 1 else []
    )


def _attribute_doc(attribute):
    parts = [attribute.doc] if attribute.doc else []
    default = attribute.default
    if default not in (None, "", []):
        parts.append(f"Default {default!r}.")
    parts.append(f"USD name `{attribute.name}`.")
    return " ".join(parts)


def _keywords(prim_type):
    keywords = [
        f"{a.py_name}: {_ARGUMENTS.get(a.kind, _TYPES[a.kind])} = ..." for a in prim_type.all_attributes()
    ]
    return keywords + list(_XFORM)


def main(output):
    lines = [
        "# SPDX-License-Identifier: BSD-3-Clause",
        "# Generated from the prim type registry by ernie/python/stub_schema.py; do not edit.",
        "from collections.abc import Sequence",
        "from typing import Any, ClassVar, Self",
        "",
        "from ._ernie import AttributeDef, Path, Prim, PrimType, Quat, Scene",
        "",
        f"__all__ = {['Schema', *[t.name for t in ernie.prim_types()]]!r}",
        "",
        "classes: dict[str, type[Schema]]",
        "",
        "class Schema:",
        *_doc(ernie.Schema.__doc__ or "", "    "),
        "    prim: Prim",
        "    prim_type: ClassVar[PrimType]",
        "    def __init__(self, prim: Prim) -> None: ...",
        "    @classmethod",
        "    def define(cls, parent: Scene | Prim | Schema, path: str | Path, **attributes: Any) -> Self: ...",
        "    @classmethod",
        "    def attributes(cls) -> list[AttributeDef]: ...",
        "    def set(self, *attribute: Any, **attributes: Any) -> Self: ...",
        "    @property",
        "    def path(self) -> Path: ...",
        "    @property",
        "    def type(self) -> str: ...",
        "    def get(self, attribute: str, default: Any = None) -> Any: ...",
        "    def write(self, attribute: str, array: Any) -> Prim: ...",
        "    def read(self, attribute: str) -> Any: ...",
        "",
    ]
    for prim_type in ernie.prim_types():
        base = prim_type.base.name if prim_type.base is not None else "Schema"
        lines.append(f"class {prim_type.name}({base}):")
        lines += _doc(prim_type.doc, "    ")
        for attribute in prim_type.attributes:
            lines.append(f"    {attribute.py_name}: {_TYPES[attribute.kind]}")
            lines += _doc(_attribute_doc(attribute), "    ")
        keywords = ", ".join(_keywords(prim_type))
        if not prim_type.abstract:
            lines.append("    @classmethod")
            lines.append(
                f"    def define(cls, parent: Scene | Prim | Schema, path: str | Path, *, {keywords}) -> Self: ..."
            )
        lines.append(f"    def set(self, *attribute: Any, {keywords}) -> Self: ...")
        lines.append("")
    with open(output, "w") as stub:
        stub.write("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
