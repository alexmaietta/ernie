# SPDX-License-Identifier: BSD-3-Clause
"""Writes ernie/schema.pyi: the typed prim classes `ernie.schema`
builds from the prim type registry at import, as stubs IDEs, type
checkers and the API docs can read.

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
    AttributeKind.VEC4: "tuple[float, float, float, float]",
    AttributeKind.TOKEN: "str",
    AttributeKind.TOKEN_ARRAY: "list[str]",
    AttributeKind.RELATIONSHIP: "Path | None",
    AttributeKind.RELATIONSHIPS: "list[Path]",
    AttributeKind.FLOAT_ARRAY: "list[float]",
    AttributeKind.TENSOR: "Any",
    AttributeKind.QUAT: "Quat",
    AttributeKind.MATRIX: "tuple[tuple[float, ...], ...]",
}

# What setting accepts, where it is wider than what reading returns.
_ARGUMENTS = {
    AttributeKind.DOUBLE: "float",
    AttributeKind.VEC2: "Sequence[float]",
    AttributeKind.VEC3: "Sequence[float]",
    AttributeKind.VEC4: "Sequence[float]",
    AttributeKind.RELATIONSHIP: "str | Path | Prim | Schema",
    AttributeKind.RELATIONSHIPS: "str | Path | Prim | Schema | Sequence[str | Path | Prim | Schema]",
    AttributeKind.FLOAT_ARRAY: "float | Sequence[float]",
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
    return [
        f'{indent}"""{lines[0]}' + ('"""' if len(lines) == 1 else "")
    ] + (
        [f"{indent}{line}" for line in lines[1:]] + [f'{indent}"""']
        if len(lines) > 1
        else []
    )


def _attribute_doc(attribute):
    parts = [attribute.doc] if attribute.doc else []
    default = attribute.default
    # Defaults may be arrays, which do not compare as one truth value.
    if default is not None and not (
        hasattr(default, "__len__") and len(default) == 0
    ):
        parts.append(f"Default {default!r}.")
    parts.append(f"USD name `{attribute.name}`.")
    return " ".join(parts)


def _keywords(prim_type):
    keywords = []
    for a in prim_type.all_attributes():
        argument = _ARGUMENTS.get(a.kind, _TYPES[a.kind]) + (
            " | Any" if a.takes_array else ""
        )
        keywords.append(f"{a.py_name}: {argument} = ...")
        if a.plural:
            keywords.append(f"{a.plural}: {argument} = ...")
    return keywords + list(_XFORM)


def main(output):
    lines = [
        "# SPDX-License-Identifier: BSD-3-Clause",
        "# Generated from the prim type registry by ernie/python/stub_schema.py; do not edit.",
        "import os",
        "from collections.abc import Sequence",
        "from typing import Any, ClassVar, Protocol, Self, overload",
        "",
        "from ._ernie import AttributeDef, Path, Prim, PrimType, Quat, Scene",
        "",
        f"__all__ = {['AttributeName', 'Schema', 'typed', 'load_scene', *[t.name for t in ernie.prim_types()], 'Stage']!r}",
        "",
        "classes: dict[str, type[Schema]]",
        "",
        "class DefineAt(Protocol):",
        '    """`define` on a Scene, Prim or typed prim: the prim at `path` (below it, for a prim)."""',
        "    @overload",
        "    def __call__[S: Schema](self, path: str | Path, type: type[S], /, **attributes: Any) -> S: ...",
        "    @overload",
        "    def __call__(self, path: str | Path, type: str, /, **attributes: Any) -> Prim: ...",
        "",
        "class _Define[F]:",
        '    """`define` on a typed prim class (F, a prim of its type) or on a typed prim (a child)."""',
        "    @overload",
        "    def __get__(self, instance: None, owner: Any, /) -> F: ...",
        "    @overload",
        "    def __get__(self, instance: Schema, owner: Any, /) -> DefineAt: ...",
        "",
        "class _SchemaDefine(Protocol):",
        "    def __call__(self, parent: Scene | Prim | Schema, path: str | Path, **attributes: Any) -> Schema: ...",
        "",
        "class AttributeName(str):",
        *_doc(ernie.schema.AttributeName.__doc__ or "", "    "),
        "    definition: AttributeDef",
        "    def __getattr__(self, member: str) -> Any: ...",
        "",
        "class Schema:",
        *_doc(ernie.Schema.__doc__ or "", "    "),
        "    prim: Prim",
        "    prim_type: ClassVar[PrimType]",
        "    def __init__(self, prim: Prim) -> None: ...",
        "    define: ClassVar[_Define[_SchemaDefine]]",
        "    @classmethod",
        "    def at(cls, parent: Scene | Prim | Schema, path: str | Path) -> Self: ...",
        "    @classmethod",
        "    def attributes(cls) -> list[AttributeDef]: ...",
        "    @classmethod",
        "    def write_rows(cls, scene: Scene, prims: Any, *, stream: Any = None, **arrays: Any) -> None: ...",
        "    def set(self, *attribute: Any, **attributes: Any) -> Self: ...",
        # Forwarded to the prim, as at run time (Schema.__getattr__).
        "    def set_xform(self, *, translate: Any = None, rotate: Any = None, rotation_order: str = 'XYZ', orient: Any = None, scale: Any = None) -> Prim: ...",
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
        base = (
            prim_type.base.name
            if prim_type.base is not None
            else "Schema"
        )
        lines += _class_lines(prim_type, base)
        if prim_type.name == "Volume":
            # The field helpers schema.py adds.
            lines[-1:] = [
                "    def set_field(self, name: str, field: Prim | Schema | Path | str) -> Self: ...",
                "    def get_field(self, name: str) -> Path | None: ...",
                "    def field_names(self) -> list[str]: ...",
                "",
            ]
    # The stage metadata's view, of the root prim (not a prim type).
    lines += _class_lines(ernie.stage_type(), "Schema")[:-1]
    lines += [
        "    time: float",
        '    """The animation\'s time, seconds: the time code over the time codes per second."""',
        "",
    ]
    lines.append(
        "def typed(prim: Prim | Schema) -> Schema | Prim: ..."
    )
    lines.append(
        "def load_scene(description: dict[str, Any] | str | os.PathLike[str], "
        "scene: Scene | None = None) -> Scene: ..."
    )
    lines.append("")
    with open(output, "w") as stub:
        stub.write("\n".join(lines))


def _class_lines(prim_type, base):
    """The stub of `prim_type`'s class, deriving from `base`, ending
    in a blank line."""
    lines = []
    keywords = ", ".join(_keywords(prim_type))
    if not prim_type.abstract:
        # Its class's define, with its attributes as keywords.
        lines.append(f"class _{prim_type.name}Define(Protocol):")
        lines.append(
            f"    def __call__(self, parent: Scene | Prim | Schema, path: str | Path, *, {keywords}) -> {prim_type.name}: ..."
        )
        lines.append("")
    lines.append(f"class {prim_type.name}({base}):")
    lines += _doc(prim_type.doc, "    ")
    for attribute in prim_type.attributes:
        if attribute.plural:
            lines.append(f"    {attribute.py_name}: Path | None")
            lines += _doc(
                "The first target. " + _attribute_doc(attribute),
                "    ",
            )
            lines.append(
                f"    {attribute.plural}: {_TYPES[attribute.kind]}"
            )
            lines += _doc(
                "Every target. " + _attribute_doc(attribute),
                "    ",
            )
            continue
        # One that takes an array in place of its value reads as the
        # array when one is written.
        array = " | Any" if attribute.takes_array else ""
        lines.append(
            f"    {attribute.py_name}: {_TYPES[attribute.kind]}{array}"
        )
        lines += _doc(_attribute_doc(attribute), "    ")
    if not prim_type.abstract:
        lines.append(
            f"    define: ClassVar[_Define[_{prim_type.name}Define]]"
        )
    lines.append(
        f"    def set(self, *attribute: Any, {keywords}) -> Self: ..."
    )
    lines.append("")
    return lines


if __name__ == "__main__":
    main(sys.argv[1])
