# SPDX-License-Identifier: BSD-3-Clause
"""Typed prim classes, one per registered prim type
(``ernie.prim_types()``).

Each class wraps a ``Prim`` and has a property per attribute its type
and bases declare, read as the declared type (the default while unset)
and converted to it when set::

    import ernie

    scene = ernie.Scene()
    sun = scene.define(
        "/World/Sun",
        ernie.DistantLight,
        angle=0.53,
        irradiance=1361.0,
    )
    lamp = scene.define(
        "/World/Lamp",
        ernie.SphereLight,
        radius=0.1,
        translate=(0, 0, 2),
    )
    lamp.intensity = 40.0
    lamp.radius                      # 0.1
    lamp.color_temperature           # 6500.0, the default
    # on the class, a property is its AttributeDef
    ernie.SphereLight.radius.doc

``scene.define(path, cls, **attributes)`` and ``prim.define(relative,
cls, **attributes)`` return an instance of ``cls``; with a type name
instead of a class they return the ``Prim`` as before, taking keyword
attributes too when the type is registered. Unknown attributes raise,
with the closest names. ``translate``, ``rotate``, ``rotation_order``,
``orient`` and ``scale`` set the transform (``Prim.set_xform``).
Anything else a ``Prim`` has (``path``, ``get``, ``write``,
``set_xform``, ``define``) is available on the instance, and
``prim.set(token, value)`` still sets any attribute by USD name.

An existing prim is typed with ``cls.at(scene, path)``, or
``cls.at(prim, relative)`` below a prim, which raise if there is no
prim there or it is of another type; ``ernie.typed(prim)`` is the
instance of the prim's own (most derived) class, or the ``Prim``
itself for a type with no
class::

    sun = ernie.DistantLight.at(scene, "/World/Sun")
    surface = ernie.Shader.at(material, "Surface")
    # SphereLight('/World/Lamp')
    ernie.typed(scene.get_prim("/World/Lamp"))
"""

from __future__ import annotations

import builtins
import difflib
import functools
import json
import os
from typing import Any, ClassVar

import numpy as np

from ._ernie import (
    AttributeDef,
    AttributeKind,
    Path,
    Prim,
    PrimType,
    Quat,
    Scene,
    _load_scene_file,
    _load_scene_text,
    find_prim_type,
    prim_types,
    stage_type,
)

__all__ = ["AttributeName", "Schema", "typed"]

_XFORM_KEYS = (
    "translate",
    "rotate",
    "rotation_order",
    "orient",
    "scale",
)
# The transform ops `Schema.write_rows` writes as rows.
_ROW_XFORM_OPS = {
    "translate": "xformOp:translate",
    "orient": "xformOp:orient",
    "scale": "xformOp:scale",
}
_scene_define = Scene.define
_prim_define = Prim.define


def _closest(name: str, names) -> str:
    matches = difflib.get_close_matches(name, list(names), n=3)
    return (
        f"; did you mean {', '.join(repr(m) for m in matches)}?"
        if matches
        else ""
    )


def _write(definition: AttributeDef, prim: Prim, value: Any) -> None:
    """Sets ``definition`` on ``prim``: a prim given for a target as
    its path, and nested lists for one that takes an array in place of
    its value (per-vertex colours) as a float32 array."""
    if (
        definition.takes_array
        and isinstance(value, (list, tuple))
        and value
        and isinstance(value[0], (list, tuple))
    ):
        value = np.asarray(value, dtype=np.float32)
    definition.write(prim, _target(value))


def _target(value: Any) -> Any:
    """A relationship target given as a prim or a typed prim, as its
    path; in a list of targets, each, and the strs among paths as
    paths."""
    if isinstance(value, Schema):
        return value.prim.path
    if isinstance(value, Prim):
        return value.path
    if isinstance(value, (list, tuple)):
        targets = [_target(item) for item in value]
        if any(isinstance(item, Path) for item in targets):
            return [
                Path(item) if isinstance(item, str) else item
                for item in targets
            ]
        return targets
    return value


class AttributeName(str):
    """A declared attribute as a typed prim class gives it
    (``ernie.PointInstancer.positions``): its USD name, a str, so any
    function taking an attribute takes it (``scene.write(prims,
    ernie.PointInstancer.positions, rows)``), that also describes it
    with its ``AttributeDef``'s members (``doc``, ``kind``,
    ``default``, ``py_name``...)."""

    def __new__(cls, definition: AttributeDef) -> AttributeName:
        name = super().__new__(cls, definition.name)
        name.definition = definition
        return name

    def __getattr__(self, member: str) -> Any:
        if member == "definition":
            raise AttributeError(member)
        return getattr(self.definition, member)

    def __repr__(self) -> str:
        return repr(self.definition)


class _Attribute:
    """One declared attribute as a property: its `AttributeName` on
    the class, its value on an instance."""

    __slots__ = ("definition", "name")

    def __init__(self, name: AttributeName) -> None:
        self.definition = name.definition
        self.name = name

    def __get__(
        self, instance: Schema | None, owner: type | None = None
    ) -> Any:
        if instance is None:
            return self.name
        return self.definition.read(instance.prim)

    def __set__(self, instance: Schema, value: Any) -> None:
        _write(self.definition, instance.prim, value)


class _FirstTarget(_Attribute):
    """A list of targets that usually holds one, by its singular name
    (``camera``): its first target, or None; set to one target or a
    list. Its ``plural`` reads the whole list."""

    __slots__ = ()

    def __get__(
        self, instance: Schema | None, owner: type | None = None
    ) -> Any:
        if instance is None:
            return self.name
        targets = self.definition.read(instance.prim)
        return targets[0] if targets else None


def _define_type(
    cls: type, parent: Any, path: str | Path, **attributes: Any
) -> Any:
    """Defines a prim of `cls`'s type at ``path`` in a Scene, or
    relative to a prim, and sets ``attributes``."""
    if cls.prim_type.abstract:
        raise TypeError(
            f"{cls.__name__} is abstract; define one of the types "
            "derived from it"
        )
    if isinstance(parent, Schema):
        parent = parent.prim
    define = (
        _scene_define if isinstance(parent, Scene) else _prim_define
    )
    return cls(define(parent, path, cls.prim_type.name)).set(
        **attributes
    )


class _Define:
    """``define`` on a typed prim class defines a prim of its type
    (``ernie.Sphere.define(scene, "/World/Rock")``); on a typed prim,
    a child of it, as ``Prim.define`` does
    (``material.define("Surface", ernie.Shader)``)."""

    def __get__(self, instance: Any, owner: type) -> Any:
        if instance is None:
            return functools.partial(_define_type, owner)
        return functools.partial(_define, instance.prim)


class Schema:
    """Base of the typed prim classes: a ``Prim`` of the class's
    registered type or one derived from it."""

    __slots__ = ("prim",)
    prim_type: ClassVar[PrimType]
    _attributes: ClassVar[dict[str, AttributeDef]] = {}

    def __init__(self, prim: Prim) -> None:
        if not isinstance(prim, Prim) or not prim.valid:
            raise ValueError(
                f"{type(self).__name__} needs a valid Prim, "
                f"not {prim!r}"
            )
        found = find_prim_type(prim.type)
        if found is None or not found.is_a(self.prim_type.name):
            what = f"a {prim.type}" if prim.type else "untyped"
            raise TypeError(
                f"'{prim.path}' is {what}, "
                f"not a {self.prim_type.name}"
            )
        object.__setattr__(self, "prim", prim)

    define = _Define()

    @classmethod
    def at(
        cls, parent: Scene | Prim | Schema, path: str | Path
    ) -> Any:
        """The existing prim at ``path`` in a Scene, or at ``path``
        relative to a prim ("Surface", "Rig/Arm"), as this class.

        Raises:
            LookupError: There is no prim there.
            TypeError: The prim is not of this class's type or one
                derived from it.
        """
        if isinstance(parent, Schema):
            parent = parent.prim
        if not isinstance(parent, (Scene, Prim)):
            raise TypeError(
                f"{cls.__name__}.at looks in a Scene or below a "
                f"prim, not {parent!r}"
            )
        prim = parent.get_prim(path)
        if prim is None:
            where = (
                f"'{path}'"
                if isinstance(parent, Scene)
                else f"'{path}' below '{parent.path}'"
            )
            raise LookupError(
                f"No prim at {where} for {cls.__name__}.at"
            )
        return cls(prim)

    @classmethod
    def attributes(cls) -> list[AttributeDef]:
        """Every attribute of the type, its root base's first."""
        return list(
            {
                definition.name: definition
                for definition in cls._attributes.values()
            }.values()
        )

    @classmethod
    def write_rows(
        cls,
        scene: Scene,
        prims: Any,
        *,
        stream: Any = None,
        **arrays: Any,
    ) -> None:
        """Writes row i of each array, by Python name, as that
        attribute of prim i of ``prims`` (a PrimSet, or a list of
        paths), every step of a simulation:
        ``ernie.PointInstancer.write_rows(scene, prims, positions=p,
        orientations=q)``. ``translate``, ``orient`` (x, y, z, w) and
        ``scale`` write the transform ops, which from CUDA arrays pose
        the prims on the GPU. The arrays are DLPack arrays written
        without copying; ``stream`` is as for ``Scene.write``.

        Raises:
            TypeError: A name is not an array attribute of this class
                or a transform op.
        """
        names = dict(_ROW_XFORM_OPS)
        names.update(
            (name, definition.name)
            for name, definition in cls._attributes.items()
            if definition.kind == AttributeKind.TENSOR
        )
        for name in arrays:
            if name not in names:
                raise TypeError(
                    f"{cls.__name__} has no array attribute {name!r}"
                    + _closest(name, names)
                )
        for name, rows in arrays.items():
            scene.write(prims, names[name], rows, stream=stream)

    def set(self, *attribute: Any, **attributes: Any) -> Any:
        """Sets attributes by Python name, and the transform from
        ``translate``, ``rotate``, ``rotation_order``, ``orient`` and
        ``scale``; ``set(token, value)`` sets one by USD name. Returns
        this prim."""
        if attribute:
            self.prim.set(*attribute)
        xform = {
            key: attributes.pop(key)
            for key in _XFORM_KEYS
            if key in attributes
        }
        for name, value in attributes.items():
            definition = self._attributes.get(name)
            if definition is None:
                raise TypeError(
                    f"{type(self).__name__} has no attribute {name!r}"
                    + _closest(
                        name, [*self._attributes, *_XFORM_KEYS]
                    )
                )
            _write(definition, self.prim, value)
        if xform:
            self.prim.set_xform(**xform)
        return self

    def __getattr__(self, name: str) -> Any:
        if name == "prim":  # not set yet: __init__ raised
            raise AttributeError(name)
        try:
            return getattr(self.prim, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
                + _closest(name, self._attributes)
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self._attributes:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
                + _closest(name, self._attributes)
            )
        object.__setattr__(self, name, value)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Schema)
            and other.prim.path == self.prim.path
        )

    def __hash__(self) -> int:
        return hash(self.prim.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.prim.path}')"


def _make_class(
    prim_type: PrimType, base: type[Schema]
) -> type[Schema]:
    reserved = set(dir(Schema)) | set(dir(Prim))
    namespace: dict[str, Any] = {
        "__slots__": (),
        "__doc__": prim_type.doc,
        "__module__": "ernie",
        "prim_type": prim_type,
        "_attributes": dict(base._attributes),
    }
    for definition in prim_type.attributes:
        names = [definition.py_name] + (
            [definition.plural] if definition.plural else []
        )
        for name in names:
            if name in reserved:
                raise RuntimeError(
                    f"{prim_type.name}.{name} hides a Prim member"
                )
            namespace["_attributes"][name] = definition
        name = AttributeName(definition)
        if definition.plural:
            namespace[definition.py_name] = _FirstTarget(name)
            namespace[definition.plural] = _Attribute(name)
        else:
            namespace[definition.py_name] = _Attribute(name)
    return builtins.type(prim_type.name, (base,), namespace)


def _build() -> dict[str, type[Schema]]:
    classes: dict[str, type[Schema]] = {}
    for prim_type in prim_types():
        base = (
            classes[prim_type.base.name]
            if prim_type.base is not None
            else Schema
        )
        classes[prim_type.name] = _make_class(prim_type, base)
    return classes


classes = _build()
globals().update(classes)
__all__ += list(classes)


class _StageView(Schema):
    """A scene's stage metadata: a view of its root prim, which stays
    untyped."""

    __slots__ = ()

    def __init__(self, scene: Scene | Prim) -> None:
        prim = (
            scene.get_prim("/") if isinstance(scene, Scene) else scene
        )
        if not isinstance(prim, Prim) or not prim.valid:
            raise ValueError(
                f"Stage needs a Scene or its root Prim, not {scene!r}"
            )
        if str(prim.path) != "/":
            raise TypeError(
                f"'{prim.path}' is not the root prim, which holds the "
                "stage metadata"
            )
        object.__setattr__(self, "prim", prim)

    @property
    def time(self) -> float:
        """The animation's time, seconds: the time code over the time
        codes per second; setting it sets the time code."""
        return self.time_code / self.time_codes_per_second

    @time.setter
    def time(self, seconds: float) -> None:
        self.time_code = seconds * self.time_codes_per_second

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "time":
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)


Stage = _make_class(stage_type(), _StageView)
Stage.__doc__ = (
    f"{stage_type().doc} ``ernie.Stage(scene)`` is the scene's: "
    "``ernie.Stage(scene).up_axis``, ``ernie.Stage(scene).time = "
    "2.0``."
)
__all__ += ["Stage"]


def typed(prim: Prim | Schema) -> Schema | Prim:
    """``prim`` as an instance of its type's class, the most derived
    (a SphereLight's ``ernie.SphereLight``, a Light given as
    ``ernie.Light``), or the ``Prim`` itself when its type has no
    class (untyped, or not registered)."""
    if isinstance(prim, Schema):
        prim = prim.prim
    if not isinstance(prim, Prim) or not prim.valid:
        raise TypeError(
            f"typed needs a valid Prim or typed prim, not {prim!r}"
        )
    cls = classes.get(prim.type)
    return cls(prim) if cls is not None else prim


def _define(
    self: Scene | Prim,
    path: str | Path,
    type: Any,
    /,
    **attributes: Any,
) -> Any:
    if isinstance(type, builtins.type) and issubclass(type, Schema):
        return type.define(self, path, **attributes)
    define = (
        _scene_define if isinstance(self, Scene) else _prim_define
    )
    prim = define(self, path, type)
    if attributes:
        cls = classes.get(type)
        if cls is None:
            raise TypeError(
                f"{type!r} is not a registered prim type, so it "
                "takes no keyword attributes; set them with "
                ".set(token, value)"
            )
        cls(prim).set(**attributes)
    return prim


def load_scene(description: Any, scene: Scene | None = None) -> Scene:
    """Builds prims from a description, into ``scene`` or a new Scene,
    and returns the scene.

    The description is a dict (or JSON text, or the path of a
    ``.json`` file) from prim paths to prims; each prim is a dict
    whose ``"type"`` names its type (none for an untyped prim, or to
    leave an existing prim's type alone), whose dict values are its
    children by name, and whose other values are its attributes, by
    Python name (``radius``) or USD name (``"inputs:radius"``), with
    ``translate``, ``rotate``, ``rotation_order``, ``orient`` and
    ``scale`` setting its transform ops (``orient`` a ``Quat`` or [w,
    x, y, z]) and ``transform`` a 4 x 4 matrix (rows of four); C++
    reads the same format (``ernie/scene/scene_json.h``)::

        ernie.load_scene({
            "/World": {"type": "Xform",
                "Rock": {"type": "Sphere", "radius": 0.5,
                    "translate": [0, 0, 1]},
                "Sun": {"type": "DistantLight", "angle": 0.53}},
            "/Render/Settings": {"type": "RenderSettings",
                "resolution": [640, 480]},
        })

    C++ loads it (``LoadScene``), so both read a description the same
    way: each value is read as its attribute's declared kind, checked
    as setting it is. Lists for array attributes become arrays: int32
    for those of whole numbers (``face_vertex_indices``,
    ``proto_indices``; ``AttributeDef.is_integer``), float32 for the
    rest (``points``, ``heights``), whatever the numbers look like;
    nested lists for one that takes an array in place of its value
    (``AttributeDef.takes_array``: per-vertex ``display_color``) are a
    float32 array. A dict may hold ``Quat``s, ``Path``s and prims too,
    and arrays (NumPy, or any DLPack array, a CUDA one included):
    where an array goes, written as it is, shared and not copied, its
    dtype kept, as ``Scene.write`` writes one; elsewhere
    (``translate``, a ``radius``) read as the numbers it holds, which
    needs a CPU array.

    Raises:
        ValueError: A value its attribute does not take, or JSON that
            does not parse, naming the prim and key (or the line and
            column).
        TypeError: The description is not a dict, JSON text or a file
            path."""
    scene = scene if scene is not None else Scene()
    if isinstance(description, (str, os.PathLike)):
        text = str(description)
        if text.lstrip().startswith("{"):
            _load_scene_text(scene, text)
        else:
            _load_scene_file(scene, description)
        return scene
    if not isinstance(description, dict):
        raise TypeError(
            "A scene description is a dict of prim paths, not "
            f"{type(description).__name__}"
        )
    # Arrays go to C++ beside the text, not through it: {"$array": i}
    # stands for arrays[i].
    arrays: list[Any] = []

    def json_value(value: Any) -> Any:
        if isinstance(value, Quat):
            return [value.w, value.x, value.y, value.z]
        if isinstance(value, (Path, Prim, Schema)):
            return str(
                value.path
                if isinstance(value, (Prim, Schema))
                else value
            )
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray) or hasattr(
            value, "__dlpack__"
        ):
            arrays.append(value)
            return {"$array": len(arrays) - 1}
        raise TypeError(
            "A scene description has no value of type "
            f"{type(value).__name__}"
        )

    _load_scene_text(
        scene, json.dumps(description, default=json_value), arrays
    )
    return scene


__all__ += ["load_scene"]

_define.__doc__ = (
    "Defines the prim at `path` with `type`, a type name or a typed "
    "prim class (ernie.SphereLight), and sets `attributes` by Python "
    "name; returns the Prim, or an instance of the class."
)
Scene.define = _define
Prim.define = _define
