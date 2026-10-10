# SPDX-License-Identifier: BSD-3-Clause
"""Typed prim classes, one per registered prim type (``ernie.prim_types()``).

Each class wraps a ``Prim`` and has a property per attribute its type and bases declare, read as
the declared type (the default while unset) and converted to it when set::

    import ernie

    scene = ernie.Scene()
    sun = scene.define("/World/Sun", ernie.DistantLight, angle=0.53, irradiance=1361.0)
    lamp = scene.define("/World/Lamp", ernie.SphereLight, radius=0.1, translate=(0, 0, 2))
    lamp.intensity = 40.0
    lamp.radius                      # 0.1
    lamp.color_temperature           # 6500.0, the default
    ernie.SphereLight.radius.doc     # on the class, a property is its AttributeDef

``scene.define(path, cls, **attributes)`` and ``prim.define(relative, cls, **attributes)`` return
an instance of ``cls``; with a type name instead of a class they return the ``Prim`` as before,
taking keyword attributes too when the type is registered. Unknown attributes raise, with the
closest names. ``translate``, ``rotate``, ``rotation_order``, ``orient`` and ``scale`` set the
transform (``Prim.set_xform``). Anything else a ``Prim`` has (``path``, ``get``, ``write``,
``set_xform``, ``define``) is available on the instance, and ``prim.set(token, value)`` still
sets any attribute by USD name.

An existing prim is typed with ``cls.at(scene, path)``, or ``cls.at(prim, relative)`` below a
prim, which raise if there is no prim there or it is of another type; ``ernie.typed(prim)`` is
the instance of the prim's own (most derived) class, or the ``Prim`` itself for a type with no
class::

    sun = ernie.DistantLight.at(scene, "/World/Sun")
    surface = ernie.Shader.at(material, "Surface")
    ernie.typed(scene.get_prim("/World/Lamp"))   # SphereLight('/World/Lamp')
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
    find_prim_type,
    prim_types,
    tokens,
)

__all__ = ["Schema", "typed"]

_XFORM_KEYS = ("translate", "rotate", "rotation_order", "orient", "scale")
_scene_define = Scene.define
_prim_define = Prim.define


def _closest(name: str, names) -> str:
    matches = difflib.get_close_matches(name, list(names), n=3)
    return f"; did you mean {', '.join(repr(m) for m in matches)}?" if matches else ""


def _target(value: Any) -> Any:
    """A relationship target given as a prim or a typed prim, as its path; in a list of targets,
    each, and the strs among paths as paths."""
    if isinstance(value, Schema):
        return value.prim.path
    if isinstance(value, Prim):
        return value.path
    if isinstance(value, (list, tuple)):
        targets = [_target(item) for item in value]
        if any(isinstance(item, Path) for item in targets):
            return [Path(item) if isinstance(item, str) else item for item in targets]
        return targets
    return value


class _Attribute:
    """One declared attribute as a property: its AttributeDef on the class, its value on an
    instance."""

    __slots__ = ("definition",)

    def __init__(self, definition: AttributeDef) -> None:
        self.definition = definition

    def __get__(self, instance: Schema | None, owner: type | None = None) -> Any:
        if instance is None:
            return self.definition
        return self.definition.read(instance.prim)

    def __set__(self, instance: Schema, value: Any) -> None:
        self.definition.write(instance.prim, _target(value))


def _define_type(cls: type, parent: Any, path: str | Path, **attributes: Any) -> Any:
    """Defines a prim of `cls`'s type at ``path`` in a Scene, or relative to a prim, and sets
    ``attributes``."""
    if cls.prim_type.abstract:
        raise TypeError(f"{cls.__name__} is abstract; define one of the types derived from it")
    if isinstance(parent, Schema):
        parent = parent.prim
    define = _scene_define if isinstance(parent, Scene) else _prim_define
    return cls(define(parent, path, cls.prim_type.name)).set(**attributes)


class _Define:
    """``define`` on a typed prim class defines a prim of its type
    (``ernie.Sphere.define(scene, "/World/Rock")``); on a typed prim, a child of it, as
    ``Prim.define`` does (``material.define("Surface", ernie.Shader)``)."""

    def __get__(self, instance: Any, owner: type) -> Any:
        if instance is None:
            return functools.partial(_define_type, owner)
        return functools.partial(_define, instance.prim)


class Schema:
    """Base of the typed prim classes: a ``Prim`` of the class's registered type or one derived
    from it."""

    __slots__ = ("prim",)
    prim_type: ClassVar[PrimType]
    _attributes: ClassVar[dict[str, AttributeDef]] = {}

    def __init__(self, prim: Prim) -> None:
        if not isinstance(prim, Prim) or not prim.valid:
            raise ValueError(f"{type(self).__name__} needs a valid Prim, not {prim!r}")
        found = find_prim_type(prim.type)
        if found is None or not found.is_a(self.prim_type.name):
            what = f"a {prim.type}" if prim.type else "untyped"
            raise TypeError(f"'{prim.path}' is {what}, not a {self.prim_type.name}")
        object.__setattr__(self, "prim", prim)

    define = _Define()

    @classmethod
    def at(cls, parent: Scene | Prim | Schema, path: str | Path) -> Any:
        """The existing prim at ``path`` in a Scene, or at ``path`` relative to a prim ("Surface",
        "Rig/Arm"), as this class.

        Raises:
            LookupError: There is no prim there.
            TypeError: The prim is not of this class's type or one derived from it.
        """
        if isinstance(parent, Schema):
            parent = parent.prim
        if not isinstance(parent, (Scene, Prim)):
            raise TypeError(f"{cls.__name__}.at looks in a Scene or below a prim, not {parent!r}")
        prim = parent.get_prim(path)
        if prim is None:
            where = f"'{path}'" if isinstance(parent, Scene) else f"'{path}' below '{parent.path}'"
            raise LookupError(f"No prim at {where} for {cls.__name__}.at")
        return cls(prim)

    @classmethod
    def attributes(cls) -> list[AttributeDef]:
        """Every attribute of the type, its root base's first."""
        return list(cls._attributes.values())

    def set(self, *attribute: Any, **attributes: Any) -> Any:
        """Sets attributes by Python name, and the transform from ``translate``, ``rotate``,
        ``rotation_order``, ``orient`` and ``scale``; ``set(token, value)`` sets one by USD name.
        Returns this prim."""
        if attribute:
            self.prim.set(*attribute)
        xform = {key: attributes.pop(key) for key in _XFORM_KEYS if key in attributes}
        for name, value in attributes.items():
            definition = self._attributes.get(name)
            if definition is None:
                raise TypeError(
                    f"{type(self).__name__} has no attribute {name!r}"
                    + _closest(name, [*self._attributes, *_XFORM_KEYS])
                )
            definition.write(self.prim, _target(value))
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
                f"{type(self).__name__} has no attribute {name!r}" + _closest(name, self._attributes)
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self._attributes:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}" + _closest(name, self._attributes)
            )
        object.__setattr__(self, name, value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Schema) and other.prim.path == self.prim.path

    def __hash__(self) -> int:
        return hash(self.prim.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.prim.path}')"


def _build() -> dict[str, type[Schema]]:
    reserved = set(dir(Schema)) | set(dir(Prim))
    classes: dict[str, type[Schema]] = {}
    for prim_type in prim_types():
        base = classes[prim_type.base.name] if prim_type.base is not None else Schema
        namespace: dict[str, Any] = {
            "__slots__": (),
            "__doc__": prim_type.doc,
            "__module__": "ernie",
            "prim_type": prim_type,
            "_attributes": dict(base._attributes),
        }
        for definition in prim_type.attributes:
            if definition.py_name in reserved:
                raise RuntimeError(f"{prim_type.name}.{definition.py_name} hides a Prim member")
            namespace[definition.py_name] = _Attribute(definition)
            namespace["_attributes"][definition.py_name] = definition
        classes[prim_type.name] = builtins.type(prim_type.name, (base,), namespace)
    return classes


classes = _build()
globals().update(classes)
__all__ += list(classes)


def typed(prim: Prim | Schema) -> Schema | Prim:
    """``prim`` as an instance of its type's class, the most derived (a SphereLight's
    ``ernie.SphereLight``, a Light given as ``ernie.Light``), or the ``Prim`` itself when its type
    has no class (untyped, or not registered)."""
    if isinstance(prim, Schema):
        prim = prim.prim
    if not isinstance(prim, Prim) or not prim.valid:
        raise TypeError(f"typed needs a valid Prim or typed prim, not {prim!r}")
    cls = classes.get(prim.type)
    return cls(prim) if cls is not None else prim


def _define(self: Scene | Prim, path: str | Path, type: Any, /, **attributes: Any) -> Any:
    if isinstance(type, builtins.type) and issubclass(type, Schema):
        return type.define(self, path, **attributes)
    define = _scene_define if isinstance(self, Scene) else _prim_define
    prim = define(self, path, type)
    if attributes:
        cls = classes.get(type)
        if cls is None:
            raise TypeError(
                f"{type!r} is not a registered prim type, so it takes no keyword attributes; set "
                "them with .set(token, value)"
            )
        cls(prim).set(**attributes)
    return prim


def load_scene(description: Any, scene: Scene | None = None) -> Scene:
    """Builds prims from a description, into ``scene`` or a new Scene, and returns the scene.

    The description is a dict (or JSON text, or the path of a ``.json`` file) from prim paths to
    prims; each prim is a dict whose ``"type"`` names its type (none for an untyped prim, or to
    leave an existing prim's type alone), whose dict values are its children by name, and whose
    other values are its attributes, by Python name (``radius``) or USD name
    (``"inputs:radius"``), with ``translate``, ``rotate``, ``rotation_order``, ``orient`` and
    ``scale`` setting its transform ops (``orient`` a ``Quat`` or [w, x, y, z]) and ``transform``
    a 4 x 4 matrix (rows of four); C++ reads the same format (``ernie/scene/scene_json.h``)::

        ernie.load_scene({
            "/World": {"type": "Xform",
                "Rock": {"type": "Sphere", "radius": 0.5, "translate": [0, 0, 1]},
                "Sun": {"type": "DistantLight", "angle": 0.53}},
            "/Render/Settings": {"type": "RenderSettings", "resolution": [640, 480]},
        })

    Values are checked as setting them is; lists for array attributes (``points``, ``heights``)
    become arrays, float32 or, of whole numbers, int32."""
    if isinstance(description, (str, os.PathLike)):
        text = str(description)
        if text.lstrip().startswith("{"):
            description = json.loads(text)
        else:
            with open(description) as file:
                description = json.load(file)
    if not isinstance(description, dict):
        raise TypeError(f"A scene description is a dict of prim paths, not {type(description).__name__}")
    scene = scene if scene is not None else Scene()
    for key, body in description.items():
        _load_prim(scene, key if key.startswith("/") else "/" + key, body)
    return scene


def _load_prim(scene: Scene, path: str, body: Any) -> None:
    if not isinstance(body, dict):
        raise TypeError(f"'{path}' must be described by a dict of attributes and children")
    type_name = body.get("type")
    existing = scene.get_prim(path)
    if type_name is not None or existing is None:
        prim = _scene_define(scene, path, type_name or "")
    else:
        prim = existing
    cls = classes.get(prim.type)
    attributes = {k: v for k, v in body.items() if k != "type" and not isinstance(v, dict)}
    transform = attributes.pop("transform", None)
    xform = {key: attributes.pop(key) for key in _XFORM_KEYS if key in attributes}
    if isinstance(xform.get("orient"), (list, tuple)):
        xform["orient"] = Quat(*xform["orient"])  # [w, x, y, z], as JSON writes one
    keywords = {}
    for name, value in attributes.items():
        definition = cls._attributes.get(name) if cls is not None else None
        if definition is None and ":" not in name and cls is not None:
            definition = next((d for d in cls._attributes.values() if d.name == name), None)
        if definition is not None and definition.kind == AttributeKind.TENSOR:
            value = _array(value)
        if definition is not None and name == definition.py_name:
            keywords[name] = value
        else:
            prim.set(definition.name if definition is not None else name, _target(value))
    if cls is not None:
        cls(prim).set(**keywords)
    if xform:
        prim.set_xform(**xform)
    if transform is not None:
        prim.set(tokens.TRANSFORM, transform)
    for name, child in body.items():
        if isinstance(child, dict):
            _load_prim(scene, f"{path.rstrip('/')}/{name}", child)


def _array(value: Any) -> Any:
    """A list from a description as an array: int32 of whole numbers, else float32."""
    if not isinstance(value, (list, tuple)):
        return value
    array = np.asarray(value)
    return array.astype(np.int32 if np.issubdtype(array.dtype, np.integer) else np.float32)


__all__ += ["load_scene"]

_define.__doc__ = (
    "Defines the prim at `path` with `type`, a type name or a typed prim class "
    "(ernie.SphereLight), and sets `attributes` by Python name; returns the Prim, or an instance "
    "of the class."
)
Scene.define = _define
Prim.define = _define
