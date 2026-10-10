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
"""

from __future__ import annotations

import builtins
import difflib
from typing import Any, ClassVar

from ._ernie import AttributeDef, Path, Prim, PrimType, Scene, find_prim_type, prim_types

__all__ = ["Schema"]

_XFORM_KEYS = ("translate", "rotate", "rotation_order", "orient", "scale")
_scene_define = Scene.define
_prim_define = Prim.define


def _closest(name: str, names) -> str:
    matches = difflib.get_close_matches(name, list(names), n=3)
    return f"; did you mean {', '.join(repr(m) for m in matches)}?" if matches else ""


def _target(value: Any) -> Any:
    """A relationship target given as a prim or a typed prim, as its path."""
    if isinstance(value, Schema):
        return value.prim.path
    if isinstance(value, Prim):
        return value.path
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

    @classmethod
    def define(cls, parent: Scene | Prim | Schema, path: str | Path, **attributes: Any) -> Any:
        """Defines a prim of this type at ``path`` in a Scene, or relative to a prim, and sets
        ``attributes``."""
        if cls.prim_type.abstract:
            raise TypeError(f"{cls.__name__} is abstract; define one of the types derived from it")
        if isinstance(parent, Schema):
            parent = parent.prim
        define = _scene_define if isinstance(parent, Scene) else _prim_define
        return cls(define(parent, path, cls.prim_type.name)).set(**attributes)

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


_define.__doc__ = (
    "Defines the prim at `path` with `type`, a type name or a typed prim class "
    "(ernie.SphereLight), and sets `attributes` by Python name; returns the Prim, or an instance "
    "of the class."
)
Scene.define = _define
Prim.define = _define
