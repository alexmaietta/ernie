# SPDX-License-Identifier: BSD-3-Clause
from ernie._ernie import Colormap, Overlay
from ernie._viewer import (
    Anchor,
    ColorbarDesc,
    Dock,
    HudItem,
    InputAction,
    KeyEvent,
    LightingMode,
    Modifiers,
    MouseEvent,
    OverlayEvent,
    OverlayGesture,
    Placement,
    RenderDelegate,
    RenderFrame,
    RenderRequest,
    Screen,
    TextStyle,
    Viewer,
    ViewportDesc,
    ViewportRect,
    WorldLayout,
    WorldPick,
    imgui,
)

__all__ = [
    "Anchor",
    "Manipulator",
    "ManipulatorModel",
    "ColorbarDesc",
    "Colormap",
    "Dock",
    "HudItem",
    "InputAction",
    "KeyEvent",
    "LightingMode",
    "Modifiers",
    "MouseEvent",
    "OverlayEvent",
    "OverlayGesture",
    "Placement",
    "RenderDelegate",
    "RenderFrame",
    "RenderRequest",
    "Screen",
    "TextStyle",
    "Viewer",
    "ViewportDesc",
    "ViewportRect",
    "WorldLayout",
    "WorldPick",
    "closest_on_axis",
    "hit_plane",
    "imgui",
]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def closest_on_axis(ray, origin, axis):
    d = ray.direction
    w = (origin[0] - ray.origin[0], origin[1] - ray.origin[1], origin[2] - ray.origin[2])
    a = _dot(axis, axis)
    b = _dot(axis, d)
    c = _dot(d, d)
    denominator = a * c - b * b
    if abs(denominator) < 1e-12:
        return None
    return (b * _dot(d, w) - c * _dot(axis, w)) / denominator


def hit_plane(ray, origin, normal):
    facing = _dot(ray.direction, normal)
    if abs(facing) < 1e-12:
        return None
    w = (origin[0] - ray.origin[0], origin[1] - ray.origin[1], origin[2] - ray.origin[2])
    s = _dot(w, normal) / facing
    if s < 0:
        return None
    return tuple(ray.origin[i] + ray.direction[i] * s for i in range(3))


class ManipulatorModel:
    def __init__(self, **items):
        self._items = dict(items)
        self._listeners = []

    def __contains__(self, name):
        return name in self._items

    def get(self, name):
        return self._items[name]

    def set(self, name, value):
        self._items[name] = value
        for listener in list(self._listeners):
            listener(name)

    def subscribe(self, listener):
        self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener):
        self._listeners.remove(listener)


class Manipulator:
    def __init__(self, viewer, name, model=None, renderer=None, viewport=""):
        self.viewer = viewer
        self.name = name
        self.model = model if model is not None else ManipulatorModel()
        self.renderer = renderer if renderer is not None else viewer.renderer(viewport)
        if self.renderer is None:
            raise ValueError("A manipulator draws through a renderer; pass renderer=")
        self.hovered = None
        self.pressed = None
        self._handler = viewer.add_overlay_handler(self._handle)
        self._listener = self.model.subscribe(self.on_model_changed)
        self.invalidate()

    def build(self, overlay):
        pass

    def invalidate(self):
        overlay = Overlay()
        self.build(overlay)
        self.renderer.set_overlay(self.name, overlay)

    def on_model_changed(self, name):
        self.invalidate()

    def on_hover(self, event):
        pass

    def on_leave(self, event):
        pass

    def on_press(self, event):
        return False

    def on_drag(self, event):
        pass

    def on_release(self, event):
        pass

    def on_click(self, event):
        pass

    def close(self):
        self.viewer.remove_handler(self._handler)
        self.model.unsubscribe(self._listener)
        self.renderer.remove_overlay(self.name)

    def _handle(self, event):
        if event.hit.overlay != self.name:
            return False
        gesture = event.gesture
        if gesture == OverlayGesture.HOVER:
            self.hovered = event.hit.id
            self.on_hover(event)
            self.invalidate()
            return False
        if gesture == OverlayGesture.LEAVE:
            self.hovered = None
            self.on_leave(event)
            self.invalidate()
            return False
        if gesture == OverlayGesture.PRESS:
            claimed = bool(self.on_press(event))
            if claimed:
                self.pressed = event.hit.id
            return claimed
        if gesture == OverlayGesture.DRAG:
            self.on_drag(event)
            return True
        if gesture == OverlayGesture.RELEASE:
            self.pressed = None
            self.on_release(event)
            self.invalidate()
            return True
        self.on_click(event)
        return True
