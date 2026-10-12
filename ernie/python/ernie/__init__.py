# SPDX-License-Identifier: BSD-3-Clause
"""Ernie: a standalone Vulkan renderer with a USD-style scene.

Build a Scene of typed prims named by path (`ernie.Terrain`,
`ernie.Camera`, ...; see `ernie.schema`), set their attributes by
name, write bulk data as arrays from any DLPack-capable framework
(NumPy, PyTorch, JAX, CuPy, Warp) without copying, and render it::

    import numpy as np
    import ernie
    from ernie import tokens

    scene = ernie.Scene()
    scene.define(
        "/World/Terrain",
        ernie.Terrain,
        heights=heights,
        cell_size=2.0,
    )
    camera = scene.define(
        "/World/Camera", ernie.Camera, focal_length=35.0
    )
    camera.set(
        tokens.TRANSFORM,
        ernie.look_at_transform((0, -500, 300), (0, 0, 0)),
    )
    image = ernie.Renderer().render(scene)  # uint8 (rows, cols, 4)

`ernie.Camera.at(scene, path)` types an existing prim, and
`ernie.typed(prim)` gives a prim its own class. Any attribute can also
be set by USD name, `prim.set(tokens.FOCAL_LENGTH, 35.0)`. The
authoring functions (`IsdLineScanner.author_strip`,
`author_singlet_chromatic`, ...) and render products take a prim as a
str, a Path, a Prim or a typed prim.

Attribute values map from Python as follows:

- bool, int, float and str become bool, int, double and a token or
  asset path;
- a Path is a relationship target, such as material:binding;
- a sequence of two ints becomes an int2, such as a resolution; two,
  three or four numbers become a 2-, 3- or 4-vector; more become a
  float array;
- four rows of four numbers (or a 4 x 4 array) become a matrix in math
  convention, with column vectors and the translation in the last
  column. This is the transpose of USD's Gf.Matrix4d layout, which
  stores the translation in the last row.

Arrays given to write() are shared, not copied, and kept alive by the
scene; arrays returned by read() are read-only views. Write again
after changing an array in place, so the change is tracked. CUDA
arrays are copied into Ernie's GPU buffers device to device, ordered
on the GPU after the work that produced them.

Rendering is by steps: renderer.step(scene,
render_products={"/Render/Camera"}) returns the outputs by product,
and
product.frames[0].render_vars["/Render/Camera/LdrColor"]
.map(device=...)
delivers each output to Device.CPU, Device.CUDA (linear, filled on the
GPU) or Device.CUDA_ARRAY (the rendered image itself, no copy);
unmap() hands a CUDA_ARRAY back. step_async() returns an Operation to
wait(timeout_ns=...) on or poll, then fetch().

A simulation updates the same arrays every step in place through the
scene: `with scene.map(path, attribute, stream=...) as array:` hands
out the written array (a writable view, on its device) and commits the
writes on exit, so renderers read it again; the stream is the CUDA
stream the writes are queued on, and renderers copy the array in order
after them, without the host. scene.write(path, attribute, array,
stream=...) replaces an array the same way.

For reinforcement learning, ernie.batch.BatchCamera renders a view per
environment through one camera, poses and worlds in and GPU
observations out, without the host waiting.

render(scene, device="cuda") returns a CudaArray in GPU memory shared
with Vulkan. It returns as soon as the render is submitted;
torch.from_dlpack(result) (or CuPy, JAX) makes the consumer's stream
wait for the render on the GPU, so the host never blocks.
"""

from . import schema
from ._ernie import (
    AttributeDef,
    AttributeKind,
    AttributeMapping,
    CahvoreModel,
    ChangeCursor,
    CudaArray,
    CudaArrayMapping,
    CylindricalDem,
    Device,
    Dirty,
    Frame,
    IsdFrame,
    IsdLineScanner,
    IsdPushFrame,
    Operation,
    OverlayHit,
    Path,
    PolarStereographicDem,
    Prim,
    PrimSet,
    PrimType,
    Quat,
    Renderer,
    RenderProductOutput,
    RenderVarOutput,
    Scene,
    Wireframe,
    author_singlet_chromatic,
    camera_image_corners,
    celestial_to_local,
    color_index_to_temperature,
    dense_nanovdb_grid,
    dirty_of,
    distant_light_transform,
    fbm_noise,
    find_prim_type,
    greenwich_sidereal_time,
    julian_date,
    load_image,
    local_terrain_of,
    look_at_transform,
    noise_permutation,
    prim_types,
    read_cahvore_models,
    read_cylindrical_dem,
    read_polar_stereographic_dem,
    reflectance_library,
    save_scene,
    stage_type,
    star_illuminance,
    synthetic_stars,
    tokens,
)

# The typed prim classes, such as SphereLight.
from .schema import *  # noqa: F403

__all__ = [
    "AttributeDef",
    "AttributeKind",
    "AttributeMapping",
    "CahvoreModel",
    "ChangeCursor",
    "CudaArray",
    "CudaArrayMapping",
    "CylindricalDem",
    "Device",
    "Dirty",
    "Frame",
    "IsdFrame",
    "IsdLineScanner",
    "IsdPushFrame",
    "Operation",
    "OverlayHit",
    "Path",
    "PolarStereographicDem",
    "Prim",
    "PrimSet",
    "PrimType",
    "Quat",
    "Renderer",
    "RenderProductOutput",
    "RenderVarOutput",
    "Scene",
    "Wireframe",
    "author_singlet_chromatic",
    "camera_image_corners",
    "celestial_to_local",
    "color_index_to_temperature",
    "dense_nanovdb_grid",
    "dirty_of",
    "distant_light_transform",
    "fbm_noise",
    "find_prim_type",
    "greenwich_sidereal_time",
    "julian_date",
    "load_image",
    "local_terrain_of",
    "look_at_transform",
    "noise_permutation",
    "prim_types",
    "read_cahvore_models",
    "read_cylindrical_dem",
    "read_polar_stereographic_dem",
    "reflectance_library",
    "save_scene",
    "stage_type",
    "star_illuminance",
    "synthetic_stars",
    "tokens",
    *schema.__all__,
]
