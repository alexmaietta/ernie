# Debug drawing and overlays

Overlays draw debug geometry and text over rendered images: lines, arrows, boxes, markers and labels in the world, text and panels on the image itself, and many items at once from arrays a simulation writes on the GPU. They are drawn after the sensor filters, into LdrColor (and the viewport), so they never change the observations of products that do not ask for them.

## Layers

An overlay is a named layer of the scene. Items stay until the layer is cleared; a simulation either clears and redraws a layer each step, or keeps batches of arrays it writes in place.

```python
debug = scene.overlay("debug")
debug.line((0, 0, 0), (1, 0, 0), color=(1, 0, 0), width=3)
debug.label((0, 0, 1), "goal")
debug.clear()
debug.visible = False  # hidden everywhere, items kept
```

A render product draws the layers its RenderSettings name in `ernie:overlays` (`overlays="*"` for every one, or a list of names); products that set nothing draw none. The viewer shows every visible layer.

In C++ the same layer is `scene.GetOverlay("debug")`, with `Line`, `Label`, `Clear` and `SetVisible`.

## Shapes

| Method | Draws |
| --- | --- |
| `line`, `lines`, `polyline` | Segments `width` pixels wide; `lines` colours each segment |
| `arrow` | A line with a cone for its head |
| `box`, `sphere`, `circle` | Box edges (placed by a transform), three great circles, a circle facing a normal |
| `arc` | Part of a circle, or with `filled=True` the sector it sweeps |
| `curve` | A cubic Bezier curve |
| `axes` | A transform's x, y and z axes in red, green and blue |
| `wire_mesh`, `triangles` | A mesh's edges, or its filled triangles |
| `marker`, `markers` | Circles, rings, squares, diamonds, crosses or pluses of `size` pixels |
| `label` | Text beside a point, over a backing box or with a halo |

Every method takes style keywords:

- `color`: RGB or RGBA display colour; alpha blends over the image.
- `width`: line width in pixels. `dash`: dashes and gaps this many pixels long, 0 for solid.
- `size`: marker diameter, or text height in pixels.
- `metric=True`: markers' and text's `size` in metres instead, so in the world they shrink with distance like the things they mark, still facing the camera.
- `depth`: how items meet the scene's surfaces. `"test"` hides them behind surfaces, `"xray"` draws them faded where surfaces hide them, `"always"` draws them over everything.
- `world`: the world whose views show the item in a batched camera, -1 for every view.
- `background`: the backing box of labels and text; alpha 0 for none.
- `halo`: text's outline, this many pixels of `background`'s colour around each character, in place of the box.
- `anchor`: a prim the item's coordinates are relative to (see below).
- `id` and `hover`: a number that hit tests and picks report, and the colour items with that id take while hovered (see below).

Lines and markers keep their size in pixels at any distance, and markers and text always face the camera. `polyline(points, colors=...)` blends a colour per point along the line, and joined segments meet with round ends.

## Text

Text is drawn in [Inter](https://rsms.me/inter/), baked once into signed distance fields, so it stays sharp at any size, scale or angle. It is kerned, and covers Latin-1: strings are UTF-8, and anything outside Latin-1 draws as `?`. A halo outlines each character instead of putting a box behind the text, which keeps labels legible over busy ground without hiding it:

```python
names.label(position, "Mare Tranquillitatis", size=14, background=(0, 0, 0, 0.8), halo=1.5)
```

A halo can be up to a fifth of the text's size.

## Text on the image

`text` and `rect` draw on the image itself, a heads-up display, over everything. `at` is a point of the image as fractions of its width and height from the top left, `offset` moves it in pixels (y down), and `align` is the item's own point that lands there, as fractions of its size. Without `align` each axis snaps to the nearer edge, so `at=(1, 0)` keeps text inside the top right corner, and `at=(0, 0.3)` hangs it down the left side.

```python
hud = scene.overlay("hud")
hud.clear()
hud.text(f"t = {t:.2f} s", offset=(12, 12), size=16)
hud.text("5 balls", at=(1, 0), offset=(-12, 12))
hud.rect((240 * share, 14), at=(0.5, 1), offset=(-120, -16), align=(0, 1), color=(0.3, 0.9, 0.4))
```

On a camera rendered through cube faces (a fisheye), screen items are drawn after the lens, so it bends the world's items but not the HUD.

## Buttons and picking

`button` draws a box with a centred label on the image. Items with an `id` can be hit: `screen_item_at(pixel, extent)` returns the id of the top screen item at a pixel, and items with a `hover` colour take it while `hovered` is their id. The viewer handles both: under the pointer it sets `hovered`, and `viewer.add_overlay_click_handler(clicked)` calls `clicked(viewer, overlay, id)` when a click lands on one, in place of picking the scene. Any other pointer, such as a VR controller's ray, can do the same through `screen_item_at` and `hovered`.

```python
menu = scene.overlay("menu")
menu.button("Pause", at=(0, 1), offset=(12, -12), id=1, background=(0.2, 0.2, 0.2, 0.8), hover=(0.3, 0.5, 0.9, 0.9))
viewer.add_overlay_click_handler(lambda viewer, overlay, id: toggle_pause())
```

Items in the world are picked on the GPU: `renderer.pick_overlay(scene, x, y)` draws the overlays once more into a single pixel and returns an `OverlayHit`, with the layer's name, and either the batch and its item index or, for an item drawn one at a time, its `id`. `viewer.add_overlay_pick_handler(picked)` calls `picked(viewer, hit)` for clicks on them.

## Images

`image(corners, image)` draws a (rows, cols, 3 or 4) uint8 image on a quad between four world points (top left, top right, bottom right, bottom left), with perspective-correct filtering. `screen_image(image, at=..., size=...)` places one on the screen instead, like `rect`. The image can be another camera's LdrColor, mapped on the GPU each step: `ernie.camera_image_corners(scene, camera, distance)` gives the corners of a camera's image plane `distance` metres in front of it, so its picture sits in its frustum where it was taken.

## Following prims

With `anchor`, an item's coordinates are in a prim's space, and it moves with the prim every frame without being redrawn:

```python
debug.label((0, 0, 0.5), "ball 0", anchor="/World/Ball0", depth="always")
debug.box((-0.3,) * 3, (0.3,) * 3, anchor="/World/Robot/Link3")
```

Prims posed on the GPU (transform ops written as rows of CUDA arrays) are followed there: the renderer reads their poses on the GPU, so overlays on a robot whose links a simulator writes in place cost the host nothing per step.

## Batches from arrays

Batch methods draw many items of one kind from arrays with a row per item. The arrays may be on the host (NumPy) or on the renderer's CUDA device (PyTorch, JAX, CuPy, Warp). GPU arrays are read as they are every frame, copied device to device on their `stream`, so writing into them is enough to redraw. Host arrays written in place need `touch()`.

| Method | Items |
| --- | --- |
| `segments`, `arrows` | Lines from `starts` to `ends` |
| `points` | Markers |
| `boxes` | Box edges around centres, turned by quaternions |
| `spheres` | Three great circles each |
| `frames` | Coordinate axes |
| `capsules` | Wireframe capsules along their local z axis |
| `trails` | A polyline per item from a ring buffer |
| `numbers` | Values printed as text, formatted on the GPU |
| `texts` | A string per item: a list of strings, or an (N, K) int32 array of character codes a program may rewrite |

Every batch takes optional per-item `colors`, line `widths` or marker `sizes`, and `worlds`. With `worlds`, a batched camera's views each draw only their own world's items, sorted by world on the GPU, so thousands of worlds cost no more per view than one. Items without `worlds` or a style's `world` take their anchors' worlds: a prim's `ernie:world` (from it or an ancestor), or `pose_worlds` for items posed under pose arrays. Coordinates are float32 relative to `origin`, so a point far from the world's origin keeps its precision.

## Physics engines

Batches can be fed straight from a physics engine's GPU arrays, so debug drawing needs no work on the host each step:

- `count`: a one-element int32 array of the rows in use, such as a contact counter. The GPU draws only those, through indirect draws, so arrays can be allocated at their capacity and the count changed on the GPU.
- `poses`: rows of position then quaternion x, y, z, w, as Warp's `transform` and Newton's `body_q` hold them. The items are posed under them, item i under pose `pose_indices[i]` (or pose i), turned on the GPU every frame and then placed by `pose_frame` (for a Y-up model, say). With `pose_worlds` (Newton's `body_world`), each item is in its body's world.
- `directions=True`: for segments and arrows, `ends` are vectors from the starts, scaled by `scale`. Velocities and forces draw from the simulation's own arrays.
- `trails`: (N, L, 3) points, a ring buffer per item whose newest point is at `heads`. A segment with a NaN end is left out, so NaN rows start a trail afresh. `fade` fades each trail to its oldest point, and `colors` may be one per point (a speed gradient).
- `numbers`: values printed as `prefix`, the value with `decimals` digits, then `suffix`, formatted on the GPU. Beside points in the world, or with `screen=True` on the image, a line apart from `at`.

```python
contacts = scene.overlay("contacts")
contacts.points(contact_points, shape="ring", size=10, count=contact_count, depth="always")
contacts.arrows(contact_points, contact_normals, directions=True, scale=0.5, count=contact_count)
scene.overlay("velocities").arrows(positions, velocities, directions=True, scale=0.15)
scene.overlay("frames").frames(zeros, length=0.5, poses=body_q)
scene.overlay("trails").trails(trail_points, heads=heads, width=2)
scene.overlay("hud").numbers(stats[:1], prefix="t = ", suffix=" s", screen=True, offset=(12, 12))
scene.overlay("names").texts(body_names, points=label_offsets, poses=body_q, pose_worlds=body_world)
```

## Writing in place

`Renderer.map_overlay(scene, overlay, batch, array)` maps a batch's array into the renderer's own memory as a CUDA array: a simulation writing it there skips even the device-to-device copy. The batch must have been drawn once. Map again every step (the same memory, with no copy), so the stream waits for the frames still reading it. `Overlay.batch_count` gives the index of the last batch added.

## Demos

- `demo_overlays_hud`: screen text and an energy bar redrawn each step, labels following bouncing balls, heights formatted on the GPU.
- `demo_overlays_physics`: every line style, colour scheme and depth mode over a small simulation: trails coloured by speed, dashed predictions in x-ray, force arrows, fading contacts, bounds, springs coloured by strain, frames, arcs and curves, and a legend.
- `demo_overlays_menu`: a menu of buttons with hover colours, handled by click handlers.
- `demo_overlays_camera_billboard`: a drone camera's picture on its image plane in the world and as a picture in picture, with its frustum.
- `demo_overlays_moon_names`: the IAU's lunar feature names, fetched in the background, with halos, chosen each frame by a level of detail (how many, how small, how far).
- `demo_interop_physics_debug`: a Warp simulation whose trails, velocities, frames, bounds, contacts, swept capsules and HUD are all read from its GPU arrays.
