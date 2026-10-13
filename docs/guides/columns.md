# Columns

A simulation keeps its bodies' state as arrays: every body's position in one array, every colour in another. A column is one attribute of many prims held as the rows of one such array, prim i's value row i. You write the whole array at once, or map it and write it in place, and the scene records one change for it however many prims it covers. Nothing visits the prims one by one: a renderer that reads the rows whole poses 100 000 bodies from one buffer.

Columns are Ernie's answer to the batched reads and writes of Omniverse Fabric. Arrays cross in and out through DLPack, so NumPy, PyTorch, CuPy and JAX arrays on the CPU or on CUDA are taken without copying.

## Writing a column

Get a column with `scene.column(prims, attribute)`, for a `PrimSet` (the prims, in row order) and an attribute name. `write` takes an array of one row per prim:

```python
import numpy as np
import ernie
from ernie import tokens

scene = ernie.Scene()
paths = [f"/World/Cube{i}" for i in range(1000)]
for i, path in enumerate(paths):
    scene.define(path, ernie.Cube, size=0.5, translate=(i, 0, 0))

bodies = ernie.PrimSet(paths)
poses = scene.column(bodies, tokens.TRANSLATE)
colours = scene.column(bodies, tokens.DISPLAY_COLOR)

poses.write(np.random.rand(1000, 3).astype(np.float32))
colours.write(np.random.rand(1000, 3).astype(np.float32))
```

The scene keeps the array itself, not a copy, so keep writing a fresh array, or the same one again after changing it, once per step. A CUDA array is read after the work already queued on its `stream` (`poses.write(rows, stream=torch.cuda.current_stream().cuda_stream)`).

`read()` returns the rows as they are. Each prim reads its own row too: `scene.read("/World/Cube7", tokens.TRANSLATE)` and the typed getters see the column's value.

A transform op column needs the op in the prim's `xformOpOrder`, so give every prim its transform once, as `translate=` does above, before writing the column.

## Mapping a column in place

`map(device)` gives the rows on a device to write into, and the changes count once you unmap. As a context manager it unmaps on exit:

```python
with poses.map("cpu") as rows:
    rows[:, 2] += 0.1
```

With a CUDA renderer, mapping on `"cuda"` hands you the renderer's own buffers for the attributes it reads from the GPU (transform ops of bodies it poses there, `displayColor`, temperatures, visibility), so a kernel writes straight into what the next frame draws:

```python
import torch

renderer = ernie.Renderer()
renderer.render(scene)  # the renderer now offers its buffers
with poses.map("cuda") as rows:
    torch.as_tensor(rows, device="cuda")[:, 2] += 0.1
```

When the renderer offers none for that attribute, mapping returns the rows written last if they live on that device, or on the CPU the prims' values gathered into a new array. Map again each step: the memory behind a mapping may alternate between frames to keep the last step's values for motion blur.

## Which attributes

Any numeric attribute can be a column: each row holds what the prim's attribute would (a number, a 3-vector, a quaternion as four numbers). These are read straight from the rows on the GPU, with no work per prim on the CPU:

| Attribute | Row | Read as |
| --- | --- | --- |
| `xformOp:translate`, `xformOp:orient`, `xformOp:scale` | 3 or 4 numbers | the body's pose, composed on the GPU |
| `primvars:displayColor` | 3 numbers | an instance colour |
| `ernie:thermal:temperature` | 1 number | an instance temperature, in kelvin |
| `visibility` | 1 number | 0 hides the instance |

Other attributes work as well, read through the prims' rows when the renderer syncs them.

## Following column changes

`scene.changes(cursor)` reports each prim a column write touched, as if each had been set alone. A consumer that takes the rows whole asks for them as columns instead:

```python
cursor = ernie.ChangeCursor()
changes, columns = scene.column_changes(cursor)
for column in columns:
    print(column.attribute, len(column), column.bits)
```

`changes` then holds only the prims changed one by one. Each `ColumnChange` gives the attribute, the dirty bits it raises and the prims in row order. When a consumer cannot use a column whole, `scene.expand_columns(changes, columns)` merges its prims, their descendants and their dependents back in, as `changes` would have listed them. A column whose prims have dependents (prims whose relationships target them, as prims bound to a material are its dependents) comes expanded already.

## C++

The same in C++, from `Scene::GetColumn`:

```cpp
#include <ernie/scene/scene.h>
#include <ernie/scene/tokens.h>

using namespace ernie;

PrimSet bodies(paths);
Column poses = scene.GetColumn(bodies, tokens::kTranslate);
poses.Write(Tensor::Zeros(DType::Float32, {count, 3}));

// In place on the CPU (or Device{DeviceKind::Cuda, 0} on a GPU).
float* rows = poses.Map(Device{}).As<float>();
rows[3 * 7 + 2] += 0.1f;
poses.Unmap();

// One ColumnChange for the whole write.
std::vector<ColumnChange> columns;
std::vector<Change> changes = scene.Changes(cursor, columns);
```

A renderer offers its buffers to the scenes it draws through `Scene::SetColumnStorage` and a `ColumnStorage`; the raster delegate does this itself. `RenderDelegate::MapArray` of a `PrimSet`, which mapped transform op rows directly, is deprecated in favour of `Column::Map`.

## Demos

- `demo_batched_columns` (C++ and Python): a field of cubes on a travelling wave, its poses, colours and temperatures written each frame as three columns; `--rows cuda` maps them on the GPU.
- `demo_batched_edits` (C++ and Python): the other way, Conway's Game of Life on 10 000 cubes recoloured prim by prim, only the cells that flipped, so a frame syncs only what changed.
