# Points

The plan for drawing USD Points in Ernie, aimed at particles from GPU simulations such as fluids, granular media, dust plumes and spray. Points render as spheres (or disks once normals are supported) and are seen by cameras, lidar and radar through the same intersector.

## Design

### One BLAS per Points prim

Each Points prim gets one AABB BLAS with one procedural primitive per point. The TLAS holds one instance per prim, so its size and rebuild cost don't grow with the particle count. That matters for batched worlds, where every world has its own TLAS.

The other options were rejected for these reasons.

- One TLAS instance per point, as the analytic shapes do with the unit box, makes every world's TLAS grow with the particle count and get rebuilt every step.
- PointInstancer copies of a sphere prototype pay the same per-instance TLAS cost. They stay the right tool for shaped grains such as DEM rocks, not for spheres.

### Data flow

- `PointsTable` (in `ernie/scene/tables/points.cpp`) owns a box buffer and a BLAS per prim, plus a `PointsRecord` per Points row holding the BLAS handle, the points and widths addresses, their counts, and local bounds.
- When points or widths change, `WritePointBoxes` writes one box per point straight from the column store and reduces the local bounds in the same pass. The BLAS is then rebuilt on the GPU, with no host readback.
- A Points slot in the TLAS has `ShapeKind::Points`, and its `ShapeRecord::row` names the Points row. `ShapeIntersector` intersects the point given by the primitive index as a sphere, recentering the ray at its closest approach first so tiny particles far away stay precise.
- At a hit, the primitive index is the point index. Vertex primvars read by it, and the shading normal points from the center.
- The table records `TableId::Points` when records change (the instance table then rebuilds the TLAS) and `TableId::PointsMoved` when BLASes are rebuilt (the TLAS refits).

### GPU simulations

GPU simulations already have two zero-copy paths. `Map(MapAccess::Write)` hands a simulation a tensor aliasing the column store, and `Attach` in place makes the column read the simulation's own buffer. Either one feeds `WritePointBoxes` directly. The gaps are changing counts on attached arrays and overlapping simulation writes with tracing, covered in milestone 5.

## Milestones

### 1. Spheres (done)

- `Points : PointBased` schema with `widths`. One width applies to every point, a width per point applies to each, and empty widths draw points 1 cm wide (`kDefaultPointWidth`).
- `PointsTable` with box writes, local bounds and a BLAS rebuild on every change.
- Sphere intersection in both tracers, shading from the hit, vertex primvars by point index, and prim and slot bounds.
- Points under prototype roots draw through PointInstancers like any gprim.
- Ray cast tests for per-point, shared and default widths, moved points, changing counts and emptied points, passing on hardware and software traversal.

### 2. Benchmarks (done)

`tests/bench/render/btest_render_points.cpp` in `ernie_render_bench`. Every case renders a 64×64 shadowed DisplayColor view per world, with the camera nudged each step so tracing is always measured.

- **Single world.** `Render/Points/World/<layout>/motion:<motion>` with 1k, 10k, 100k, 1M, 4M and 16M points, on hardware and software traversal.
- **Batched worlds.** `Render/Points/Worlds/box/motion:<motion>` with 16, 256, 1024 and 4096 worlds of 1k, 10k and 50k points each, cloned from one world, capped at 16M points in total.
- **Motion modes.**
    - `stationary` leaves the points alone and only moves the camera.
    - `moving` copies device-side frames into the points through `Map(MapAccess::Write)`, which is exactly what a GPU simulation does, so no host upload is counted.
    - `uploaded` sets new positions from the host every step.
    - `recounted` alternates between all the points and 90% of them, so the BLAS and box buffer are recreated each step.
    - `resized` rewrites only widths through `Map`.
- **Layouts.** `box` is uniform like gas or dust, `splash` is eight dense Gaussian blobs like fluid, and `bed` is a thin layer like granular media. The splash and bed layouts run the stationary and moving modes only.
- **Baselines.** `Render/Points/Baseline/spheres` (one Sphere prim per point, up to 100k) and `Render/Points/Baseline/instancer` (a PointInstancer of one sphere, up to 1M), stationary and moving.
- **Counters.** CPU ms per step, GPU ms split into `gpu boxes ms` (`Points/Boxes`), `gpu blas ms` (`Points/Blas`), `gpu tlas ms` (the `Bvh/` and `Instances/` zones), `gpu render ms` and the rest, plus points/s, px/s and device bytes per point, net of the motion frames.

The first numbers on an RTX 5070 Laptop GPU with hardware traversal set the targets for milestone 4.

| Case | Step | Where the time goes |
| --- | --- | --- |
| 1M points, moving | 6.1 ms | 5.6 ms BLAS rebuild, 0.1 ms boxes |
| 16M points, moving | 86 ms | 74 ms BLAS rebuild, 2.9 ms boxes |
| 16M points, stationary | 0.3 ms | 154 B/point |
| 4096 worlds × 1k, moving | 77 ms | 45 ms CPU per step, 32 ms BLAS builds |
| 4096 worlds × 1k, recounted | 137 ms | 67 ms CPU per step |
| 1M instancer copies, moving | 8.4 ms | 6.4 ms TLAS, 8.1 ms CPU |
| 1k points, moving, software | 1.6 ms | 1.1 ms BLAS, mostly fixed cost |

So a moving cloud is bound by the BLAS rebuild, many small clouds are bound by per-cloud CPU work in `PointsTable` and by per-BLAS build overhead, and Points beat the instancer on moving particles while using a sixth of the memory.

### 3. Shading and sensors (done)

- Disks from `normals`. One normal applies to every point, a normal per point applies to each, and each disk is as wide as its point. The sphere box still bounds it, so boxes don't change.
- Per-point colour from a vertex `displayColor` primvar, which every gprim now honours, with instancer primvars still taking precedence.
- Stochastic opacity from `displayOpacity`, constant or as a vertex primvar. Each candidate hit is kept with probability equal to its opacity, from a hash of the object-space ray and the point index, inside the shape intersector, so camera, shadow, lidar and radar rays all see it in both tracers without non-opaque ray flags. The same ray always makes the same choice, so a still camera without jitter sees a fixed pattern that pixel samples average out.
- `ids` on Points, reported per point by the `InstanceId` output, with each point's index when ids are empty, and the Points prim's stable id as `InstancerStableId`.
- Lidar tests for dust spheres, dust disks, transparent dust and a moved point, and a radar test for a moving plume.

Still open from this milestone.

- Colour and opacity as primvars are packed on the host, so a simulation can't write them every step through `Map`. Device-writable per-point colour and opacity arrays belong with milestone 5.
- Point clouds from lidar don't carry an instance id channel, only the prim's stable id.

### 4. Efficiency

- A refit or rebuild policy per Points prim (done). Moved points refit their BLAS and rebuild after `SceneDesc::maxRefits` refits, and a count change rebuilds. On hardware traversal a moving step drops from 6.1 to 1.3 ms at 1M points, 86 to 19 ms at 16M, and 77 to 30 ms for 4096 worlds of 1k, and on software traversal from 1.6 to 0.4 ms at 1k.
- Per-cloud CPU work in `PointsTable::Run` and `Build` that scales to thousands of worlds, and batched BLAS builds for many small clouds.
- A lower fixed cost for small software BLAS builds.
- Capacity with inactive boxes for dead slots (done). Capacity grows by half again when exceeded and shrinks to the count when under a quarter is used, and slots past the count get NaN boxes, which Vulkan treats as inactive and the software LBVH now leaves out of its leaves. A count change within capacity rebuilds the same BLAS instead of reallocating it, since refits can't change which primitives are active. Recounting 4096 worlds of 1k points dropped from 137 to about 87 ms, with what remains mostly host array uploads.
- Clustered leaves, with 4 to 8 Morton-adjacent points per box and the intersector looping over them, for dense fluids and batched worlds.
- Hardware spheres through `VK_NV_ray_tracing_linear_swept_spheres` where the adapter has them, with boxes staying the portable path.

### 5. Simulation handoff and motion

- Changing counts on attached and mapped points, through a capacity-sized buffer whose dead slots the box kernel skips.
- Simulation writes that overlap tracing the previous step, through the snapshot ring planned in `TODO.md`.
- Motion blur from `velocities`, growing each box by the shutter and intersecting the moving sphere.

## Open questions

- Is 1 cm the right default width? USD leaves it to the renderer, and Storm uses 1.
- Are hardware spheres worth adding, given that the AMD presentation favours the portable box path?
