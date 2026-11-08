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

### 2. Benchmarks

A `btest_render_points.cpp` suite in `tests/bench/render`, registered like the scaling and instancer suites and with capped run times.

- **Single world scaling.** One world, one camera, point counts of 1k, 10k, 100k, 1M, 4M and 16M, on hardware and software traversal.
- **Batched worlds.** 16, 256, 1024 and 4096 worlds with 1k, 10k and 50k points each, one camera per world at a small resolution, cloned from one world as the scaling suite does.
- **Motion modes.**
    - Stationary, with points written once and only the camera moving.
    - Moving, with every point rewritten each step from frames kept on the device, so host uploads don't count.
    - Mapped, with a kernel writing positions through `Map(MapAccess::Write)` as a simulation would.
    - Recounted, spawning and killing a fraction of the points each step so counts change.
    - Resized, with only widths changing.
- **Layouts.** A uniform box like gas or dust, a dense clustered splash like a fluid, and a flat bed like granular media, since they stress BVH quality differently.
- **Baselines.** The same counts as Sphere prims and as a PointInstancer of one sphere, up to where those stop scaling, to show what the Points path buys.
- **Measurements.** Step time split into box writes, BLAS builds, TLAS work and tracing (the `Points/Boxes`, `Points/Blas` and existing zones), points per second, rays per second, and box and BLAS memory per point.

These are the baselines milestone 4 is measured against.

### 3. Shading and sensors

- Disks from `normals`, facing each point's normal, following USD.
- Per-point colour and opacity as device arrays a simulation can write every step. The packed primvars table is built on the host, so it doesn't suit per-step particle data. Opacity goes through the non-opaque procedural path.
- `ids` on Points for instance id and segmentation outputs.
- Lidar and radar coverage tests, for dust, ejecta and rain.

### 4. Efficiency

- A refit or rebuild policy per Points prim, refitting up to a limit and then rebuilding as the instance table's `maxRefits` does, chosen from the moving benchmark.
- Capacity that grows geometrically, with inactive boxes for dead slots, so spawning and killing particles doesn't reallocate the BLAS.
- Clustered leaves, with 4 to 8 Morton-adjacent points per box and the intersector looping over them, for dense fluids and batched worlds.
- Hardware spheres through `VK_NV_ray_tracing_linear_swept_spheres` where the adapter has them, with boxes staying the portable path.

### 5. Simulation handoff and motion

- Changing counts on attached and mapped points, through a capacity-sized buffer whose dead slots the box kernel skips.
- Simulation writes that overlap tracing the previous step, through the snapshot ring planned in `TODO.md`.
- Motion blur from `velocities`, growing each box by the shutter and intersecting the moving sphere.

## Open questions

- Is 1 cm the right default width? USD leaves it to the renderer, and Storm uses 1.
- Are hardware spheres worth adding, given that the AMD presentation favours the portable box path?
