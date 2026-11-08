# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Release candidates. `ERNIE_PRERELEASE` in `CMakeLists.txt` names one, such as `rc1`, and a matching `v0.1.1rc1` tag publishes `0.1.1rc1` to PyPI and as a GitHub pre-release, with the Docker image tagged `0.1.1rc1` alone and Debian and RPM packages versioned `0.1.1~rc1` so the final release upgrades them. CI also runs on pushes to `release-*` branches.
- `Scene::WorldTransforms`, `Scene::ReadWorldTransformsAsync` and Python's `scene.world_transforms(paths)` read world transforms as of the latest publish.
- `Scene::PrimBounds`, `Scene::ReadPrimBoundsAsync` and Python's `scene.prim_bounds(paths)` read each prim's world box, relative to its world's origin, as of the latest publish.
- Snapshots carry `tables`, the device address of the render tables header, and keep every table it points to alive until they're destroyed.

### Changed

- The `Normal` render var is the surface normal from the geometry, interpolated from its normals before materials and normal maps, in every render mode. Real-time products used to report the normal-mapped normal, and products without color outputs didn't.
- The `Normal` render var has three float channels instead of four. Misses write a zero normal in place of the old fourth channel, and `ColorizeNormals` shows them as transparent.
- Every TLAS is placed relative to a render origin that follows the first sensor a render product points at, so geometry near a camera stays precise up to solar system distances.
- Render products are resolved by the scene when it publishes. Editing prims that a product doesn't read, or adding and removing them, no longer prepares every product again or reallocates their outputs. Only products whose settings, vars or sensors changed are prepared again.
- Editing display colors, primvars or authored normals no longer rewrites instance records, and only rewrites the surfaces of the worlds that show the edited prims.

### Removed

- The derived attributes `ernie:worldTransform`, `ernie:computedVisibility`, `ernie:computedPurpose` and `ernie:computedMaterial`. The hierarchy system keeps them in its own tables, and world transforms are read through `WorldTransforms` instead.
- The derived mesh attributes `ernie:blas`, `ernie:triangleCorners`, `ernie:triangleFaces`, `ernie:pointTriangleStarts`, `ernie:pointTriangles`, `ernie:pointBounds`, `ernie:smoothNormals`, `ernie:tangents` and `ernie:tangentPrimvar`. The geometry and tangents systems keep them in mesh and tangent records.
- The derived attributes `ernie:worldMin` and `ernie:worldMax` on boundable prims, and `ernie:textures` and `ernie:textureSize` on materials and lights. The bounds and textures systems keep them in their own tables, and prim boxes are read through `PrimBounds`. No derived attributes remain in the schemas, and the `derived` schema keyword is gone, so a schema that uses it fails to build. The catalog's `instance` and `next` columns on multiple-apply APIs are flagged `Internal` and no longer show in the viewer's properties panel.

### Fixed

- Ernie builds with a newer GCC paired with binutils older than 2.36, as HPC compiler modules often are, because configuring skips libdeflate's AVX-VNNI checksum when the assembler can't encode it.
- Ernie builds with GCC 12, whose standard library has no `<format>`, because it formats text with the header-only fmt library on every compiler. CMake 3.25.1 also accepts the VMA download again.
- `ernie --version`, the viewer's About window and `ernie.build_info().version` show the full version with its patch and any release candidate, such as `0.1.0` or `0.1.1rc1`, instead of `0.1`.
- Loading a HIP kernel no longer leaves `hipErrorNotFound` as the thread's last HIP error when the module lacks optional globals, so PyTorch on ROCm no longer raises `named symbol not found` at its next call after Ernie builds a scene or renderer. An error that was already pending stays pending.
- hipRTC loads from the folder of the HIP runtime Ernie uses, such as the one bundled in PyTorch, instead of from another ROCm install that would pull in a second comgr and HSA runtime. Runtime libraries already in the process are also found under their unversioned names.
- Minimal mode's `full` shading lights surfaces with their normal maps, which it used to drop because the surface was copied before its material bent the normal.
- A publish that throws in a system no longer loses the writes made before the next publish, because the store's stamp advances and systems record what they've seen only after they run.
- Changing how many copies a point instancer draws in the same publish as moving a prim every world holds rebuilds each TLAS instead of refitting it with the old instance counts.
- Contact shadows and occlusion hold far from a world's origin, because new rays start from the hit's own floating point error bound instead of an offset that grew with distance.

## [0.1.0] - 2026-11-06

The first release of Ernie, a real-time spectral GPU renderer for planetary robotics and sensing with C++ and Python APIs.

### Added

- The initial import, with Vulkan, CUDA and HIP device backends, Slang kernels compiled at runtime, software and hardware BVHs, a USD-style scene with batched worlds, real-time and minimal render modes, camera, lidar and radar sensors, scene importers, the interactive viewer, tests, benchmarks and CI.
- A single `ernie` command in `ernie/app` that opens the viewer when run with no options, loads the viewer library at runtime when it is installed, and takes `-V`, `-v` and `-q` for the version and log level.
- The `ernie-renderer` wheel ships the viewer, so `pip install ernie-renderer` brings the full interactive app with no separate package.
- The `ernie` command renders without a window. `--headless`, `--frames` and `--output` render the default scene and write its AOVs, `--benchmark` with `--warmup` and `--json` times steps, and `--profile` prints the profiler's report.
- `--worlds`, `--camera`, `--product`, `--preset` and `--aovs` set up what the command renders, in the viewer too.
- Kernels are cached in `kernels/<version>` under the cache folder, which `ERNIE_CACHE_DIR` moves, and stay under 1 GiB by dropping the least recently used.
- CI fails on any compiler or linker warning.

### Fixed

- CUDA scenes build their BVHs when NVRTC 12 compiles the kernels, for example in a process that already loaded PyTorch's NVRTC, instead of crashing with an illegal or misaligned address.
- Python calls that take DLPack tensors import each one on its own producer stream, so pose writes from a busy CUDA stream no longer read the tensor before its producer finishes writing it.
- Framing a selected prim reads its world bounds instead of failing, and framing the whole scene leaves out infinite planes.
- Infinite planes no longer stretch scene and world bounds, and finite planes get flat boxes.
- Cylinders, cones and capsules no longer show black speckles on their lit sides when seen from far away.

[unreleased]: https://github.com/alexmaietta/ernie/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alexmaietta/ernie/releases/tag/v0.1.0
