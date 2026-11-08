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
- `OverlayStyle::screen` anchors any overlay shape, label, number or image to a point of the image, from `{0, 0}` at the top left to `{1, 1}` at the bottom right, and reads its points as pixel offsets from there. Screen shapes draw over everything and can still be picked, so logos, credits, plots and debug readouts land in rendered frames. Python's `OverlayStyle(screen=(x, y))` does the same.
- `Overlay::Prims` takes a color per prim or group, and Python's `overlay.prims(..., colors=)` does the same.
- `OverlayStyle::screenSized` on prim frames gives their length in pixels, so joint and prim axes keep their size on screen.
- `Overlays` holds named overlay layers apart from any renderer, with `Set`, `Remove`, `Names` and `Pick`. `RendererDesc::overlays` shares one between renderers, and `Renderer::GetOverlays` returns the one a renderer draws. Python's `ernie.Overlays`, `Renderer(..., overlays=)` and `renderer.overlays` do the same.
- `RenderProduct:overlays` names the overlay layers a camera product draws, with a trailing `*` matching by prefix and `*` drawing them all. The overlay still comes back as `FrameOutput::overlay`, beside the render vars.
- The viewer draws the world's axes as the camera sees them in the viewport's bottom left corner, and its eye menu adds selection axes, every geometry prim's bounds in a color each, and a checkbox for each scene overlay.
- The viewer's grid has two levels of lines that fade between zoom levels and toward its edge, and its settings set a fixed spacing, the lines to each major line, its extent, its height and the brightness of its minor and major lines.
- The viewer's properties panel resets a field to its schema default with the square at its end, shows translations, orientations, scales and other vectors as red, green and blue x, y and z fields, and picks relationship targets from a stage tree as well as by path or by dropping prims.
- The viewer's render resolution menu offers the viewport's size, sensor and display presets, or a custom size with a locked aspect. Its render scale menu offers 25 to 200 percent, and Fill Viewport fits the image to the viewport or shows it a texel per pixel.

### Changed

- The `Normal` render var is the surface normal from the geometry, interpolated from its normals before materials and normal maps, in every render mode. Real-time products used to report the normal-mapped normal, and products without color outputs didn't.
- The `Normal` render var has three float channels instead of four. Misses write a zero normal in place of the old fourth channel, and `ColorizeNormals` shows them as transparent.
- Every TLAS is placed relative to a render origin that follows the first sensor a render product points at, so geometry near a camera stays precise up to solar system distances.
- Render products are resolved by the scene when it publishes. Editing prims that a product doesn't read, or adding and removing them, no longer prepares every product again or reallocates their outputs. Only products whose settings, vars or sensors changed are prepared again.
- Editing display colors, primvars or authored normals no longer rewrites instance records, and only rewrites the surfaces of the worlds that show the edited prims.
- The scene owns the simulation clock. `PublishDesc::dt` advances it, `Scene::Time` and `Scene::SetTime` read and set it, and Python's `scene.time` and `scene.publish(dt=...)` do the same. `Renderer::Step` publishes with its `dt`, and the scene decides which products are due from their sensors' frame rates, so schedules advance with the scene's time even for products a step doesn't render.
- `Renderer::Step` traces a product only when the scene changed since that output set last traced it, and otherwise returns the same outputs with only their overlays drawn again. Overlay edits, picks and guide changes no longer re-render a still scene. Outputs come back with the pixels they were traced with, so a tensor written in place keeps those writes until the scene changes.
- `Scene::At` is now `Scene::ByIndex`, and `Image::At` is now `Image::Pixel`.
- Public scene headers moved to match the module's layout. `ernie/scene/access/` is now `ernie/scene/views/`, `ernie/scene/catalog/tokens.h` is now `ernie/scene/structure/tokens.h`, `ernie/scene/kernels/schema/` is now `ernie/scene/schema/`, `semantics.h` and `primvars.h` moved from `ernie/scene/kernels/` to `ernie/scene/records/`, `ernie/scene/kernels/ids.h` is now `ernie/scene/schema/ids.h`, and `ernie/scene/kernels/paged.h` is now `ernie/scene/columns/kernels/paged.h`.
- The overlay is its own module. `ernie/render/overlay/overlay.h` is now `ernie/overlay/overlay.h` and `ernie/render/kernels/overlay/style.h` is now `ernie/overlay/kernels/style.h`, and `ernie/ernie.h` still includes the overlay.
- The viewer's settings menu groups render resolution, selection, gizmo and grid settings into submenus.
- The viewer shows the memory, capture and profiler panels beside info and log by default, and hides light guides by default.
- The viewer's perspective view starts a little further out from the scene, and the default scene's camera looks straight at the middle of its shapes from lower down.

### Removed

- `Renderer::Time`, `Renderer::Reset` and Python's `renderer.time` and `renderer.reset`. The scene's clock replaces them.
- The public headers `ernie/scene/kernels/scene.h` and `ernie/scene/kernels/textures.h`. The authored scene data they described is internal to the scene module, and render kernels read the scene's records instead.
- `Snapshot::sceneData` and Python's `snapshot.scene_data`. Render kernels read only the render tables a snapshot's `tables` points at, so the authored columns no longer travel with a snapshot.
- The derived attributes `ernie:worldTransform`, `ernie:computedVisibility`, `ernie:computedPurpose` and `ernie:computedMaterial`. The hierarchy system keeps them in its own tables, and world transforms are read through `WorldTransforms` instead.
- The derived mesh attributes `ernie:blas`, `ernie:triangleCorners`, `ernie:triangleFaces`, `ernie:pointTriangleStarts`, `ernie:pointTriangles`, `ernie:pointBounds`, `ernie:smoothNormals`, `ernie:tangents` and `ernie:tangentPrimvar`. The geometry and tangents systems keep them in mesh and tangent records.
- The derived attributes `ernie:worldMin` and `ernie:worldMax` on boundable prims, and `ernie:textures` and `ernie:textureSize` on materials and lights. The bounds and textures systems keep them in their own tables, and prim boxes are read through `PrimBounds`. No derived attributes remain in the schemas, and the `derived` schema keyword is gone, so a schema that uses it fails to build. The catalog's `instance` and `next` columns on multiple-apply APIs are flagged `Internal` and no longer show in the viewer's properties panel.
- `Renderer::SetOverlay`, `Renderer::RemoveOverlay`, `Renderer::ShowOverlays`, `Renderer::HideOverlays`, `Renderer::PickOverlay` and their Python counterparts. `Overlays` holds and picks the layers, and a product's `overlays` attribute shows them.
- The viewer's eco mode, with `ViewportDesc::eco`, `ViewportDesc::ecoFrames`, `ViewerDesc::eco`, `ViewerDesc::ecoFrames`, `Viewer::EnableEco`, `Viewer::Eco`, `Viewer::SetEcoFrames`, `Viewer::EcoFrames`, `Viewer::EcoPaused` and their Python counterparts. Viewports render every frame, so overlay and guide changes show at once.

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
- The viewer frames the scene when it opens, which it skipped because it read the scene's bounds before the first publish.
- A selection outline traces only the edges where the selected prims are in front, so selecting a ground no longer outlines everything standing on it.

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
