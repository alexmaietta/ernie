# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-08

### Fixed

- Loading a HIP kernel no longer leaves `hipErrorNotFound` as the thread's last HIP error when the module lacks optional globals, so PyTorch on ROCm no longer raises `named symbol not found` at its next call after Ernie builds a scene or renderer. An error that was already pending stays pending.
- hipRTC loads from the folder of the HIP runtime Ernie uses, such as the one bundled in PyTorch, instead of from another ROCm install that would pull in a second comgr and HSA runtime. Runtime libraries already in the process are also found under their unversioned names.

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

[unreleased]: https://github.com/alexmaietta/ernie/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/alexmaietta/ernie/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/alexmaietta/ernie/releases/tag/v0.1.0
