# Contributing to Ernie

Thanks for helping. This page covers building, testing and the conventions the code follows.

## Building

You need CMake 3.24 or later, a C++20 compiler, Ninja, and a Vulkan 1.3 driver. CMake fetches every other dependency (see [third_party/README.md](third_party/README.md)). On Linux the viewer also needs the X11 and Wayland development headers:

```sh
sudo apt install ninja-build libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libwayland-dev libxkbcommon-dev
```

```sh
cmake -S . -B build -DERNIE_BUILD_PYTHON=ON -DERNIE_BUILD_APP=ON
cmake --build build
```

| Option | Default | What it does |
| --- | --- | --- |
| `ERNIE_BUILD_TESTS` | on, top level | GoogleTest suites and the Python tests |
| `ERNIE_BUILD_DEMOS` | on, top level | C++ demos in `demos/cpp` |
| `ERNIE_BUILD_PYTHON` | off | The `ernie` Python module |
| `ERNIE_BUILD_APP` | off | The viewer (`ernie` executable and `ernie::app`) |
| `ERNIE_INSTALL` | on, top level | libernie, the install rules and packages |
| `ERNIE_TRACY` | off | Tracy profiler zones |

Python tooling uses [uv](https://docs.astral.sh/uv/): `uv venv && uv pip install numpy pytest nanobind`, then configure with `-DPython_EXECUTABLE=$PWD/.venv/bin/python`.

## Testing

```sh
ctest --test-dir build -j4 --output-on-failure
```

Tests carry labels, so you can pick a subset with `-L`:

| Label | Needs |
| --- | --- |
| `unit` | Nothing: CPU only |
| `gpu` | A Vulkan device |
| `python` | The Python module (`ERNIE_BUILD_PYTHON`) |
| `install` | `ERNIE_INSTALL`: installs Ernie and builds an application against it |

Without a GPU, use Mesa's software driver, lavapipe, as CI does. CUDA interop tests skip themselves there.

```sh
sudo apt install mesa-vulkan-drivers vulkan-validationlayers xvfb
VK_DRIVER_FILES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json) xvfb-run -a ctest --test-dir build -j4
```

Performance changes: `tools/benchmark.py --check` compares batched rendering against the baselines recorded for your GPU.

Packages: `tools/packaging/` builds the DEB, RPM and TGZ packages in a manylinux container and installs them on clean Ubuntu and Rocky Linux; each script's header shows how to run it.

## Code style

- Formatting is enforced by [pre-commit](https://pre-commit.com): clang-format (`.clang-format`, 100 columns), cmake-format, cppcheck, and whitespace fixes. Run `pre-commit install` once, or `pre-commit run --all-files` before pushing.
- Each module in `ernie/` is a library that depends only on the modules before it (see `ernie/CMakeLists.txt`). Headers sit next to their sources and are included as `"ernie/<module>/<file>.h"`.
- Document public API with `///` comments. When code implements a published technique, cite it in a `References:` list in the doc comment.
- Scene attributes use physical units (meters, kelvin, lux and so on). Say which unit in the doc comment.
- Demos go in `demos/cpp` or `demos/python` and are named `demo_<category>_<description>`.
- Tests are named `<module>_<topic>_test.cpp`. GPU tests use the `_gpu_test.cpp` suffix.

## Pull requests

- Keep a pull request to one change, with tests for new behavior.
- Write a one-line, imperative commit message describing what changed.
- CI must pass: formatting, the build and tests on lavapipe, and the wheel.

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
