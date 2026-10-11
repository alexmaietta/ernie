# Contributing to Ernie

Thanks for helping. This page covers building, testing and the conventions the code follows.

## Building

You need CMake 3.24 or later, a C++20 compiler, Ninja, and a Vulkan 1.3 driver. CMake fetches every other dependency (see [third_party/README.md](third_party/README.md)). On Linux the viewer also needs the X11 and Wayland development headers:

```sh
sudo apt install ninja-build libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libwayland-dev libxkbcommon-dev
```

```sh
cmake -S . -B build -DERNIE_BUILD_TESTS=ON -DERNIE_BUILD_DEMOS=ON -DERNIE_INSTALL=ON -DERNIE_BUILD_PYTHON=ON -DERNIE_BUILD_APP=ON
cmake --build build
```

Every option is off by default, so a plain configure builds just the libraries.

| Option | Default | What it does |
| --- | --- | --- |
| `ERNIE_BUILD_TESTS` | off | GoogleTest suites and the Python tests |
| `ERNIE_BUILD_DEMOS` | off | A program per C++ demo, `demo_<category>_<name>` (the `ernie` program has them all either way) |
| `ERNIE_BUILD_PYTHON` | off | The `ernie` Python module |
| `ERNIE_BUILD_APP` | off | The viewer (`ernie` executable and `ernie::app`) |
| `ERNIE_INSTALL` | off | libernie, the install rules and packages (not on Windows yet) |
| `ERNIE_TRACY` | off | Tracy profiler zones |

## Development environment

Python work uses a virtual environment in the repository, made with [uv](https://docs.astral.sh/uv/). The `dev` extra brings the formatting, linting and type checking tools (ruff, mypy, pre-commit), `test` brings pytest, and `demos` brings Warp for the interop demos:

```sh
uv venv
uv pip install numpy nanobind ruff mypy pre-commit pytest pillow warp-lang
pre-commit install
```

Then configure with `-DPython_EXECUTABLE=$PWD/.venv/bin/python` so the module, its stubs and the Python tests use the same interpreter. The build writes the package to `build/python/ernie`: put it on the path with `PYTHONPATH=build/python` to run demos and tests against it.

| Tool | What it checks | Run it by hand |
| --- | --- | --- |
| clang-format | C++ layout (`.clang-format`) | `pre-commit run clang-format --all-files` |
| cmake-format | CMake layout (`.cmake-format.yaml`) | `pre-commit run cmake-format --all-files` |
| cppcheck | C++ static analysis | `pre-commit run cppcheck --all-files` |
| ruff | Python lint and layout (`[tool.ruff]` in `pyproject.toml`) | `ruff check --fix . && ruff format .` |
| mypy | Python types against the built package's stubs (`[tool.mypy]`) | `mypy` (after a build) |
| pre-commit | All of the above on the files a commit changes | `pre-commit run --all-files` |

mypy reads `build/python`, so build first: it checks the package as built (its `.py` files with the extension's generated `.pyi` stubs) and the demos.

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

Packages: `cmake --preset package-deb` (or `package-rpm`), then `cmake --build --preset package-deb` and `cpack --preset package-deb`. CI builds them in a manylinux_2_34 container and installs them on clean Ubuntu, Debian, Rocky Linux and Fedora (the `packages` and `package-test` jobs).

## Code style

- Code wraps at 70 columns: C++, Slang, CMake and Python alike, comments and docstrings included. Fill comment paragraphs to the limit rather than leaving lines short. Markdown is the exception: one line per paragraph or list item.
- Formatting is enforced by [pre-commit](https://pre-commit.com): clang-format, cmake-format, cppcheck, ruff, mypy and whitespace fixes. Run `pre-commit install` once, or `pre-commit run --all-files` before pushing.
- Python is typed: annotate public functions, and keep `mypy` clean. The package ships `py.typed` and stubs generated from the bindings (nanobind's stubgen for `ernie._ernie`, `ernie/python/stub_schema.py` for the typed prim classes); give a binding an `nb::sig` where its generated signature is too loose. Follow ruff's import order and modern syntax (`X | None`, built-in generics).
- Each module in `ernie/` is a library that depends only on the modules before it (see `ernie/CMakeLists.txt`). Headers sit next to their sources and are included as `"ernie/<module>/<file>.h"`.
- Document public API with `///` comments. When code implements a published technique, cite it in a `References:` list in the doc comment.
- Scene attributes use physical units (meters, kelvin, lux and so on). Say which unit in the doc comment.
- Demos go in `demos/<category>/` as a pair, `<name>.cpp` and `<name>.py`, the same demo in each language (registered with `ERNIE_REGISTER_DEMO` in C++); only `demos/interop/` is Python alone. `ernie/test/python/test_demos.py` checks every pair renders the same image.
- Tests are named `<module>_<topic>_test.cpp`. GPU tests use the `_gpu_test.cpp` suffix.

## Pull requests

- Keep a pull request to one change, with tests for new behavior.
- Write a one-line, imperative commit message describing what changed.
- CI must pass: formatting, the build and tests on lavapipe, and the wheel.

By contributing, you agree that your contributions are licensed under the [BSD 3-Clause License](LICENSE).
