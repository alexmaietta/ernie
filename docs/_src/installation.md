# Installation

Ernie ships as a Python package on PyPI, as C++ packages on GitHub releases, as a Docker image, and builds from source everywhere else.

## Requirements

| | Supported |
| --- | --- |
| Linux | x86_64 and aarch64 (glibc 2.34) |
| Windows | x64 and ARM64 |
| macOS | 26 or later on Apple silicon |
| Python | 3.10 or later |

Ernie renders on any device it has a backend for.

| Backend | Devices | Driver |
| --- | --- | --- |
| Vulkan | NVIDIA, AMD and Intel GPUs, Apple GPUs through MoltenVK, and CPUs through lavapipe | Vulkan 1.3 or later |
| CUDA | NVIDIA GPUs | CUDA 12.0 or later |
| HIP | AMD GPUs, both CDNA and RDNA | ROCm 6.2 or later |
| Metal | Apple GPUs | In progress |
| CPU | x86_64 and aarch64 CPUs through Embree | In progress |

Ray tracing hardware is optional. On devices without it, Ernie builds and traverses its own BVH in software, so every backend renders the same scenes. Hardware texture sampling is optional too, and devices without it filter textures in software.

On macOS, the Vulkan backend runs on Apple GPUs through MoltenVK, which Homebrew installs along with the Vulkan loader. macOS doesn't search Homebrew's library folder, so point `DYLD_FALLBACK_LIBRARY_PATH` at it too, and add that line to your shell profile to keep it.

```bash
brew install molten-vk vulkan-loader
export DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix)/lib
```

## Python

```bash
pip install ernie-renderer
```

Wheels are built for Linux, Windows and macOS. Check the install by listing the devices Ernie can render with.

```bash
ernie
python -c "import ernie; print(ernie.__version__)"
```

The `dev` release also holds wheels rebuilt from main on every push. They are versioned like `0.1.0.post0.dev42`, which pip treats as a pre-release, so ask for one with `--pre`.

```bash
pip install --pre ernie-renderer \
    --find-links https://github.com/alexmaietta/ernie/releases/expanded_assets/dev
```

### Viewer

The interactive viewer is a separate wheel that the `viewer` extra installs. It needs a GPU that can present to a window.

```bash
pip install "ernie-renderer[viewer]"
ernie-viewer
```

The `ernie-viewer` command opens the default scene, and `ernie-viewer --help` lists its options. From Python, open a viewer on your own scene.

```python
import ernie
import ernie.viewer

device = ernie.Device(backend=ernie.Backend.VULKAN, presentation=True)
scene = ernie.Scene(device)
viewer = ernie.viewer.Viewer(scene, product="/Render/View")
viewer.run()
```

## C++

Download the archive for your platform from the [latest release](https://github.com/alexmaietta/ernie/releases/latest). It holds the library, its C++17 headers, the CMake package and a pkg-config file. The Linux archives are built against glibc 2.34, so they run on any distribution from the [requirements](#requirements).

Releases follow tags, so `latest` points at the newest one, with `dev` rebuilt from main on every push. The commands below install the latest release from the command line. On Linux and macOS, look up its version first.

```bash
VERSION=$(curl -fsSL https://api.github.com/repos/alexmaietta/ernie/releases/latest \
    | sed -nE 's/.*"tag_name": *"v([^"]+)".*/\1/p')
```

Set `VERSION` yourself to install another release.

=== "Linux"

    ```bash
    URL=https://github.com/alexmaietta/ernie/releases/download/v$VERSION
    curl -fL $URL/ernie-$VERSION-Linux-$(uname -m).tar.gz \
        | tar -xz -C ~/.local --strip-components=1
    ```

=== "macOS"

    ```bash
    URL=https://github.com/alexmaietta/ernie/releases/download/v$VERSION
    mkdir -p ~/.local
    curl -fL $URL/ernie-$VERSION-macOS-aarch64.tar.gz \
        | tar -xz -C ~/.local --strip-components=1
    ```

=== "Windows"

    ```powershell
    $Release = Invoke-RestMethod https://api.github.com/repos/alexmaietta/ernie/releases/latest
    $VERSION = $Release.tag_name.TrimStart("v")
    $URL = "https://github.com/alexmaietta/ernie/releases/download/v$VERSION"
    $ARCH = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
    Invoke-WebRequest "$URL/ernie-$VERSION-Windows-$ARCH.zip" -OutFile ernie.zip
    Expand-Archive ernie.zip .
    Rename-Item "ernie-$VERSION-Windows-$ARCH" ernie
    ```

The `dev` release leaves the version out of its file names, so it installs from fixed URLs. Its builds are versioned like `0.1.0+dev.42`, which sorts after `0.1.0` and before the next release.

=== "Linux"

    ```bash
    URL=https://github.com/alexmaietta/ernie/releases/download/dev
    curl -fL $URL/ernie-Linux-$(uname -m).tar.gz \
        | tar -xz -C ~/.local --strip-components=1
    ```

=== "macOS"

    ```bash
    URL=https://github.com/alexmaietta/ernie/releases/download/dev
    mkdir -p ~/.local
    curl -fL $URL/ernie-macOS-aarch64.tar.gz \
        | tar -xz -C ~/.local --strip-components=1
    ```

=== "Windows"

    ```powershell
    $URL = "https://github.com/alexmaietta/ernie/releases/download/dev"
    $ARCH = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
    Invoke-WebRequest "$URL/ernie-Windows-$ARCH.zip" -OutFile ernie.zip
    Expand-Archive ernie.zip .
    Rename-Item (Get-Item "ernie-*-Windows-$ARCH") ernie
    ```

Each release, `dev` included, also lists a `SHA256SUMS` file to check the downloads against.

The Windows ZIP holds both Debug and Release builds, so either configuration links against it. Add the `ernie` folder to `CMAKE_PREFIX_PATH` so CMake finds it, and its `bin` folder to `PATH` so programs built against it find the DLLs.

Then find it from CMake and link against `Ernie::ernie`.

```cmake
find_package(Ernie REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE Ernie::ernie)
```

Ernie's headers need only C++17, so projects on C++17 or later can use the packages. `Ernie::ernie` asks CMake for C++17 at least, and with pkg-config pass `-std=c++17` or later yourself. Building Ernie itself needs C++20.

### Viewer

The archives also hold the viewer library and the `ernie` command line tool. The viewer is a separate library, so projects that only link `Ernie::ernie` don't load it or the windowing libraries it opens. To use it, ask CMake for the viewer component and link against `Ernie::viewer`.

```cmake
find_package(Ernie REQUIRED CONFIG COMPONENTS viewer)
target_link_libraries(my_app PRIVATE Ernie::viewer)
```

## Docker

The image is Ubuntu 24.04 with Ernie installed in `/opt/ernie`.

```bash
docker run --rm -it ghcr.io/alexmaietta/ernie
```

Tags follow releases, so `0.1.0`, `0.1` and `latest`, with `dev` built from main. Add `--gpus all` to render on an NVIDIA GPU, which needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

`CMAKE_PREFIX_PATH` already points at Ernie, so a project mounted into the container can build against it.

```bash
docker run --rm -it -v "$PWD:/app" -w /app ghcr.io/alexmaietta/ernie \
    sh -c "cmake -S . -B build -G Ninja && cmake --build build"
```

## Building from source

Ernie needs CMake 3.25 or later and a C++20 compiler, such as GCC 13, Clang 18 or Visual Studio 2022. Its other dependencies are downloaded while it configures.

```bash
git clone https://github.com/alexmaietta/ernie.git
cd ernie
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DERNIE_INSTALL=ON
cmake --build build
cmake --install build --prefix ~/.local
```

| Option | Default | Builds |
| --- | --- | --- |
| `ERNIE_BUILD_PYTHON` | `OFF` | The Python bindings |
| `ERNIE_BUILD_VIEWER` | `OFF` | The viewer and the `ernie` command line tool, which need X11 or Wayland development files on Linux |
| `ERNIE_BUILD_TESTS` | `OFF` | The C++ and Python tests |
| `ERNIE_BUILD_BENCHMARKS` | `OFF` | The benchmarks |
| `ERNIE_INSTALL` | `OFF` | The install rules and packages |
| `ERNIE_COVERAGE` | `OFF` | Code coverage instrumentation, with GCC or Clang |

The Python package builds from the checkout with pip.

```bash
pip install .
```

### Presets

The presets build the way CI does, each in its own folder.

| Preset | Builds |
| --- | --- |
| `ci` | Release with the tests, in `build-ci` |
| `bench` | Release with the benchmarks, in `build-bench` |
| `coverage` | Debug with coverage and the tests, in `build-coverage` |
| `package` | The TGZ archive on Linux and macOS, viewer included, in `build-package` |
| `package-windows` | The Windows ZIP with Debug and Release builds, in `build-package-windows` |

```bash
cmake --preset ci
cmake --build --preset ci
ctest --preset ci
```

The tests run on the device Ernie picks by default, which `ERNIE_ADAPTER` and `ERNIE_BACKEND` change. To run them on others, list the adapter aliases that `ernie` prints in `ERNIE_TEST_ADAPTERS`, separated by commas, or set it to `all`.

```bash
ERNIE_TEST_ADAPTERS=vulkan:0,cuda:0 ctest --preset ci
```

CI sets `ERNIE_TEST_FULL=1` to run the full suite. It tests every device and also compiles every kernel for every NVIDIA and AMD target, which the tests otherwise skip.

```bash
cmake --preset package
cmake --build --preset package
cpack --preset package
```

## Troubleshooting

If `ernie` lists no devices, none of its backends found a driver to load. Install your GPU vendor's driver, or lavapipe for the CPU as in [Requirements](#requirements).
