> 🚧 **This project is in active beta development.** This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

[![CI](https://github.com/alexmaietta/ernie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alexmaietta/ernie/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/alexmaietta/ernie?include_prereleases&sort=semver&color=orange)](https://github.com/alexmaietta/ernie/releases)
[![PyPI](https://img.shields.io/pypi/v/ernie-renderer?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/ernie-renderer/)
[![Docs](https://img.shields.io/badge/docs-latest-8A2BE2)](https://alexmaietta.github.io/ernie/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-2E8B57)](LICENSE)

# Efficient Renderer for Natural and Interplanetary Environments (Ernie)

![]()

Ernie is a physically accurate, real-time, spectral renderer for interplanetary environments.

It is a Vulkan rasterizer for Linux, Windows and macOS (through MoltenVK), with C++ and Python APIs and a Hydra-like scene API. Ernie is designed around physical quantities, efficient batched rendering, and interoperability.

Ernie is built to be customized and to work alongside other tools. You can add your own BRDFs, sensor filters and light patterns. Rendered outputs and scene attributes (transforms, mesh vertices, light parameters and more) are shared zero copy with any DLPack tensor (PyTorch, JAX, Warp, NumPy), which makes Ernie a good fit for real-time physics simulation and reinforcement learning. Support for OpenUSD and glTF scenes, and for MDL and MaterialX materials, is planned.

Its key features are aimed at planetary robotics and sensing:

- **Spectral light**
    - Light carried in bands (16 across 380-780 nm by default, up to 64 into the infrared) from the source, through materials and media, to the sensor
    - Sun, sky, lamps, domes and lasers in physical units, with measured spectra such as the TSIS-1 sun
    - Distant lights as limb-darkened discs whose angular size sets every penumbra
- **Planets and space**
    - Terrain and whole planets with continuous level of detail from orbit to the ground, their heights and maps streamed from rasters larger than GPU memory
    - Adaptive spheres and ellipsoids, exact from a pebble to a star, in double precision with a floating origin and reverse-Z depth
    - Earth and Mars atmospheres from physical models, with refraction, seen from the ground or from orbit
    - Eclipses cast by real geometry, and star fields from your own catalogues
- **Materials**
    - Planetary photometry: Hapke, Lommel-Seeliger and lunar-Lambert, with parameters from texture maps
    - OpenPBR Surface and UsdPreviewSurface, spectral throughout
    - Your own BRDFs written in Slang
- **Cameras and sensors**
    - Perspective, fisheye, orthographic and line-scan pushbroom cameras, with CAHVORE and USGS CSM instrument models
    - The EMVA 1288 sensor model: quantum efficiency, shot and read noise, dark current, PRNU, DSNU, full well and ADC
    - Filter chains for optics and detectors (diffraction, lens flare, blooming, smear, TDI, Bayer, companding), with your own in Slang
    - Polarization cameras
- **Thermal infrared**
    - Planck emission from surfaces and volumes, with emissivity from Kirchhoff's law or measured spectra
    - Temperatures from your own solver, textures, or models run per pixel: spacecraft energy balance and diurnal tables for airless bodies
    - Microbolometer cameras with NETD, fixed pattern noise and false-colour palettes
- **Volumes**
    - Smoke, dust, exhaust plumes and clouds from VDB files or GPU arrays, ray marched spectrally and casting shadows
    - Procedural cloud layers, with Mie droplets for silver linings, fogbows and glories
- **Lasers and structured light**
    - Gaussian beams and diffractive patterns (lines, grids, dot fields or your own), pulses, speckle, and beams seen in dust and air
- **Simulation and learning**
    - Thousands of worlds and cameras rendered in one batched step for RL and Monte Carlo workloads
    - Inputs and outputs shared without copies with PyTorch, JAX, Warp and NumPy, through CUDA and HIP
    - Columns and incremental sync: thousands of prims' poses written in one call, and an edit redraws only what it reaches
- **Tools and integrations**
    - A viewer with your own panels and debug drawing, and C++ and Python APIs on Linux, Windows and macOS
    - OpenUSD import, renderer backends for Newton Viewer, IsaacLab and Chrono::Sensor, and TerrainGen built on the viewer


For a more detailed overview of the features, see our [documentation](https://alexmaietta.github.io/ernie/).

Future work:

- A fully USD- and glTF-compliant renderer, loading OpenUSD, glTF, OBJ, STL and 3D Tiles, with the glTF material system, UsdPreviewSurface shaders and MDL and MaterialX support
- Differentiable rendering
- Reflections and global illumination
- OpenXR integrations
- WebGPU backend
- Gaussian splat and particle fields
- Lidar and radar sensors
- A real-time ray-tracing renderer alongside the rasterizer
- Integrations into Newton Viewer, Chrono Sensor, and IsaacLab
- Benchmark against RTX, Cycles, Newton Warp Raytrace, Madrona both in speed, throughput, and image quality

## Installation

The Python package is on PyPI as `ernie-renderer` (Python 3.12 or later, with a Vulkan 1.3 driver). It bundles everything else, the viewer and the native `ernie` program included:

```sh
pip install ernie-renderer
```

Each GitHub release also has an `ernie` package for Ubuntu 22.04+, Debian 12+ and RHEL 9+ (and a tarball for other distributions), with the library, the viewer, and the headers and CMake package for building against it in C++. Download it from the [latest release](https://github.com/alexmaietta/ernie/releases/latest) and install it with a Vulkan driver: Mesa's below, which includes the lavapipe software driver for machines without a GPU, or your GPU vendor's.

```sh
# Debian, Ubuntu
sudo apt install mesa-vulkan-drivers
sudo apt install ./ernie_*.deb

# RHEL, Rocky, Fedora
sudo dnf install mesa-vulkan-drivers
sudo dnf install ./ernie-*.rpm
```

### Docker

The `ernie` program is also published as a Docker image.

```sh
docker run --rm --gpus all -v "$PWD:/out" ghcr.io/alexmaietta/ernie \
  --demo demo_atmosphere_earth --viewer null --output /out/earth.png
```

### Build from source

Ernie needs CMake 3.24+, a C++20 compiler and a Vulkan 1.3 driver; it fetches its other dependencies (see [third_party/README.md](third_party/README.md)).

```sh
cmake -S . -B build -DERNIE_BUILD_TESTS=ON -DERNIE_INSTALL=ON -DERNIE_BUILD_PYTHON=ON -DERNIE_BUILD_APP=ON
cmake --build build
ctest --test-dir build -j4              # C++, GPU and Python tests
build/bin/ernie                         # the viewer
cmake --install build --prefix ~/.local # libernie, headers and the CMake package
```

To build the Python wheel instead, run `uv build --wheel --out-dir dist .` and `uv pip install dist/ernie_renderer-*.whl`; for the Linux packages, `cmake --preset package-deb` (or `package-rpm`), `cmake --build --preset package-deb` and `cpack --preset package-deb`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the build options, testing without a GPU on lavapipe, and benchmarks.

## Quick Start

Run `ernie` to open the Ernie Viewer. Its Demos panel runs any of the built-in demos, each written in both C++ and Python, covering cameras and sensors, lighting, atmospheres, terrain, lasers, batched rendering and the viewer; the two versions of a demo render the same image.

```sh
ernie                               # the viewer, with the Demos panel
ernie --list                        # every demo's name
ernie --demo demo_atmosphere_earth  # run one demo
ernie --help                        # every option
```

`python -m ernie` opens the same viewer and takes the same options, and `python -m ernie.demos` runs the Python version of a demo:

```sh
python -m ernie.demos --list
python -m ernie.demos demo_atmosphere_earth
```

Every demo takes common options, such as `--viewer null` to render without a window and `--output frame.png` to save the last frame; `--help` lists them.

## Your First Render

### Python

```python
from PIL import Image

import ernie
from ernie import tokens

scene = ernie.Scene()
scene.define("/World/Ground", ernie.GroundPlane)
scene.define("/World/Rock", ernie.Sphere, translate=(0, 0, 1))  # 1 m up
scene.define("/World/Sun", ernie.DistantLight).set(
    tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
)
scene.define("/World/Camera", ernie.Camera).set(
    tokens.TRANSFORM, ernie.look_at_transform((0, -8, 3), (0, 0, 1))
)
scene.define(
    "/Render/Settings", ernie.RenderSettings, camera="/World/Camera", resolution=(640, 480)
)

renderer = ernie.Renderer()
image = renderer.render(scene)  # a (480, 640, 4) uint8 NumPy array
Image.fromarray(image).save("ernie.png")
```

### C++

With the `ernie` package or your own build installed, find it with CMake:

```cmake
find_package(Ernie REQUIRED)
target_link_libraries(my_app PRIVATE Ernie::ernie)
```

```cpp
#include <ernie/scene/camera_schemas.h>
#include <ernie/scene/geometry_schemas.h>
#include <ernie/scene/light_schemas.h>
#include <ernie/scene/render_schemas.h>
#include <ernie/scene/schema.h>
#include <ernie/scene/tokens.h>
#include <ernie/scene/xform.h>
#include <ernie/session/renderer.h>

int main() {
    using namespace ernie;
    Scene scene;
    GroundPlane::Define(scene, "/World/Ground");
    Sphere::Define(scene, "/World/Rock");
    SetXform(scene, "/World/Rock", {.translate = glm::dvec3(0.0, 0.0, 1.0)});  // 1 m up
    DistantLight::Define(scene, "/World/Sun");
    scene.Set("/World/Sun", tokens::kTransform, DistantLightTransform({0.4, -0.6, 0.7}));
    Camera::Define(scene, "/World/Camera");
    scene.Set("/World/Camera", tokens::kTransform,
              LookAtTransform({0.0, -8.0, 3.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}));
    RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution({640, 480});

    Renderer renderer;
    Tensor image = renderer.Render(scene);  // 480 x 640 x 4 bytes of LdrColor on the CPU
}
```

## Citation

If you use Ernie in your research, please cite it. GitHub's "Cite this repository" button gives the same citation from [CITATION.cff](CITATION.cff).

```bibtex
@software{maietta2026ernie,
  author  = {Maietta, Alex},
  title   = {Ernie: Efficient Renderer for Natural and Interplanetary Environments},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/alexmaietta/ernie},
  license = {BSD-3-Clause}
}
```

## License

Ernie is licensed under the [BSD 3-Clause License](LICENSE). The third-party libraries it bundles keep their own licenses, listed in [third_party/README.md](third_party/README.md).
