> 🚧 **This project is in active beta development.** This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

[![CI](https://github.com/alexmaietta/ernie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alexmaietta/ernie/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Falexmaietta.github.io%2Fernie%2Fcoverage%2Fbadge.json)](https://alexmaietta.github.io/ernie/coverage/)
[![Release](https://img.shields.io/github/v/release/alexmaietta/ernie?include_prereleases&sort=semver&color=orange)](https://github.com/alexmaietta/ernie/releases)
[![PyPI](https://img.shields.io/pypi/v/ernie-renderer?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/ernie-renderer/)
[![Docs](https://img.shields.io/badge/docs-latest-8A2BE2)](https://alexmaietta.github.io/ernie/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-2E8B57)](LICENSE)

# Efficient Renderer for Natural and Interplanetary Environments (Ernie)

![]()

Ernie is a physically accurate, real-time, spectral renderer for interplanetary environments.

It is a Vulkan ray tracer for Linux, Windows and macOS (through MoltenVK), with C++ and Python APIs and a Hydra-like scene API. Ernie is designed around physical quantities, efficient batched rendering, and interoperability.

Ernie is built to be extensible, customizable, and to work alongside other tools. You can add your own BRDFs, sensor filters and light patterns. Rendered outputs and scene attributes (transforms, mesh vertices, light parameters and more) are shared zero copy with any DLPack tensor (PyTorch, JAX, Warp, NumPy), which makes Ernie a good fit for real-time physics simulation and reinforcement learning. Support for OpenUSD and glTF scenes, and for MDL and MaterialX materials, is planned.

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
    - Planetary photometry with Hapke, Lommel-Seeliger and lunar-Lambert, with parameters from texture maps
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
- Validation with real-world sensors and scenes.

## Quick Start

We recommend starting with the Python package. Install it and open the Ernie Viewer:

```sh
pip install ernie-renderer
python -m ernie
```

Ernie ships with built-in demos covering cameras and sensors, lighting, atmospheres, terrain, lasers and batched rendering. Open the Demos panel in the viewer and explore. Each demo is written in both C++ and Python, and the two versions render the same image.

From the command line, `python -m ernie --list` names every demo, `python -m ernie --demo demo_atmosphere_earth` runs one, and `--help` lists every option.

## Installation

Ernie needs a Vulkan 1.3 driver: your GPU vendor's, or Mesa's (`mesa-vulkan-drivers`), which includes the lavapipe software driver for machines without a GPU.

### Python

Python 3.10 or later. `pip` installs `ernie-renderer` from [PyPI](https://pypi.org/project/ernie-renderer/); the package bundles everything else, the viewer included.

```sh
pip install ernie-renderer
```

### C++

Each [GitHub release](https://github.com/alexmaietta/ernie/releases/latest) has an `ernie` package for Ubuntu 22.04+, Debian 12+ and RHEL 9+ (and a tarball for other distributions) with the library, the viewer, the headers and the CMake package.

```sh
sudo apt install ./ernie_*.deb   # Debian, Ubuntu
sudo dnf install ./ernie-*.rpm   # RHEL, Rocky, Fedora
```

The viewer is also a Docker image: `docker run --rm --gpus all ghcr.io/alexmaietta/ernie --help`.

### Build from source

CMake 3.24+, a C++20 compiler and Python 3.10+ with its development headers (or `-DERNIE_BUILD_PYTHON=OFF`). Small dependencies are vendored; Slang and OpenEXR are fetched on first configure.

```sh
cmake -S . -B build -DERNIE_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build -j4                # C++, GPU and Python tests
cmake --install build --prefix ~/.local   # libernie, headers and the CMake package
```

For the Python wheel, `uv build --wheel --out-dir dist .`; for Linux packages, `cmake --preset package-deb` (or `package-rpm`), then `cmake --build --preset package-deb` and `cpack --preset package-deb`.

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
#include <ernie/render/renderer.h>
#include <ernie/scene/schema/camera.h>
#include <ernie/scene/schema/geom.h>
#include <ernie/scene/schema/lux.h>
#include <ernie/scene/schema/planet.h>
#include <ernie/scene/schema/render.h>
#include <ernie/scene/tokens.h>
#include <ernie/scene/xform.h>

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

## Acknowledgements

Ernie is built on these open-source projects. Versions and licenses are in [third_party/README.md](third_party/README.md).

- [Vulkan Headers](https://github.com/KhronosGroup/Vulkan-Headers), [volk](https://github.com/zeux/volk) and [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) for the GPU layer
- [Slang](https://github.com/shader-slang/slang) for shaders
- [DLPack](https://github.com/dmlc/dlpack) for sharing tensors with PyTorch, JAX, Warp and NumPy
- [nanobind](https://github.com/wjakob/nanobind) for the Python bindings
- [GLM](https://github.com/g-truc/glm) for math
- [OpenEXR and Imath](https://github.com/AcademySoftwareFoundation/openexr), [stb](https://github.com/nothings/stb), [Basis Universal](https://github.com/BinomialLLC/basis_universal) and [bc7enc_rdo](https://github.com/richgel999/bc7enc_rdo) for images and textures
- [NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb) for volumes
- [nlohmann/json](https://github.com/nlohmann/json) for JSON
- [OpenUSD](https://openusd.org/) for scene interchange, with Ernie's schemas published as a USD schema plugin
- [Dear ImGui](https://github.com/ocornut/imgui) and [GLFW](https://github.com/glfw/glfw) for the viewer
- [Tracy](https://github.com/wolfpld/tracy) for profiling
- [GoogleTest](https://github.com/google/googletest) for tests

Ernie also leverages ideas from these renderers and projects:

- [Madrona](https://madrona-engine.github.io/) for batch rendering across many worlds
- [Chrono::Sensor](https://api.projectchrono.org/group__sensor.html) for sensor simulation inside Chrono
- [Cycles](https://www.cycles-renderer.org/), [Mitsuba 3](https://www.mitsuba-renderer.org/) and [PBRT](https://pbrt.org/) for physically based and spectral rendering
- [Unreal Engine](https://www.unrealengine.com/)'s Render Dependency Graph and Frostbite's FrameGraph for the frame graph
- [OpenUSD](https://openusd.org/) and Omniverse Fabric for the schema-based, columnar scene
- [nvpro-samples](https://github.com/nvpro-samples) for Vulkan ray tracing and cluster LOD
- [PANGU](https://pangu.software/) for planetary and asteroid surface simulation
- [IRIS](https://dartslab.jpl.nasa.gov/References/pdf/IRIS_aiaa2022SciTech.pdf) for real-time, closed-loop rendering of space sensors
- [CosmoScout VR](https://github.com/cosmoscout/cosmoscout-vr) for planet-scale scenes and solar-system visualization
- [three-geospatial](https://github.com/takram-design-engineering/three-geospatial) for atmospheres and clouds
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [Newton](https://github.com/newton-physics/newton) for tiled cameras in robot learning
