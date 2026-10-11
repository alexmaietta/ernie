> 🚧 **This project is in active beta development.** This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

[![CI](https://github.com/alexmaietta/ernie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alexmaietta/ernie/actions/workflows/ci.yml) [![Release](https://img.shields.io/github/v/release/alexmaietta/ernie?include_prereleases&sort=semver&color=orange)](https://github.com/alexmaietta/ernie/releases) [![PyPI](https://img.shields.io/pypi/v/ernie-renderer?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/ernie-renderer/) [![Docs](https://img.shields.io/badge/docs-latest-8A2BE2)](https://alexmaietta.github.io/ernie/) [![License](https://img.shields.io/badge/license-BSD--3--Clause-2E8B57)](LICENSE)

# Efficient Renderer for Natural and Interplanetary Environments (Ernie)

![]()

Ernie is a physically accurate, real-time, spectral renderer for interplanetary environments.

It is a Vulkan rasterizer for Linux, Windows and macOS (through MoltenVK), with C++ and Python APIs and a Hydra-like scene API. Ernie is designed around physical quantities, efficient batched rendering and modularity.

Ernie is built to be customized and to work alongside other tools. You can add your own BRDFs, sensor filters and light patterns. Rendered outputs and scene attributes (transforms, mesh vertices, light parameters and more) are shared zero copy with any DLPack tensor (PyTorch, JAX, Warp, NumPy), which makes Ernie a good fit for real-time physics simulation and reinforcement learning. Support for OpenUSD and glTF scenes, and for MDL and MaterialX materials, is planned.

Its key features are aimed at planetary robotics and sensing:

- **Cameras and sensors**
  - Exposure, aperture, motion blur, rolling shutter, lens distortion and vignetting.
  - Perspective, fisheye and orthographic projections, and line-scan pushbroom cameras like LROC's.
  - Real sensor specs: quantum efficiency curves, full well, read noise, dark current, PRNU, DSNU, gain and ADC bit depth.
  - USGS CSM camera models for non-standard instruments.
  - Sensor filter chains (noise models, lens flare, vignetting), with your own filters written in Slang.
- **Planetary environments**
  - Terrain as a built-in prim, with continuous level of detail through quadtrees, updated in place where its heights change.
  - Whole planets and moons from a Terrain with a radius, drawn from orbit down to the ground.
  - Earth and Mars atmospheres from physical models, seen from the ground or from orbit, with aerial perspective.
  - Star fields from your own catalogs, with brightness from magnitudes and color from B-V indices.
- **Light and materials**
  - Spectral rendering: light is carried in bands (16 across 380-780 nm by default) from the sources, through the materials, to the sensor.
  - Distant lights with an angular size, for soft penumbrae from the sun's width in the sky.
  - Illuminants (blackbody, CIE daylight, the measured solar spectrum) and custom spectral curves, for example Earthshine on the lunar surface.
  - Hapke, Lommel-Seeliger, lunar-Lambert, Oren-Nayar and Lambert BRDFs built in, and your own for the surfaces you study.
- **Simulation and learning**
  - Batch rendering: many worlds, each with its own camera, rendered in one step for RL and Monte Carlo workloads.
  - Meshes, instances and points written from CUDA or Warp arrays without copies, for rover and particle simulations.
  - A floating origin, so large scenes keep their precision.
- **Tools**
  - A viewer with customizable panels, to drive your own simulations and workflows.
  - Debug drawing: labels, trajectories and overlays in the scene.

For a more detailed overview of the features, see our [documentation](https://alexmaietta.github.io/ernie/).

Future work:

- A fully USD- and glTF-compliant renderer, loading OpenUSD, glTF, OBJ, STL and 3D Tiles, with the glTF material system, UsdPreviewSurface shaders and MDL and MaterialX support
- Dust and other volumes from particle simulations through NanoVDB
- Differentiable rendering
- Global illumination
- OpenXR integrations
- WebGPU backend
- Gaussian splat and particle fields.
- Lidar and radar sensors
- A real-time ray-tracing renderer alongside the rasterizer
- Integrations into Newton Viewer, Chrono Sensor, and IsaacLab
- Compare benchmarks with RTX, Cycles, Newton Warp Raytrace, Madrona both in speed, throughput, and image quality

## Getting Started

### Python

```sh
pip install ernie-renderer
```

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

With Ernie installed (see Installation), find it with CMake:

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

For the prim types and their attributes, transforms, relationships, arrays, validation, loading scenes from JSON and declaring your own types, see [Describing scenes](docs/guides/scenes.md).

### Examples

Planned future examples.

**Demos**

- Cornell box

**Cameras**

- Pinhole, fisheye, orthographic
- Distortion
- LRO NAC and WAC
- Shadowcam
- Mars2020 Navcam, Mastcam
- LWIR, NIR, UV
- Narrowband camera

**Camera Filters and Noise**

- Vignetting
- Lens flare
- Bloom
- Narrowband filter

**Terrain**

- SLDEM global and multiple tiles
- Mars terrains
- Asteroid shape files
- Saturn and its rings
- Pluto

**Lighting**

- Distant light
- Custom radiance curves to model Earthshine
- Dome light shadows
- Illuminants across the same scene
- Laser lines competing with sunlight

**Star Field**

- Hipparcos star catalog
- Stars viewed from another galaxy

**Atmosphere**

- Earth atmosphere (from orbit and ground)
- Mars atmosphere (compare against existing images)

**Volumes**

- NanoVDB interop
- Clouds in Earth orbit
- Dust clouds on lunar surface

**Batched and RL**

- Batched camera rendering and setting poses

**Viewer**

- Adding a custom panel
- Hiding default panels and menus
- Extending HUD options
- Physics debug draw overlays

## Installation

### Precompiled binaries

The Python wheel is on PyPI as `ernie-renderer` (`pip install ernie-renderer`, then `import ernie`; Python 3.12+). It bundles everything, the viewer included, but a Vulkan driver.

Each GitHub release also has an `ernie` package for Ubuntu 22.04+, Debian 12+ and RHEL 9+ (and a tarball for other distributions), with the library, the viewer, and the headers and CMake package for building against it.

```sh
sudo apt install ./ernie_*.deb    # Debian, Ubuntu
sudo dnf install ./ernie-*.rpm    # RHEL, Rocky, Fedora
```

Both need a Vulkan 1.3 driver: your GPU vendor's, or Mesa's (`mesa-vulkan-drivers`, which includes the lavapipe software driver for machines without a GPU).

### Build from source

Ernie needs CMake 3.24+, a C++20 compiler and a Vulkan 1.3 driver; it fetches its other dependencies (see [third_party/README.md](third_party/README.md)).

```sh
cmake -S . -B build -DERNIE_BUILD_TESTS=ON -DERNIE_INSTALL=ON -DERNIE_BUILD_PYTHON=ON -DERNIE_BUILD_APP=ON
cmake --build build
ctest --test-dir build -j4              # C++, GPU and Python tests
build/bin/ernie                         # the viewer
cmake --install build --prefix ~/.local # libernie, headers and the CMake package
```

A Python wheel bundles the module with the Slang runtime it loads:

```sh
uv build --wheel --out-dir dist .
uv pip install dist/ernie_renderer-*.whl
```

Packages are built with `cmake --preset package-deb` (or `package-rpm`), `cmake --build --preset package-deb` and `cpack --preset package-deb`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the build options, testing without a GPU on lavapipe, and benchmarks.

## License

Ernie is licensed under the [BSD 3-Clause License](LICENSE). The third-party libraries it bundles keep their own licenses, listed in [third_party/README.md](third_party/README.md).
