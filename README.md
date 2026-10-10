# Efficient Renderer for Natural and Interplanetary Environments (Ernie)

Ernie is a physically accurate, real-time, spectral renderer for interplanetary environments, with C++ and Python APIs. It runs on any Vulkan 1.3 GPU.

Scenes are described in physical quantities rather than arbitrary parameters. A light has a blackbody temperature and a power in lumens, not an intensity. A camera has exposure, aperture and focal length, but also quantum efficiency curves, full well capacity, read noise and dark current, with sensor filter chains that model a real sensor's noise and effects. This seems like more work at first, but it makes scenes much simpler to design.

Ernie has a Hydra-like scene and render delegate API. It renders batches of cameras at once and exchanges data with any DLPack tensor (NumPy, PyTorch, JAX, Warp), so it fits into physics and reinforcement learning workflows. Today it is a Vulkan rasterizer, tested on Linux; macOS should work through MoltenVK but is untested.

Key features with Ernie, mostly for planetary robotics:

- **Batch rendering**: many environments ("worlds"), each with its own camera, rendered in one step for RL and Monte Carlo workloads.
- **Camera effects and modeling**: exposure, aperture, motion blur, rolling shutter, lens distortion and vignetting, with perspective, fisheye and orthographic projections.
- **Real camera specs**: from quantum efficiency curves to full well, read noise, dark current, PRNU, DSNU, gain and ADC bit depth, and line-scan pushbroom cameras like LROC's, enabling users to model the noise and effects of those instruments.
- **Terrain**: a first-class citizen, with a built-in Terrain prim type with continuous level of detail through quadtrees, updated in place where its heights change. With a radius, a Terrain is a whole planet or moon, drawn from orbit down to the ground with cube-sphere level of detail.
- **GPU interop**: deforming meshes, instances and points from CUDA or Warp arrays without copies, useful for rover simulations and particle simulations.
- **Spectral-first rendering**: light is carried in spectral bands (16 across 380-780 nm by default) from the light sources, through the materials, to the sensor.
- **Star fields**: a built-in StarField prim allowing users to plug in their own star catalogs, with brightness from magnitudes and color from B-V indices, so you can see how camera exposure affects the visible stars.
- **Lighting**: distant lights have an angle parameter, giving smooth penumbrae based on the light's width in the sky. Lights also take illuminants (blackbody, CIE daylight) and custom spectral curves, for example to model Earthshine on the lunar surface.
- **Customizable and extendable**: write your own sensor filters in Slang and build up your own chains (noise models, lens flare, vignetting), designed for efficiency and throughput, and plug in custom BRDFs for the planetary reflectances you encounter. Hapke, Lommel-Seeliger, lunar-Lambert, Oren-Nayar and Lambert come built in.
- **Debug drawing**: an easy-to-use API to add custom labels, trajectories and overlays to the scene.
- **Floating origin**: prevents precision issues in large-scale scenes.
- **Viewer**: a viewer class to easily spin up a GUI with customizable panels to integrate with your own simulations or workflows.

For a more detailed overview of the features, see our documentation, which covers:

- Scene description
- Spectral
- Lighting
- Cameras
- Sensor filter graph
- Materials and BRDFs
- Shadows
- Terrains and LOD
- Star fields
- Batched rendering
- GPU interop
- Debug draw
- Misc rendering techniques (culling, etc)
- USGS CSM
- Profling
- The life of a pixel

Future work:

- A fully USD- and glTF-compliant renderer, loading OpenUSD, glTF, OBJ, STL and 3D Tiles, with the glTF material system, UsdPreviewSurface shaders and MDL.
- Atmosphere rendering, both on and above a planet, for LEO simulations or Mars rovers.
- Dust and other volumes from particle simulations through NanoVDB.
- Global illumination for reflections
- USGS CSM camera models for advanced, non-standard cameras and their noise.
- OpenXR, so users can easily connect headsets to an Ernie viewer.
- A ray-tracing backend alongside the rasterizer.
- Windows support, and tested macOS support.
- Continued test assets and benchmark scenes.
- More sensor support for Lidars and Radars.
- Integrations into Newton Viewer, Chrono Sensor, IsaacLab.
- Compare benchmarks with Newton Warp Raytrace, Madrona both in speed and image quality.

## Getting Started

### Python

```sh
pip install ernie
```

```python
import numpy as np
from PIL import Image

import ernie
from ernie import tokens

scene = ernie.Scene()
scene.define("/World/Ground", tokens.GROUND_PLANE)
rock = np.eye(4)
rock[2, 3] = 1.0  # 1 m up
scene.define("/World/Rock", tokens.SPHERE).set(tokens.TRANSFORM, rock)
scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
    tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
)
scene.define("/World/Camera", tokens.CAMERA).set(
    tokens.TRANSFORM, ernie.look_at_transform((0, -8, 3), (0, 0, 1))
)
scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(
    tokens.CAMERA_REL, ernie.Path("/World/Camera")
).set(tokens.RESOLUTION, (640, 480))

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
#include <ernie/scene/schema.h>
#include <ernie/scene/tokens.h>
#include <ernie/session/renderer.h>

#include <glm/gtc/matrix_transform.hpp>

int main() {
    using namespace ernie;
    Scene scene;
    scene.Define("/World/Ground", tokens::kGroundPlane);
    scene.Define("/World/Rock", tokens::kSphere)
        .Set(tokens::kTransform, glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, 1.0)));
    scene.Define("/World/Sun", tokens::kDistantLight)
        .Set(tokens::kTransform, DistantLightTransform({0.4, -0.6, 0.7}));
    scene.Define("/World/Camera", tokens::kCamera)
        .Set(tokens::kTransform, LookAtTransform({0.0, -8.0, 3.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}));
    scene.Define("/Render/Settings", tokens::kRenderSettings)
        .Set(tokens::kCameraRel, Path("/World/Camera"))
        .Set(tokens::kResolution, glm::ivec2(640, 480));

    Renderer renderer;
    Tensor image = renderer.Render(scene);  // 480 x 640 x 4 bytes of LdrColor on the CPU
}
```

### Examples

Planned future examples.

**Cameras**

- LRO NAC and WAC
- Mars2020 Navcam, Mastcam
- LWIR, NIR, UV
- Pinhole, fisheye, orthographic
- Narrowband camera
- Distortion

**Camera Filters and Noise**

- Vignetting
- Lens flare

**Terrain**

- SLDEM global and multiple tiles
- Mars terrains
- Asteroid shape files
- Saturn and its rings
- Pluto

**Lighting**

- Distant light
- Custom radiance curves to model Earthshine
- 

**Star Field**

- Hipparcos star catalog
- Stars viewed from another galaxy

**Atmosphere**

- Earth atmosphere (from orbit and ground)
- Mars atmosphere

**Batched and RL**

- Batched camera rendering and setting poses

**Viewer**

- Adding a custom panel
- Hiding default panels and menus
- Extending HUD options
- Physics debug draw overlays

## Installation

### Precompiled binaries

The Python wheel is on PyPI (`pip install ernie`, Python 3.12+). It bundles everything, the viewer included, but a Vulkan driver.

Each GitHub release also has an `ernie` package for Ubuntu 22.04+, Debian 12+ and RHEL 9+ (and a tarball for other distributions), with the library, the viewer, and the headers and CMake package for building against it:

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
uv pip install dist/ernie-*.whl
```

Packages are built with `cmake --preset package-deb` (or `package-rpm`) and `cpack --preset package-deb`; `tools/packaging/` builds them in the same container as our releases.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the build options, testing without a GPU on lavapipe, and benchmarks.

## License

Ernie is licensed under the [BSD 3-Clause License](LICENSE). The third-party libraries it bundles keep their own licenses, listed in [third_party/README.md](third_party/README.md).
