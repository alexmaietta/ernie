> 🚧 **This project is in active beta development.** This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

[![CI](https://github.com/alexmaietta/ernie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alexmaietta/ernie/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/alexmaietta/ernie?include_prereleases&sort=semver&color=orange)](https://github.com/alexmaietta/ernie/releases)
[![PyPI](https://img.shields.io/pypi/v/ernie-renderer?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/ernie-renderer/)
[![Docs](https://img.shields.io/badge/docs-latest-8A2BE2)](https://alexmaietta.github.io/ernie/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-2E8B57)](LICENSE)

# Efficient Renderer for Natural and Interplanetary Environments (Ernie)

![]()

Ernie is a physically accurate, real-time, spectral renderer for interplanetary environments.

It is a Vulkan rasterizer for Linux, Windows and macOS (through MoltenVK), with C++ and Python APIs and a Hydra-like scene API. Ernie is designed around physical quantities, efficient batched rendering, and user experience.

Ernie is built to be customized and to work alongside other tools. You can add your own BRDFs, sensor filters and light patterns. Rendered outputs and scene attributes (transforms, mesh vertices, light parameters and more) are shared zero copy with any DLPack tensor (PyTorch, JAX, Warp, NumPy), which makes Ernie a good fit for real-time physics simulation and reinforcement learning. Support for OpenUSD and glTF scenes, and for MDL and MaterialX materials, is planned.

Its key features are aimed at planetary robotics and sensing:

| | |
| --- | --- |
| **Cameras** | Exposure, aperture, motion blur, rolling shutter, lens distortion and vignetting<br>Perspective, fisheye, orthographic and line-scan pushbroom cameras like LROC's<br>USGS CSM camera models for non-standard instruments |
| **Sensors** | Quantum efficiency curves, full well, read noise, dark current, PRNU, DSNU, gain and ADC bit depth<br>Sensor filter chains (noise models, lens flare, vignetting), with your own filters written in Slang |
| **Planets** | Terrain as a built-in prim, with continuous level of detail, updated in place where its heights change<br>Whole planets and moons drawn from orbit down to the ground<br>Earth and Mars atmospheres from physical models, from the ground or from orbit, with aerial perspective<br>Star fields from your own catalogs, with brightness from magnitudes and color from B-V indices |
| **Light** | Spectral rendering: light carried in bands (16 across 380-780 nm by default) from the sources, through the materials, to the sensor<br>Distant lights with an angular size, for soft penumbrae from the sun's width in the sky<br>Illuminants (blackbody, CIE daylight, the measured solar spectrum) and custom spectral curves, such as Earthshine on the Moon |
| **Materials** | Hapke, Lommel-Seeliger, lunar-Lambert, Oren-Nayar and Lambert BRDFs built in, and your own for the surfaces you study<br>OpenPBR Surface materials, spectral throughout: coat, fuzz, thin film, anisotropy, transmission and subsurface scattering ([below](#physically-based-materials)) |
| **Volumes** | Smoke, dust, exhaust plumes and clouds from VDB files, GPU arrays or procedural cloud layers, lit by the sun, sky and ground and casting shadows ([below](#volumes)) |
| **Simulation and learning** | Batch rendering: many worlds, each with its own camera, rendered in one step for RL and Monte Carlo workloads<br>Meshes, instances and points written from CUDA or Warp arrays without copies<br>Columns: one attribute of thousands of prims, such as a simulation's poses and colours, written whole from host, NumPy, PyTorch or CUDA arrays, or mapped in place in the renderer's own buffers, one change per step however many prims (`demo_batched_columns`)<br>Incremental sync: an edit redraws only the prims it reaches, with inheritance and relationships declared in the schemas, so scattered edits among thousands of prims stay cheap (`demo_batched_edits`)<br>A floating origin, so large scenes keep their precision |
| **Tools** | A viewer with customizable panels, to drive your own simulations and workflows<br>Debug drawing for labels, trajectories and overlays in the scene |
| **Integrations** | TerrainGen builds on Ernie's viewer, adding its own panels to render synthetic terrain<br>Newton Viewer and IsaacLab renderer backends, and a viewer and renderer for Project Chrono's Chrono Sensor |

### Volumes

Volumes are UsdVol-style prims: a `Volume` names its fields and a material gives them density, albedo, emission and a phase function. They are ray marched per pixel, spectrally, and composited with the surfaces, the sky and the atmosphere.

| Effect | What it does | Demo |
| --- | --- | --- |
| **Grids** | NanoVDB and OpenVDB files, NanoVDB arrays, and dense fields that CUDA or Warp kernels write on the GPU each frame, read without a copy to the host | `volumes_plume`, `volumes_dust_devil`, `volumes_lunar_dust` |
| **Procedural cloud layers** | Up to four layers of cumulus and cirrus carved at each sample from a weather map and Perlin-Worley noise, drifting with the wind | `volumes_cloud_layers` |
| **Phase functions** | Henyey-Greenstein, double and dual Henyey-Greenstein, Rayleigh, isotropic, and cloud droplets by Mie theory per band: silver linings, fogbows and glories | `volumes_cloud_droplets` |
| **Multiple scattering** | Wrenninge's octaves, or Bouthors' collectors for the light clouds scatter many times, with a powder effect at their edges | `volumes_clouds`, `volumes_cloud_droplets` |
| **Emission** | Black-body glow from temperature fields, emission fields and colours, and the Temperature var for thermal sensors | `volumes_plume`, `volumes_rcs` |
| **Lighting** | Sun, sky and ground light through a lighting cache; shadows from surfaces onto volumes and from volumes onto surfaces and each other | `volumes_lunar_dust`, `volumes_clouds` |
| **Clouds and the atmosphere** | Cloud layers lit through their own shadow maps from the sun, light shafts from the sun through the air past clouds and terrain, and aerial perspective in front of the clouds | `volumes_cloud_layers` |
| **Performance** | Empty space skipped by bricks, temporal amortisation and half-resolution marching, and volumes outside every view left unmarched | |

### Physically based materials

OpenPBR Surface (`open_pbr`) is built in alongside UsdPreviewSurface. Its colours are uplifted to spectra on the GPU, so the thin film's interference and the subsurface radius vary by wavelength. Layers that no material in the scene uses are compiled out of the shaders, and the G-buffer's tangents, the transparent pass and the subsurface pass run only while a material needs them.

| Feature | OpenPBR inputs | Demo |
| --- | --- | --- |
| **Rough diffuse** | `base_diffuse_roughness`: energy-preserving Oren-Nayar (EON), flat like the full Moon when rough | `pbr_diffuse_roughness` |
| **Metals** | `base_metalness`, `specular_color` as the F82 edge tint | `pbr_metals` |
| **Clear coat** | `coat_weight`, `coat_roughness`, `coat_ior`, `coat_color`, `coat_darkening` | `pbr_coat` |
| **Fuzz** | `fuzz_weight`, `fuzz_color`, `fuzz_roughness`: Zeltner's sheen, for velvet and dust | `pbr_fuzz` |
| **Thin film** | `thin_film_weight`, `thin_film_thickness`, `thin_film_ior`: iridescence computed per band | `pbr_thin_film` |
| **Anisotropy** | `specular_roughness_anisotropy`, along the texture's u, with smooth mesh tangents, in direct and sky light | `pbr_anisotropy` |
| **Transmission** | `transmission_weight`, `transmission_color`, `geometry_thin_walled`: glass with screen-space refraction, and alpha blending for partly opaque materials | `pbr_transmission` |
| **Subsurface scattering** | `subsurface_weight`, `subsurface_color`, `subsurface_radius` and its scale per channel, spread across the image by Burley's profile per band | `pbr_subsurface` |

Run a demo with `python -m ernie --demo demo_<name>` (`--list` names them all), or open it from the viewer's Demos panel.

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

The `ernie` program is also published as a Docker image, `ghcr.io/alexmaietta/ernie`, with Mesa's Vulkan drivers. With the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) and `--gpus all` it renders on your GPU; without them, drop `--gpus all` and it renders on lavapipe. To render a demo into the current directory:

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

### Next steps

- [Describing scenes](docs/guides/scenes.md): the prim types and their attributes, transforms, relationships, arrays, validation, loading scenes from JSON and declaring your own types.
- [The demos](demos/), as `demos/<category>/demo_<category>_<name>.{cpp,py}`: a good starting point for your own; subclass `ernie.demo.Demo` in Python or `ernie::app::Demo` in C++.
- The [documentation](https://alexmaietta.github.io/ernie/) for every feature in depth.

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
