> 🚧 **This project is in active beta development.** This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

[![CI](https://github.com/alexmaietta/ernie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alexmaietta/ernie/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Falexmaietta.github.io%2Fernie%2Fcoverage%2Freport%2Fbadge.json)](https://alexmaietta.github.io/ernie/coverage/)
[![Kernel coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Falexmaietta.github.io%2Fernie%2Fcoverage%2Freport%2Fkernel-badge.json)](https://alexmaietta.github.io/ernie/coverage/)
[![Release](https://img.shields.io/github/v/release/alexmaietta/ernie?include_prereleases&sort=semver&color=orange)](https://github.com/alexmaietta/ernie/releases)
[![PyPI](https://img.shields.io/pypi/v/ernie-renderer?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/ernie-renderer/)
[![Docs](https://img.shields.io/badge/docs-latest-8A2BE2)](https://alexmaietta.github.io/ernie/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-2E8B57)](https://github.com/alexmaietta/ernie/blob/main/LICENSE)

# Efficient Renderer for Natural and Interplanetary Environments (Ernie)

Ernie is a physically-based, real-time, spectral renderer for planetary robotics and sensing.

It is cross-platform and designed with portability in mind, supporting Linux, Windows, and macOS with C++ and Python APIs, and both hardware and software ray tracing support across NVIDIA and AMD GPUs and Apple Metal.

A brief overview of its features:

<!-- - **Extensible:** custom materials, BRDFs, sensor filters, camera models, and lights -->
- **Configiurable fidelity:** built-in rendering modes ranging from low-fidelity RL high-throughput cases to predictive spectral path tracing
- **Zero-copy tensors:** render outputs and scene data (transforms, vertices, lights) are shared via DLPack with PyTorch, JAX, Warp and Numpy
- **Batched rendering:** many cameras and worlds rendered in one step for RL
<!-- - **Standard formats:** import OpenUSD and glTF scenes or OBJ, STL meshes
- **Spectral light:** 16 bands across 380–780 nm by default, up to 64 into the infrared, carried from source to sensor
- **Physical lighting:** sun, sky, lamps, domes and lasers in physical units, with measured spectra such as the TSIS-1 sun
- **Materials:** UsdPreviewSurface, Lambert and GGX, OpenPBR, partial MaterialX and MDL support
- **Planetary photometry:** Hapke, Lommel-Seeliger and lunar-Lambert, with parameters from texture maps
- **Cameras:** perspective, fisheye, orthographic and pushbroom, with CAHVORE and USGS CSM models
- **Sensor noise:** the EMVA 1288 model (QE, shot and read noise, dark current, PRNU, DSNU, full well, ADC)
- **Camera distortion:**
- **Optics and detector effects:** diffraction, lens flare, blooming, smear, TDI, Bayer and companding, with your own filters in Slang
- **Polarization:** polarization-sensitive cameras
- **Thermal infrared:** Planck emission, emissivity from Kirchhoff's law or measured spectra, and temperatures from your solver or per-pixel models
- **Microbolometers:** NETD, fixed-pattern noise and false-colour palettes
- **Lidar:**
- **Radar:**
- **Volumes:** dust, smoke, plumes and clouds from VDB files or GPU arrays, with spectral scattering and shadows
- **Lasers and structured light:** Gaussian beams, diffractive patterns, pulses, speckle, and beams visible in dust
- **Planet-scale terrain:** continuous level of detail from orbit to the ground, streamed from rasters larger than GPU memory
- **Eclipses and stars:** shadows cast by real geometry, and star fields from your own catalogues
- **Atmospheres:** physically modelled Earth and Mars skies with refraction, seen from the ground or from orbit
- **Precision at any scale:** floating rendering origin
- **XR/VR:**
- **Gaussian Splats:** -->

For a more detailed overview of the features, see our [documentation](https://alexmaietta.github.io/ernie/).

## Quick Start

We recommend installing Ernie from PyPi. We also ship a GUI application that provides interactive viewing and editing of scenes. The following commands install Ernie, check it was installed sucessfully, and opens the GUI.

```sh
pip install ernie-renderer
python -c "import ernie; print(ernie.__version__)"
ernie
```

In Python, basic API usage involves creating a `Scene` with some objects. Then creating a `Renderer` with that `Scene` and calling `Renderer.step` with the desired render products. See below for an example.

```python
import ernie

scene = ernie.Scene()

ernie.Plane.define(
    scene,
    "/Ground",
    width=20,
    length=20,
    checkerboard=True,
)
ernie.Cube.define(
    scene,
    "/Box",
    translate=(-1.5, 0, 0.75),
    size=1.5,
    display_color=(0.8, 0.2, 0.2),
)
ernie.Cylinder.define(
    scene,
    "/Post",
    translate=(1.5, 1, 1),
    radius=0.4,
    height=2,
    display_color=(0.2, 0.4, 0.8),
)
ball = ernie.Sphere.define(
    scene,
    "/Ball",
    radius=0.5,
    display_color=(0.9, 0.8, 0.2),
    semantics_labels={"class": ["ball"]},
)

ernie.DistantLight.define(
    scene,
    "/Sun",
    orient=ernie.from_azimuth_elevation(135, 45),
    intensity=50000,
)

ernie.Camera.define(
    scene,
    "/Front",
    translate=(0, -9, 5),
    orient=ernie.look_at((0, -9, 5), (0, 0, 0.5)),
    focal_length=35,
)
ernie.Camera.define(
    scene,
    "/Top",
    translate=(0, 0, 14),
    orient=ernie.look_at((0, 0, 14), (0, 0, 0), up=(0, 1, 0)),
    focal_length=24,
)

ernie.GradientBackground.define(scene, "/Sky")
ernie.Scope.define(scene, "/Render")
ernie.RenderVar.define(scene, "/Render/Color", source_name="LdrColor")
ernie.RenderVar.define(scene, "/Render/Depth", source_name="DistanceToCamera")
ernie.RenderVar.define(scene, "/Render/Normal", source_name="Normal")
ernie.RenderVar.define(scene, "/Render/Semantic", source_name="SemanticSegmentation")
ernie.RenderProduct.define(
    scene,
    "/Render/Views",
    camera=["/Front", "/Top"],
    resolution=(640, 480),
    ordered_vars=["/Render/Color", "/Render/Depth", "/Render/Normal", "/Render/Semantic"],
    render_mode="RealTime",
    background="/Sky",
)

renderer = ernie.Renderer(scene)
for step in range(4):
    ball.translate = (step - 1.5, -1.5, 0.5)
    output = renderer.step(["/Render/Views"], dt=1 / 30)
    output.wait()
    [frame] = output["/Render/Views"].frames
    for var in frame.render_vars:
        aov = frame.render_var(var)
        for camera, name in enumerate(["front", "top"]):
            ernie.write_render_var(f"{name}_{aov.source_name}_{step}", aov, camera)
```

<table>
  <tr>
    <th></th>
    <th>Color</th>
    <th>Depth</th>
    <th>Normal</th>
    <th>Semantic</th>
  </tr>
  <tr>
    <th>Front</th>
    <td><img src="https://raw.githubusercontent.com/alexmaietta/ernie/main/docs/_src/images/readme_front_LdrColor.png" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/alexmaietta/ernie/main/docs/_src/images/readme_front_DistanceToCamera.png" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/alexmaietta/ernie/main/docs/_src/images/readme_front_Normal.png" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/alexmaietta/ernie/main/docs/_src/images/readme_front_SemanticSegmentation.png" width="200"></td>
  </tr>
  <tr>
    <th>Top</th>
    <td><img src="https://raw.githubusercontent.com/alexmaietta/ernie/main/docs/_src/images/readme_top_LdrColor.png" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/alexmaietta/ernie/main/docs/_src/images/readme_top_DistanceToCamera.png" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/alexmaietta/ernie/main/docs/_src/images/readme_top_Normal.png" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/alexmaietta/ernie/main/docs/_src/images/readme_top_SemanticSegmentation.png" width="200"></td>
  </tr>
</table>

For more advanced code snippets and API usage such as zero-copy interop, batched rendering, and cloning, see our [guides](https://alexmaietta.github.io/ernie/guides).

For more information about Ernie's requirements, C++ installation, or other installation methods such as Docker and building from source, see our [installation guide](https://alexmaietta.github.io/ernie/installation).

## Citation

If you use Ernie in your research, please cite it. GitHub's "Cite this repository" button gives the same citation from [CITATION.cff](https://github.com/alexmaietta/ernie/blob/main/CITATION.cff).

```bibtex
@software{maietta2026ernie,
  author  = {Maietta, Alex},
  title   = {Ernie: Efficient Renderer for Natural and Interplanetary Environments},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/alexmaietta/ernie},
  license = {BSD-3-Clause}
}
```

## License

Ernie is licensed under the [BSD 3-Clause License](https://github.com/alexmaietta/ernie/blob/main/LICENSE).

## Acknowledgements

<!-- --8<-- [start:acknowledgements] -->
Ernie is built on these open-source projects.

- [Vulkan Headers](https://github.com/KhronosGroup/Vulkan-Headers), [volk](https://github.com/zeux/volk) and [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) for the GPU layer
- [Slang](https://github.com/shader-slang/slang) for shaders
- [DLPack](https://github.com/dmlc/dlpack) for sharing tensors with PyTorch, JAX, Warp and NumPy
- [nanobind](https://github.com/wjakob/nanobind) for the Python bindings
- [OpenEXR and Imath](https://github.com/AcademySoftwareFoundation/openexr), [stb](https://github.com/nothings/stb), [Basis Universal](https://github.com/BinomialLLC/basis_universal) and [bc7enc_rdo](https://github.com/richgel999/bc7enc_rdo) for images and textures
- [NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb) for volumes
- [nlohmann/json](https://github.com/nlohmann/json) for JSON
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) for OBJ import
- [MikkTSpace](https://github.com/mmikk/MikkTSpace) for tangents
- [NVTX](https://github.com/NVIDIA/NVTX) for profiling
- [OpenUSD](https://openusd.org/) for scene interchange
- [Dear ImGui](https://github.com/ocornut/imgui) and [GLFW](https://github.com/glfw/glfw) for the viewer
- [GoogleTest](https://github.com/google/googletest) for tests
- [Google Benchmark](https://github.com/google/benchmark) for benchmarks

Ernie also leverages ideas from these renderers and projects:

- [Madrona](https://madrona-engine.github.io/) for batch rendering across many worlds
- [Chrono::Sensor](https://api.projectchrono.org/group__sensor.html) for sensor simulation inside Chrono
- [Cycles](https://www.cycles-renderer.org/), [Mitsuba 3](https://www.mitsuba-renderer.org/) and [PBRT](https://pbrt.org/) for physically based and spectral rendering
- [OpenUSD](https://openusd.org/) and Omniverse Fabric for the schema-based scene
- [ovrtx](https://github.com/NVIDIA-Omniverse/ovrtx) for zero-copy interops and USD-based rendering
- [nvpro-samples](https://github.com/nvpro-samples) for Vulkan ray tracing and cluster LOD
- [PANGU](https://pangu.software/) for planetary and asteroid surface simulation
- [IRIS](https://dartslab.jpl.nasa.gov/References/pdf/IRIS_aiaa2022SciTech.pdf) for real-time, closed-loop rendering of space sensors
- [CosmoScout VR](https://github.com/cosmoscout/cosmoscout-vr) for planet-scale scenes and solar-system visualization
- [three-geospatial](https://github.com/takram-design-engineering/three-geospatial) for atmospheres and clouds
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [Newton](https://github.com/newton-physics/newton) for tiled cameras in robot learning
<!-- --8<-- [end:acknowledgements] -->
