# Ernie features

A complete list of what Ernie does, grouped by concept. Each section opens with the idea behind it, then lists what is implemented. Everything here exists in the code today; planned work is at the end under [Not yet](#not-yet).

Ernie is a Vulkan rasterizer. It evaluates direct light from analytic emitters, the sky and the environment, with visibility from shadow maps, and carries that light spectrally through materials and media to a model of the sensor. It does not trace rays.

## Contents

- [Light as a spectrum](#light-as-a-spectrum)
- [Light units](#light-units)
- [Light sources](#light-sources)
- [Extended light sources and the sun](#extended-light-sources-and-the-sun)
- [Stars](#stars)
- [Atmospheres](#atmospheres)
- [Eclipses and long-range shadows](#eclipses-and-long-range-shadows)
- [Shadows](#shadows)
- [Lasers](#lasers)
- [Polarization](#polarization)
- [Materials and BRDFs](#materials-and-brdfs)
- [Planetary photometry](#planetary-photometry)
- [Physically based materials](#physically-based-materials)
- [Textures and filtering](#textures-and-filtering)
- [Thermal infrared](#thermal-infrared)
- [Volumes](#volumes)
- [Cameras](#cameras)
- [Lens and instrument models](#lens-and-instrument-models)
- [The radiometric chain](#the-radiometric-chain)
- [Photographic output](#photographic-output)
- [Optical effects](#optical-effects)
- [Detector and readout effects](#detector-and-readout-effects)
- [Thermal cameras](#thermal-cameras)
- [Outputs and AOVs](#outputs-and-aovs)
- [Planets and terrain](#planets-and-terrain)
- [Large scales and precision](#large-scales-and-precision)
- [Streaming](#streaming)
- [Scene description](#scene-description)
- [Animation and time](#animation-and-time)
- [GPU-driven rendering](#gpu-driven-rendering)
- [Batched rendering for learning](#batched-rendering-for-learning)
- [Interoperability](#interoperability)
- [Extending Ernie](#extending-ernie)
- [Viewer and tools](#viewer-and-tools)
- [Astronomy and mission data](#astronomy-and-mission-data)
- [Not yet](#not-yet)

## Light as a spectrum

Ernie carries light as a spectrum from the emitter to the sensor, not as RGB. A camera's spectral range is split into bands, and every light, material and medium quantity is reduced to one value per band, so effects that depend on wavelength (a sensor's quantum efficiency, Rayleigh scattering, thin-film colour, thermal emission) come out right without colour-space tricks.

- Equal bands over a camera's range: 16 across 380-780 nm by default, up to 64, reaching into the infrared for thermal sensors.
- RGB mode: 3 bands whose values are linear sRGB channels, for speed when spectra do not matter.
- Spectra averaged into each band; RGB inputs (material colours, light tints, dome maps) uplifted to smooth spectra by Jakob and Hanika's sigmoid model, fitted on the CPU and looked up from a coefficient table on the GPU.
- Colour from the CIE 1931 2° observer (an analytic fit), with band-to-XYZ weights fitted by least squares for coarse bands and a Bradford adaptation to D65.
- Illuminant shapes: black body (Planck), CIE daylight D-series, flat white, the measured TSIS-1 solar spectrum, and custom tabulated curves.

## Light units

Every light ends as a spectral quantity in W·nm⁻¹ per band, whichever convention it was authored in.

- Physical: irradiance, power or a spectrum in SI units (`ernie:irradiance`, `ernie:power`, `ernie:spectrum`).
- Relative: UsdLux intensity × 2^exposure, read as luminance in cd/m².
- Photometric: Omniverse-style lumens, or lux at a distance.
- Authored surface emission in cd/m² (UsdPreviewSurface `emissiveColor`), converted to spectral radiance.

## Light sources

All UsdLux light types, lit analytically per pixel.

- Distant lights with an angular diameter (see [extended sources](#extended-light-sources-and-the-sun)).
- Sphere and point lights, and spot lights with a smooth cone, UsdLux focus and IES LM-63 candela profiles.
- Rect, disk and cylinder area lights, lit by the exact form factor of the emitter clipped to the receiver's horizon (Arvo's vector form factor).
- Dome lights from latitude-longitude maps: diffuse light from spherical-harmonic irradiance, glossy light from prefiltered mips. They can also extract the HDR sun as a real, shadowed distant light, or replace the map by median-cut distant lights.
- Uniform ambient light (`ernie:ambient`).
- Pulsed lights: rate, width, delay, exponential decay, jitter, triggered by the sensor clock or the exposure. Each view is lit by its pulses' overlap with its own exposure, so rolling-shutter flash banding appears.
- Light linking: lights restricted to chosen prims, as UsdLux collections.

## Extended light sources and the sun

The sun is not a point: from Earth or the Moon it is a disc about half a degree across, brighter at the centre than at the limb. Its width sets how soft every shadow is, how a partial eclipse dims the light, and how sunrise creeps across terrain. Ernie treats every shadowed distant light as an extended source with an angular size, not just "the sun".

- Angular diameter per distant light (0.53° by default), with irradiance given at normal incidence above any atmosphere.
- Limb-darkened discs seen directly by cameras. Their integral equals the light's irradiance, and a disc smaller than a pixel conserves its irradiance in one pixel.
- Soft shadows whose penumbra width follows the light's true angular size and the occluder distance (percentage-closer soft shadows).
- Partial occlusion of the disc by moons, planets, terrain horizons and distant geometry, integrated over rings of the limb-darkened disc ([eclipses](#eclipses-and-long-range-shadows)).
- Sunrise and sunset on terrain: the fraction of the disc above the local horizon.

## Stars

- `StarField` prims from your own catalogues: right ascension and declination or directions, V magnitude, B-V colour index or temperature.
- Each star a black body at infinity with irradiance from its magnitude, splatted with a tent filter so sub-pixel motion stays smooth.
- Stars lifted by atmospheric refraction and dimmed by atmospheric extinction.
- Demos read the Bright Star Catalogue, Hipparcos and Tycho-2.

## Atmospheres

Atmospheres are physical models of what the air is made of, not fitted sky colours, so the same model gives the sky from the ground and the limb from orbit.

- A spherical shell of constituents, each with a density profile and wavelength-dependent cross sections and phase functions.
- Earth: air on the U.S. Standard Atmosphere 1976, with ozone and bimodal haze. Mars: a CO₂ atmosphere with feldspar and hematite dust.
- Rayleigh scattering (after Bodhaine), Mie scattering, and Maxwell Garnett mixing for dust.
- Precomputed tables: transmittance to the top of the atmosphere, Hillaire's multiple scattering, sky irradiance on level ground, refraction, and the planet's shadow.
- Per-pixel sky: the atmosphere ray-marched for each background pixel, for up to 4 distant lights.
- Aerial perspective on surfaces, with light shafts from shadow maps.
- Sky and ground light on surfaces; above the air it falls off with distance as light from the planet below.
- Atmospheric refraction: the sun, stars and dome lifted by the bending through the layered air, and light reaching below the geometric horizon.
- Lamps lighting the air, and laser beams seen in it ([lasers](#lasers)).

## Eclipses and long-range shadows

Eclipses are shadows of real geometry at astronomical distances: a moon tens of thousands of kilometres away covering part of the sun's disc. Cascaded shadow maps cannot reach that far, so each distant light gets a separate long-range map.

- Casters more than 10 km closer to the light than the receivers are drawn into a long-range orthographic map.
- A coverage mip chain integrates the limb-darkened disc over the casters ring by ring, for any caster shape at any distance.
- Evaluated at 4 wavelengths and 2 receiver depths, so limb darkening's colour comes through.
- The planet's own shadow with its refracting atmosphere: light bent into the umbra gives the red of a lunar eclipse.
- Terrain sun-horizon maps for sunrise and sunset over a whole terrain ([shadows](#shadows)).

## Shadows

Visibility comes from shadow maps, built to keep shadows correct from a rover's wheel out to the horizon.

- Cascaded shadow maps for distant lights. Unreal's distribution runs to 200 m, then far cascades each about 8 times deeper until the farthest depth seen, up to 10 cascades.
- Stable cascades: bounding spheres and texel snapping, so shadows do not shimmer as the camera moves.
- Constant and slope-scaled bias, 5×5 PCF, and a cross-fade between cascades.
- Percentage-closer soft shadows at the light's true angular size, ported from Unreal's directional PCSS with blocker-covariance elongation.
- Soft shadows from sphere, spot and area lights.
- Terrain sun-horizon maps: horizon elevation toward the sun, marched over a max-height pyramid, for sun shadows at any distance on terrain.
- Terrain horizon occlusion of the sky: horizons in 8 azimuths, integrated for sky visibility and a bent normal.
- Shadows from volumes onto surfaces through transmittance shadow maps, and from surfaces onto volumes.
- Per-world shadow maps for batched rendering ([batched rendering](#batched-rendering-for-learning)).

## Lasers

- `LaserLight`: a Gaussian beam with datasheet parameters: one spectral line with linewidth, power, elliptical waist radii and position, M² per axis, an aperture clip, and polarization.
- Patterns from diffractive optics: spot, line (Powell flat or Gaussian), grid of beamlets, pseudo-random dot fields (Kinect-like), images, Lissajous scans, and your own patterns written in Slang.
- Irradiance averaged over each pixel's footprint, so sub-pixel lines and dots keep their power.
- Pulse trains and sequences of patterns, integrated exactly over each exposure.
- Speckle: grain size set by the camera aperture, with contrast reduced by averaging, polarization and wavelength diversity, and by smooth surfaces. Fixed to the scene, it decorrelates as the camera moves.
- Beams seen in air, smoke and dust, as closed-form line integrals so millimetre beams light the medium without depending on march steps.
- Sodium laser guide stars: resonant fluorescence in the mesospheric sodium layer near 589 nm.
- Structured light: a laser pattern sequence seen by an ordinary camera; the Gray-code demo decodes it.

## Polarization

- Lights with a degree and angle of linear polarization.
- Fresnel polarization of specular reflection. Diffuse reflection keeps a per-material share of the incident polarization.
- The sky's Rayleigh polarization.
- Polarizer filters and 2×2 polarization mosaics (Sony IMX250MZR layout), with Stokes, degree and angle of linear polarization outputs.

## Materials and BRDFs

Every reflectance model is a Slang struct returning the radiance factor I/F per band. Models are registered at run time with their parameters, ranges and references, and each compiles into its own shading pipeline. Rough models share one treatment of roughness below the pixel: slope variance lost in normal-map mips widens each model's roughness consistently.

- Lambert.
- Oren-Nayar (qualitative).
- Lommel-Seeliger with a Henyey-Greenstein phase function.
- Lunar-Lambert (McEwen), the phase-dependent blend used for photoclinometry.
- Hapke ([planetary photometry](#planetary-photometry)).
- UsdPreviewSurface (glTF 2.0 metallic-roughness and specular workflows, clearcoat).
- OpenPBR Surface ([physically based materials](#physically-based-materials)).
- Your own BRDFs in Slang ([extending](#extending-ernie)).
- MDL OmniPBR and OmniSurface materials approximated by UsdPreviewSurface on USD import.

## Planetary photometry

Regolith reflects light unlike paint or metal: brightness surges toward zero phase angle and shadows hide between grains. These are the models planetary scientists fit to orbital images.

- Hapke as parameterised by Sato et al. (2014) for the LROC WAC parameter maps: single-scattering albedo, porosity, shadow-hiding opposition effect, double Henyey-Greenstein phase function (b, c), multiple scattering through Hapke's 2002 H-function approximation, and Hapke's 1984 macroscopic roughness.
- Lommel-Seeliger and lunar-Lambert for photoclinometry and quick fits.
- Parameters as textures, so maps of albedo or Hapke parameters drive each pixel.
- The same phase functions available to volumes, so dust clouds match the regolith.

## Physically based materials

OpenPBR Surface is built in alongside UsdPreviewSurface, spectral throughout. Layers no material uses are compiled out.

- Rough diffuse: energy-preserving Oren-Nayar (EON).
- GGX specular with multiple-scattering energy compensation, kept reciprocal.
- Metals with an F82 edge tint.
- Thin-film interference computed per band.
- Anisotropic roughness along smooth mesh tangents.
- Clear coat with its own roughness, IOR, colour and darkening.
- Fuzz: Zeltner's sheen, for dust and fabric.
- Transmission: screen-space refraction for glass, alpha blending for partly opaque materials.
- Subsurface scattering spread across the image by a Burley-like profile, per band.

## Textures and filtering

- Normal maps mip-filtered in slope space, with the lost slope variance kept to widen roughness, so distant terrain darkens correctly instead of turning glossy.
- Hex-tiling and triplanar projection for detail textures, without visible repetition.
- Pixel footprints from ray differentials, for analytic filtering of lasers and patterns.
- Virtual textures larger than GPU memory ([streaming](#streaming)).
- BC7 compression.

## Thermal infrared

Everything glows. At thermal wavelengths a surface's own emission, set by its temperature and emissivity, outshines reflected sunlight.

- Planck emission from surfaces and volumes, averaged over each band by Gauss-Legendre quadrature.
- Emissivity from Kirchhoff's law (one minus reflectance) or from measured spectra (e.g. ECOSTRESS).
- Temperature sources, in order of precedence: an external solver writing per mesh point (from host or CUDA), per-prim temperatures, temperature textures, temperature models, and material constants.
- Temperature models computed per pixel as the scene renders:
  - zero-capacitance energy balance for spacecraft surfaces;
  - diurnal lookup tables for airless bodies, read at each pixel's thermal inertia, albedo, latitude, slope and local time;
  - your own models in Slang.
- A regolith thermal solver (after Hayne et al. 2017) that bakes the diurnal tables.
- Microbolometer cameras ([thermal cameras](#thermal-cameras)).

## Volumes

Volumes are UsdVol-style prims: a `Volume` names its fields, and a material gives them density, albedo, emission and a phase function. They are ray marched per pixel, spectrally, and composited with surfaces, sky and atmosphere.

- NanoVDB and OpenVDB files (OpenVDB is converted on load), NanoVDB arrays, and dense fields written on the GPU each frame by CUDA or Warp.
- Emission-absorption with single scattering of every light type, self-shadowed.
- Phase functions: isotropic, Henyey-Greenstein, double and dual Henyey-Greenstein (Hapke's b and c), Rayleigh, and Mie tables for cloud droplets per band (silver linings, fogbows, glories).
- Multiple scattering approximated by Wrenninge's octaves or Bouthors' collectors, with a powder effect at cloud edges.
- Black-body glow from temperature fields, plus emission fields and colours.
- Sun, sky and ground light through a lighting cache.
- Shadows from volumes onto surfaces and onto each other.
- Procedural cloud layers: up to four layers of cumulus and cirrus from weather maps and Perlin-Worley noise, drifting with the wind, lit through their own shadow maps.
- Exact empty-space skipping (NanoVDB tiles and distance-field bricks), step merging, prefiltered grid levels, checkerboard marching with temporal reprojection, half-resolution marching, and culling of volumes outside every view.

## Cameras

Cameras follow UsdGeomCamera: focal length and apertures in millimetres, with aperture offsets for the principal point.

- Perspective and orthographic (telecentric) projections.
- Reverse-Z depth with an optional infinite far plane ([precision](#large-scales-and-precision)).
- Fisheye and wide fields beyond 120°, rendered on cube faces.
- Line-scan pushbroom cameras like LROC NAC: each row exposed at its own pose along the trajectory, with a projection that finds that pose per vertex.
- Exposure time, f-number, ISO and EV, as USD's exposure attributes.
- Thin-lens depth of field, per band with axial colour.
- Motion blur from sub-frames (2 to 64), weighted over each row's exposure.
- Rolling shutter.
- Batches of cameras sharing one lens and sensor ([batched rendering](#batched-rendering-for-learning)).

## Lens and instrument models

Every lens model is a mapping from sensor pixels to rays. Ernie renders a pinhole or cube-face raster and resamples it through that mapping.

- Brown-Conrady radial and tangential distortion (OpenCV convention).
- Kannala-Brandt fisheye (OpenCV fisheye).
- Lens ray tables: a unit ray at every n-th pixel corner, interpolated. They render JPL CAHV, CAHVOR and CAHVORE (Mars rover cameras) exactly, and a CAHV model can be fitted to any Ernie lens.
- USGS Community Sensor Model ISDs from ALE: frame cameras (with rolling-shutter jitter), line scanners, push-frame cameras (LROC WAC framelets), and the ISD distortion models (radial, transverse, radtan, LROC NAC, Kaguya, Dawn FC, CAHVOR and others).
- Chromatic aberration of a thin singlet (Cauchy dispersion): lateral colour scales each band's image; axial colour shifts each band's focus.
- Natural vignetting from the lens's projection.

## The radiometric chain

A sensor counts photons, not colours. Ernie follows the light from scene radiance to digital numbers the way a detector does, with the noise of the EMVA 1288 linear camera model.

- Camera equation: scene radiance to focal-plane irradiance through the f-number and lens transmission, with natural vignetting.
- Photons per band over the pixel area and exposure time, times each channel's quantum efficiency (mono, RGB or RGB+NIR channels, or measured QE curves).
- Dark current, with temperature doubling.
- Photon shot noise (Poisson) and read noise (Gaussian), seeded per frame and view.
- Fixed pattern noise: PRNU (gain) and DSNU (offset), fixed per pixel by the sensor's seed.
- Full well clipping, ISO-dependent conversion gain, black level, and ADC bit depth.
- Electrons and raw DN as outputs.

## Photographic output

- Linear sRGB in cd/m² (HdrColor) from the CIE observer.
- USD photographic exposure, then clip or AgX filmic tone mapping, and sRGB encoding (LdrColor).
- White balance, a 3×3 colour matrix and a display tone curve.
- Automatic exposure from a log-luminance histogram, adapted over time, metered on the GPU.

## Optical effects

- Diffraction by the aperture: an Airy pattern with central obstruction, or a Gaussian fit, per band at the f-number, integrated over the pixel.
- Multi-Gaussian scatter PSFs (PANGU-style), up to 8 terms.
- Glare: a power-law halo around bright sources.
- Lens flare ghosts in screen space, shaped by the aperture's disc or polygon. Light taken by the ghosts leaves the direct image.
- Stray light from bright sources outside the field, from a point source transmittance curve.

## Detector and readout effects

Sensor filters are compute passes in a chain, each acting on spectra, electrons, digital numbers or display colour. They can be reordered, and you can add your own.

- Spectral transmission of filters and windows.
- Measured QE curves.
- Dark current with temperature.
- Dead and hot pixels.
- Column and row gain and offset patterns.
- Blooming along columns, with an anti-blooming drain.
- Frame-transfer smear.
- Time delay integration (TDI) with drift.
- Image lag.
- Radiation hits: cosmic-ray tracks with path length, straggling and charge diffusion.
- Bayer colour filter arrays, demosaiced bilinearly or by Malvar-He-Cutler.
- Readout taps with per-tap gain and offset.
- LROC NAC nonlinearity tables.
- Companding: square-root, piecewise-linear (LROC NAC) and lookup tables (LROC WAC).
- Downlink errors: lost packets, pixel slips and bit flips.
- JPEG compression at a given quality.
- Temporal averaging.

## Thermal cameras

- Microbolometers measuring in-band radiance, with a first-order thermal time constant, NETD and fixed-pattern noise.
- Brightness temperature for a given object emissivity and reflected temperature.
- White-hot, black-hot, ironbow and rainbow palettes, with automatic gain.
- A Temperature output from surfaces and volumes.

## Outputs and AOVs

- Spectral radiance per band, electrons, raw DN, HdrColor, LdrColor.
- Distance to the image plane and to the camera, normals, prim and instance ids, and motion vectors.
- Linear Stokes, degree and angle of linear polarization.
- Temperature.
- Eight extra AOV slots that custom filters write.
- Outputs shared with PyTorch, JAX, Warp and NumPy without copies ([interoperability](#interoperability)).

## Planets and terrain

- Terrain as a built-in prim: a height-field quadtree drawn by continuous distance-dependent LOD (CDLOD). Vertices geomorph between levels, so nothing pops and no skirts are needed. Edited heights update in place.
- Whole planets and moons as equiangular cube-sphere CDLOD bodies, from orbit to the ground, with horizon culling.
- Ellipsoids, as spheres with non-uniform scale, and planets with triaxial radii; heights rise along the geodetic normal.
- Adaptive spheres: any sphere prim drawn with an exact silhouette from a pebble to a star. Also UV, icosphere, cube-sphere and HEALPix tessellations.
- DEMs from PDS3-labelled rasters such as LOLA LDEM, in simple-cylindrical and polar-stereographic projections, resampled to cube faces with prefiltering.
- Terrain colour modes, including a GIS hillshade.

## Large scales and precision

A float has about seven digits, enough for a millimetre at ten kilometres but not at a planet's radius. Ernie keeps the scene in double precision and gives the GPU small numbers.

- Scene transforms, camera poses and planet geometry in double precision.
- A floating origin: the GPU draws relative to an origin near the camera, rebased when the eye moves 1024 m away.
- Planet patches anchored in double precision, with vertex offsets computed without catastrophic cancellation.
- Reverse-Z 32-bit float depth with an infinite far plane, for even depth precision from centimetres to astronomical distances.
- Viewer navigation in double precision out to 10²¹ m.

## Streaming

- Planet heights paged from large rasters as the views need them, finer pages built on worker threads, LRU-evicted.
- Virtual textures larger than GPU memory. Shaders request the tiles their footprint needs; tiles load coarsest first.
- Sparse residency with a min-LOD clamp where the GPU supports it, with a page-table fallback; hardware anisotropic filtering kept.
- Asynchronous uploads through per-frame staging.

## Scene description

The scene is a USD-shaped tree of typed prims, without a dependency on OpenUSD in the core.

- Prims named by paths, attributes by their USD names: UsdGeom, UsdLux, UsdShade, UsdVol and UsdSkel shapes.
- Prim types declared once, from which come typed C++ views, a runtime registry with validation, Python bindings and the viewer's property panels.
- Inherited attributes, such as `ernie:world`, and relationships whose target edits reach their holders.
- Change tracking per prim, so an edit redraws only what it reaches. A single edit among 100,000 prims costs milliseconds.
- Columns: one attribute of thousands of prims written whole from host, NumPy, PyTorch or CUDA arrays, or mapped in place in the renderer's own buffers.
- OpenUSD stage import through an optional plugin, keeping cameras' exposure and shutter.

## Animation and time

- Time samples on any attribute, evaluated at the stage time code: linear interpolation for numbers, vectors, matrices and arrays; spherical interpolation for quaternions; held values for the rest.
- UsdGeomXformable transform ops composed in double precision; instancer transforms composed on the GPU.
- Skinning on the GPU, linear-blend or dual-quaternion, with blend shapes.
- Catmull-Clark and Loop subdivision (OpenSubdiv).

## GPU-driven rendering

- Deferred shading: a G-buffer, then a compute pass that shades every band.
- GPU frustum culling, two-pass hierarchical-Z occlusion culling and screen-space-error mesh LOD (meshoptimizer), writing compacted, order-preserving draw lists.
- Multi-draw indirect, with parameters fetched by draw index.
- Bindless textures and buffer device addresses.
- Vertex pulling for terrain and planets, with no vertex buffers.
- Points and spheres ray-cast as quadrics in the fragment shader.
- Shaders in Slang, compiled at run time.

## Batched rendering for learning

Reinforcement learning and Monte Carlo studies want thousands of images per step, each of a different world from a different camera. Ernie renders them all in one submission.

- A batch of camera poses per render product, one image layer per view, all drawn by the same draws.
- Worlds: prims tagged with `ernie:world`, or shared by all; each view sees its world and the shared prims.
- Culling per draw and view, with the world filter applied there.
- Poses written from device memory each step, as matrices or position and quaternion rows.
- Per-world shadow policies: separate maps per world, a shared map, or one merged map; each map fitted to its world's casters.
- Volumes and adaptive spheres respect worlds.

## Interoperability

- Zero-copy sharing with CUDA and HIP: Vulkan memory exported to the compute device, ordered by timeline semaphores in both directions, with no host waits.
- DLPack tensors in and out: PyTorch, JAX, Warp and NumPy read outputs and write poses, vertices and light parameters.
- Meshes, instances and points written from CUDA or Warp arrays without copies.
- Renderer backends for Newton Viewer and IsaacLab, and a viewer and renderer for Project Chrono's Chrono::Sensor.
- C++ and Python APIs; Linux, Windows and macOS (through MoltenVK).

## Extending Ernie

- BRDFs in Slang, registered at run time with their parameters and references.
- Sensor filters in Slang, at any domain of the chain.
- AOVs written by custom filters.
- Lens models.
- Spectra and illuminants.
- Star catalogue readers.
- Temperature models.
- Laser patterns.
- Render delegates.
- Viewer panels, menus and frame callbacks.
- Debug drawing: labels, trajectories and overlays.

## Viewer and tools

- A viewer with customisable panels, a property panel showing inherited attributes at their ancestor's value, and a demos panel.
- Double-precision orbit navigation from the ground to interplanetary distances.
- GPU timing from timestamp queries, Tracy profiling, and the Vulkan memory budget.
- Demos in C++ and Python pairs (`python -m ernie --demo demo_<name>`, `--list`), each generating its own data or fetching small public assets.

## Astronomy and mission data

- Julian dates, sidereal time and the sky over a site.
- Sun and spacecraft positions from ALE image support data (SPICE).
- Mean Keplerian elements for the solar-system demo.
- PDS3 DEMs and LROC instrument models for rendering real images.

## Not yet

Planned, not implemented:

- Global illumination and reflections between surfaces; a ray-traced renderer alongside the rasterizer.
- Lidar, radar and time-of-flight sensors.
- Hapke's coherent backscatter opposition effect.
- OGC 3D Tiles streaming (via cesium-native).
- Full OpenUSD and glTF compliance, and MDL and MaterialX compilation.
- Differentiable rendering.
- Gaussian splats and particle fields.
- OpenXR and a WebGPU backend.
