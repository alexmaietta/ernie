// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/vulkan_context.h"
#include "ernie/geometry/mesh_data.h"
#include "ernie/scene/brdf.h"
#include "ernie/scene/camera.h"
#include "ernie/scene/material.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/sensor_filters.h"
#include "ernie/session/frame_params.h"
#include "ernie/util/image.h"
#include "ernie/util/spectrum.h"
#include "ernie/util/tensor.h"

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// How every render delegate reads a `Scene`: which prims a render
/// uses, its render products, and the camera, lights, materials,
/// meshes and instances as plain data, so backends differ only in how
/// they draw.
namespace ernie {

/// The prims a render of `scene` uses, as every render delegate
/// resolves them:
/// - `settings`: the given RenderSettings prim, or the first one
///   outside `tokens::kSessionRoot` when the given path is empty;
///   empty if there is none, which leaves every option at its
///   default.
/// - `camera`: the settings' `camera` relationship, or else the first
///   Camera outside the session root.
/// - `terrain`: the first visible Terrain.
/// - `material`: the terrain's `material:binding` when it names a
///   Material.
/// - the lights, from the settings' `ernie:lights` relationship when
///   it is set (its visible light prims), or else from every visible
///   light outside the session root: `lights` every DistantLight,
///   SphereLight, RectLight, DiskLight, CylinderLight and
///   LaserLight, sorted by path, no light above another; `dome` the
///   first
///   DomeLight. A scene without
///   lights renders black.
/// - `ground`: the first visible GroundPlane.
struct RenderPrims {
    Path settings;
    Path camera;
    Path terrain;
    Path material;
    std::vector<Path> lights;
    Path dome;
    Path ground;
    Path stars;       ///< The first visible StarField.
    Path atmosphere;  ///< The first visible Atmosphere.
    /// The first visible CelestialBodies, at most
    /// `FrameParams::kMaxBodies`.
    std::vector<Path> bodies;
    /// A stand-in for the camera's world transform, for a camera
    /// posed on the GPU (`CameraParamsOf`).
    std::optional<glm::dmat4> cameraTransform;
    /// Stand-ins for the world transforms of `lights` posed on the
    /// GPU (`LightParamsOf`).
    std::map<Path, glm::dmat4> lightTransforms;
};

/// A render product to draw, resolved from UsdRender prims.
struct RenderProductSpec {
    Path product;  ///< The RenderProduct prim, or the settings' path
                   ///< for the default product.
    Path camera;   ///< The first of `cameras`; empty to use the
                   ///< settings' camera.
    /// Every camera of the product's `camera` relationship: several
    /// render one image each, a batch of views (`ViewPosesOf`).
    std::vector<Path> cameras;
    VkExtent2D resolution{2048, 1080};
    /// Render vars as (path, sourceName).
    std::vector<std::pair<Path, std::string>> vars;
    /// The product's sensor filter chain, with each filter's inputs;
    /// empty for `DefaultFilterChain`. Custom filter types are
    /// checked when rendering, against the renderer's registry.
    std::vector<SensorFilterStep> filters;
    /// Per var of `vars`, the chain of its own `ernie:filters` when
    /// it has one: that var comes from it rather than the product's
    /// (the same chain twice runs once), each source name from one
    /// chain.
    std::vector<std::optional<std::vector<SensorFilterStep>>>
        varFilters;
    /// The product's own `ernie:overlays` (`tokens::kOverlays`), in
    /// place of its settings'.
    std::optional<std::string> overlays;
    /// The product's own minimal shading and colour
    /// (`tokens::kMinimalMode`, `tokens::kMinimalConstantColor`), in
    /// place of its settings'.
    std::optional<MinimalMode> minimal;
    std::optional<glm::vec3> minimalColor;
};

/// The minimal shading `prim` asks for (`tokens::kMinimalMode`), or
/// `fallback` without it.
/// @throws std::invalid_argument for an unknown mode.
MinimalMode MinimalModeOf(const Scene& scene, const Path& prim,
                          MinimalMode fallback);

/// The views a render product draws through `cameras`
/// (`RenderProductSpec::cameras`), for images of `width` x `height`:
/// none for a single camera without `ernie:poses`, which draws the
/// one view of its own transform; else the camera-to-world pose of
/// every view in a batch that shares the first camera's lens and
/// sensor, one image each:
/// - several cameras: each camera's world transform;
/// - one camera with `ernie:poses` (`tokens::kPoses`): each pose in
///   place of the camera's own transform, under its parent's.
/// @throws std::invalid_argument if a camera is not a Camera prim,
/// cameras of a batch differ in projection or spectral bands, or the
/// poses are not a CPU (views, 4, 4) float array. `hostPoses`, when
/// given, stands in for one camera's `ernie:poses`, such as a host
/// copy of poses written as a CUDA array.
std::vector<glm::dmat4>
ViewPosesOf(const Scene& scene, std::span<const Path> cameras,
            std::uint32_t width, std::uint32_t height,
            const Tensor* hostPoses = nullptr);

/// Checks that `cameras` can render as one batch: the same projection
/// and spectral bands.
/// @throws std::invalid_argument naming a camera that differs.
void CheckBatchCameras(const Scene& scene,
                       std::span<const Path> cameras,
                       std::uint32_t width, std::uint32_t height);

/// The parent world transform the poses of `camera`
/// (`tokens::kPoses`) are under: its parent's world transform, or the
/// identity at the root.
glm::dmat4 PosesParentOf(const Scene& scene, const Path& camera);

/// `camera`'s poses (`tokens::kPoses`) if they stay on the GPU
/// (`tokens::kPosesDevice`): a contiguous float32 CUDA array of
/// (views, 4, 4), (views, 16) or (views, 7), and "gpu" (the default).
/// Null for poses on the host or downloaded first.
/// @throws std::invalid_argument for poses of another shape or an
/// unknown device token.
const Tensor* GpuPosesOf(const Scene& scene, const Path& camera);

/// Whether poses (`tokens::kPoses`) are rows of a position and a unit
/// quaternion, (views, 7) as x, y, z, qx, qy, qz, qw, rather than 4 x
/// 4 matrices.
bool PosesAreQuaternions(const Tensor& poses);

/// The world of the prim at `path` (`tokens::kWorld`), from it or its
/// nearest ancestor with one; -1, shared by every world, if none has
/// one.
std::int32_t WorldOf(const Scene& scene, const Path& path);

/// A host copy of array `attribute` of `path`, written as `device` on
/// a GPU: for the few arrays the CPU plans with, such as worlds. A
/// delegate with compute interop gives one, which may keep the copy
/// until the attribute is written again.
using HostArrayReader =
    std::function<Tensor(const Path& path, std::string_view attribute,
                         const Tensor& device)>;

/// The world each of the `views` views through `cameras` sees (see
/// `ViewPosesOf`): each camera's world for several cameras; for one
/// camera, its `ernie:worlds` per pose (`tokens::kWorlds`), or else
/// its world for every view. Worlds written on a GPU are read through
/// `read`.
/// @throws std::invalid_argument if `ernie:worlds` is not an integer
/// array of one world per view, or is on a GPU without `read`.
std::vector<std::int32_t>
ViewWorldsOf(const Scene& scene, std::span<const Path> cameras,
             std::size_t views, const HostArrayReader& read = {});

/// The world of each of the `count` instances of the PointInstancer
/// at `instancer`: its `ernie:worlds`, or else the instancer's world
/// for every instance. Worlds written on a GPU are read through
/// `read`.
/// @throws std::invalid_argument if `ernie:worlds` is not an integer
/// array of one world per instance, or is on a GPU without `read`.
std::vector<std::int32_t>
InstanceWorldsOf(const Scene& scene, const Path& instancer,
                 std::size_t count, const HostArrayReader& read = {});

/// The render products a step of `scene` draws, following UsdRender:
/// the given RenderProduct prims, or else the RenderSettings'
/// `products`, or else one default product keyed by the settings'
/// path (the root without settings). Each product takes its `camera`
/// (one or several, see `ViewPosesOf`) and `resolution` from its
/// prim, falling back to the settings' and then 2048 x 1080; its vars
/// are its `orderedVars` RenderVar prims by `sourceName` (any of
/// `kRenderVarNames`), or `LdrColor` alone; its sensor filters are
/// the SensorFilter prims of its `ernie:filters`, by
/// `ernie:filter:type`.
/// @throws std::invalid_argument for a path that is not a
/// RenderProduct, a var that is not a RenderVar, a filter that is not
/// a SensorFilter or of an unknown type, an invalid filter chain
/// (`CheckFilterChain`), or a var the chain does not write.
std::vector<RenderProductSpec>
ResolveRenderProducts(const Scene& scene,
                      std::span<const Path> products,
                      const Path& settings);

/// Resolves the prims a render of `scene` with the RenderSettings
/// prim `settings` uses.
/// @throws std::invalid_argument if `settings` is not empty and not a
/// RenderSettings prim.
RenderPrims ResolveRenderPrims(const Scene& scene,
                               const Path& settings);

/// The frame number the RenderSettings prim `settings` authors in
/// `ernie:frame`, if it does: the frame an image's noise and samples
/// derive from (`FrameSeed`), so a simulation can render any frame
/// again without the ones before it.
/// @throws std::invalid_argument for a negative or non-integer frame.
std::optional<std::uint64_t> FrameOf(const Scene& scene,
                                     const Path& settings);

/// The Material prim bound to `prim` by `material:binding`, or the
/// empty path.
Path BoundMaterial(const Scene& scene, const Path& prim);

/// The sun's spectral irradiance at normal incidence per band of
/// `bands`, in W m^-2 nm^-1, from the DistantLight at `light`:
/// - physical, when it has `ernie:irradiance` or `ernie:spectrum`:
///   the spectrum binned (scaled to the irradiance when both are
///   set), or else a black body at the enabled colour temperature or
///   the Sun's 5772 K carrying the irradiance (default 1361 W/m^2,
///   the solar constant);
/// - otherwise relative, following UsdLux: `inputs:intensity` times
///   2^`inputs:exposure` is the luminance (cd/m^2) of a white
///   Lambertian surface facing the light, with the spectrum's shape
///   from `inputs:color` (upsampled with `RgbToSigmoid`) times a
///   black body at the colour temperature when
///   `inputs:enableColorTemperature` is set.
/// @throws std::invalid_argument for invalid bands or spectrum pairs.
std::vector<float> SunIrradianceOf(const Scene& scene,
                                   const Path& light,
                                   const SpectralBands& bands);

/// The DistantLight or SphereLight at `path` as a `Light` for
/// `bands`:
/// - distant: towards the light along its +z axis, its spectrum as
///   `SunIrradianceOf`;
/// - sphere: at its world position with its cone
///   (`inputs:shaping:cone:angle` and `:softness`, default 180
///   degrees: unshaped) around its -z axis, radius `inputs:radius`
///   (default 0.5; 0 with `treatAsPoint`). Physical with
///   `ernie:power` or `ernie:spectrum` (`tokens::kLightPower`);
///   otherwise relative, following UsdLux: `inputs:intensity` times
///   2^`inputs:exposure` is the luminance (cd/m^2) of its surface, so
///   its luminous intensity is that times pi r^2 (times 1 for a
///   point), with the shape of `inputs:color` and the enabled colour
///   temperature.
/// - rect and disk: centred at its world position, emitting along its
///   -z axis from a face of `inputs:width` x `inputs:height` (default
///   1 x 1) or radius `inputs:radius` (default 0.5), scaled by its
///   transform. Physical with `ernie:power` or `ernie:spectrum` as a
///   sphere; otherwise relative: `inputs:intensity` times
///   2^`inputs:exposure` is its luminance (cd/m^2). Shadows follow
///   `inputs:shadow:enable`.
/// - laser (Ernie's LaserLight): a Gaussian beam along its -z axis,
///   its waist and Rayleigh ranges in `beam`, its line binned exactly
///   into the bands as W nm^-1 of the power through its aperture.
/// @throws std::invalid_argument for another prim type, a negative
/// radius or power, an area light without area, a degenerate
/// transform, invalid bands or spectrum pairs.
LightParams LightParamsOf(const Scene& scene, const Path& path,
                          const SpectralBands& bands);

/// `LightParamsOf` with `world` standing in for the light's world
/// transform, for a light posed on the GPU
/// (`RenderPrims::lightTransforms`).
LightParams LightParamsOf(const Scene& scene, const Path& path,
                          const SpectralBands& bands,
                          const glm::dmat4& world);

/// Diffuse irradiance over pi of a latitude-longitude RGB map
/// (`tokens::kLightTexture` layout) turned by `rotation` into the
/// world, as order-2 spherical harmonics (Ramamoorthi and Hanrahan
/// 2001): the map's projection weighted by each texel's exact solid
/// angle, convolved with the clamped cosine.
/// @throws std::invalid_argument unless `latlong` is a float32 (rows,
/// cols, 3 or 4) CPU array.
std::array<glm::vec3, 9> DomeIrradianceSh(const Tensor& latlong,
                                          const glm::dmat3& rotation);

/// The DomeLight at `path` for `bands`: radiance `ernie:spectrum` (W
/// m^-2 sr^-1 nm^-1), or in relative units `inputs:intensity` times
/// 2^`inputs:exposure` as luminance (cd/m^2) with the shape of
/// `inputs:color` and the enabled colour temperature; its map
/// `ernie:texture` (uniform without one) rotated by its transform.
/// @throws std::invalid_argument for another prim type, an invalid
/// map, bands or spectrum pairs.
DomeParams DomeParamsOf(const Scene& scene, const Path& path,
                        const SpectralBands& bands);

/// The DomeLight at `path`'s map (`tokens::kLightTexture`) as a
/// linear RGB image for drawing behind the scene; none without one.
/// @throws std::invalid_argument for an invalid map.
std::optional<Image> DomeMapOf(const Scene& scene, const Path& path);

/// A sun found in a dome's map (`ExtractDomeSun`), in the map's
/// frame.
struct DomeSun {
    glm::dvec3 direction{
        0.0, 0.0,
        1.0};  ///< Unit vector towards it, in the dome's frame.
    double angle = 0.53;  ///< Angular diameter in degrees.
    /// Its RGB irradiance at normal incidence above the sky around
    /// it: the map's values over the sky's, times each texel's solid
    /// angle, summed (the map's units times steradians).
    glm::dvec3 irradiance{0.0};
};

/// The sun of a latitude-longitude dome map (`tokens::kLightTexture`
/// layout) and the map without it, for lighting the sun as a shadowed
/// distant light while the dome keeps the rest of the sky.
struct DomeSunExtraction {
    DomeSun sun;
    /// The map with the sun's texels replaced by the sky around it.
    Image lighting;
};

/// The sun of dome map `map`, if it has one: the brightest texel's
/// connected region of texels above `kDomeSunThreshold` of its
/// luminance, when that region is compact (under 5 degrees across),
/// much brighter than the sky (its peak over 20 times the map's
/// median luminance) and carries at least 5% of the map's irradiance.
/// HDR skies with a sun disc have one; overcast skies, interiors and
/// low-dynamic-range maps do not.
///
/// References:
/// - P. Debevec, "A Median Cut Algorithm for Light Probe Sampling",
///   SIGGRAPH 2005 posters (lighting from the energy of an
///   environment map's regions).
///   https://vgl.ict.usc.edu/Research/MedianCut/MedianCutSampling.pdf
std::optional<DomeSunExtraction> ExtractDomeSun(const Image& map);
inline constexpr double kDomeSunThreshold = 0.1;

/// One of a dome map's regions of about equal energy
/// (`DomeDirectionsOf`), in the map's frame.
struct DomeDirection {
    glm::dvec3 direction{
        0.0, 0.0,
        1.0};  ///< The region's energy-weighted mean direction.
    glm::dvec3 irradiance{0.0};  ///< Its RGB values times each
                                 ///< texel's solid angle, summed.
};

/// `count` directions standing for latitude-longitude dome map `map`,
/// for lighting with it as shadowed distant lights: Debevec's median
/// cut, splitting the most energetic region along its longer side (in
/// angle) where half its energy (luminance times solid angle) lies on
/// each side, until there are `count`. Their irradiance adds up to
/// the map's. Fewer when the map has fewer texels with energy.
///
/// Reference: P. Debevec, "A Median Cut Algorithm for Light Probe
/// Sampling", SIGGRAPH 2005 posters.
/// https://vgl.ict.usc.edu/Research/MedianCut/MedianCutSampling.pdf
std::vector<DomeDirection> DomeDirectionsOf(const Image& map,
                                            std::size_t count);

/// The stars of the StarField at `path`: each one's place from
/// `ernie:stars:rightAscension` and `ernie:stars:declination` (or
/// `ernie:stars:direction`), and its black-body spectrum from its
/// temperature (`ernie:stars:temperature`, else
/// `ernie:stars:colorIndex` through `ColorIndexToTemperature`, else
/// the Sun's) scaled to its illuminance from `ernie:stars:magnitude`
/// (`StarIlluminance` of `ernie:stars:zeroPoint`, times
/// 2^`inputs:exposure`). Stars fainter than
/// `ernie:stars:magnitudeLimit` are left out.
///
/// References:
/// - A. N. Cox (ed.), "Allen's Astrophysical Quantities", 4th
///   edition, Springer, 2000 (the illuminance of a magnitude 0 star).
/// - F. J. Ballesteros, "New insights into black bodies", EPL
///   97(3):34008, 2012 (B-V to temperature).
/// - H. W. Jensen, F. Durand, J. Dorsey, M. M. Stark, P. Shirley and
///   S. Premoze, "A Physically-Based Night Sky Model", SIGGRAPH 2001,
///   399-408 (stars as black bodies by magnitude and colour index).
///   https://doi.org/10.1145/383259.383306
/// @throws std::invalid_argument for another prim type, or arrays of
/// mismatched lengths.
std::vector<Star> StarsOf(const Scene& scene, const Path& path);

/// Where the StarField at `path`'s catalogue frame lies in the world,
/// from its transform.
/// @throws std::invalid_argument for another prim type.
StarFieldParams StarFieldParamsOf(const Scene& scene,
                                  const Path& path);

/// The mean radius (metres) of an Atmosphere preset's planet, as
/// `AtmosphereParamsOf` takes it; 0 for an unknown preset.
double AtmospherePresetRadius(std::string_view preset);

/// The Atmosphere at `path` around the planet Terrain `terrain`
/// (empty for none): with a planet Terrain anchored on the ground the
/// planet is at the Terrain's translation and has its radius unless
/// `ernie:atmosphere:planetRadius` sets one; without, it is that
/// radius, else the preset planet's mean radius, centred that far
/// below the Atmosphere's origin.
///
/// The air is a spherical shell. On an ellipsoid Terrain
/// (`ernie:radii`) its sphere, of the Atmosphere's radius or else the
/// preset's, touches the ellipsoid at the foot of the normal through
/// `eye` (world), sharing its tangent plane there, so the ground and
/// the air's bottom agree under the viewer and the altitude is right
/// (three-geospatial's altitude correction); without an eye, it is
/// centred on the Terrain.
///
/// References:
/// - takram-design-engineering, three-geospatial,
///   `Ellipsoid.getOsculatingSphereCenter` and
///   `getAltitudeCorrectionOffset` (its sphere touches the ellipsoid
///   rather than osculating it).
///   https://github.com/takram-design-engineering/three-geospatial
/// - H. Moritz, "Geodetic Reference System 1980", Journal of Geodesy
///   74(1):128-133, 2000 (the Earth's mean radius R1, 6371.0088 km).
///   https://doi.org/10.1007/s001900050278
/// - B. A. Archinal et al., "Report of the IAU Working Group on
///   Cartographic Coordinates and Rotational Elements: 2015",
///   Celestial Mechanics and Dynamical Astronomy 130:22, 2018 (Mars's
///   mean radius, 3389.5 km).
///   https://doi.org/10.1007/s10569-017-9805-5
/// @throws std::invalid_argument for another prim type or an unknown
/// preset.
AtmosphereParams AtmosphereParamsOf(
    const Scene& scene, const Path& path, const Path& terrain,
    const std::optional<glm::dvec3>& eye = std::nullopt);

/// The planet an Atmosphere wraps, in world space.
struct PlanetSphere {
    glm::dvec3 centre{0.0};
    double radius = 0.0;  ///< Metres, to the ground.
    double top = 0.0;     ///< Metres, to the top of its air.
};

/// The planet the Atmosphere at `path` wraps (`AtmosphereParamsOf`,
/// over the scene's first visible Terrain), or none for another prim
/// or one that cannot resolve.
std::optional<PlanetSphere> PlanetOf(const Scene& scene,
                                     const Path& path);

/// The planet of the scene's first visible Atmosphere (`PlanetOf`),
/// or none.
std::optional<PlanetSphere> PlanetOf(const Scene& scene);

/// Of the scene's visible Atmospheres, the one whose air is nearest
/// `point` (world; inside it, distance 0, the smallest planet's
/// first): the one a frame renders, as a camera flies between
/// planets. Empty without one.
Path NearestAtmosphere(const Scene& scene, const glm::dvec3& point);

/// A CelestialBody's centre (its transform's origin), radius and
/// albedo.
/// @throws std::invalid_argument for another prim type.
BodyParams BodyParamsOf(const Scene& scene, const Path& path);

/// How much wider than the sensor's the raster image's field of view
/// must be for every sensor pixel's undistorted ray to land in it: 1
/// without distortion, else the largest ratio along the sensor's
/// border, plus one percent.
double OverscanOf(const CameraParams& camera);

/// The token naming `mode` in `ernie:wireframe`: "off", "overlay" or
/// "hiddenLine".
std::string_view WireframeToken(Wireframe mode);

/// The wireframe mode of the RenderSettings prim `settings`
/// (`ernie:wireframe`), `Off` when unset; a bool means `Overlay` when
/// true.
/// @throws std::invalid_argument for a token other than those of
/// `WireframeToken`.
Wireframe WireframeOf(const Scene& scene, const Path& settings);

/// The occlusion culling of the RenderSettings prim `settings`
/// (`ernie:occlusion`), `Occlusion::Off` when unset.
/// @throws std::invalid_argument for a token other than "off",
/// "occluders" or "twoPass".
Occlusion OcclusionOf(const Scene& scene, const Path& settings);

/// The culling of the RenderSettings prim `settings`
/// (`ernie:culling`), `Culling::Gpu` when unset.
/// @throws std::invalid_argument for a token other than "off", "cpu"
/// or "gpu".
Culling CullingOf(const Scene& scene, const Path& settings);

/// The frame parameters for `prims` of `scene`, for an image of
/// `width` x `height`:
/// - the camera from the Camera prim (`CameraParamsOf`) and the
///   overscan its lens distortion needs (`OverscanOf`);
/// - sun direction from the light's transform (`TowardsLight`), its
///   spectral irradiance (`SunIrradianceOf`), angular diameter
///   (`inputs:angle`, default 0.53 degrees) and shadows from its
///   `inputs:shadow:enable`; without a sun, no sunlight and no
///   ambient;
/// - the other lights (`LightParamsOf`, each in its world, `WorldOf`)
///   and the dome (`DomeParamsOf`); the sun and the dome shine in
///   every world;
/// - ambient, wireframe (`WireframeOf`), culling (`CullingOf`,
///   `OcclusionOf`), mesh level-of-detail error, clear colour
///   (linear) and shadow options from the render settings.
/// @throws std::invalid_argument if there is no camera, it is not
/// perspective, or it or the light are invalid.
FrameParams FrameParamsOf(const Scene& scene,
                          const RenderPrims& prims,
                          std::uint32_t width, std::uint32_t height);

/// The settings of the Material prim at `material` (empty for
/// defaults): BRDF from `ernie:brdf`, its parameters from
/// `inputs:<parameter>`, albedo and detail options. A Material whose
/// `outputs:surface` connects to a UsdPreviewSurface takes its albedo
/// from diffuseColor, a constant or a UsdUVTexture
/// (`MaterialSettings::albedoTexture`), and a normal map from normal.
/// @throws std::invalid_argument if the material names a BRDF that is
/// not in `brdfs`, or its shader network is malformed.
MaterialSettings MaterialSettingsOf(const Scene& scene,
                                    const Path& material,
                                    const BrdfRegistry& brdfs);

/// The geometry of the Mesh prim at `mesh`: `points`,
/// `faceVertexCounts` and `faceVertexIndices` (triangulated as fans),
/// per-vertex `normals`, and per-vertex `primvars:displayColor` when
/// it has one colour per point. Points, normals and colours may be
/// float32 or float64, shaped (n, 3) or flat; indices int32, int64 or
/// uint32. Normals and colours of other sizes (such as face-varying
/// ones) are ignored. Points on a CUDA device come back as zeros of
/// the right number, for the caller to copy device to device. Texture
/// coordinates come per triangle corner from `primvars:<set>`
/// (constant, uniform, vertex or faceVarying, optionally indexed; see
/// tokens.h), the set being the one the bound material's textures
/// read, else `st`.
/// @throws std::invalid_argument naming the attribute for arrays of
/// the wrong type, size or device, or invalid topology. For a UsdGeom
/// primitive (Cube, Sphere, Cylinder, Cone, Capsule, Plane) the
/// generated mesh, `PrimitiveMeshOf`.
MeshData MeshDataOf(const Scene& scene, const Path& mesh);

/// Whether prims of `type` draw as meshes: Mesh and the UsdGeom
/// primitives.
bool IsMeshType(std::string_view type);

/// The mesh of the UsdGeom primitive at `path` from its attributes
/// and their UsdGeom defaults: Cube `size` (2); Sphere `radius` (1);
/// Cylinder and Cone `radius` (1), `height` (2) and `axis` ("Z");
/// Capsule `radius` (0.5), `height` (1) and `axis`; Plane `width`
/// (2), `length` (2) and `axis`. See `BoxMesh` and the other
/// generators.
/// @throws std::invalid_argument for another prim type, an unknown
/// axis or invalid sizes.
MeshData PrimitiveMeshOf(const Scene& scene, const Path& path);

/// The per-instance arrays of the PointInstancer prim at `instancer`:
/// `positions`, and optional `orientations` (quaternions as x, y, z,
/// w, imaginary part first as in USD's memory layout) and `scales`.
/// @throws std::invalid_argument naming the attribute for arrays of
/// the wrong type or size.
InstanceData InstanceDataOf(const Scene& scene,
                            const Path& instancer);

/// The instances of each prototype of the PointInstancer at
/// `instancer`: entry k lists the instances whose `protoIndices`
/// entry is k, for `prototypes` prototypes. Without `protoIndices`
/// every instance uses prototype 0; instances with an index out of
/// range are dropped.
/// @throws std::invalid_argument if `protoIndices` is not an integer
/// array.
std::vector<std::vector<std::uint32_t>>
InstancesByPrototype(const Scene& scene, const Path& instancer,
                     std::size_t prototypes);

/// The prim's `extent`: six numbers, low then high corner, as a float
/// array value or a CPU array; none if unset or malformed.
std::optional<Bounds3> ExtentOf(const Scene& scene, const Path& prim);

/// A prim posed on the GPU: its transform ops are rows of GPU arrays,
/// one row per prim of one `Scene::Write` of many paths, so a
/// renderer composes its local transform per instance from the arrays
/// whole, as a PointInstancer's: translate, then orient, then scale.
struct GpuXform {
    bool reset = false;  ///< `!resetXformStack!`: ancestors'
                         ///< transforms do not apply.
    /// Every op in order, the first outermost: its attribute
    /// (`xformOp:translate`...) and its whole array, (n, width)
    /// float32.
    std::vector<std::pair<std::string, Tensor>> ops;
    /// For ops that are translate, orient and scale (each optional,
    /// in that order), the arrays as a PointInstancer's: translations
    /// (n, 3), orientations (n, 4) and scales (n, 3), each empty when
    /// the prim lacks the op. Otherwise `composed`: the ops are
    /// multiplied on the GPU.
    Tensor translate;
    Tensor orient;
    Tensor scale;
    bool composed = false;
    std::int64_t row = 0;  ///< The prim's row of them.
    std::uintptr_t stream =
        1;                ///< The CUDA stream they were written on.
    bool wFirst = false;  ///< `orient` is w, x, y, z
                          ///< (`tokens::kQuaternionOrder`).
};

/// The prim's GPU pose, or none when none of its transform ops is a
/// GPU array.
/// @throws std::invalid_argument when some are but the prim cannot be
/// posed on the GPU: a suffixed or inverted op, an op that is not a
/// row of a GPU array written with the others, an array of the wrong
/// width for its op (3 for translate, scale and Euler angles, or 1
/// for a uniform scale; 4 for orient; 1 for rotateX/Y/Z; 16 for
/// transform), or arrays that are not contiguous float32 of the same
/// number of rows.
std::optional<GpuXform> GpuXformOf(const Scene& scene,
                                   const Path& path);

/// Whether the quaternions in the prim's arrays put w first
/// (`tokens::kQuaternionOrder`).
/// @throws std::invalid_argument for an order other than "xyzw" or
/// "wxyz".
bool QuaternionsWFirst(const Scene& scene, const Path& path);

/// The nearest prim from `path` up whose pose is on the GPU
/// (`GpuXformOf`) and whose transform reaches `path` (no
/// `!resetXformStack!` between), or none.
std::optional<Path> GpuXformAnchorOf(const Scene& scene,
                                     const Path& path);

/// The prim's bounds in its own space: its `extent` if authored, else
/// those of its geometry, a Mesh's CPU points or a UsdGeom
/// primitive's shape (`PrimitiveMeshOf`), or an Atmosphere's planet
/// to the top of its air (`PlanetOf`); none for other prims or
/// geometry that cannot be read.
std::optional<Bounds3> LocalBoundsOf(const Scene& scene,
                                     const Path& prim);

/// A detail texture tensor as an `Image`: float32 values are linear;
/// uint8 values are sRGB-encoded when `encoding` is `Srgb` and linear
/// otherwise. Grey replicates into rgb and a missing alpha reads as
/// 1.
/// @throws std::invalid_argument for a tensor that is not (rows,
/// cols) or (rows, cols, 1, 3 or 4) float32 or uint8 on the CPU.
Image ImageOfTensor(const Tensor& tensor, ImageEncoding encoding);

}  // namespace ernie
