// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/camera.h"
#include "ernie/scene/sensor_filters.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {

/// A pushbroom camera's strip for one frame
/// (`CameraParams::pushbroom`), as its shaders project with it
/// (projection_pushbroom.slang): its rows' poses run from `from` (the
/// top row's camera position) to `to` (the bottom row's), turning
/// from `fromRotation` to `toRotation` along the great arc. Positions
/// are relative to the frame's origin.
struct PushbroomStrip {
    glm::vec3 from{0.0f};
    glm::vec3 to{0.0f};
    glm::quat fromRotation{1.0f, 0.0f, 0.0f,
                           0.0f};  ///< Camera to world.
    glm::quat toRotation{
        1.0f, 0.0f, 0.0f,
        0.0f};  ///< On `fromRotation`'s side of the sphere.
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
    float scale =
        1.0f;  ///< 2 f / width: NDC x per normalised image x.
    float shift = 0.0f;  ///< 2 cx / width - 1.
    float line = 0.0f;   ///< The detector line's normalised image y.

    /// What the shaders read in place of a view-projection: columns
    /// (from, near), (to, far), fromRotation and toRotation as (x, y,
    /// z, w).
    glm::mat4 Packed() const;

    /// (scale, shift, line, 0).
    glm::vec4 Intrinsics() const {
        return {scale, shift, line, 0.0f};
    }
};

/// A pinhole view that holds every ray of pushbroom `camera`'s strip
/// from pose `from` to `to` between its clipping planes, for culling,
/// levels of detail and shadow cascades: its eye at the middle pose
/// pulled back along the view by the strip's length, its field of
/// view the widest the rays reach from there, sampled along the strip
/// and padded.
CameraView PushbroomEnvelope(const CameraParams& camera,
                             const glm::dmat4& from,
                             const glm::dmat4& to);

/// Which casters shadow a batch of views of several worlds
/// (`ernie:shadows:worlds`).
enum class WorldShadows : std::uint32_t {
    /// A map per world of the views: what every world shares and that
    /// world's casters.
    Separate = 0,
    /// One map of what every world shares: worlds' own geometry casts
    /// no shadows. The cheapest.
    Shared = 1,
    /// One map of every world's casters: right when worlds do not
    /// overlap in space, such as a grid of environments.
    Merged = 2,
};

/// What bounds a single view's sun cascades (`ernie:shadows:fit`).
enum class ShadowFit : std::uint32_t {
    Scene = 0,    ///< The casters and where their shadows land;
                  ///< deterministic.
    Visible = 1,  ///< The depths a recent frame saw (sample
                  ///< distribution shadow maps).
};

/// What a world's own shadow map covers (`ernie:shadows:worldFit`).
enum class WorldShadowFit : std::uint32_t {
    Scene = 0,  ///< The whole scene, as every map of the batch.
    /// The world's own casters' bounds grown by
    /// `ShadowSettings::worldMargin`, for sharper shadows in worlds
    /// far apart; receivers outside it are not shadowed.
    World = 1,
};

/// Cascaded shadow map options.
struct ShadowSettings {
    bool enabled = true;
    int cascades = 3;  ///< Number of cascades, 1 to 4.
    std::uint32_t resolution =
        2048;  ///< Texels along each side of every cascade's map.
    float softness =
        1.5f;  ///< Poisson filter radius, in shadow map texels.
    float normalOffset =
        1.5f;  ///< Receiver offset along the normal, in texels.
    float splitLambda =
        0.8f;  ///< Logarithmic weight of the practical split scheme.
    ShadowFit fit =
        ShadowFit::Scene;  ///< What bounds a single view's cascades.
    bool showCascades =
        false;  ///< Tint fragments by cascade, for tuning.
    std::uint32_t lightResolution =
        1024;  ///< Texels along each side of other lights' maps.
    /// Texels along each side of the sun's map of each world in a
    /// batch of views of several worlds, as a batch holds one map per
    /// world; 0 for automatic, `AutoWorldResolution` of the image
    /// size.
    std::uint32_t worldResolution = 0;

    /// The automatic `worldResolution` for images of `width` by
    /// `height`: four texels per pixel along the larger side, as a
    /// power of two from 256 to 2048, so small RL observations of
    /// many worlds do not each clear and render a large map.
    static std::uint32_t AutoWorldResolution(std::uint32_t width,
                                             std::uint32_t height) {
        std::uint32_t texels = 256;
        while (texels < 4 * std::max(width, height) &&
               texels < 2048) {
            texels *= 2;
        }
        return texels;
    }
    WorldShadows worlds =
        WorldShadows::Separate;  ///< Which casters a batch's worlds'
                                 ///< maps hold.
    WorldShadowFit worldFit =
        WorldShadowFit::Scene;  ///< What each world's map covers.
    float worldMargin = 1.0f;   ///< Metres a world-fitted map reaches
                                ///< past the world's casters.
};

/// How the rasterizer draws triangle edges (`ernie:wireframe`).
enum class Wireframe : std::uint32_t {
    Off = 0,      ///< Shaded surfaces only.
    Overlay = 1,  ///< Triangle edges drawn over the shaded surfaces.
    HiddenLine = 2,  ///< Triangle edges on flat, unlit surfaces that
                     ///< hide the edges behind them.
};

/// How the rasterizer culls what lies outside a view
/// (`ernie:culling`).
enum class Culling : std::uint32_t {
    Off = 0,  ///< Draw everything every view sees by world.
    Cpu = 1,  ///< Skip draws (meshes, and each world's run of an
              ///< instancer) outside the frustum.
    Gpu = 2,  ///< As `Cpu`, and test every instance on the GPU,
              ///< drawing only the visible ones.
};

/// Whether the rasterizer also culls instances hidden behind others
/// (`ernie:occlusion`), with GPU culling and for views only (not
/// shadow views).
enum class Occlusion : std::uint32_t {
    Off = 0,  ///< Frustum culling alone.
    /// Each view first draws everything but the GPU-culled instances
    /// (terrain, ground, meshes), builds a depth pyramid of it and
    /// draws only the instances it does not hide: exact, and
    /// independent of earlier frames.
    Occluders = 1,
    /// As `Occluders`, and instances visible in the view's previous
    /// frame draw first too, so instances hide each other; the rest
    /// are tested against the pyramid. Needs coherent frames: a view
    /// that jumps draws everything new in the second pass.
    TwoPass = 2,
};

/// Minimal shading (`ernie:minimal:mode`), after RTX Minimal: for
/// throughput, a product draws in one forward pass straight into its
/// outputs, without the spectral shading, sensor filters, sky or
/// dome; lit by the sun (the first distant light) and the ambient
/// fraction.
///
/// Reference: NVIDIA, "RTX - Minimal Mode", Omniverse Materials and
/// Rendering documentation.
/// https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_minimal.html
enum class MinimalMode : std::uint32_t {
    Off = 0,   ///< The full spectral pipeline.
    None = 1,  ///< No colour (black): only depth, normals and ids; no
               ///< material evaluated.
    Constant = 2,  ///< Every surface lit in one colour
                   ///< (`FrameParams::minimalColor`).
    Textured = 3,  ///< Surfaces lit in their diffuse colour:
                   ///< material, textures, vertex colours.
};

/// Kinds of `LightParams`.
enum class LightType : std::uint32_t {
    Distant = 0,  ///< Parallel light from a direction, such as
                  ///< Earthshine (UsdLux DistantLight).
    Sphere = 1,   ///< A point or sphere, optionally limited to a cone
                  ///< (UsdLux SphereLight with ShapingAPI): lamps and
                  ///< spot lights.
    Rect = 2,     ///< A one-sided Lambertian rectangle (UsdLux
                  ///< RectLight): panels, windows.
    Disk = 3,     ///< A one-sided Lambertian disk (UsdLux DiskLight).
    /// A Lambertian cylinder's side (UsdLux CylinderLight): tubes,
    /// `size.x` long along `tangent` and `radius` wide, shaded
    /// through its outline from the shaded point.
    Cylinder = 4,
};

/// A light besides the sun, resolved for a frame from its prim
/// (`LightParamsOf`). In the rasterizer, distant lights get one
/// orthographic shadow map over the scene; sphere lights a
/// perspective map when their cone is at most `kMaxSpotConeDegrees`,
/// else six (a cube); rect and disk lights a cube at their centre.
///
/// The rasterizer shades rect and disk lights with their exact
/// Lambertian irradiance, the vector form factor of the emitter
/// clipped to the receiver's horizon (Lambert's polygon formula; Arvo
/// 1994), and the receiver's BRDF evaluated along that vector. Disks
/// are integrated as 16-gons of the same area.
///
/// Reference: J. Arvo, "The Irradiance Jacobian for Partially
/// Occluded Polyhedral Sources", SIGGRAPH '94, 343-350, 1994.
/// https://doi.org/10.1145/192161.192250
struct LightParams {
    LightType type = LightType::Sphere;
    glm::dvec3 position{
        0.0};  ///< Sphere, rect and disk: world centre.
    /// Distant: unit vector towards the light. Others: unit axis the
    /// light emits along (its -z): the cone's axis, or the normal of
    /// the emitting face.
    glm::vec3 direction{0.0f, 0.0f, 1.0f};
    /// Rect and disk: unit vector along the emitter's width, the
    /// light's +x; its height runs along the light's +y, `tangent` x
    /// `direction`.
    glm::vec3 tangent{1.0f, 0.0f, 0.0f};
    glm::vec2 size{1.0f, 1.0f};  ///< Rect: width and height.
    /// One value per band of the camera. Distant: spectral irradiance
    /// at normal incidence, W m^-2 nm^-1. Sphere: spectral radiant
    /// intensity, W sr^-1 nm^-1, the same in every direction inside
    /// the cone. Rect and disk: spectral radiance of the emitting
    /// face, W m^-2 sr^-1 nm^-1.
    std::vector<float> spectrum;
    /// Sphere radius (irradiance follows the inverse square law
    /// outside it and is capped at its surface value inside), or disk
    /// radius.
    float radius = 0.0f;
    float coneAngle = 180.0f;  ///< Degrees off the axis that are lit;
                               ///< 180 lights every direction.
    float coneSoftness = 0.0f;  ///< Fraction of the cone, at its
                                ///< edge, over which light fades out.
    /// Not distant (UsdLux's shaping focus): `spectrum` is the
    /// emission along the axis, and off it the emission per band is
    /// spectrum x lerp(focusTint, 1, |cos|^focus), cos of the angle
    /// between the axis and the direction from the light. An empty
    /// tint is black.
    float focus = 0.0f;
    std::vector<float> focusTint;
    /// Not distant, with an IES profile: its candela, `kProfileRows`
    /// rows of `kProfileColumns`, row i at i x 180 / (kProfileRows -
    /// 1) degrees off the axis (its angle scale applied) and column j
    ///    at j x 360 / kProfileColumns degrees around it from the
    ///    light's +x towards +y; the emission per band is spectrum x
    ///    the profile, interpolated, besides cone and focus. Empty
    ///    without one.
    std::vector<float> profile;
    static constexpr int kProfileRows = 129;
    static constexpr int kProfileColumns = 64;
    bool shadows = true;
    /// A direction of a dome (`DomeLightMode::Directions`): it lights
    /// with the model's diffuse response to surrounding light
    /// (`IBrdf::environment`), as the dome's diffuse light would,
    /// while the dome keeps its specular light.
    bool domeDiffuse = false;
    /// The world the light shines in (`tokens::kWorld`): views of a
    /// batch see their world's lights and the shared ones (-1).
    std::int32_t world = -1;
    /// Posed on the GPU under a prim posed there (a lamp on a robot's
    /// link): `position`, `direction` and `tangent` are the CPU's
    /// estimate, and the renderer takes the pose the GPU composes for
    /// the light and its shadow views
    /// (`RasterRenderer::LightPoseBuffer`). Its size, radius and
    /// spectrum keep the estimate's scale.
    bool posedOnGpu = false;

    /// Cones up to this half angle (degrees) get one perspective
    /// shadow map, wider ones a cube.
    static constexpr float kMaxSpotConeDegrees = 75.0f;
};

/// An environment light at infinity (UsdLux DomeLight) resolved for a
/// frame (`DomeParamsOf`): radiance from every direction, the
/// spectrum `radiance` (W m^-2 sr^-1 nm^-1 per camera band) times a
/// relative RGB map whose cosine-convolved spherical harmonics are
/// `sh`. It lights surfaces diffusely, with the BRDF's albedo, and is
/// not occluded.
///
/// Reference: R. Ramamoorthi and P. Hanrahan, "An Efficient
/// Representation for Irradiance Environment Maps", SIGGRAPH 2001,
/// 497-500. https://doi.org/10.1145/383259.383317
struct DomeParams {
    std::vector<float> radiance;
    /// Irradiance over pi of the RGB map as order-2 spherical
    /// harmonics in world space (`DomeIrradianceSh`): a uniform map
    /// of 1 has sh[0] = sqrt(4 pi) and zeros elsewhere.
    std::array<glm::vec3, 9> sh{};
    /// Whether camera rays that hit nothing see the dome
    /// (`tokens::kVisibleInPrimaryRay`): its radiance times its map
    /// (`DomeMapOf`) in their direction.
    bool visible = true;
    /// The rotation from world directions to the dome's frame, in
    /// which its map lies.
    glm::mat3 toDome{1.0f};
    /// Whether it may be shadowed (`tokens::kShadowEnable`), in
    /// `DomeLightMode::Full`.
    bool shadow = true;
};

/// How dome lights light the scene (`tokens::kDomeLightMode`), as
/// Omniverse RTX's modes.
enum class DomeLightMode {
    Approximate,  ///< Diffuse and specular, unoccluded.
    Full,     ///< Also shadowed: ambient occlusion from the G-buffer.
    Limited,  ///< Reflections only: no diffuse dome light.
    /// Full, with the diffuse dome light from shadowed directions
    /// (`DomeDirectionsOf`).
    Directions,
};

/// One star of a `StarField`, as renders take it: where it is and the
/// black-body spectrum it shines with.
struct Star {
    float rightAscension =
        0.0f;                  ///< Radians, in the catalogue's frame.
    float declination = 0.0f;  ///< Radians.
    float temperature = 5772.0f;  ///< Kelvin.
    /// Spectral irradiance at 555 nm, W m^-2 nm^-1; at other
    /// wavelengths, this times the black body's spectral radiance
    /// there over its value at 555 nm.
    float irradiance555 = 0.0f;
};

/// The stars of a StarField prim (`StarsOf`) are uploaded when they
/// change; each frame takes only where the catalogue's frame lies
/// (`StarFieldParamsOf`).
struct StarFieldParams {
    /// The rotation from world directions to the catalogue's frame.
    glm::mat3 toCatalogue{1.0f};
};

/// Per-frame parameters of a render, whatever the backend: the
/// camera, the sun, the other lights, the sensor filters and the
/// frame's place in the random streams. `shadows` tunes the
/// rasterizer's shadow maps; tracers ignore it.
struct FrameParams {
    /// The camera: pose, projection, spectral bands, exposure, lens
    /// and sensor. Its `width` and `height` are the image size, which
    /// the render targets must have.
    CameraParams camera;
    /// How much wider than the camera's the raster view's field of
    /// view is, so that with lens distortion it covers the
    /// undistorted rays of every sensor pixel (`OverscanOf`).
    double overscan = 1.0;
    bool sun = true;  ///< Off lights the scene with `lights` alone
                      ///< (no sun, ambient or cascades).
    glm::vec3 sunDirection{
        0.0f, 0.0f, 1.0f};  ///< Unit vector pointing towards the sun.
    /// Angular diameter of the sun's disk in degrees, which widens
    /// shadow penumbrae with the distance from the occluder
    /// (percentage-closer soft shadows); 0 keeps `shadows.softness`.
    float sunAngle = 0.53f;
    /// The sun's spectral irradiance at normal incidence per band of
    /// `camera.bands`, in W m^-2 nm^-1. Empty means a flat spectrum
    /// scaled so a white Lambertian surface facing the sun has a
    /// luminance of 1 cd/m^2 (relative units).
    std::vector<float> sunIrradiance;
    float ambient = 0.15f;  ///< Fraction of sunlight reaching
                            ///< surfaces without shading.
    /// Other lights; the rasterizer takes
    /// `RasterRenderer::kMaxLights`.
    std::vector<LightParams> lights;
    std::optional<DomeParams> dome;
    DomeLightMode domeMode = DomeLightMode::Approximate;
    /// World distance within which geometry shadows a dome in
    /// `DomeLightMode::Full`.
    float domeShadowRadius = 1.0f;
    /// How many shadowed directions light the dome's diffuse light in
    /// `DomeLightMode::Directions` (`tokens::kDomeDirections`), and
    /// whether `lights` holds them this frame, so the dome's own
    /// diffuse light stays off.
    std::uint32_t domeDirections = 16;
    bool domeDirectionsLit = false;
    std::optional<StarFieldParams>
        stars;  ///< Stars behind everything, as last uploaded.
    float exposure =
        1.0f;  ///< Extra scale on the photographic output.
    bool meter =
        false;  ///< Measure the scene's luminance (`tokens::kMeter`).
    Culling culling =
        Culling::Gpu;  ///< How draws outside a view are culled.
    Occlusion occlusion =
        Occlusion::Off;  ///< Whether views cull instances hidden by
                         ///< others.
    /// The simplification error, in pixels, up to which instanced
    /// meshes draw a coarser level of detail with GPU culling; 0
    /// draws the full meshes.
    float meshLodError = 1.0f;
    Wireframe wireframe =
        Wireframe::Off;  ///< How triangle edges are drawn.
    ShadowSettings shadows;
    /// The sensor filters to run, with their inputs
    /// (`CheckFilterChain`); empty runs `DefaultFilterChain`.
    std::vector<SensorFilterStep> filters;
    /// The render vars wanted from the frame; empty for all. Only
    /// they are held, and filters that produce none of them do not
    /// run (the sensor fuses the rest, `PlanSensor`).
    std::vector<std::string_view> vars;
    /// Several sensor chains over the one raster image, each writing
    /// its own vars (render vars with their own `ernie:filters`);
    /// empty runs `filters` for `vars`. Each var comes from one
    /// chain; each chain's AOV slots start at its `aovOffset`.
    struct SensorChain {
        std::vector<SensorFilterStep> filters;
        std::vector<std::string_view> vars;
        std::uint32_t aovOffset = 0;
    };
    std::vector<SensorChain> chains;
    /// The overlays drawn over LdrColor (`tokens::kOverlays`): "*",
    /// names, or empty for none.
    std::string overlays;
    /// Minimal shading (`MinimalMode`), and its constant colour
    /// (linear RGB).
    MinimalMode minimal = MinimalMode::Off;
    glm::vec3 minimalColor{1.0f};
    glm::vec3 clearColor{0.0f};  ///< Linear display colour of the
                                 ///< background in LdrColor.
    /// The frame number temporal noise derives from (`FrameSeed`):
    /// rendering the same frame again gives the same image, and
    /// consecutive frames independent noise.
    std::uint64_t frame = 0;
    /// Decorrelates images rendered with the same seed and frame,
    /// such as the products of one step: `StreamOf` the render
    /// product's path.
    std::uint32_t stream = 0;

    /// Camera-to-world poses of a batch of views that share
    /// `camera`'s lens and sensor, one image each, such as one camera
    /// per environment in reinforcement learning; empty for the
    /// single view of `camera.transform`.
    std::vector<glm::dmat4> poses;

    /// Whether the views' poses come from the GPU (`ernie:poses`
    /// written as a CUDA array and copied device to device), each the
    /// camera's parent world transform `posesParent` after its pose;
    /// `poses` are then only estimates (the last such frame's, or the
    /// camera's own transform) for work the CPU plans, such as
    /// terrain levels of detail, and CPU frustum tests of the views
    /// are skipped.
    bool posesOnGpu = false;
    /// With `posesOnGpu`: the GPU poses are rows of seven numbers, a
    /// position and a unit quaternion (x, y, z, qx, qy, qz, qw),
    /// rather than row-major 4 x 4 matrices.
    bool posesQuaternions = false;
    glm::dmat4 posesParent{1.0};

    /// The views' camera-to-world poses at the last frame (time -1),
    /// one per view, for motion blur; empty when there is no last
    /// frame, which holds the camera still.
    std::vector<glm::dmat4> previousPoses;

    /// The world each view sees (`tokens::kWorld`): its own and the
    /// shared one, or every world for -1; empty for every view seeing
    /// every world.
    std::vector<std::int32_t> worlds;

    /// The world view `view` sees, -1 for every world.
    std::int32_t ViewWorld(std::size_t view) const {
        return view < worlds.size() ? worlds[view] : -1;
    }

    /// The number of views: `poses`' or 1.
    std::size_t Views() const {
        return poses.empty() ? 1 : poses.size();
    }

    /// The camera of view `view`: `camera` at that view's pose.
    CameraParams ViewCamera(std::size_t view) const {
        CameraParams result = camera;
        if (!poses.empty()) {
            result.transform = poses.at(view);
        }
        return result;
    }

    /// The world position the GPU's coordinates are relative to
    /// (camera-relative rendering, a floating origin): passes see
    /// world points as `point - origin`, so float precision follows
    /// the camera instead of the world's origin, as planet-scale
    /// scenes need. Chosen near the camera and moved only when the
    /// camera gets far from it (`RasterDelegate`), so what is
    /// expressed relative to it is recomputed rarely.
    ///
    /// References:
    /// - C. Thorne, "Using a Floating Origin to Improve Fidelity and
    ///   Performance of Large, Distributed Virtual Worlds", CW 2005.
    ///   https://doi.org/10.1109/CW.2005.94
    /// - P. Cozzi and K. Ring, "3D Engine Design for Virtual Globes",
    ///   CRC Press, 2011, chapter 5 (rendering relative to center and
    ///   to eye).
    glm::dvec3 origin{0.0};

    /// World transform `world` relative to `origin`, in float for the
    /// GPU.
    glm::mat4 Relative(const glm::dmat4& world) const {
        glm::dmat4 relative = world;
        relative[3] -= glm::dvec4(origin, 0.0);
        return glm::mat4(relative);
    }

    /// World point `world` relative to `origin`.
    glm::vec3 Relative(const glm::dvec3& world) const {
        return glm::vec3(world - origin);
    }

    /// The raster view the passes draw view `view` with,
    /// `ViewCamera(view).RasterView(overscan)` relative to `origin`
    /// and without copying the camera. A pushbroom's is a pinhole
    /// view around its whole strip (`PushbroomEnvelope`), for what
    /// the CPU plans: culling, levels of detail and shadow cascades.
    CameraView View(std::size_t view = 0) const {
        glm::dmat4 pose =
            poses.empty() ? camera.transform : poses.at(view);
        pose[3] -= glm::dvec4(origin, 0.0);
        if (camera.pushbroom) {
            return PushbroomEnvelope(camera, PreviousPose(view),
                                     pose);
        }
        return camera.RasterView(pose, overscan);
    }

    /// View `view`'s pose at the last frame relative to `origin`: its
    /// pose now without one.
    glm::dmat4 PreviousPose(std::size_t view = 0) const;

    /// A pushbroom camera's strip of view `view`, from its last
    /// frame's pose to this one's.
    PushbroomStrip Strip(std::size_t view = 0) const;

    /// The frame as its sub-frames see it, for motion blur
    /// (`CameraParams::Subframes`): a view per sub-frame and view,
    /// sub-frame `k`'s from `k * Views()` on, each posed at
    /// `camera.SubframeTime(k)` between `previousPoses` and its pose
    /// (`InterpolatePose`) and seeing its view's world. A copy for a
    /// frame of one sub-frame.
    /// @throws std::invalid_argument for `previousPoses` of another
    /// view count.
    FrameParams Subframes() const;

    /// The frame as the `faces` cube faces of a wide camera see it
    /// (wide_view.h): a view per view and face, view `v`'s face `f`
    /// at `v * faces + f`, posed at the view's pose turned onto the
    /// face (`CubeFaceAxes`, its previous pose too) and seeing the
    /// view's world, and `camera` a centred pinhole of the same image
    /// whose square of 90 degrees plus a margin (`CubeFaceExtent`)
    /// spans the image's shorter side. The raster passes draw these;
    /// the sensor follows the original camera's lens rays onto them.
    /// A copy for one face. With the poses on the GPU, `poses` are
    /// their estimates, and the GPU turns the poses themselves
    /// (views.slang).
    /// @throws std::invalid_argument for faces other than 1,
    /// `kCubeFaces` and `kAllCubeFaces`.
    FrameParams Faces(int faces) const;

    /// The tangent-plane focal length, in raster pixels, and the
    /// principal point of a cube face of a camera of `width` x
    /// `height` pixels (`Faces`): the face's image-plane coordinates
    /// (x, y) land at (cx + f x, cy - f y).
    static glm::dvec3 FaceIntrinsics(std::uint32_t width,
                                     std::uint32_t height);
};

/// Camera-to-world pose `s` of the way from `from` (0) to `to` (1),
/// extrapolating outside: the rotation along the great arc between
/// them (slerp), the position and scale along the line.
///
/// Reference: K. Shoemake, "Animating Rotation with Quaternion
/// Curves", SIGGRAPH 1985, 245-254.
/// https://doi.org/10.1145/325165.325242
glm::dmat4 InterpolatePose(const glm::dmat4& from,
                           const glm::dmat4& to, double s);

/// Axis-aligned world bounds, low and high corners.
using Bounds3 = std::pair<glm::vec3, glm::vec3>;

}  // namespace ernie
