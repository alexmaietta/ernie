#pragma once

#include <string_view>

/// Prim type and attribute names, following UsdGeom, UsdLux, UsdShade and UsdRender where those
/// schemas have the concept, with Ernie's own under `ernie:`.
///
/// References:
/// - Pixar, OpenUSD schema domains, https://openusd.org/release/api/index.html
/// - NVIDIA, Omniverse lighting, photometric units,
///   https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
namespace ernie::tokens {

/// Root of the prims interactive tools define for their own use, such as the GUI viewer's cameras
/// and lights. Render delegates skip it when falling back to the scene's first RenderSettings or
/// Camera and when gathering the scene's lights.
inline constexpr std::string_view kSessionRoot = "/__session";

// Prim types.
inline constexpr std::string_view kXform = "Xform";
inline constexpr std::string_view kScope = "Scope";
inline constexpr std::string_view kMesh = "Mesh";
inline constexpr std::string_view kCube = "Cube";
inline constexpr std::string_view kSphere = "Sphere";
inline constexpr std::string_view kCylinder = "Cylinder";
inline constexpr std::string_view kCone = "Cone";
inline constexpr std::string_view kCapsule = "Capsule";
inline constexpr std::string_view kPlane = "Plane";
inline constexpr std::string_view kPointInstancer = "PointInstancer";
inline constexpr std::string_view kVolume = "Volume";
inline constexpr std::string_view kCamera = "Camera";
inline constexpr std::string_view kDistantLight = "DistantLight";
inline constexpr std::string_view kSphereLight = "SphereLight";
inline constexpr std::string_view kRectLight = "RectLight";
inline constexpr std::string_view kDiskLight = "DiskLight";
inline constexpr std::string_view kDomeLight = "DomeLight";
inline constexpr std::string_view kMaterial = "Material";
inline constexpr std::string_view kTerrain = "Terrain";
inline constexpr std::string_view kGroundPlane = "GroundPlane";  ///< Ernie: an infinite floor.
inline constexpr std::string_view kRenderSettings = "RenderSettings";
inline constexpr std::string_view kRenderProduct = "RenderProduct";
inline constexpr std::string_view kRenderVar = "RenderVar";

// Imageable and Xformable (UsdGeom).
inline constexpr std::string_view kTransform = "xformOp:transform";  ///< dmat4, column vectors.
inline constexpr std::string_view kVisibility = "visibility";        ///< kInherited or kInvisible.
inline constexpr std::string_view kInherited = "inherited";
inline constexpr std::string_view kInvisible = "invisible";

// Geometry (UsdGeom Mesh, PointInstancer).
inline constexpr std::string_view kPoints = "points";
inline constexpr std::string_view kNormals = "normals";
inline constexpr std::string_view kFaceVertexCounts = "faceVertexCounts";
inline constexpr std::string_view kFaceVertexIndices = "faceVertexIndices";
inline constexpr std::string_view kDisplayColor = "primvars:displayColor";
inline constexpr std::string_view kPositions = "positions";
inline constexpr std::string_view kOrientations = "orientations";
inline constexpr std::string_view kScales = "scales";
inline constexpr std::string_view kVelocities = "velocities";
/// Bounds in the prim's own frame (UsdGeomBoundable): a float array of low then high x, y, z.
inline constexpr std::string_view kExtent = "extent";
inline constexpr std::string_view kProtoIndices = "protoIndices";
inline constexpr std::string_view kPrototypes = "prototypes";

// Implicit primitives (UsdGeom Cube, Sphere, Cylinder, Cone, Capsule, Plane), drawn as meshes
// (`ernie::BoxMesh` and the other generators).
inline constexpr std::string_view kSize = "size";  ///< Cube edge length, default 2.
inline constexpr std::string_view kGeomRadius = "radius";
inline constexpr std::string_view kGeomHeight = "height";
inline constexpr std::string_view kGeomWidth = "width";    ///< Plane, default 2.
inline constexpr std::string_view kGeomLength = "length";  ///< Plane, default 2.
inline constexpr std::string_view kAxis = "axis";          ///< "X", "Y" or "Z" (default).

// Material binding (UsdShade MaterialBindingAPI): a Path.
inline constexpr std::string_view kMaterialBinding = "material:binding";

// Camera (UsdGeom Camera). Focal length and apertures share a unit, conventionally millimetres;
// only their ratio sets the field of view.
inline constexpr std::string_view kFocalLength = "focalLength";
inline constexpr std::string_view kHorizontalAperture = "horizontalAperture";
inline constexpr std::string_view kVerticalAperture = "verticalAperture";
inline constexpr std::string_view kClippingRange = "clippingRange";  ///< vec2 near, far.
inline constexpr std::string_view kProjection = "projection";  ///< kPerspective or kOrthographic.
inline constexpr std::string_view kPerspective = "perspective";
inline constexpr std::string_view kOrthographic = "orthographic";
inline constexpr std::string_view kCameraExposure = "exposure";  ///< Stops; scales by 2^exposure.
inline constexpr std::string_view kHorizontalApertureOffset = "horizontalApertureOffset";
inline constexpr std::string_view kVerticalApertureOffset = "verticalApertureOffset";
inline constexpr std::string_view kExposureTime = "exposure:time";  ///< Seconds.
inline constexpr std::string_view kExposureIso = "exposure:iso";
inline constexpr std::string_view kExposureFStop = "exposure:fStop";
inline constexpr std::string_view kExposureResponsivity = "exposure:responsivity";

// Physical camera (Ernie): spectral sampling, lens and sensor. Spectra are float arrays of
// interleaved (wavelength in nm, value) pairs.
inline constexpr std::string_view kSpectralRange = "ernie:spectral:range";    ///< vec2 nm.
inline constexpr std::string_view kSpectralBands = "ernie:spectral:bands";    ///< int, 1 to 64.
inline constexpr std::string_view kLensDistortion = "ernie:lens:distortion";  ///< k1 k2 k3 p1 p2.
inline constexpr std::string_view kLensVignetting = "ernie:lens:vignetting";  ///< bool.
inline constexpr std::string_view kLensTransmission = "ernie:lens:transmission";  ///< fraction.
inline constexpr std::string_view kSensorType = "ernie:sensor:type";  ///< "rgb" or "mono".
/// Quantum efficiency spectrum of one channel: append the channel name ("r", "g", "b", "mono").
inline constexpr std::string_view kSensorQePrefix = "ernie:sensor:qe:";
inline constexpr std::string_view kSensorFullWell = "ernie:sensor:fullWell";        ///< electrons.
inline constexpr std::string_view kSensorReadNoise = "ernie:sensor:readNoise";      ///< electrons.
inline constexpr std::string_view kSensorDarkCurrent = "ernie:sensor:darkCurrent";  ///< e-/s.
inline constexpr std::string_view kSensorPrnu = "ernie:sensor:prnu";                ///< fraction.
inline constexpr std::string_view kSensorDsnu = "ernie:sensor:dsnu";                ///< electrons.
inline constexpr std::string_view kSensorGain = "ernie:sensor:electronsPerDn";      ///< at ISO 100.
inline constexpr std::string_view kSensorAdcBits = "ernie:sensor:adcBits";
inline constexpr std::string_view kSensorBlackLevel = "ernie:sensor:blackLevel";  ///< DN.
inline constexpr std::string_view kSensorNoise = "ernie:sensor:noise";            ///< bool.
inline constexpr std::string_view kSensorSeed = "ernie:sensor:seed";              ///< int.

// Lights (UsdLux). A DistantLight shines along the -z axis of its transform, and a SphereLight's
// cone (ShapingAPI) is around its -z axis.
inline constexpr std::string_view kIntensity = "inputs:intensity";
inline constexpr std::string_view kExposure = "inputs:exposure";  ///< Stops; scales by 2^exposure.
inline constexpr std::string_view kColor = "inputs:color";
inline constexpr std::string_view kAngle = "inputs:angle";  ///< Degrees subtended.
inline constexpr std::string_view kRadius = "inputs:radius";
inline constexpr std::string_view kTreatAsPoint = "treatAsPoint";  ///< bool: radius 0.
/// Degrees off the axis that a light reaches, and the fraction of that angle over which it fades.
inline constexpr std::string_view kShapingConeAngle = "inputs:shaping:cone:angle";
inline constexpr std::string_view kShapingConeSoftness = "inputs:shaping:cone:softness";
inline constexpr std::string_view kWidth = "inputs:width";
inline constexpr std::string_view kHeight = "inputs:height";
inline constexpr std::string_view kTextureFile = "inputs:texture:file";
inline constexpr std::string_view kShadowEnable = "inputs:shadow:enable";
inline constexpr std::string_view kColorTemperature = "inputs:colorTemperature";  ///< Kelvin.
inline constexpr std::string_view kEnableColorTemperature = "inputs:enableColorTemperature";
/// Physical sunlight (Ernie): total irradiance at normal incidence in W/m^2, and a spectrum of
/// (wavelength nm, W m^-2 nm^-1) pairs. Either one makes the light physical; without a spectrum
/// it is a black body (the colour temperature when enabled, else the Sun's 5772 K) and without an
/// irradiance a black body carries the solar constant, 1361 W/m^2.
inline constexpr std::string_view kLightIrradiance = "ernie:irradiance";
inline constexpr std::string_view kLightSpectrum = "ernie:spectrum";
/// Physical lamp (Ernie): radiant flux over 360 to 830 nm in W. A SphereLight emits it equally in
/// every direction (a cone only masks it); a RectLight or DiskLight from its face as a Lambertian
/// emitter. Its spectrum's shape is `ernie:spectrum`, else a black body at the enabled colour
/// temperature, else flat; `ernie:spectrum` alone gives the radiant intensity in W sr^-1 nm^-1
/// (sphere) or radiance in W m^-2 sr^-1 nm^-1 (rect, disk, dome).
inline constexpr std::string_view kLightPower = "ernie:power";
/// A DomeLight's map (Ernie): float32 (rows, cols, 3 or 4) linear RGB scaling its radiance, in
/// latitude-longitude layout around the dome's +z axis: rows run from +z (top) to -z, columns
/// from +x through +y. (USD's `inputs:texture:file` maps are not read.)
inline constexpr std::string_view kLightTexture = "ernie:texture";

// Physical lighting (Omniverse's PhysicalLightIlluminantAPI and Photometric*LightAPI): the shape of
// a light's spectrum, and its brightness in photometric units. Authoring any `photometric:` value
// puts the light in photometric units: its brightness then comes from them alone, not from
// `inputs:intensity`, `inputs:exposure`, `ernie:power` or `ernie:irradiance`.
//
/// token: the shape of the light's spectrum: "white" (flat, Ernie's neutral; the default),
/// "blackbody" or "illuminant D" (CIE daylight) at `inputs:colorTemperature`, or "custom"
/// (`physical:customIlluminant`). `inputs:color` still tints it. Unset, the shape is a black body
/// when `inputs:enableColorTemperature` is on, else flat.
inline constexpr std::string_view kPhysicalIlluminant = "physical:illuminant";
inline constexpr std::string_view kWhite = "white";
inline constexpr std::string_view kBlackbody = "blackbody";
inline constexpr std::string_view kIlluminantD = "illuminant D";
inline constexpr std::string_view kCustom = "custom";
/// float array: (nm, value) pairs, wavelengths increasing, of a "custom" illuminant's shape; zero
/// outside them.
inline constexpr std::string_view kPhysicalCustomIlluminant = "physical:customIlluminant";
/// float, lux: for a DistantLight, on a surface facing it (default 10000); for a DomeLight, on an
/// upward-facing surface (default 10000); for an area light, on a surface facing it at
/// `photometric:illuminance:distance` metres, when that is not 0.
inline constexpr std::string_view kPhotometricIlluminance = "photometric:illuminance";
inline constexpr std::string_view kPhotometricIlluminanceDistance =
    "photometric:illuminance:distance";
/// float, lumens: an area light's (SphereLight, RectLight, DiskLight) luminous flux, default
/// 1600; a SphereLight emits it in every direction, a RectLight or DiskLight from its face.
inline constexpr std::string_view kPhotometricPower = "photometric:power";

// Terrain (Ernie): a north-up heightfield, x east, y north, z up.
inline constexpr std::string_view kHeights = "heights";          ///< float32 tensor (rows, cols).
inline constexpr std::string_view kOrigin = "ernie:origin";      ///< dvec2 north-west corner.
inline constexpr std::string_view kCellSize = "ernie:cellSize";  ///< double, world units.
inline constexpr std::string_view kNoData = "ernie:noData";      ///< float marking missing cells.
inline constexpr std::string_view kExaggeration = "ernie:exaggeration";  ///< float.

// Ground plane (Ernie): an infinite horizontal plane facing +z, drawn with its material binding.
inline constexpr std::string_view kGroundHeight = "ernie:height";       ///< World z, default 0.
inline constexpr std::string_view kChecker = "ernie:checker";           ///< bool, default off.
inline constexpr std::string_view kCheckerSize = "ernie:checker:size";  ///< Square size, default 1.
/// Linear colours of the checker squares, multiplying the material's albedo; default 1 and 0.5.
inline constexpr std::string_view kCheckerColor0 = "ernie:checker:color0";
inline constexpr std::string_view kCheckerColor1 = "ernie:checker:color1";

// Material (Ernie surface). BRDF parameters are `inputs:<parameter id>`.
inline constexpr std::string_view kBrdf = "ernie:brdf";             ///< BRDF model id.
inline constexpr std::string_view kAlbedo = "inputs:albedo";        ///< vec3 linear albedo.
inline constexpr std::string_view kAlbedoMap = "inputs:albedoMap";  ///< float32 (rows, cols).
inline constexpr std::string_view kAlbedoMapOrigin = "inputs:albedoMap:origin";
inline constexpr std::string_view kAlbedoMapCellSize = "inputs:albedoMap:cellSize";
inline constexpr std::string_view kAlbedoMapNoData = "inputs:albedoMap:noData";
inline constexpr std::string_view kAlbedoMapEnable = "inputs:albedoMap:enable";  ///< bool.
/// Detail textures: float32 (rows, cols, channels) linear, or uint8 with sRGB albedo and linear
/// normals; 1, 3 or 4 channels.
inline constexpr std::string_view kDetailAlbedo = "inputs:detailAlbedo";
inline constexpr std::string_view kDetailNormal = "inputs:detailNormal";
inline constexpr std::string_view kDetailTileSize = "inputs:detail:tileSize";
inline constexpr std::string_view kDetailAlbedoStrength = "inputs:detail:albedoStrength";
inline constexpr std::string_view kDetailNormalStrength = "inputs:detail:normalStrength";
inline constexpr std::string_view kDetailDirectXNormals = "inputs:detail:directXNormals";
inline constexpr std::string_view kDetailHexTiling = "inputs:detail:hexTiling";
inline constexpr std::string_view kDetailHexRotation = "inputs:detail:hexRotation";
inline constexpr std::string_view kDetailHexContrast = "inputs:detail:hexContrast";
inline constexpr std::string_view kDetailHexSharpness = "inputs:detail:hexSharpness";

// Render settings (UsdRender), with Ernie's rasterizer options under `ernie:`.
inline constexpr std::string_view kCameraRel = "camera";       ///< Path of the camera.
inline constexpr std::string_view kResolution = "resolution";  ///< ivec2 pixels.
/// A Camera's batch of view poses (Ernie): float32 or float64 (views, 4, 4), each a local
/// transform in the math convention (column vectors, translation in the last column) that
/// replaces the camera's own. A render product through the camera renders one image per pose.
inline constexpr std::string_view kPoses = "ernie:poses";
/// token: on a Camera with `kPoses` written as a CUDA array (`Scene::Write`), where the views'
/// poses go: "gpu" (the default) copies them device to device and the views' matrices are
/// computed on the GPU, with no wait for the host; "host" downloads them first, so every CPU
/// stage (frustum tests, terrain levels of detail, single-view shadow cascades) sees the exact
/// poses. Only contiguous float32 arrays stay on the GPU; others download.
inline constexpr std::string_view kPosesDevice = "ernie:poses:device";
inline constexpr std::string_view kHost = "host";
/// The world a prim and its descendants belong to (Ernie): int, inherited from the nearest
/// ancestor that has one. A batch of views renders environments ("worlds") that may overlap in
/// space, such as one per reinforcement learning environment, each view seeing the prims of its
/// own world and those of no world (-1, the default), which every world shares. On a Camera, the
/// world its views see; -1 sees every world.
inline constexpr std::string_view kWorld = "ernie:world";
/// Per-element worlds (Ernie): int array. On a PointInstancer, each instance's world (in place of
/// the instancer's `ernie:world`); on a Camera with `ernie:poses`, each view's world.
inline constexpr std::string_view kWorlds = "ernie:worlds";
/// int: the frame number noise and samples derive from (Ernie); unset counts the images of each
/// render product a delegate renders.
inline constexpr std::string_view kFrame = "ernie:frame";
/// Paths of the lights to render with (Ernie); unset renders with every visible light outside
/// `kSessionRoot`.
inline constexpr std::string_view kLightsRel = "ernie:lights";
inline constexpr std::string_view kProducts = "products";        ///< Paths of RenderProducts.
inline constexpr std::string_view kOrderedVars = "orderedVars";  ///< Paths of RenderVars.
/// Sensor filter chain of a RenderProduct (Ernie): paths of SensorFilter prims in order.
inline constexpr std::string_view kFilters = "ernie:filters";
inline constexpr std::string_view kSensorFilter = "SensorFilter";  ///< Prim type.
/// A SensorFilter prim's filter: "lens", "qe", "noise", "adc", "cie", "exposure", or the id of a
/// custom filter (`SensorFilterModel`), whose parameters are the prim's inputs:* attributes.
inline constexpr std::string_view kFilterType = "ernie:filter:type";
inline constexpr std::string_view kSourceName = "sourceName";  ///< RenderVar name, e.g. "LdrColor".
inline constexpr std::string_view kAmbient = "ernie:ambient";  ///< float fraction of sunlight.
inline constexpr std::string_view kColorMode = "ernie:colorMode";
inline constexpr std::string_view kHillshade = "hillshade";
inline constexpr std::string_view kColormap = "colormap";
inline constexpr std::string_view kShadedColormap = "shadedColormap";
/// token: "off", "overlay" (edges over shaded surfaces) or "hiddenLine" (edges on flat, unlit
/// surfaces); a bool means "overlay" when true and "off" when false.
inline constexpr std::string_view kWireframe = "ernie:wireframe";
inline constexpr std::string_view kOff = "off";
inline constexpr std::string_view kOverlay = "overlay";
inline constexpr std::string_view kHiddenLine = "hiddenLine";
/// string, on a RenderProduct or else its RenderSettings: the overlays (`Scene::GetOverlay`) drawn
/// over its LdrColor, "*" for all, else names separated by commas or spaces; none by default, so
/// observations stay free of debug drawing.
inline constexpr std::string_view kOverlays = "ernie:overlays";
/// token: how the rasterizer culls what lies outside a view: "off", "cpu" (whole draws) or
/// "gpu" (whole draws, then every instance on the GPU; the default).
inline constexpr std::string_view kCulling = "ernie:culling";
inline constexpr std::string_view kCpu = "cpu";
inline constexpr std::string_view kGpu = "gpu";
/// token: whether views also cull instances hidden behind others with GPU culling: "off" (the
/// default), "occluders" (behind terrain, ground and meshes) or "twoPass" (also behind instances
/// visible in the view's previous frame).
inline constexpr std::string_view kOcclusion = "ernie:occlusion";
inline constexpr std::string_view kOccluders = "occluders";
inline constexpr std::string_view kTwoPass = "twoPass";
inline constexpr std::string_view kClearColor = "ernie:clearColor";                ///< vec3 linear.
inline constexpr std::string_view kShadowCascades = "ernie:shadows:cascades";      ///< int.
inline constexpr std::string_view kShadowResolution = "ernie:shadows:resolution";  ///< int.
/// int: texels along each side of the sun's map of each world in a batch of views; 0 or unset for
/// automatic, four per pixel of the image's larger side from 256 to 2048.
inline constexpr std::string_view kShadowWorldResolution = "ernie:shadows:worldResolution";
/// token: which casters shadow a batch of views of several worlds: "separate" (a map per world,
/// the default), "shared" (one map of what the worlds share) or "merged" (one map of every
/// world's casters).
inline constexpr std::string_view kShadowWorlds = "ernie:shadows:worlds";
inline constexpr std::string_view kSeparate = "separate";
inline constexpr std::string_view kShared = "shared";
inline constexpr std::string_view kMerged = "merged";
/// token: what each world's own map covers: "scene" (the default) or "world" (the world's casters
/// grown by `ernie:shadows:worldMargin` metres, default 1).
inline constexpr std::string_view kShadowWorldFit = "ernie:shadows:worldFit";
inline constexpr std::string_view kScene = "scene";
inline constexpr std::string_view kWorldFit = "world";
inline constexpr std::string_view kShadowWorldMargin = "ernie:shadows:worldMargin";
/// token: what bounds a single view's sun cascades: "scene" (the default: the casters and where
/// their shadows land, the same every time a frame renders) or "visible" (sample distribution
/// shadow maps: the depths a recent frame of the same product saw, tighter and sharper, but an
/// image then depends on the frames before it; the viewer's choice).
inline constexpr std::string_view kShadowFit = "ernie:shadows:fit";
/// float, stops: exposure compensation on the photographic outputs (HdrColor's display, LdrColor),
/// on top of the camera's own exposure; default 0.
inline constexpr std::string_view kExposureCompensation = "ernie:exposure";
/// bool: measure each frame's scene luminance (its histogram) for automatic exposure and exposure
/// meters, read back from a later frame (`RenderDelegate::MeteredLuminance`); default off.
inline constexpr std::string_view kMeter = "ernie:meter";
inline constexpr std::string_view kVisible = "visible";
inline constexpr std::string_view kShadowSoftness = "ernie:shadows:softness";
inline constexpr std::string_view kShadowNormalOffset = "ernie:shadows:normalOffset";
inline constexpr std::string_view kShadowSplitLambda = "ernie:shadows:splitLambda";
inline constexpr std::string_view kShadowShowCascades = "ernie:shadows:showCascades";
inline constexpr std::string_view kLodEnable = "ernie:lod:enable";
inline constexpr std::string_view kLodDetail = "ernie:lod:detail";
inline constexpr std::string_view kLodMorphStart = "ernie:lod:morphStart";
inline constexpr std::string_view kLodFreeze = "ernie:lod:freeze";
inline constexpr std::string_view kLodShowLevels = "ernie:lod:showLevels";
/// float: the simplification error, in pixels, up to which instanced meshes draw a coarser level
/// of detail (with GPU culling); 0 always draws the full meshes. Default 1.
inline constexpr std::string_view kLodMeshError = "ernie:lod:meshError";

/// RenderVar source names.
inline constexpr std::string_view kLdrColor = "LdrColor";

}  // namespace ernie::tokens
