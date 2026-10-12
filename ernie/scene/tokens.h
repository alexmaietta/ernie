// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string_view>

/// Prim type and attribute names, following UsdGeom, UsdLux, UsdShade
/// and UsdRender where those schemas have the concept, with Ernie's
/// own under `ernie:`.
///
/// References:
/// - Pixar, OpenUSD schema domains,
///   https://openusd.org/release/api/index.html
/// - NVIDIA, Omniverse lighting, photometric units,
///   https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html
namespace ernie::tokens {

/// Root of the prims interactive tools define for their own use, such
/// as the GUI viewer's cameras and lights. Render delegates skip it
/// when falling back to the scene's first RenderSettings or Camera
/// and when gathering the scene's lights.
inline constexpr std::string_view kSessionRoot = "/__session";

// Stage metadata (UsdStage's, which USD keeps on the stage's
// pseudo-root; Ernie on its root prim `/`, `Stage`): the axis that
// points up ("Y" or "Z"), metres and kilograms per scene unit, the
// time codes and frames per second, the time codes animation plays
// between, and the prim a reference to the scene brings in.
inline constexpr std::string_view kUpAxis = "upAxis";
inline constexpr std::string_view kMetersPerUnit = "metersPerUnit";
inline constexpr std::string_view kKilogramsPerUnit =
    "kilogramsPerUnit";
inline constexpr std::string_view kTimeCodesPerSecond =
    "timeCodesPerSecond";
inline constexpr std::string_view kFramesPerSecond =
    "framesPerSecond";
inline constexpr std::string_view kStartTimeCode = "startTimeCode";
inline constexpr std::string_view kEndTimeCode = "endTimeCode";
inline constexpr std::string_view kDefaultPrim = "defaultPrim";
/// double: the time code the scene's animation is at (Ernie), as
/// ovstage's `apply_usd_time` sets it: apart from a render delegate's
/// sensor clock (`RenderDelegate::Time`).
inline constexpr std::string_view kTimeCode = "ernie:timeCode";
inline constexpr std::string_view kY = "Y";
inline constexpr std::string_view kZ = "Z";

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
/// UsdGeomPoints: `points` drawn as spheres `widths` across, coloured
/// by their displayColor.
inline constexpr std::string_view kGeomPoints = "Points";
inline constexpr std::string_view kVolume = "Volume";

// Volumes (UsdVol). A Volume names its fields with relationships
// `field:<name>` (`field:density`, `field:temperature`), each to a
// field prim: an OpenVDBAsset, a grid of a NanoVDB file (.nvdb; .vdb
// files are converted first). Its bound Material's volume inputs
// below say how the fields shade.
//
// Reference: Pixar, UsdVol,
// https://openusd.org/release/api/usd_vol_page_front.html
inline constexpr std::string_view kFieldBase = "FieldBase";
inline constexpr std::string_view kFieldAsset = "FieldAsset";
inline constexpr std::string_view kOpenVDBAsset = "OpenVDBAsset";
/// The namespace of a Volume's field relationships.
inline constexpr std::string_view kField = "field";
inline constexpr std::string_view kFilePath =
    "filePath";  ///< asset: the grid's file.
inline constexpr std::string_view kFieldName =
    "fieldName";  ///< token: the grid's name in the file.
inline constexpr std::string_view kFieldIndex =
    "fieldIndex";  ///< int: which of the grids of that name.
inline constexpr std::string_view kFieldDataType =
    "fieldDataType";  ///< token: the grid's value type.
inline constexpr std::string_view kVectorDataRoleHint =
    "vectorDataRoleHint";  ///< token: what a vector grid holds.
inline constexpr std::string_view kFieldClass =
    "fieldClass";  ///< token: levelSet, fogVolume, staggered.
/// Ernie's field of NanoVDB grids written as an array, such as Warp's
/// `Volume.array()`, rather than read from a file.
inline constexpr std::string_view kNanoVDBField = "NanoVDBField";
inline constexpr std::string_view kNanoVdbBuffer =
    "ernie:buffer";  ///< array: the grids, back to back.
/// Ernie's field of a dense array of values, such as a grid solver's.
inline constexpr std::string_view kDenseField = "DenseField";
inline constexpr std::string_view kDenseValues =
    "ernie:values";  ///< array: (x, y, z) float32.
inline constexpr std::string_view kVoxelSize =
    "ernie:voxelSize";  ///< double: world units between values.
// The volume inputs of a Material. The density field times
// `inputs:density` is the extinction coefficient in 1/m, of which
// `inputs:volumeAlbedo` scatters; the rest absorbs, and a temperature
// field makes what absorbs glow as a black body.
inline constexpr std::string_view kVolumeDensity = "inputs:density";
inline constexpr std::string_view kVolumeDensityField =
    "inputs:densityField";
inline constexpr std::string_view kVolumeAlbedo =
    "inputs:volumeAlbedo";
inline constexpr std::string_view kVolumeAnisotropy =
    "inputs:anisotropy";
inline constexpr std::string_view kVolumeScatteringOctaves =
    "inputs:scatteringOctaves";
inline constexpr std::string_view kVolumeOctaveAttenuation =
    "inputs:octaveAttenuation";
inline constexpr std::string_view kVolumeOctaveExtinction =
    "inputs:octaveExtinction";
inline constexpr std::string_view kVolumeOctavePhase =
    "inputs:octavePhase";
inline constexpr std::string_view kVolumeSkyLight = "inputs:skyLight";
inline constexpr std::string_view kVolumeEmissionColor =
    "inputs:emissionColor";
inline constexpr std::string_view kVolumeEmissionStrength =
    "inputs:emissionStrength";
inline constexpr std::string_view kVolumeEmissionField =
    "inputs:emissionField";
inline constexpr std::string_view kVolumeBlackbody =
    "inputs:blackbody";
inline constexpr std::string_view kVolumeTemperature =
    "inputs:temperature";
inline constexpr std::string_view kVolumeTemperatureField =
    "inputs:temperatureField";
inline constexpr std::string_view kVolumeTemperatureScale =
    "inputs:temperatureScale";
/// token: how what scatters spreads its light: henyeyGreenstein (by
/// `inputs:anisotropy`), doubleHenyeyGreenstein (Hapke's, by the
/// `inputs:b` and `inputs:c` a Hapke surface reads), rayleigh or
/// isotropic.
inline constexpr std::string_view kVolumePhaseFunction =
    "inputs:phaseFunction";
inline constexpr std::string_view kHenyeyGreenstein =
    "henyeyGreenstein";
inline constexpr std::string_view kDoubleHenyeyGreenstein =
    "doubleHenyeyGreenstein";
inline constexpr std::string_view kRayleigh = "rayleigh";
inline constexpr std::string_view kIsotropic = "isotropic";
/// Hapke's double Henyey-Greenstein lobe width b and backscatter
/// weight c (`brdf.cpp`'s hapke model's parameters).
inline constexpr std::string_view kPhaseB = "inputs:b";
inline constexpr std::string_view kPhaseC = "inputs:c";
inline constexpr std::string_view kCamera = "Camera";
inline constexpr std::string_view kDistantLight = "DistantLight";
inline constexpr std::string_view kSphereLight = "SphereLight";
inline constexpr std::string_view kRectLight = "RectLight";
inline constexpr std::string_view kDiskLight = "DiskLight";
inline constexpr std::string_view kCylinderLight = "CylinderLight";
inline constexpr std::string_view kLaserLight = "LaserLight";
inline constexpr std::string_view kDomeLight = "DomeLight";
inline constexpr std::string_view kMaterial = "Material";
inline constexpr std::string_view kTerrain = "Terrain";
inline constexpr std::string_view kGroundPlane =
    "GroundPlane";  ///< Ernie: an infinite floor.
inline constexpr std::string_view kStarField =
    "StarField";  ///< Ernie: stars behind everything.
inline constexpr std::string_view kAtmosphere =
    "Atmosphere";  ///< Ernie: a planet's air and aerosols.
inline constexpr std::string_view kCelestialBody =
    "CelestialBody";  ///< Ernie: a moon or planet far off.
inline constexpr std::string_view kRenderSettings = "RenderSettings";
inline constexpr std::string_view kRenderProduct = "RenderProduct";
inline constexpr std::string_view kRenderVar = "RenderVar";

// Imageable and Xformable (UsdGeom).
inline constexpr std::string_view kTransform =
    "xformOp:transform";  ///< dmat4, column vectors.
/// UsdGeomXformable's transform ops (ernie/scene/xform.h): a prim's
/// local transform is the product of the ops `xformOpOrder` lists, in
/// that order (the last applies to points first). Without an
/// `xformOpOrder`, `xformOp:transform` alone is the local transform.
/// An op may carry a suffix, as in `xformOp:translate:pivot`, and the
/// order may name it `!invert!xformOp:translate:pivot`.
inline constexpr std::string_view kXformOpOrder =
    "xformOpOrder";  ///< Token array.
/// Leading `xformOpOrder` entry: the prim ignores its ancestors'
/// transforms.
inline constexpr std::string_view kResetXformStack =
    "!resetXformStack!";
inline constexpr std::string_view kTranslate =
    "xformOp:translate";  ///< dvec3, meters.
inline constexpr std::string_view kScale =
    "xformOp:scale";  ///< dvec3, or one number.
/// dquat: a rotation as a unit quaternion (w, x, y, z); as a tensor,
/// rows of x, y, z, w.
inline constexpr std::string_view kOrient = "xformOp:orient";
/// Rotations in degrees about one axis.
inline constexpr std::string_view kRotateX = "xformOp:rotateX";
inline constexpr std::string_view kRotateY = "xformOp:rotateY";
inline constexpr std::string_view kRotateZ = "xformOp:rotateZ";
/// Euler angles in degrees (dvec3, one angle per axis as x, y, z),
/// applied about fixed axes in the order the name gives: rotateXYZ
/// rotates about X first, then Y, then Z.
inline constexpr std::string_view kRotateXYZ = "xformOp:rotateXYZ";
inline constexpr std::string_view kRotateXZY = "xformOp:rotateXZY";
inline constexpr std::string_view kRotateYXZ = "xformOp:rotateYXZ";
inline constexpr std::string_view kRotateYZX = "xformOp:rotateYZX";
inline constexpr std::string_view kRotateZXY = "xformOp:rotateZXY";
inline constexpr std::string_view kRotateZYX = "xformOp:rotateZYX";
/// Ernie: on a prim whose transform ops are rows of GPU arrays
/// (`Scene::Write` of many paths), the bounds of it and its
/// descendants, posed, in its parent's frame: a float array of low
/// then high x, y, z. Without it the renderer copies the poses to the
/// host to bound them each step.
inline constexpr std::string_view kXformExtent = "ernie:xform:extent";
/// Ernie: the component order of the quaternions in a prim's arrays:
/// "xyzw" (the default, USD's memory layout, as Warp and PhysX keep
/// them) or "wxyz" (w first, as Isaac Lab and MuJoCo do). It applies
/// to `xformOp:orient` written as an array and to a Camera's (views,
/// 7) `ernie:poses`; single `xformOp:orient` values are quaternions
///    either way.
inline constexpr std::string_view kQuaternionOrder =
    "ernie:quaternionOrder";
inline constexpr std::string_view kXyzw = "xyzw";
inline constexpr std::string_view kWxyz = "wxyz";
inline constexpr std::string_view kVisibility =
    "visibility";  ///< kInherited or kInvisible.
inline constexpr std::string_view kInherited = "inherited";
inline constexpr std::string_view kInvisible = "invisible";

// Geometry (UsdGeom Mesh, PointInstancer).
inline constexpr std::string_view kPoints = "points";
inline constexpr std::string_view kNormals = "normals";
/// A Points prim's diameters: one per point, or one for all (default
/// 1).
inline constexpr std::string_view kWidths = "widths";
inline constexpr std::string_view kFaceVertexCounts =
    "faceVertexCounts";
inline constexpr std::string_view kFaceVertexIndices =
    "faceVertexIndices";
/// Ernie: a Mesh's points and triangle indices in use, of those its
/// arrays hold, for topology that changes on the GPU within a
/// capacity.
inline constexpr std::string_view kActiveCounts =
    "ernie:activeCounts";
inline constexpr std::string_view kDisplayColor =
    "primvars:displayColor";
inline constexpr std::string_view kPositions = "positions";
inline constexpr std::string_view kOrientations = "orientations";
inline constexpr std::string_view kScales = "scales";
inline constexpr std::string_view kVelocities = "velocities";
/// Bounds in the prim's own frame (UsdGeomBoundable): a float array
/// of low then high x, y, z.
inline constexpr std::string_view kExtent = "extent";
inline constexpr std::string_view kProtoIndices = "protoIndices";
inline constexpr std::string_view kPrototypes = "prototypes";

// Implicit primitives (UsdGeom Cube, Sphere, Cylinder, Cone, Capsule,
// Plane), drawn as meshes (`ernie::BoxMesh` and the other
// generators).
inline constexpr std::string_view kSize =
    "size";  ///< Cube edge length, default 2.
inline constexpr std::string_view kGeomRadius = "radius";
inline constexpr std::string_view kGeomHeight = "height";
inline constexpr std::string_view kGeomWidth =
    "width";  ///< Plane, default 2.
inline constexpr std::string_view kGeomLength =
    "length";  ///< Plane, default 2.
inline constexpr std::string_view kAxis =
    "axis";  ///< "X", "Y" or "Z" (default).
/// token, on a Sphere (Ernie): how it is divided into triangles
/// (`ernie::SphereTessellation`), and int: its level, 0 for the
/// default.
inline constexpr std::string_view kTessellation =
    "ernie:tessellation";
inline constexpr std::string_view kTessellationLevel =
    "ernie:tessellation:level";
inline constexpr std::string_view kUvSphere = "uv";
inline constexpr std::string_view kIcosphere = "icosphere";
inline constexpr std::string_view kCubeSphere = "cubeSphere";
inline constexpr std::string_view kHealpix = "healpix";

// Material binding (UsdShade MaterialBindingAPI): a Path.
inline constexpr std::string_view kMaterialBinding =
    "material:binding";

// Camera (UsdGeom Camera). Focal length and apertures share a unit,
// conventionally millimetres; only their ratio sets the field of
// view.
inline constexpr std::string_view kFocalLength = "focalLength";
inline constexpr std::string_view kHorizontalAperture =
    "horizontalAperture";
inline constexpr std::string_view kVerticalAperture =
    "verticalAperture";
inline constexpr std::string_view kClippingRange =
    "clippingRange";  ///< vec2 near, far.
inline constexpr std::string_view kProjection =
    "projection";  ///< kPerspective or kOrthographic.
inline constexpr std::string_view kPerspective = "perspective";
inline constexpr std::string_view kOrthographic = "orthographic";
inline constexpr std::string_view kCameraExposure =
    "exposure";  ///< Stops; scales by 2^exposure.
inline constexpr std::string_view kHorizontalApertureOffset =
    "horizontalApertureOffset";
inline constexpr std::string_view kVerticalApertureOffset =
    "verticalApertureOffset";
inline constexpr std::string_view kExposureTime =
    "exposure:time";  ///< Seconds.
inline constexpr std::string_view kExposureIso = "exposure:iso";
inline constexpr std::string_view kExposureFStop = "exposure:fStop";
inline constexpr std::string_view kExposureResponsivity =
    "exposure:responsivity";
/// How the photographic outputs are exposed: kManual by the camera's
/// exposure settings, or kAuto keyed to middle grey from the metered
/// scene, over `ernie:exposure:adaptation` frames.
inline constexpr std::string_view kExposureMode =
    "ernie:exposure:mode";
inline constexpr std::string_view kExposureAdaptation =
    "ernie:exposure:adaptation";
inline constexpr std::string_view kManual = "manual";
inline constexpr std::string_view kAuto = "auto";
/// UsdGeomCamera's lens: its f-number for depth of field and
/// diffraction (0, the default, is a pinhole with everything in
/// focus, and diffraction then uses `exposure:fStop`), and the
/// distance in scene units it focuses at.
inline constexpr std::string_view kFStop = "fStop";
inline constexpr std::string_view kFocusDistance = "focusDistance";
/// UsdGeomCamera's shutter interval, in frames relative to the
/// frame's time (0): motion blurs over it
/// (`CameraParams::shutterOpen`).
inline constexpr std::string_view kShutterOpen = "shutter:open";
inline constexpr std::string_view kShutterClose = "shutter:close";
/// Sub-frames rendered across the shutter and readout for motion blur
/// (int, default 8).
inline constexpr std::string_view kShutterSamples =
    "ernie:shutter:samples";
/// Rolling shutter: frames between the top row's exposure and the
/// bottom row's (0, global).
inline constexpr std::string_view kShutterRolling =
    "ernie:shutter:rolling";
/// How the camera reads out: "frame" (the default) or "pushbroom", a
/// line scanner whose rows are read one after another along the
/// camera's motion (`CameraParams::pushbroom`).
inline constexpr std::string_view kCameraScan = "ernie:camera:scan";
inline constexpr std::string_view kFrameScan = "frame";
inline constexpr std::string_view kPushbroom = "pushbroom";
/// A pushbroom's detector line along y on the focal plane from the
/// principal point, mm.
inline constexpr std::string_view kPushbroomLineOffset =
    "ernie:pushbroom:lineOffset";

// Physical camera (Ernie): spectral sampling, lens and sensor.
// Spectra are float arrays of interleaved (wavelength in nm, value)
// pairs.
inline constexpr std::string_view kSpectralRange =
    "ernie:spectral:range";  ///< vec2 nm.
inline constexpr std::string_view kSpectralBands =
    "ernie:spectral:bands";  ///< int, 1 to 64.
/// token: "spectral" (the default), bands over `kSpectralRange` as
/// `kSpectralBands` gives, or "rgb", three bands carrying linear sRGB
/// blue, green and red (`SpectralBands::Rgb`), which keep colours as
/// a 3-channel renderer does, faster than many bands.
inline constexpr std::string_view kSpectralMode =
    "ernie:spectral:mode";
inline constexpr std::string_view kSpectral = "spectral";
inline constexpr std::string_view kRgb = "rgb";
/// The lens's projection: "perspective" (the default) or "fisheye"
/// (`LensProjection`).
inline constexpr std::string_view kLensProjection =
    "ernie:lens:projection";
inline constexpr std::string_view kFisheye = "fisheye";
inline constexpr std::string_view kLensDistortion =
    "ernie:lens:distortion";  ///< k1 k2 k3 p1 p2.
/// A fisheye's Kannala-Brandt coefficients k1 k2 k3 k4 (float array;
/// none is equidistant).
inline constexpr std::string_view kLensFisheye = "ernie:lens:fisheye";
/// The lens as its rays (`LensRayTable`), in place of its projection
/// and distortion: a float32 (rows, columns, 3) CPU tensor of unit
/// directions in the camera's frame at every `ernie:lens:raysStep`
/// (int) pixels across the image, corners included.
inline constexpr std::string_view kLensRays = "ernie:lens:rays";
inline constexpr std::string_view kLensRaysStep =
    "ernie:lens:raysStep";
/// Chromatic aberration (`LensModel::chromaticScale`,
/// `chromaticFocusMm`), float arrays of (wavelength nm, value) pairs:
/// the image's scale per wavelength (lateral colour), and the focal
/// length's shift in millimetres (axial colour, which the `defocus`
/// filter blurs by).
inline constexpr std::string_view kLensChromaticScale =
    "ernie:lens:chromatic:scale";
inline constexpr std::string_view kLensChromaticFocus =
    "ernie:lens:chromatic:focus";
inline constexpr std::string_view kLensVignetting =
    "ernie:lens:vignetting";  ///< bool.
inline constexpr std::string_view kLensTransmission =
    "ernie:lens:transmission";  ///< fraction.
inline constexpr std::string_view kSensorType =
    "ernie:sensor:type";  ///< "rgb" or "mono".
/// Quantum efficiency spectrum of one channel: append the channel
/// name ("r", "g", "b", "mono").
inline constexpr std::string_view kSensorQePrefix =
    "ernie:sensor:qe:";
inline constexpr std::string_view kSensorQeR = "ernie:sensor:qe:r";
inline constexpr std::string_view kSensorQeG = "ernie:sensor:qe:g";
inline constexpr std::string_view kSensorQeB = "ernie:sensor:qe:b";
inline constexpr std::string_view kSensorQeMono =
    "ernie:sensor:qe:mono";
inline constexpr std::string_view kSensorFullWell =
    "ernie:sensor:fullWell";  ///< electrons.
inline constexpr std::string_view kSensorReadNoise =
    "ernie:sensor:readNoise";  ///< electrons.
inline constexpr std::string_view kSensorDarkCurrent =
    "ernie:sensor:darkCurrent";  ///< e-/s.
inline constexpr std::string_view kSensorPrnu =
    "ernie:sensor:prnu";  ///< fraction.
inline constexpr std::string_view kSensorDsnu =
    "ernie:sensor:dsnu";  ///< electrons.
inline constexpr std::string_view kSensorGain =
    "ernie:sensor:electronsPerDn";  ///< at ISO 100.
inline constexpr std::string_view kSensorAdcBits =
    "ernie:sensor:adcBits";
inline constexpr std::string_view kSensorBlackLevel =
    "ernie:sensor:blackLevel";  ///< DN.
/// An orthographic (telecentric) camera's pixel pitch on its sensor,
/// in micrometres: its apertures span the scene, so they do not give
/// it (`CameraParams::orthographicPixelPitchUm`).
inline constexpr std::string_view kSensorPixelPitch =
    "ernie:sensor:pixelPitch";
inline constexpr std::string_view kSensorNoise =
    "ernie:sensor:noise";  ///< bool.
inline constexpr std::string_view kSensorSeed =
    "ernie:sensor:seed";  ///< int.

// Lights (UsdLux). A DistantLight shines along the -z axis of its
// transform, and a SphereLight's cone (ShapingAPI) is around its -z
// axis.
inline constexpr std::string_view kIntensity = "inputs:intensity";
inline constexpr std::string_view kExposure =
    "inputs:exposure";  ///< Stops; scales by 2^exposure.
inline constexpr std::string_view kColor = "inputs:color";
/// bool: divides `inputs:intensity` by the light's size (UsdLux's
/// normalize): a sphere, rect or disk light's world surface area; a
/// distant light's by its disk's projected solid angle, so its
/// intensity becomes its illuminance at normal incidence. Domes,
/// powers and photometric units ignore it.
inline constexpr std::string_view kNormalize = "inputs:normalize";
inline constexpr std::string_view kAngle =
    "inputs:angle";  ///< Degrees subtended.
inline constexpr std::string_view kRadius = "inputs:radius";
inline constexpr std::string_view kTreatAsPoint =
    "treatAsPoint";  ///< bool: radius 0.
/// A CylinderLight's length along its x axis, and whether it may be
/// drawn as a line (a hint).
inline constexpr std::string_view kLength = "inputs:length";
inline constexpr std::string_view kTreatAsLine = "treatAsLine";
/// A LaserLight (Ernie): a Gaussian beam along its -z axis.
/// Wavelength and linewidth (FWHM) in nm; average power in W out of
/// its aperture; 1/e^2 waist radius in m along its x and y (double2);
/// the waist's distance along the beam from the aperture in m; beam
/// quality M^2 along x and y (double2, 1 for an ideal beam, ISO
/// 11146); an aperture radius in m clipping the beam (0 none).
inline constexpr std::string_view kLaserWavelength =
    "ernie:laser:wavelength";
inline constexpr std::string_view kLaserLinewidth =
    "ernie:laser:linewidth";
inline constexpr std::string_view kLaserPower = "ernie:laser:power";
inline constexpr std::string_view kLaserWaistRadius =
    "ernie:laser:waistRadius";
inline constexpr std::string_view kLaserWaistDistance =
    "ernie:laser:waistDistance";
inline constexpr std::string_view kLaserBeamQuality =
    "ernie:laser:beamQuality";
inline constexpr std::string_view kLaserApertureRadius =
    "ernie:laser:apertureRadius";
/// Whether a laser's light shows speckle (bool): the grain coherent
/// light takes on optically rough surfaces, its size set by the
/// camera's aperture and its contrast lowered by the pixel's area,
/// depolarization and the laser's linewidth.
inline constexpr std::string_view kLaserSpeckle =
    "ernie:laser:speckle";
/// A laser's pattern: "spot" (its beam), "line" (fanned over
/// `fanAngle` degrees along its x axis, its profile along the line
/// "flat" as a Powell lens gives or "gaussian" as a cylindrical lens
/// does, the beam's across it), "grid" (`gridColumns` x `gridRows`
/// beamlets `gridPitch` degrees apart along x and y, as a diffractive
/// element splits it), "dots" (a seeded field of `dotColumns` x
/// `dotRows` dots per tile, `dotTileColumns` x `dotTileRows` tiles of
/// the same dots over `pattern:fieldOfView`, as a depth camera's
/// projector makes), "image" (`pattern:image` over
/// `pattern:fieldOfView`, as a projector shows it) or a registered
/// custom pattern (`LightPatternModel`). `efficiency` of the power
/// goes into the pattern and `zeroOrder` stays in an undiffracted
/// beam along the axis; the rest is lost.
inline constexpr std::string_view kLaserPattern =
    "ernie:laser:pattern";
inline constexpr std::string_view kLaserFanAngle =
    "ernie:laser:fanAngle";
inline constexpr std::string_view kLaserLineProfile =
    "ernie:laser:lineProfile";
inline constexpr std::string_view kLaserGridColumns =
    "ernie:laser:gridColumns";
inline constexpr std::string_view kLaserGridRows =
    "ernie:laser:gridRows";
inline constexpr std::string_view kLaserGridPitch =
    "ernie:laser:gridPitch";
/// "dots": dots per tile along x and y, one in each cell of a grid,
/// jittered by up to `dotJitter` of a cell from its centre; tiles of
/// the same dots along x and y; the seed the dots derive from.
inline constexpr std::string_view kLaserDotColumns =
    "ernie:laser:dotColumns";
inline constexpr std::string_view kLaserDotRows =
    "ernie:laser:dotRows";
inline constexpr std::string_view kLaserDotTileColumns =
    "ernie:laser:dotTileColumns";
inline constexpr std::string_view kLaserDotTileRows =
    "ernie:laser:dotTileRows";
inline constexpr std::string_view kLaserDotJitter =
    "ernie:laser:dotJitter";
inline constexpr std::string_view kLaserDotSeed =
    "ernie:laser:dotSeed";
/// double2: a scan's Lissajous frequencies along x and y, turns a
/// period; double: its phase along x, degrees.
inline constexpr std::string_view kLaserScanFrequencies =
    "ernie:laser:scanFrequencies";
inline constexpr std::string_view kLaserScanPhase =
    "ernie:laser:scanPhase";
/// A pulsed laser (0 rate: continuous): pulses `pulse:width` seconds
/// long at `pulse:rate` per second, the first starting
/// `pulse:delay` seconds into the sensor clock
/// (`RenderDelegate::Time`); its `power` is their average.
inline constexpr std::string_view kLaserPulseRate =
    "ernie:laser:pulse:rate";
inline constexpr std::string_view kLaserPulseWidth =
    "ernie:laser:pulse:width";
inline constexpr std::string_view kLaserPulseDelay =
    "ernie:laser:pulse:delay";
/// Any other light's pulses, a strobe or flash (0 rate: continuous):
/// `pulse:width` seconds long at `pulse:rate` per second from
/// `pulse:delay` on the sensor clock; or with `pulse:trigger`
/// "exposure", one a frame, `pulse:delay` after the camera's
/// exposure opens (its top row's, with a rolling shutter), as a
/// camera's strobe output fires a flash. Its intensity is its level
/// as a pulse starts. Lasers keep `ernie:laser:pulse:*` and take
/// the rest below.
inline constexpr std::string_view kPulseRate = "ernie:pulse:rate";
inline constexpr std::string_view kPulseWidth = "ernie:pulse:width";
inline constexpr std::string_view kPulseDelay = "ernie:pulse:delay";
/// A pulse's decay (seconds): its level falls as exp(-t / decay)
/// over its width, as a xenon flash's; 0 flat, as an LED strobe's.
inline constexpr std::string_view kPulseDecay = "ernie:pulse:decay";
/// Each pulse's start strays uniformly within +-`pulse:jitter`
/// seconds, the trigger's latency spread, seeded per pulse (or per
/// frame when triggered by the exposure).
inline constexpr std::string_view kPulseJitter = "ernie:pulse:jitter";
inline constexpr std::string_view kPulseTrigger =
    "ernie:pulse:trigger";
inline constexpr std::string_view kClock = "clock";
inline constexpr std::string_view kExposureTrigger = "exposure";
/// A sequence of `sequence:count` patterns (1: one), each shown for
/// `sequence:period` seconds in turn from `sequence:delay` seconds
/// into the sensor clock, as structured light's Gray codes and phase
/// shifts are: the pattern's frames (`LightPatternInput::frame`), an
/// image's stacked top to bottom.
inline constexpr std::string_view kLaserSequenceCount =
    "ernie:laser:sequence:count";
inline constexpr std::string_view kLaserSequencePeriod =
    "ernie:laser:sequence:period";
inline constexpr std::string_view kLaserSequenceDelay =
    "ernie:laser:sequence:delay";
inline constexpr std::string_view kLaserEfficiency =
    "ernie:laser:efficiency";
inline constexpr std::string_view kLaserZeroOrder =
    "ernie:laser:zeroOrder";
/// float[]: a custom pattern's parameters (`LightPatternModel`), in
/// its order; missing ones take its defaults.
inline constexpr std::string_view kLaserPatternParams =
    "ernie:laser:pattern:params";
/// float[]: data a custom pattern reads (`p.data(i)`), such as a
/// diffractive element's beamlet angles.
inline constexpr std::string_view kLaserPatternData =
    "ernie:laser:pattern:data";
/// asset: an image of the pattern's relative power (linear, colour
/// channels averaged), drawn by "image" and read by custom patterns
/// (`p.image(uv, footprint)`).
inline constexpr std::string_view kLaserPatternImage =
    "ernie:laser:pattern:image";
/// Tensor float32 (rows, cols): the pattern image as an array, rows
/// top to bottom, in place of `pattern:image`'s file: on the host, or
/// on the GPU, where the renderer reads it in place when mapped
/// (`RenderDelegate::MapArray`), so procedural patterns need no copy.
inline constexpr std::string_view kLaserPatternImageArray =
    "ernie:laser:pattern:imageArray";
/// double2: the full angles, degrees, that "image" spreads its image
/// over along x and y.
inline constexpr std::string_view kLaserPatternFieldOfView =
    "ernie:laser:pattern:fieldOfView";
inline constexpr std::string_view kSpot = "spot";
inline constexpr std::string_view kLine = "line";
inline constexpr std::string_view kGrid = "grid";
inline constexpr std::string_view kImage = "image";
inline constexpr std::string_view kDots = "dots";
inline constexpr std::string_view kScan = "scan";
inline constexpr std::string_view kFlat = "flat";
inline constexpr std::string_view kGaussian = "gaussian";
/// Degrees off the axis that a light reaches, and the fraction of
/// that angle over which it fades.
inline constexpr std::string_view kShapingConeAngle =
    "inputs:shaping:cone:angle";
inline constexpr std::string_view kShapingConeSoftness =
    "inputs:shaping:cone:softness";
/// float: a sphere, rect, disk or cylinder light's emission falls off
/// as |cos|^focus of the angle off its -z axis; color3f: the tint its
/// falloff fades to instead of black (white undoes it).
inline constexpr std::string_view kShapingFocus =
    "inputs:shaping:focus";
inline constexpr std::string_view kShapingFocusTint =
    "inputs:shaping:focusTint";
/// asset: an IES LM-63 profile (`IesProfile`) shaping a sphere, rect,
/// disk or cylinder light, its nadir along the light's -z axis and
/// its 0 degree plane through +x. float: scales its vertical angle
/// about the nadir (above 0) or the zenith (below 0); 0 leaves it.
/// bool: normalize keeps the light's power as without the profile;
/// otherwise its candela multiply the intensity.
inline constexpr std::string_view kShapingIesFile =
    "inputs:shaping:ies:file";
inline constexpr std::string_view kShapingIesAngleScale =
    "inputs:shaping:ies:angleScale";
inline constexpr std::string_view kShapingIesNormalize =
    "inputs:shaping:ies:normalize";
inline constexpr std::string_view kWidth = "inputs:width";
inline constexpr std::string_view kHeight = "inputs:height";
inline constexpr std::string_view kTextureFile =
    "inputs:texture:file";
inline constexpr std::string_view kShadowEnable =
    "inputs:shadow:enable";
/// UsdLuxLightAPI's light and shadow linking (UsdCollectionAPI, its
/// prims expanded): the geometry a light lights, and the geometry
/// that casts its shadows. bool: whether the whole scene is included
/// unless excluded (default true); relationships: the prims, with
/// their descendants, included or excluded, the nearest listed
/// ancestor deciding.
inline constexpr std::string_view kLightLinkIncludeRoot =
    "collection:lightLink:includeRoot";
inline constexpr std::string_view kLightLinkIncludes =
    "collection:lightLink:includes";
inline constexpr std::string_view kLightLinkExcludes =
    "collection:lightLink:excludes";
inline constexpr std::string_view kShadowLinkIncludeRoot =
    "collection:shadowLink:includeRoot";
inline constexpr std::string_view kShadowLinkIncludes =
    "collection:shadowLink:includes";
inline constexpr std::string_view kShadowLinkExcludes =
    "collection:shadowLink:excludes";
inline constexpr std::string_view kColorTemperature =
    "inputs:colorTemperature";  ///< Kelvin.
inline constexpr std::string_view kEnableColorTemperature =
    "inputs:enableColorTemperature";
/// Physical sunlight (Ernie): total irradiance at normal incidence in
/// W/m^2, and a spectrum of (wavelength nm, W m^-2 nm^-1) pairs.
/// Either one makes the light physical; without a spectrum it has its
/// `physical:illuminant`'s shape, else is a black body (the colour
/// temperature when enabled, else the Sun's 5772 K), and without an
/// irradiance a black body carries the solar constant, 1361 W/m^2.
inline constexpr std::string_view kLightIrradiance =
    "ernie:irradiance";
inline constexpr std::string_view kLightSpectrum = "ernie:spectrum";
/// A light's linear polarization (Ernie): the fraction of its light
/// polarized, 0 to 1 (a laser diode's 100:1 is 0.98), and the angle
/// of its electric field from the light's +x about its axis, degrees
/// counterclockwise looking along its light.
inline constexpr std::string_view kPolarizationDegree =
    "ernie:polarization:degree";
inline constexpr std::string_view kPolarizationAngle =
    "ernie:polarization:angle";
/// A material's share of incident polarization its diffuse
/// reflection keeps, 0 (all scattered into unpolarized light) to 1;
/// its specular reflection polarizes by Fresnel's laws regardless.
inline constexpr std::string_view kPolarizationRetention =
    "ernie:polarization:retention";
/// A material's rms surface height in m, which decides how fully
/// laser speckle develops on it (`ernie:laser:speckle`): optically
/// rough (well above a quarter wavelength, the default) shows full
/// contrast, smooth (polished, 0) none.
inline constexpr std::string_view kSpeckleRoughness =
    "ernie:speckle:roughness";
/// A camera's polarizer (Ernie): "none", "linear" (one in front of
/// the lens, at `ernie:polarizer:angle`) or "mosaic" (a polarization
/// sensor's 2 x 2 cells of 90, 45 / 135, 0 degree polarizers over its
/// pixels, as Sony's IMX250MZR); the angle of its transmission axis
/// from the image's +x towards +y, degrees; its extinction ratio,
/// the light it passes along that axis over across it; and that
/// transmission along it.
inline constexpr std::string_view kPolarizerType =
    "ernie:polarizer:type";
inline constexpr std::string_view kPolarizerAngle =
    "ernie:polarizer:angle";
inline constexpr std::string_view kPolarizerExtinction =
    "ernie:polarizer:extinction";
inline constexpr std::string_view kPolarizerTransmission =
    "ernie:polarizer:transmission";
inline constexpr std::string_view kLinear = "linear";
inline constexpr std::string_view kMosaic = "mosaic";
/// Physical lamp (Ernie): radiant flux over 360 to 830 nm in W. A
/// SphereLight emits it equally in every direction of its cone (a
/// narrower cone concentrates it); a RectLight or DiskLight from its
/// face as a Lambertian emitter. Its spectrum's shape is
/// `ernie:spectrum`, else a black body at the enabled colour
/// temperature, else flat; `ernie:spectrum` alone gives the radiant
/// intensity in W sr^-1 nm^-1 (sphere) or radiance in W m^-2 sr^-1
/// nm^-1 (rect, disk, dome).
inline constexpr std::string_view kLightPower = "ernie:power";
/// A DomeLight's map (Ernie): float32 (rows, cols, 3 or 4) linear RGB
/// scaling its radiance, in latitude-longitude layout around the
/// dome's +z axis: rows run from +z (top) to -z, columns from +x
/// through +y. (USD's `inputs:texture:file` maps are not read.)
inline constexpr std::string_view kLightTexture = "ernie:texture";

// Physical lighting (Omniverse's PhysicalLightIlluminantAPI and
// Photometric*LightAPI): the shape of a light's spectrum, and its
// brightness in photometric units. Authoring any `photometric:` value
// puts the light in photometric units: its brightness then comes from
// them alone, not from `inputs:intensity`, `inputs:exposure`,
// `ernie:power` or `ernie:irradiance`.
//
/// token: the shape of the light's spectrum: "white" (flat) or
/// "blackbody" (at `inputs:colorTemperature`), both over 380 to 780
/// nm as Omniverse's, "illuminant D" (CIE daylight), or "custom"
/// (`physical:customIlluminant`). `inputs:color` still tints it.
/// Unset, the shape is a black body when
/// `inputs:enableColorTemperature` is on, else flat, over the whole
/// spectral range.
inline constexpr std::string_view kPhysicalIlluminant =
    "physical:illuminant";
inline constexpr std::string_view kWhite = "white";
inline constexpr std::string_view kBlackbody = "blackbody";
inline constexpr std::string_view kIlluminantD = "illuminant D";
inline constexpr std::string_view kCustom = "custom";
/// The Sun above the atmosphere, measured (`SolarSpectralIrradiance`,
/// the TSIS-1 Hybrid Solar Reference Spectrum) over its whole range.
inline constexpr std::string_view kSun = "sun";
/// float array: (nm, value) pairs, wavelengths increasing, of a
/// "custom" illuminant's shape; zero outside them.
inline constexpr std::string_view kPhysicalCustomIlluminant =
    "physical:customIlluminant";
/// float, lux: for a DistantLight, on a surface facing it (default
/// 10000); for a DomeLight, on an upward-facing surface (default
/// 10000); for an area light, on a surface facing it on its axis
/// `photometric:illuminance:distance` metres away, in place of the
/// power when both are above 0.
inline constexpr std::string_view kPhotometricIlluminance =
    "photometric:illuminance";
inline constexpr std::string_view kPhotometricIlluminanceDistance =
    "photometric:illuminance:distance";
/// float, lumens: an area light's (SphereLight, RectLight, DiskLight)
/// luminous flux, default 1600, whatever its colour or cone; a
/// SphereLight emits it over its cone, a RectLight or DiskLight from
/// its face.
inline constexpr std::string_view kPhotometricPower =
    "photometric:power";

/// bool: whether a DomeLight shows behind the scene where camera rays
/// hit no geometry (Omniverse's "Visible In Primary Ray"); default
/// on. Off, it still lights the scene.
inline constexpr std::string_view kVisibleInPrimaryRay =
    "visibleInPrimaryRay";

// Star field (Ernie): stars from a catalogue as points of light at
// infinity, behind everything else, each with a black-body spectrum.
// The prim's transform takes the catalogue's frame, J2000 equatorial
// with x towards right ascension 0 and z the north celestial pole,
// into the world (`CelestialToLocal` places it over a site at a
// time). A star's illuminance is `ernie:stars:zeroPoint` times
// 10^(-0.4 magnitude) times 2^`inputs:exposure`. Renders draw the
// first visible StarField.
//
/// float arrays, degrees, one per star: right ascension and
/// declination in the catalogue's frame.
inline constexpr std::string_view kStarsRightAscension =
    "ernie:stars:rightAscension";
inline constexpr std::string_view kStarsDeclination =
    "ernie:stars:declination";
/// float array (stars, 3): unit directions to the stars in the prim's
/// frame, in place of right ascension and declination.
inline constexpr std::string_view kStarsDirection =
    "ernie:stars:direction";
/// float array: apparent visual (Johnson V) magnitudes, one per star.
inline constexpr std::string_view kStarsMagnitude =
    "ernie:stars:magnitude";
/// float array: B-V colour indices, from which each star's
/// temperature follows (Ballesteros 2012).
inline constexpr std::string_view kStarsColorIndex =
    "ernie:stars:colorIndex";
/// float array, kelvin: each star's temperature, in place of
/// `ernie:stars:colorIndex`. Without either, stars are 5772 K, the
/// Sun's.
inline constexpr std::string_view kStarsTemperature =
    "ernie:stars:temperature";
/// float: stars fainter than this magnitude are left out (default 99,
/// none).
inline constexpr std::string_view kStarsMagnitudeLimit =
    "ernie:stars:magnitudeLimit";
/// float, lux: the illuminance of a magnitude 0 star (default
/// 2.54e-6, Allen's Astrophysical Quantities), so 10^(-0.4 m) of it
/// for a star of magnitude m.
inline constexpr std::string_view kStarsZeroPoint =
    "ernie:stars:zeroPoint";

// Atmosphere (Ernie): a planet's gases and aerosols, from physical
// models (`EarthAtmosphere`, `MarsAtmosphere`), around the planet
// Terrain or, without one, a planet under the world's origin. Renders
// use the first visible Atmosphere.
//
/// token: the planet's model, "earth" or "mars".
inline constexpr std::string_view kAtmospherePreset =
    "ernie:atmosphere:preset";
inline constexpr std::string_view kEarth = "earth";
inline constexpr std::string_view kMars = "mars";
/// float, metres: the planet's radius; 0 takes the planet Terrain's,
/// or else the preset's (Earth 6371 km, Mars 3389.5 km).
inline constexpr std::string_view kAtmospherePlanetRadius =
    "ernie:atmosphere:planetRadius";
/// float: the ground's reflectance for light scattered in the air.
inline constexpr std::string_view kAtmosphereGroundAlbedo =
    "ernie:atmosphere:groundAlbedo";
/// float: Earth's vertical haze optical depth at 550 nm.
inline constexpr std::string_view kAtmosphereHazeOpticalDepth =
    "ernie:atmosphere:hazeOpticalDepth";
/// float, Dobson units: Earth's total column ozone.
inline constexpr std::string_view kAtmosphereOzoneColumn =
    "ernie:atmosphere:ozoneColumn";
/// float, atoms per square metre: Earth's mesospheric sodium column.
inline constexpr std::string_view kAtmosphereSodiumColumn =
    "ernie:atmosphere:sodiumColumn";
/// float: Mars's vertical dust optical depth at 880 nm.
inline constexpr std::string_view kAtmosphereDustOpticalDepth =
    "ernie:atmosphere:dustOpticalDepth";
/// float, metres: the clouds' tops, below which they block light
/// grazing the planet, which reddens the Moon in a lunar eclipse; and
/// the fraction of the planet's limb they cover.
inline constexpr std::string_view kAtmosphereCloudTop =
    "ernie:atmosphere:cloudTop";
inline constexpr std::string_view kAtmosphereCloudCover =
    "ernie:atmosphere:cloudCover";
/// token: where the Atmosphere's origin is on its planet: "ground"
/// (on the surface, the planet's centre a radius below along its -z;
/// the default) or "centre" (at the planet's centre), its world
/// transform then placing the planet.
inline constexpr std::string_view kAtmosphereAnchor =
    "ernie:atmosphere:anchor";
inline constexpr std::string_view kAnchorGround = "ground";
inline constexpr std::string_view kAnchorCentre = "centre";

// CelestialBody (Ernie): a sphere far beyond the scene, such as a
// moon, at its transform's origin, drawn in the sky and casting
// eclipse shadows.
//
/// float, metres.
inline constexpr std::string_view kBodyRadius = "ernie:body:radius";
/// color3f: its normal albedo, linear RGB.
inline constexpr std::string_view kBodyAlbedo = "ernie:body:albedo";

// Terrain (Ernie): a north-up heightfield, x east, y north, z up.
inline constexpr std::string_view kHeights =
    "heights";  ///< float32 tensor (rows, cols).
inline constexpr std::string_view kOrigin =
    "ernie:origin";  ///< dvec2 north-west corner.
inline constexpr std::string_view kCellSize =
    "ernie:cellSize";  ///< double, world units.
inline constexpr std::string_view kNoData =
    "ernie:noData";  ///< float marking missing cells.
inline constexpr std::string_view kExaggeration =
    "ernie:exaggeration";  ///< float.
/// double: makes the Terrain a whole planet of this radius centred at
/// the world's origin, its heights a float32 (6, n + 1, n + 1) tensor
/// of cube faces (`CubeSphereLod`, n a power of two of at least 64)
/// rather than a raster, so `ernie:origin` and `ernie:cellSize` do
/// not apply.
inline constexpr std::string_view kPlanetRadius = "ernie:radius";
/// double3: a planet's ellipsoid, its semi-axes along the Terrain's
/// x, y and z; overrides `ernie:radius` where set.
inline constexpr std::string_view kPlanetRadii = "ernie:radii";
/// double, world units: a planet's finest LOD quad
/// (`TerrainParams::finestCell`).
inline constexpr std::string_view kFinestCell = "ernie:finestCell";
/// int4 (first column, first row, columns, rows): the cells that
/// changed with the last write of heights on a GPU, so only their
/// gradients and LOD bounds are rebuilt; absent for all.
inline constexpr std::string_view kHeightsDirty =
    "ernie:heights:dirty";

// Ground plane (Ernie): an infinite horizontal plane facing +z, drawn
// with its material binding.
inline constexpr std::string_view kGroundHeight =
    "ernie:height";  ///< World z, default 0.
inline constexpr std::string_view kChecker =
    "ernie:checker";  ///< bool, default off.
inline constexpr std::string_view kCheckerSize =
    "ernie:checker:size";  ///< Square size, default 1.
/// Linear colours of the checker squares, multiplying the material's
/// albedo; default 1 and 0.5.
inline constexpr std::string_view kCheckerColor0 =
    "ernie:checker:color0";
inline constexpr std::string_view kCheckerColor1 =
    "ernie:checker:color1";

// Material (Ernie surface). BRDF parameters are `inputs:<parameter
// id>`.
inline constexpr std::string_view kBrdf =
    "ernie:brdf";  ///< BRDF model id.
inline constexpr std::string_view kAlbedo =
    "inputs:albedo";  ///< vec3 linear albedo.
/// float[] (nm, value) pairs of a reflectance spectrum, replacing the
/// albedo.
inline constexpr std::string_view kReflectance = "inputs:reflectance";
/// token: a measured reflectance spectrum Ernie embeds, by id
/// (`ReflectanceLibrary`), replacing the albedo unless
/// `inputs:reflectance` is given.
inline constexpr std::string_view kReflectanceLibrary =
    "inputs:reflectanceLibrary";
inline constexpr std::string_view kAlbedoMap =
    "inputs:albedoMap";  ///< float32 (rows, cols).
inline constexpr std::string_view kAlbedoMapOrigin =
    "inputs:albedoMap:origin";
inline constexpr std::string_view kAlbedoMapCellSize =
    "inputs:albedoMap:cellSize";
inline constexpr std::string_view kAlbedoMapNoData =
    "inputs:albedoMap:noData";
inline constexpr std::string_view kAlbedoMapEnable =
    "inputs:albedoMap:enable";  ///< bool.
/// Detail textures: float32 (rows, cols, channels) linear, or uint8
/// with sRGB albedo and linear normals; 1, 3 or 4 channels.
inline constexpr std::string_view kDetailAlbedo =
    "inputs:detailAlbedo";
inline constexpr std::string_view kDetailNormal =
    "inputs:detailNormal";
inline constexpr std::string_view kDetailTileSize =
    "inputs:detail:tileSize";
inline constexpr std::string_view kDetailAlbedoStrength =
    "inputs:detail:albedoStrength";
inline constexpr std::string_view kDetailNormalStrength =
    "inputs:detail:normalStrength";
inline constexpr std::string_view kDetailDirectXNormals =
    "inputs:detail:directXNormals";
inline constexpr std::string_view kDetailHexTiling =
    "inputs:detail:hexTiling";
inline constexpr std::string_view kDetailHexRotation =
    "inputs:detail:hexRotation";
inline constexpr std::string_view kDetailHexContrast =
    "inputs:detail:hexContrast";
inline constexpr std::string_view kDetailHexSharpness =
    "inputs:detail:hexSharpness";

// UsdShade networks. A Material's `outputs:surface.connect` names a
// UsdPreviewSurface Shader, whose inputs are constants or connections
// to other Shaders' outputs: a connection is the attribute
// `<input>.connect` holding the source's property path as text, e.g.
// "/Looks/M/Tex.outputs:rgb" (`GetConnection`). Ernie reads
// UsdPreviewSurface's diffuseColor, normal, roughness, metallic, ior,
// useSpecularWorkflow, specularColor, clearcoat and
// clearcoatRoughness, UsdUVTexture, UsdPrimvarReader_float2 and
// UsdTransform2d; a Material without a surface keeps Ernie's own
// inputs above. A Material with one shades with the `preview_surface`
// model unless it names another in `ernie:brdf`, which then reads its
// parameters from the Material as before.
//
// Reference: Pixar, "UsdPreviewSurface Specification",
// https://openusd.org/release/spec_usdpreviewsurface.html
inline constexpr std::string_view kShader = "Shader";
inline constexpr std::string_view kInfoId =
    "info:id";  ///< The shader's type, a token.
inline constexpr std::string_view kUsdPreviewSurface =
    "UsdPreviewSurface";
inline constexpr std::string_view kUsdUVTexture = "UsdUVTexture";
inline constexpr std::string_view kUsdPrimvarReaderFloat2 =
    "UsdPrimvarReader_float2";
inline constexpr std::string_view kUsdTransform2d = "UsdTransform2d";
inline constexpr std::string_view kConnectSuffix = ".connect";
inline constexpr std::string_view kSurfaceOutput = "outputs:surface";
inline constexpr std::string_view kDiffuseColor =
    "inputs:diffuseColor";  ///< vec3, 0.18 grey.
/// Tangent-space normal, +y along the texture's t; a texture's scale
/// and bias decode it.
inline constexpr std::string_view kNormalInput = "inputs:normal";
/// The `preview_surface` model's inputs (defaults 0.5, 0, 1.5, 0,
/// black, 0, 0.01); roughness and metallic may connect to one channel
/// of a texture (outputs:r, g, b or a), as glTF's metallicRoughness
/// texture does with b and g.
inline constexpr std::string_view kRoughness = "inputs:roughness";
inline constexpr std::string_view kMetallic = "inputs:metallic";
inline constexpr std::string_view kIor = "inputs:ior";
inline constexpr std::string_view kUseSpecularWorkflow =
    "inputs:useSpecularWorkflow";  ///< int.
inline constexpr std::string_view kSpecularColor =
    "inputs:specularColor";  ///< vec3.
inline constexpr std::string_view kClearcoat = "inputs:clearcoat";
inline constexpr std::string_view kClearcoatRoughness =
    "inputs:clearcoatRoughness";
/// vec3 linear RGB radiance in cd/m^2 for white (default black),
/// whatever the model.
inline constexpr std::string_view kEmissiveColor =
    "inputs:emissiveColor";
/// A material's temperature in kelvin (Ernie; default 0, none): above
/// 0 it glows by Planck's law times its emissivity, whatever the
/// model.
inline constexpr std::string_view kThermalTemperature =
    "ernie:thermal:temperature";
/// A material's emissivity as (nm, value) pairs in [0, 1] (Ernie);
/// unset, 1 minus its reflectance (Kirchhoff's law).
inline constexpr std::string_view kThermalEmissivity =
    "ernie:thermal:emissivity";
/// RenderSettings' thermal emission (Ernie): whether materials glow
/// at their temperatures (default off); the temperature in kelvin of
/// materials without their own (default 0, none); and the
/// background's (sky's or space's), an emitting environment that also
/// lights the scene (default 0, none).
inline constexpr std::string_view kThermalEnable =
    "ernie:thermal:enable";
inline constexpr std::string_view kThermalDefaultTemperature =
    "ernie:thermal:defaultTemperature";
inline constexpr std::string_view kThermalBackgroundTemperature =
    "ernie:thermal:backgroundTemperature";
/// Opacity (default 1; one texture channel, often a colour texture's
/// outputs:a) and the threshold below which a surface is cut out
/// (default 0, none).
inline constexpr std::string_view kOpacity = "inputs:opacity";
inline constexpr std::string_view kOpacityThreshold =
    "inputs:opacityThreshold";
/// UsdUVTexture: an image file (PNG, JPEG or EXR), or the tensor
/// `ernie:texture` (float32 or uint8 (rows, cols, channels), rows top
/// to bottom) in its place; texture coordinates from `inputs:st`.
inline constexpr std::string_view kFile = "inputs:file";
inline constexpr std::string_view kShaderTexture = "ernie:texture";
inline constexpr std::string_view kSt = "inputs:st";
/// "repeat" (default), "mirror", "clamp", "black" or "useMetadata"
/// (as repeat).
inline constexpr std::string_view kWrapS = "inputs:wrapS";
inline constexpr std::string_view kWrapT = "inputs:wrapT";
/// vec4 scale and bias on the texture's values (and UsdTransform2d's
/// vec2 scale).
inline constexpr std::string_view kShaderScale = "inputs:scale";
inline constexpr std::string_view kShaderBias = "inputs:bias";
/// "auto" (default: sRGB for 8-bit images, raw otherwise), "raw" or
/// "sRGB".
inline constexpr std::string_view kSourceColorSpace =
    "inputs:sourceColorSpace";
/// "auto" (default: albedo textures over 8192 texels a side), "on" or
/// "off": whether a UsdUVTexture's texture is virtual.
inline constexpr std::string_view kVirtualTexture =
    "ernie:virtualTexture";
inline constexpr std::string_view kVarname =
    "inputs:varname";  ///< Primvar reader: e.g. "st".
/// UsdTransform2d: `in` scaled, then rotated (degrees,
/// counter-clockwise), then translated.
inline constexpr std::string_view kIn = "inputs:in";
inline constexpr std::string_view kRotation = "inputs:rotation";
inline constexpr std::string_view kTranslation = "inputs:translation";

// Mesh texture coordinates (UsdGeom primvars): float32 (n, 2).
// `primvars:<name>:indices` indexes them (an indexed primvar), and
// `primvars:<name>:interpolation` (USD's metadata, as an attribute:
// "constant", "uniform", "vertex" or "faceVarying") says what each
// covers; without it the count decides: one per point is vertex, one
// per face vertex faceVarying, one per face uniform, one constant.
inline constexpr std::string_view kPrimvarsPrefix = "primvars:";
inline constexpr std::string_view kIndicesSuffix = ":indices";
inline constexpr std::string_view kInterpolationSuffix =
    ":interpolation";

// Render settings (UsdRender), with Ernie's rasterizer options under
// `ernie:`.
inline constexpr std::string_view kCameraRel =
    "camera";  ///< Path of the camera.
inline constexpr std::string_view kResolution =
    "resolution";  ///< ivec2 pixels.
/// A Camera's batch of view poses (Ernie): float32 or float64 (views,
/// 4, 4), each a local transform in the math convention (column
/// vectors, translation in the last column) that replaces the
/// camera's own; or (views, 7), each a position and a unit quaternion
/// (x, y, z, qx, qy, qz, qw), as simulators keep poses. A render
/// product through the camera renders one image per pose.
inline constexpr std::string_view kPoses = "ernie:poses";
/// token: on a Camera with `kPoses` written as a CUDA array
/// (`Scene::Write`), where the views' poses go: "gpu" (the default)
/// copies them device to device and the views' matrices are computed
/// on the GPU, with no wait for the host; "host" downloads them
/// first, so every CPU stage (frustum tests, terrain levels of
/// detail, single-view shadow cascades) sees the exact poses. Only
/// contiguous float32 arrays stay on the GPU; others download.
inline constexpr std::string_view kPosesDevice = "ernie:poses:device";
inline constexpr std::string_view kHost = "host";
/// The world a prim and its descendants belong to (Ernie): int,
/// inherited from the nearest ancestor that has one. A batch of views
/// renders environments ("worlds") that may overlap in space, such as
/// one per reinforcement learning environment, each view seeing the
/// prims of its own world and those of no world (-1, the default),
/// which every world shares. On a Camera, the world its views see; -1
/// sees every world.
inline constexpr std::string_view kWorld = "ernie:world";
/// Per-element worlds (Ernie): int array. On a PointInstancer, each
/// instance's world (in place of the instancer's `ernie:world`); on a
/// Camera with `ernie:poses`, each view's world. A GPU array is
/// copied to the host once per write, as the CPU plans the views'
/// worlds.
inline constexpr std::string_view kWorlds = "ernie:worlds";
/// int: the frame number noise and samples derive from (Ernie); unset
/// counts the images of each render product a delegate renders.
inline constexpr std::string_view kFrame = "ernie:frame";
/// Paths of the lights to render with (Ernie); unset renders with
/// every visible light outside `kSessionRoot`.
inline constexpr std::string_view kLightsRel = "ernie:lights";
inline constexpr std::string_view kProducts =
    "products";  ///< Paths of RenderProducts.
inline constexpr std::string_view kOrderedVars =
    "orderedVars";  ///< Paths of RenderVars.
/// Sensor filter chain of a RenderProduct (Ernie): paths of
/// SensorFilter prims in order.
inline constexpr std::string_view kFilters = "ernie:filters";
/// bool, on a RenderProduct (Ernie): whether its render vars mapped
/// to linear GPU tensors (`MapDevice::Gpu`) are written in place by
/// its later frames, which the sensor writes into buffers beside its
/// images so their maps copy nothing (true, the default), or copied
/// at each map (false, and no such buffers).
inline constexpr std::string_view kOutputsInPlace =
    "ernie:outputs:inPlace";
inline constexpr std::string_view kSensorFilter =
    "SensorFilter";  ///< Prim type.
/// A SensorFilter prim's filter: "lens", "qe", "noise", "adc", "cie",
/// "exposure", or the id of a custom filter (`SensorFilterModel`),
/// whose parameters are the prim's inputs:* attributes.
inline constexpr std::string_view kFilterType = "ernie:filter:type";
inline constexpr std::string_view kSourceName =
    "sourceName";  ///< RenderVar name, e.g. "LdrColor".
/// float luminance (cd/m^2) of the scene's ambient light: white,
/// uniformly from all around, besides the lights.
inline constexpr std::string_view kAmbient = "ernie:ambient";
/// float3 linear RGB of the ambient light (its luminance
/// `kAmbient`'s).
inline constexpr std::string_view kAmbientColor =
    "ernie:ambientColor";
/// How dome lights light the scene, as Omniverse RTX's dome lighting
/// modes: "approximate" (default; diffuse and specular, unoccluded),
/// "full" (also shadowed, where the dome's `inputs:shadow:enable`
/// allows: an HDR sky's sun as a shadowed sun, terrain horizon maps,
/// and ambient occlusion from the G-buffer over a world radius of
/// `ernie:domeLight:shadowRadius`, default 1), "directions" (the full
/// mode, with the dome's diffuse light from
/// `ernie:domeLight:directions` shadowed directions of about equal
/// energy, default 16, as many as light slots allow) or "limited"
/// (reflections only, no diffuse).
inline constexpr std::string_view kDomeLightMode =
    "ernie:domeLight:mode";
inline constexpr std::string_view kDomeShadowRadius =
    "ernie:domeLight:shadowRadius";
inline constexpr std::string_view kDomeDirections =
    "ernie:domeLight:directions";
/// token, on a Terrain: how it is coloured, "shaded" (its
/// material, lit; the default), "colormap" (a height ramp, unlit),
/// "shadedColormap" (the ramp times the material, lit) or
/// "hillshade".
inline constexpr std::string_view kColorMode = "ernie:colorMode";
inline constexpr std::string_view kShaded = "shaded";
inline constexpr std::string_view kColormap = "colormap";
inline constexpr std::string_view kShadedColormap = "shadedColormap";
/// A GIS hillshade, grey and unlit: the cosine of the angle between
/// the terrain's normal and a fixed light at
/// `ernie:hillshade:azimuth` (degrees clockwise from north) and
/// `ernie:hillshade:altitude` (degrees above the horizon).
inline constexpr std::string_view kHillshade = "hillshade";
inline constexpr std::string_view kHillshadeAzimuth =
    "ernie:hillshade:azimuth";
inline constexpr std::string_view kHillshadeAltitude =
    "ernie:hillshade:altitude";
/// token: "off", "overlay" (edges over shaded surfaces) or
/// "hiddenLine" (edges on flat, unlit surfaces); a bool means
/// "overlay" when true and "off" when false.
inline constexpr std::string_view kWireframe = "ernie:wireframe";
inline constexpr std::string_view kOff = "off";
inline constexpr std::string_view kOverlay = "overlay";
inline constexpr std::string_view kHiddenLine = "hiddenLine";
/// string, on a RenderProduct or else its RenderSettings: the
/// overlays (`Scene::GetOverlay`) drawn over its LdrColor, "*" for
/// all, else names separated by commas or spaces; none by default, so
/// observations stay free of debug drawing.
inline constexpr std::string_view kOverlays = "ernie:overlays";
/// token, on a RenderProduct or else its RenderSettings: minimal
/// shading for throughput (`MinimalMode`), after RTX Minimal: "off"
/// (the full pipeline, the default), "none" (black colour, for depth,
/// normals and ids alone), "constantDiffuse" (`kMinimalConstantColor`
/// lit by the sun and ambient), "texturedDiffuse" (surfaces'
/// diffuse colours so lit) or "diffuseGlossyEmission" (those, with
/// the lights' glossy highlights and surfaces' emission). Its
/// products render LdrColor,
/// DistanceToCamera, DistanceToImagePlane, Normal, PrimId and
/// InstanceId.
inline constexpr std::string_view kMinimalMode = "ernie:minimal:mode";
inline constexpr std::string_view kNone = "none";
inline constexpr std::string_view kConstantDiffuse =
    "constantDiffuse";
inline constexpr std::string_view kTexturedDiffuse =
    "texturedDiffuse";
inline constexpr std::string_view kDiffuseGlossyEmission =
    "diffuseGlossyEmission";
/// vec3 linear, on a RenderProduct or else its RenderSettings:
/// "constantDiffuse"'s colour, white by default.
inline constexpr std::string_view kMinimalConstantColor =
    "ernie:minimal:constantColor";
/// token, on RenderSettings: the precision the rasterizer keeps each
/// band's radiance in between shading and the sensor, "float32" (the
/// default) or "float16", which halves that memory and its traffic
/// (a 16-bit float's 11 bits of mantissa, to 65504).
inline constexpr std::string_view kSpectralPrecision =
    "ernie:spectral:precision";
inline constexpr std::string_view kFloat32 = "float32";
inline constexpr std::string_view kFloat16 = "float16";
/// token: how the rasterizer culls what lies outside a view: "off",
/// "cpu" (whole draws) or "gpu" (whole draws, then every instance on
/// the GPU; the default).
inline constexpr std::string_view kCulling = "ernie:culling";
inline constexpr std::string_view kCpu = "cpu";
inline constexpr std::string_view kGpu = "gpu";
/// token: whether views also cull instances hidden behind others with
/// GPU culling: "off" (the default), "occluders" (behind terrain,
/// ground and meshes) or "twoPass" (also behind instances visible in
/// the view's previous frame).
inline constexpr std::string_view kOcclusion = "ernie:occlusion";
inline constexpr std::string_view kOccluders = "occluders";
inline constexpr std::string_view kTwoPass = "twoPass";
inline constexpr std::string_view kClearColor =
    "ernie:clearColor";  ///< vec3 linear.
inline constexpr std::string_view kShadowCascades =
    "ernie:shadows:cascades";  ///< int.
inline constexpr std::string_view kShadowResolution =
    "ernie:shadows:resolution";  ///< int.
/// int: texels along each side of the sun's map of each world in a
/// batch of views; 0 or unset for automatic, four per pixel of the
/// image's larger side from 256 to 2048.
inline constexpr std::string_view kShadowWorldResolution =
    "ernie:shadows:worldResolution";
/// token: which casters shadow a batch of views of several worlds:
/// "separate" (a map per world, the default), "shared" (one map of
/// what the worlds share) or "merged" (one map of every world's
/// casters).
inline constexpr std::string_view kShadowWorlds =
    "ernie:shadows:worlds";
inline constexpr std::string_view kSeparate = "separate";
inline constexpr std::string_view kShared = "shared";
inline constexpr std::string_view kMerged = "merged";
/// token: what each world's own map covers: "scene" (the default) or
/// "world" (the world's casters grown by `ernie:shadows:worldMargin`
/// metres, default 1).
inline constexpr std::string_view kShadowWorldFit =
    "ernie:shadows:worldFit";
// RenderSettings' volume quality: steps through volumes and towards
// lights, the sky and the air, multiple scattering and shadow maps.
inline constexpr std::string_view kVolumeStepsPerVoxel =
    "ernie:volumes:stepsPerVoxel";
inline constexpr std::string_view kVolumeMaxSteps =
    "ernie:volumes:maxSteps";
inline constexpr std::string_view kVolumeShadowSteps =
    "ernie:volumes:shadowSteps";
inline constexpr std::string_view kVolumeSkyLightSteps =
    "ernie:volumes:skyLightSteps";
inline constexpr std::string_view kVolumeAirSteps =
    "ernie:volumes:airSteps";
inline constexpr std::string_view kVolumeMaxOctaves =
    "ernie:volumes:maxOctaves";
inline constexpr std::string_view kVolumeShadowResolution =
    "ernie:volumes:shadowResolution";
inline constexpr std::string_view kVolumeLightCache =
    "ernie:volumes:lightCache";
inline constexpr std::string_view kVolumeTemporal =
    "ernie:volumes:temporal";
inline constexpr std::string_view kVolumeTemporalBlock =
    "ernie:volumes:temporalBlock";
inline constexpr std::string_view kVolumeTemporalAnimated =
    "ernie:volumes:temporalAnimated";
inline constexpr std::string_view kVolumeStepsPerPixel =
    "ernie:volumes:stepsPerPixel";
inline constexpr std::string_view kVolumeThinOpticalDepth =
    "ernie:volumes:thinOpticalDepth";
inline constexpr std::string_view kScene = "scene";
inline constexpr std::string_view kWorldFit = "world";
inline constexpr std::string_view kShadowWorldMargin =
    "ernie:shadows:worldMargin";
/// bool: the first shadowed distant light of a single view shadows
/// through a virtual shadow map (`VirtualShadowMap`), about a texel a
/// pixel from near to far, rather than its cascades; default off.
inline constexpr std::string_view kShadowVirtual =
    "ernie:shadows:virtual";
/// int: the virtual shadow map's physical pages of 128² texels (64
/// KiB each); default 1024.
inline constexpr std::string_view kShadowVirtualPages =
    "ernie:shadows:virtualPages";
/// float: added to the level a virtual shadow map picks for a point
/// (each level doubles its texels); default 0, about a texel a pixel.
inline constexpr std::string_view kShadowVirtualBias =
    "ernie:shadows:virtualBias";
/// token: what the virtual shadow map's debugging shows: "off" (the
/// default), "levels" (surfaces tinted by the clipmap level that
/// shadows them) or "pages" (black where no page backs a point: past
/// the pool, or not marked).
inline constexpr std::string_view kShadowVirtualDebug =
    "ernie:shadows:virtualDebug";
inline constexpr std::string_view kLevels = "levels";
inline constexpr std::string_view kPages = "pages";
/// token: what bounds a single view's sun cascades: "scene" (the
/// default: the casters and where their shadows land, the same every
/// time a frame renders) or "visible" (sample distribution shadow
/// maps: the depths a recent frame of the same product saw, tighter
/// and sharper, but an image then depends on the frames before it).
inline constexpr std::string_view kShadowFit = "ernie:shadows:fit";
/// float, stops: exposure compensation on the photographic outputs
/// (HdrColor's display, LdrColor), on top of the camera's own
/// exposure; default 0.
inline constexpr std::string_view kExposureCompensation =
    "ernie:exposure";
/// bool: measure each frame's scene luminance (its histogram) for
/// automatic exposure and exposure meters, read back from a later
/// frame (`RenderDelegate::MeteredLuminance`); default off.
inline constexpr std::string_view kMeter = "ernie:meter";
inline constexpr std::string_view kVisible = "visible";
inline constexpr std::string_view kShadowSoftness =
    "ernie:shadows:softness";
inline constexpr std::string_view kShadowNormalOffset =
    "ernie:shadows:normalOffset";
inline constexpr std::string_view kShadowSplitLambda =
    "ernie:shadows:splitLambda";
inline constexpr std::string_view kShadowShowCascades =
    "ernie:shadows:showCascades";
inline constexpr std::string_view kLodEnable = "ernie:lod:enable";
inline constexpr std::string_view kLodDetail = "ernie:lod:detail";
inline constexpr std::string_view kLodMorphStart =
    "ernie:lod:morphStart";
inline constexpr std::string_view kLodFreeze = "ernie:lod:freeze";
inline constexpr std::string_view kLodShowLevels =
    "ernie:lod:showLevels";
/// float: the simplification error, in pixels, up to which instanced
/// meshes draw a coarser level of detail (with GPU culling); 0 always
/// draws the full meshes. Default 1.
inline constexpr std::string_view kLodMeshError =
    "ernie:lod:meshError";

/// RenderVar source names.
inline constexpr std::string_view kLdrColor = "LdrColor";

}  // namespace ernie::tokens
