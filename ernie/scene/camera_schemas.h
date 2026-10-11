// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/tensor.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

/// UsdGeom's Camera as a reflected schema (`prim_type.h`), with
/// Ernie's lens, exposure, shutter, scan, spectral and sensor
/// attributes and its batch of view poses. Each declaration's
/// fallback is the value `CameraParamsOf` resolves an unset attribute
/// to: a default-constructed `CameraParams`'s.
///
/// References:
/// - Pixar, "UsdGeomCamera",
///   https://openusd.org/release/api/class_usd_geom_camera.html

// clang-format off
#define ERNIE_CAMERA_ATTRIBUTES(X)                                                                 \
    X(std::string, Projection, ::ernie::tokens::kProjection, "perspective",                        \
      (::ernie::AttributeMeta{.doc = "Orthographic reads the apertures in tenths of a scene unit.", \
                              .group = "Lens",                                                     \
                              .options = {::ernie::tokens::kPerspective,                           \
                                          ::ernie::tokens::kOrthographic}}))                       \
    X(double, FocalLength, ::ernie::tokens::kFocalLength, 50.0,                                    \
      (::ernie::AttributeMeta{.doc = "mm.", .group = "Lens", .min = 1.0, .max = 2000.0}))          \
    X(double, VerticalAperture, ::ernie::tokens::kVerticalAperture, 15.2908,                       \
      (::ernie::AttributeMeta{.doc = "mm; tenths of a scene unit when orthographic.",            \
                              .group = "Lens", .min = 0.1, .max = 100.0}))                         \
    X(double, HorizontalAperture, ::ernie::tokens::kHorizontalAperture, 20.955,                    \
      (::ernie::AttributeMeta{.doc = "mm; not read: the image's aspect sets the horizontal "       \
                                     "extent. Written for other USD readers.",                     \
                              .group = "Lens", .min = 0.1, .max = 100.0}))                         \
    X(double, HorizontalApertureOffset, ::ernie::tokens::kHorizontalApertureOffset, 0.0,           \
      (::ernie::AttributeMeta{.label = "Horizontal offset", .doc = "mm.",.group = "Lens",          \
                              .min = -50.0, .max = 50.0}))                                         \
    X(double, VerticalApertureOffset, ::ernie::tokens::kVerticalApertureOffset, 0.0,               \
      (::ernie::AttributeMeta{.label = "Vertical offset", .doc = "mm.",.group = "Lens",            \
                              .min = -50.0, .max = 50.0}))                                         \
    X(glm::dvec2, ClippingRange, ::ernie::tokens::kClippingRange, glm::dvec2(1.0, 1e6),            \
      (::ernie::AttributeMeta{.doc = "Near and far, scene units.", .group = "Lens"}))              \
    X(double, FStop, ::ernie::tokens::kFStop, 0.0,                                                 \
      (::ernie::AttributeMeta{.label = "Lens f-number",                                            \
                              .doc = "For depth of field and diffraction; 0 is a pinhole, all in " \
                                     "focus.",                                                     \
                              .group = "Lens", .max = 64.0}))                                      \
    X(double, FocusDistance, ::ernie::tokens::kFocusDistance, 0.0,                                 \
      (::ernie::AttributeMeta{.doc = "Scene units; 0 focuses at infinity.", .group = "Lens",       \
                              .max = 1e4}))                                                        \
    X(std::string, LensProjection, ::ernie::tokens::kLensProjection, "perspective",                \
      (::ernie::AttributeMeta{.doc = "Rectilinear, or a Kannala-Brandt fisheye.",                  \
                              .group = "Lens",                                                     \
                              .options = {::ernie::tokens::kPerspective,                           \
                                          ::ernie::tokens::kFisheye}}))                            \
    X(std::vector<float>, LensDistortion, ::ernie::tokens::kLensDistortion, std::vector<float>{},  \
      (::ernie::AttributeMeta{.doc = "Brown-Conrady k1 k2 k3 p1 p2; missing ones are 0.",          \
                              .group = "Lens"}))                                                   \
    X(std::vector<float>, LensFisheye, ::ernie::tokens::kLensFisheye, std::vector<float>{},        \
      (::ernie::AttributeMeta{.doc = "Kannala-Brandt k1 k2 k3 k4 of a fisheye; none is "           \
                                     "equidistant.",                                               \
                              .group = "Lens"}))                                                   \
    X(bool, LensVignetting, ::ernie::tokens::kLensVignetting, false,                               \
      (::ernie::AttributeMeta{.label = "Vignetting",                                               \
                              .doc = "Natural falloff of irradiance off axis.", .group = "Lens"})) \
    X(double, LensTransmission, ::ernie::tokens::kLensTransmission, 1.0,                           \
      (::ernie::AttributeMeta{.label = "Transmission", .doc = "Fraction of light the optics pass.", \
                              .group = "Lens", .max = 1.0}))                                       \
    X(std::vector<float>, LensChromaticScale, ::ernie::tokens::kLensChromaticScale,                \
      std::vector<float>{},                                                                        \
      (::ernie::AttributeMeta{.doc = "(nm, scale) pairs: lateral colour, the image's scale per "   \
                                     "wavelength.",                                                \
                              .group = "Lens"}))                                                   \
    X(std::vector<float>, LensChromaticFocus, ::ernie::tokens::kLensChromaticFocus,                \
      std::vector<float>{},                                                                        \
      (::ernie::AttributeMeta{.doc = "(nm, mm) pairs: axial colour, the focal length's shift per " \
                                     "wavelength.",                                                \
                              .group = "Lens"}))                                                   \
    X(::ernie::Tensor, LensRays, ::ernie::tokens::kLensRays, nullptr,                              \
      (::ernie::AttributeMeta{.doc = "float32 (rows, columns, 3) CPU tensor of unit directions in " \
                                     "the camera's frame every rays step pixels, in place of the " \
                                     "projection and distortion.",                                 \
                              .group = "Lens"}))                                                   \
    X(int, LensRaysStep, ::ernie::tokens::kLensRaysStep, 8,                                        \
      (::ernie::AttributeMeta{.label = "Rays step", .doc = "Pixels between the lens's rays.",      \
                              .group = "Lens", .min = 1.0, .max = 256.0}))                         \
    X(double, Exposure, ::ernie::tokens::kCameraExposure, 0.0,                                     \
      (::ernie::AttributeMeta{.doc = "Stops; scales the photographic outputs.",.group = "Exposure", \
                              .min = -16.0, .max = 16.0}))                                         \
    X(double, ExposureTime, ::ernie::tokens::kExposureTime, 1.0,                                   \
      (::ernie::AttributeMeta{.doc = "Seconds.",.group = "Exposure", .min = 1e-6, .max = 10.0}))   \
    X(double, ExposureIso, ::ernie::tokens::kExposureIso, 100.0,                                   \
      (::ernie::AttributeMeta{.label = "ISO", .group = "Exposure", .min = 25.0, .max = 102400.0})) \
    X(double, ExposureFStop, ::ernie::tokens::kExposureFStop, 1.0,                                 \
      (::ernie::AttributeMeta{.label = "f-number", .group = "Exposure", .min = 0.5, .max = 64.0})) \
    X(double, ExposureResponsivity, ::ernie::tokens::kExposureResponsivity, 1.0,                   \
      (::ernie::AttributeMeta{.label = "Responsivity", .group = "Exposure", .max = 100.0}))        \
    X(std::string, ExposureMode, ::ernie::tokens::kExposureMode, "manual",                         \
      (::ernie::AttributeMeta{.label = "Mode",                                                     \
                              .doc = "\"auto\" keys the photographic outputs to middle grey "      \
                                     "from the metered scene, on top of the settings above; "      \
                                     "\"manual\" by them alone.",                                  \
                              .group = "Exposure",                                                 \
                              .options = {::ernie::tokens::kManual, ::ernie::tokens::kAuto}}))     \
    X(double, ExposureAdaptation, ::ernie::tokens::kExposureAdaptation, 0.0,                       \
      (::ernie::AttributeMeta{.label = "Adaptation",                                               \
                              .doc = "Frames automatic exposure takes to cover 63% of a "          \
                                     "change; 0 at once.",                                         \
                              .group = "Exposure", .max = 1000.0}))                                \
    X(double, ShutterOpen, ::ernie::tokens::kShutterOpen, 0.0,                                     \
      (::ernie::AttributeMeta{.label = "Open",                                                     \
                              .doc = "Frames from the frame's time; motion blurs from here to "    \
                                     "the close.",                                               \
                              .group = "Shutter", .min = -2.0, .max = 1.0}))                       \
    X(double, ShutterClose, ::ernie::tokens::kShutterClose, 0.0,                                   \
      (::ernie::AttributeMeta{.label = "Close", .doc = "Frames from the frame's time.",             \
                              .group = "Shutter", .min = -2.0, .max = 1.0}))                       \
    X(int, ShutterSamples, ::ernie::tokens::kShutterSamples, 8,                                    \
      (::ernie::AttributeMeta{.label = "Samples", .doc = "Sub-frames rendered across the shutter.", \
                              .group = "Shutter", .min = 1.0, .max = 64.0}))                       \
    X(double, ShutterRolling, ::ernie::tokens::kShutterRolling, 0.0,                               \
      (::ernie::AttributeMeta{.label = "Rolling readout",                                          \
                              .doc = "Frames between the top row's exposure and the bottom "       \
                                     "row's; 0 is global.",                                       \
                              .group = "Shutter", .max = 2.0}))                                    \
    X(std::string, Scan, ::ernie::tokens::kCameraScan, "frame",                                    \
      (::ernie::AttributeMeta{.doc = "Pushbroom: a line scanner whose rows run from the last "     \
                                     "frame's pose to this one's.",                               \
                              .group = "Shutter",                                                  \
                              .options = {::ernie::tokens::kFrameScan,                             \
                                          ::ernie::tokens::kPushbroom}}))                          \
    X(double, PushbroomLineOffset, ::ernie::tokens::kPushbroomLineOffset, 0.0,                     \
      (::ernie::AttributeMeta{.label = "Line offset",                                              \
                              .doc = "mm; the pushbroom's detector line from the principal point.", \
                              .group = "Shutter", .min = -50.0, .max = 50.0}))                     \
    X(glm::dvec2, SpectralRange, ::ernie::tokens::kSpectralRange, glm::dvec2(380.0, 780.0),        \
      (::ernie::AttributeMeta{.doc = "nm.",.group = "Sensor"}))                                    \
    X(int, SpectralBands, ::ernie::tokens::kSpectralBands, 16,                                     \
      (::ernie::AttributeMeta{.label = "Bands", .group = "Sensor", .min = 1.0, .max = 64.0}))      \
    X(std::string, SensorType, ::ernie::tokens::kSensorType, "rgb",                                \
      (::ernie::AttributeMeta{.label = "Type", .group = "Sensor", .options = {"rgb", "mono"}}))    \
    X(bool, SensorNoise, ::ernie::tokens::kSensorNoise, true,                                      \
      (::ernie::AttributeMeta{.label = "Noise", .group = "Sensor"}))                               \
    X(double, SensorFullWell, ::ernie::tokens::kSensorFullWell, 20000.0,                           \
      (::ernie::AttributeMeta{.label = "Full well", .doc = "Electrons.", .group = "Sensor",         \
                              .min = 100.0, .max = 1e6}))                                          \
    X(double, SensorReadNoise, ::ernie::tokens::kSensorReadNoise, 3.0,                             \
      (::ernie::AttributeMeta{.label = "Read noise", .doc = "Electrons, standard deviation.", .group = "Sensor",        \
                              .max = 100.0}))                                                      \
    X(double, SensorDarkCurrent, ::ernie::tokens::kSensorDarkCurrent, 0.1,                         \
      (::ernie::AttributeMeta{.label = "Dark current", .doc = "Electrons per second.",              \
                              .group = "Sensor", .max = 1000.0}))                                  \
    X(double, SensorPrnu, ::ernie::tokens::kSensorPrnu, 0.005,                                     \
      (::ernie::AttributeMeta{.label = "PRNU",                                                     \
                              .doc = "Standard deviation of the per-pixel relative gain.",          \
                              .group = "Sensor", .max = 0.2}))                                     \
    X(double, SensorDsnu, ::ernie::tokens::kSensorDsnu, 0.5,                                       \
      (::ernie::AttributeMeta{.label = "DSNU", .doc = "Electrons, standard deviation of the per-pixel offset.", .group = "Sensor",              \
                              .max = 100.0}))                                                      \
    X(double, SensorElectronsPerDn, ::ernie::tokens::kSensorGain, 0.0,                             \
      (::ernie::AttributeMeta{.label = "Electrons per DN",                                         \
                              .doc = "At ISO 100; 0 for the full well over the ADC range.",         \
                              .group = "Sensor", .max = 1000.0}))                                  \
    X(int, SensorAdcBits, ::ernie::tokens::kSensorAdcBits, 12,                                     \
      (::ernie::AttributeMeta{.label = "ADC bits", .group = "Sensor", .min = 1.0, .max = 16.0}))   \
    X(double, SensorBlackLevel, ::ernie::tokens::kSensorBlackLevel, 0.0,                           \
      (::ernie::AttributeMeta{.label = "Black level", .doc = "DN.", .group = "Sensor",              \
                              .max = 4096.0}))                                                     \
    X(double, SensorPixelPitch, ::ernie::tokens::kSensorPixelPitch, 5.0,                           \
      (::ernie::AttributeMeta{.label = "Pixel pitch", .doc = "Micrometres, of orthographic cameras.", \
                              .group = "Sensor", .min = 0.1, .max = 100.0}))                       \
    X(int, SensorSeed, ::ernie::tokens::kSensorSeed, 0,                                            \
      (::ernie::AttributeMeta{.label = "Seed", .doc = "Of the sensor's noise.",                    \
                              .group = "Sensor"}))                                                 \
    X(std::vector<float>, SensorQeR, ::ernie::tokens::kSensorQeR, std::vector<float>{},            \
      (::ernie::AttributeMeta{.doc = "(nm, electrons per photon) pairs of an rgb sensor's red "    \
                                     "channel; unset keeps the generic curve.",                    \
                              .group = "Sensor"}))                                                 \
    X(std::vector<float>, SensorQeG, ::ernie::tokens::kSensorQeG, std::vector<float>{},            \
      (::ernie::AttributeMeta{.doc = "(nm, electrons per photon) pairs of an rgb sensor's green "  \
                                     "channel; unset keeps the generic curve.",                    \
                              .group = "Sensor"}))                                                 \
    X(std::vector<float>, SensorQeB, ::ernie::tokens::kSensorQeB, std::vector<float>{},            \
      (::ernie::AttributeMeta{.doc = "(nm, electrons per photon) pairs of an rgb sensor's blue "   \
                                     "channel; unset keeps the generic curve.",                    \
                              .group = "Sensor"}))                                                 \
    X(std::vector<float>, SensorQeMono, ::ernie::tokens::kSensorQeMono, std::vector<float>{},      \
      (::ernie::AttributeMeta{.doc = "(nm, electrons per photon) pairs of a mono sensor's "        \
                                     "channel; unset keeps the generic silicon curve.",            \
                              .group = "Sensor"}))                                                 \
    X(::ernie::Tensor, Poses, ::ernie::tokens::kPoses, nullptr,                                    \
      (::ernie::AttributeMeta{.doc = "float32 or float64 (views, 4, 4) local transforms, or "      \
                                     "(views, 7) positions and unit quaternions: one image per "   \
                                     "pose, each in place of the camera's own transform.",         \
                              .group = "Views"}))                                                  \
    X(std::string, PosesDevice, ::ernie::tokens::kPosesDevice, "gpu",                              \
      (::ernie::AttributeMeta{.doc = "Where poses written as a CUDA array go: \"gpu\" keeps them " \
                                     "there, \"host\" downloads them first.",                      \
                              .group = "Views",                                                    \
                              .options = {::ernie::tokens::kGpu, ::ernie::tokens::kHost}}))        \
    X(::ernie::Tensor, Worlds, ::ernie::tokens::kWorlds, nullptr,                                  \
      (::ernie::AttributeMeta{.doc = "int (views,): each pose's world, in place of the camera's "  \
                                     "world.",                                                     \
                              .group = "Views"}))

#define ERNIE_CAMERA_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_CAMERA_ATTRIBUTES(X)
// clang-format on

namespace ernie {

/// A camera looking down its -z axis with +y up: UsdGeomCamera's
/// projection, apertures, clipping, exposure and shutter, and Ernie's
/// lens, scan, spectral sampling, sensor and view poses.
/// `CameraParamsOf` resolves it for a frame.
class Camera : public Imageable {
    ERNIE_SCHEMA_BODY(Camera, Imageable, ERNIE_CAMERA_ALL)
    ERNIE_SCHEMA_DEFINE(Camera)

    /// The quantum efficiency (nm, value) pairs of the sensor channel
    /// `channel` ("r", "g", "b" or "mono"), or none when unset or for
    /// another channel name.
    std::optional<std::vector<float>>
    GetSensorQe(std::string_view channel) const;
};

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> CameraSchemaTypes();

}  // namespace ernie
