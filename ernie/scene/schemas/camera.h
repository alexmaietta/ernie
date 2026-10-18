// SPDX-License-Identifier: BSD-3-Clause
/// @file camera.h
/// @brief UsdGeomCamera: a view's projection, lens and exposure.
///
/// References:
/// - Pixar, UsdGeomCamera, https://openusd.org/release/api/class_usd_geom_camera.html
#pragma once

#include "ernie/scene/schemas/geom.h"

// clang-format off
// Lens lengths are metres, whatever a USD stage's scale: USD's tenths of a scene unit are
// converted on import and export.
#define ERNIE_LENS_LENGTH(X, Name, usdName, fallback, text)                                        \
    X(double, Name, usdName, fallback, text ", metres.", .group = "Lens", .lower = AtLeast(0.0),   \
      .finite = true)

#define ERNIE_CAMERA_ATTRIBUTES(X)                                                                 \
    X(std::string, Projection, "projection", "perspective", "How it projects the scene.",          \
      .group = "Lens", .options = {"perspective", "orthographic"})                                 \
    ERNIE_LENS_LENGTH(X, HorizontalAperture, "horizontalAperture", 0.020955, "Film width")         \
    ERNIE_LENS_LENGTH(X, VerticalAperture, "verticalAperture", 0.0152908, "Film height")           \
    X(double, HorizontalApertureOffset, "horizontalApertureOffset", 0.0,                           \
      "The film's offset from the lens axis along x, metres.", .group = "Lens", .finite = true)    \
    X(double, VerticalApertureOffset, "verticalApertureOffset", 0.0,                               \
      "The film's offset from the lens axis along y, metres.", .group = "Lens", .finite = true)    \
    ERNIE_LENS_LENGTH(X, FocalLength, "focalLength", 0.05, "The lens's focal length")              \
    X(glm::dvec2, ClippingRange, "clippingRange", glm::dvec2(1.0, 1000000.0),                      \
      "Near and far distances in metres.", .group = "Lens", .lower = Above(0.0))                   \
    X(Tensor, ClippingPlanes, "clippingPlanes", Tensor(),                                          \
      "(n, 4) planes (a, b, c, d) keeping points where a x + b y + c z + d >= 0.",                 \
      .group = "Lens", .components = 4)                                                            \
    X(double, FStop, "fStop", 0.0, "The lens's f-number; 0 for a pinhole, all in focus.",          \
      .group = "Lens", .lower = AtLeast(0.0))                                                      \
    X(double, FocusDistance, "focusDistance", 0.0, "Metres to the plane in focus.",                \
      .group = "Lens", .lower = AtLeast(0.0))                                                      \
    X(std::string, StereoRole, "stereoRole", "mono", "Which eye it is.", .group = "Lens",          \
      .options = {"mono", "left", "right"})                                                        \
    X(double, ShutterOpen, "shutter:open", 0.0, "Frames from the time code the shutter opens.",   \
      .group = "Shutter")                                                                          \
    X(double, ShutterClose, "shutter:close", 0.0, "Frames from the time code it closes.",          \
      .group = "Shutter")                                                                          \
    X(double, Exposure, "exposure", 0.0, "Stops of exposure added to the image.",                  \
      .group = "Exposure", .finite = true)                                                         \
    X(double, ExposureIso, "exposure:iso", 100.0, "The sensor's ISO speed.", .group = "Exposure",  \
      .lower = Above(0.0))                                                                         \
    X(double, ExposureTime, "exposure:time", 1.0, "Seconds of exposure.", .group = "Exposure",     \
      .lower = AtLeast(0.0))                                                                       \
    X(double, ExposureFStop, "exposure:fStop", 1.0, "The f-number exposure is computed with.",     \
      .group = "Exposure", .lower = Above(0.0))                                                    \
    X(double, ExposureResponsivity, "exposure:responsivity", 1.0, "The sensor's responsivity.",    \
      .group = "Exposure", .lower = AtLeast(0.0))                                                  \
    X(std::string, ExposureMode, "ernie:exposure:mode", "manual",                                  \
      "auto keys the image to middle grey from the metered scene, after the settings above.",      \
      .group = "Exposure", .options = {"manual", "auto"})                                          \
    X(double, ExposureAdaptation, "ernie:exposure:adaptation", 0.0,                                \
      "Frames auto exposure takes to cover 63% of a change; 0 at once.", .group = "Exposure",      \
      .lower = AtLeast(0.0))                                                                       \
    X(int, ShutterSamples, "ernie:shutter:samples", 8, "Sub-frames rendered across the shutter.",  \
      .group = "Shutter", .lower = AtLeast(1.0))                                                   \
    X(double, ShutterRolling, "ernie:shutter:rolling", 0.0,                                        \
      "Frames between the top row's exposure and the bottom row's; 0 global.", .group = "Shutter", \
      .lower = AtLeast(0.0))                                                                       \
    X(std::string, Scan, "ernie:camera:scan", "frame",                                             \
      "pushbroom: a line scanner whose rows run from the last frame's pose to this one's.",        \
      .group = "Shutter", .options = {"frame", "pushbroom"})                                       \
    X(double, PushbroomLineOffset, "ernie:pushbroom:lineOffset", 0.0,                              \
      "Metres from the principal point to the pushbroom's detector line.",                         \
      .group = "Shutter", .finite = true)

#define ERNIE_LENS_API_ATTRIBUTES(X)                                                               \
    X(std::string, LensProjection, "ernie:lens:projection", "perspective",                         \
      "Rectilinear, or a Kannala-Brandt fisheye.", .group = "Lens",                                \
      .options = {"perspective", "fisheye"})                                                       \
    X(std::vector<float>, LensDistortion, "ernie:lens:distortion", std::vector<float>{},           \
      "Brown-Conrady k1 k2 k3 p1 p2; missing ones are 0.", .group = "Lens")                        \
    X(std::vector<float>, LensFisheye, "ernie:lens:fisheye", std::vector<float>{},                 \
      "Kannala-Brandt k1 k2 k3 k4 of a fisheye; none is equidistant.", .group = "Lens")            \
    X(bool, LensVignetting, "ernie:lens:vignetting", false, "Natural falloff off axis.",           \
      .group = "Lens")                                                                             \
    X(double, LensTransmission, "ernie:lens:transmission", 1.0, "The fraction of light passed.",   \
      .group = "Lens", .lower = AtLeast(0.0), .upper = AtMost(1.0))                                \
    X(std::vector<float>, LensChromaticScale, "ernie:lens:chromatic:scale", std::vector<float>{},  \
      "(nm, scale) pairs: lateral colour, the image's scale per wavelength.", .group = "Lens")     \
    X(std::vector<float>, LensChromaticFocus, "ernie:lens:chromatic:focus", std::vector<float>{},  \
      "(nm, metres) pairs: axial colour, the focal length's shift per wavelength.",                \
      .group = "Lens")                                                                             \
    X(Tensor, LensRays, "ernie:lens:rays", Tensor(),                                               \
      "float32 (rows, columns, 3) unit directions every rays step pixels, in place of the "        \
      "projection and distortion.",                                                                \
      .group = "Lens", .ranks = {3})                                                               \
    X(int, LensRaysStep, "ernie:lens:raysStep", 8, "Pixels between the lens's rays.",              \
      .group = "Lens", .lower = AtLeast(1.0))

#define ERNIE_POLARIZER_API_ATTRIBUTES(X)                                                          \
    X(std::string, PolarizerType, "ernie:polarizer:type", "linear",                                \
      "linear in front of the lens, or a mosaic of 90, 45 / 135, 0 degree polarizers over 2 x 2 "  \
      "pixels, as a polarization sensor's.",                                                       \
      .group = "Polarizer", .options = {"linear", "mosaic"})                                       \
    X(double, PolarizerAngle, "ernie:polarizer:angle", 0.0,                                        \
      "Degrees from the image's +x towards +y to a linear polarizer's axis.", .group = "Polarizer", \
      .finite = true)                                                                              \
    X(double, PolarizerExtinction, "ernie:polarizer:extinction", 1000.0,                           \
      "Light passed along the axis over light passed across it.", .group = "Polarizer",            \
      .lower = AtLeast(1.0))                                                                       \
    X(double, PolarizerTransmission, "ernie:polarizer:transmission", 0.85,                         \
      "The fraction of light polarized along the axis passed.", .group = "Polarizer",              \
      .lower = AtLeast(0.0), .upper = AtMost(1.0))

#define ERNIE_SENSOR_QE(X, Name, usdName, channel)                                                 \
    X(std::vector<float>, Name, usdName, std::vector<float>{},                                     \
      "(nm, electrons per photon) pairs of the " channel "; unset keeps the generic curve.",       \
      .group = "Sensor")

#define ERNIE_SENSOR_API_ATTRIBUTES(X)                                                             \
    X(glm::dvec2, SpectralRange, "ernie:spectral:range", glm::dvec2(380.0, 780.0),                 \
      "Wavelengths in nm the sensor responds to; its products sample them in bands.",              \
      .group = "Sensor", .lower = Above(0.0))                                                      \
    X(std::string, SensorType, "ernie:sensor:type", "rgb", "A colour or a mono sensor.",           \
      .group = "Sensor", .options = {"rgb", "mono"})                                               \
    X(bool, SensorNoise, "ernie:sensor:noise", true, "Whether it adds the noise below.",           \
      .group = "Sensor")                                                                           \
    X(double, SensorFullWell, "ernie:sensor:fullWell", 20000.0, "Electrons a pixel holds.",        \
      .group = "Sensor", .lower = Above(0.0))                                                      \
    X(double, SensorReadNoise, "ernie:sensor:readNoise", 3.0, "Electrons, standard deviation.",    \
      .group = "Sensor", .lower = AtLeast(0.0))                                                    \
    X(double, SensorDarkCurrent, "ernie:sensor:darkCurrent", 0.1, "Electrons per second.",         \
      .group = "Sensor", .lower = AtLeast(0.0))                                                    \
    X(double, SensorPrnu, "ernie:sensor:prnu", 0.005,                                              \
      "Standard deviation of each pixel's relative gain.", .group = "Sensor",                      \
      .lower = AtLeast(0.0))                                                                       \
    X(double, SensorDsnu, "ernie:sensor:dsnu", 0.5,                                                \
      "Electrons, standard deviation of each pixel's offset.", .group = "Sensor",                  \
      .lower = AtLeast(0.0))                                                                       \
    X(double, SensorElectronsPerDn, "ernie:sensor:electronsPerDn", 0.0,                            \
      "At ISO 100; 0 for the full well over the ADC range.", .group = "Sensor",                    \
      .lower = AtLeast(0.0))                                                                       \
    X(int, SensorAdcBits, "ernie:sensor:adcBits", 12, "The ADC's bits.", .group = "Sensor",        \
      .lower = AtLeast(1.0), .upper = AtMost(32.0))                                                \
    X(double, SensorBlackLevel, "ernie:sensor:blackLevel", 0.0, "DN added to every pixel.",        \
      .group = "Sensor", .lower = AtLeast(0.0))                                                    \
    X(double, SensorPixelPitch, "ernie:sensor:pixelPitch", 5.0,                                    \
      "Micrometres, of orthographic cameras.", .group = "Sensor", .lower = Above(0.0))             \
    X(int, SensorSeed, "ernie:sensor:seed", 0, "The seed of the sensor's noise.",                  \
      .group = "Sensor")                                                                           \
    ERNIE_SENSOR_QE(X, SensorQeR, "ernie:sensor:qe:r", "red channel")                              \
    ERNIE_SENSOR_QE(X, SensorQeG, "ernie:sensor:qe:g", "green channel")                            \
    ERNIE_SENSOR_QE(X, SensorQeB, "ernie:sensor:qe:b", "blue channel")                             \
    ERNIE_SENSOR_QE(X, SensorQeMono, "ernie:sensor:qe:mono", "mono channel")
// clang-format on

namespace ernie {

ERNIE_CONCRETE_SCHEMA(Camera, Xformable, ERNIE_CAMERA_ATTRIBUTES,
                      "A view looking down -z, y up, with a lens and film.");
ERNIE_API_SCHEMA(ErnieLensAPI, ERNIE_LENS_API_ATTRIBUTES,
                 "A camera's lens beyond USD's: fisheyes, distortion, vignetting and chromatic "
                 "aberration.");
ERNIE_API_SCHEMA(ErniePolarizerAPI, ERNIE_POLARIZER_API_ATTRIBUTES,
                 "A polarizer in front of a camera's lens, or a polarization sensor's mosaic.");
ERNIE_API_SCHEMA(ErnieSensorAPI, ERNIE_SENSOR_API_ATTRIBUTES,
                 "A camera's sensor: its spectral range, noise, ADC and quantum efficiency; its "
                 "products set how finely the range is sampled.");

}  // namespace ernie
