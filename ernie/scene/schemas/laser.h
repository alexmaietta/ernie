// SPDX-License-Identifier: BSD-3-Clause
/// @file laser.h
/// @brief Ernie's LaserLight: a Gaussian beam from datasheet parameters, and its diffractive
/// patterns and sequences.
#pragma once

#include "ernie/scene/schemas/lux.h"

// clang-format off
#define ERNIE_LASER_COUNT(X, Name, usdName, fallback, text)                                        \
    X(int, Name, usdName, fallback, text, .group = "Pattern", .lower = AtLeast(1.0))

#define ERNIE_LASER_LIGHT_ATTRIBUTES(X)                                                            \
    X(double, Wavelength, "ernie:laser:wavelength", 532.0, "Centre wavelength, nm.",               \
      .group = "Laser", .lower = Above(0.0), .finite = true)                                       \
    X(double, Linewidth, "ernie:laser:linewidth", 0.1, "Spectral width (FWHM), nm.",              \
      .group = "Laser", .lower = AtLeast(0.0), .finite = true)                                     \
    X(bool, Speckle, "ernie:laser:speckle", false,                                                 \
      "Grain on rough surfaces, sized by the camera's aperture.", .group = "Laser")                \
    X(glm::dvec2, WaistRadius, "ernie:laser:waistRadius", glm::dvec2(0.0005),                      \
      "1/e^2 radius at the waist along x and y, metres.", .group = "Beam", .lower = Above(0.0),    \
      .finite = true)                                                                              \
    X(double, WaistDistance, "ernie:laser:waistDistance", 0.0,                                     \
      "Metres from the aperture along the beam to its waist.", .group = "Beam", .finite = true)    \
    X(glm::dvec2, BeamQuality, "ernie:laser:beamQuality", glm::dvec2(1.0),                         \
      "M^2 along x and y: 1 for an ideal beam.", .label = "M^2", .group = "Beam",                  \
      .lower = AtLeast(1.0), .finite = true)                                                       \
    X(double, ApertureRadius, "ernie:laser:apertureRadius", 0.0,                                   \
      "Metres at which the aperture clips the beam's tails; 0 none.", .group = "Beam",             \
      .lower = AtLeast(0.0), .finite = true)                                                       \
    X(std::string, Pattern, "ernie:laser:pattern", "spot",                                         \
      "The beam itself, a line fanned along x, a grid of beamlets, a field of dots, an image, or " \
      "a scanned Lissajous figure.",                                                               \
      .group = "Pattern", .options = {"spot", "line", "grid", "dots", "image", "scan"})            \
    X(double, FanAngle, "ernie:laser:fanAngle", 60.0, "A line's full angle along x, degrees.",     \
      .group = "Pattern", .lower = Above(0.0), .upper = Below(170.0))                              \
    X(std::string, LineProfile, "ernie:laser:lineProfile", "flat",                                 \
      "Along a line: flat (a Powell lens) or Gaussian, 1/e^2 at the fan's edges.",                 \
      .group = "Pattern", .options = {"flat", "gaussian"})                                         \
    ERNIE_LASER_COUNT(X, GridColumns, "ernie:laser:gridColumns", 5, "A grid's beamlets along x.")  \
    ERNIE_LASER_COUNT(X, GridRows, "ernie:laser:gridRows", 5, "A grid's beamlets along y.")        \
    X(glm::dvec2, GridPitch, "ernie:laser:gridPitch", glm::dvec2(2.0),                             \
      "Degrees between a grid's beamlets along x and y.", .group = "Pattern", .lower = Above(0.0)) \
    ERNIE_LASER_COUNT(X, DotColumns, "ernie:laser:dotColumns", 64, "Dots per tile along x.")       \
    ERNIE_LASER_COUNT(X, DotRows, "ernie:laser:dotRows", 48, "Dots per tile along y.")             \
    ERNIE_LASER_COUNT(X, DotTileColumns, "ernie:laser:dotTileColumns", 1,                          \
                      "Tiles of the same dots along x.")                                           \
    ERNIE_LASER_COUNT(X, DotTileRows, "ernie:laser:dotTileRows", 1,                                \
                      "Tiles of the same dots along y.")                                           \
    X(double, DotJitter, "ernie:laser:dotJitter", 0.9,                                             \
      "How far a dot strays from its cell's centre, in cells: 0 a regular grid, 1 anywhere.",      \
      .group = "Pattern", .lower = AtLeast(0.0), .upper = AtMost(1.0))                             \
    X(int, DotSeed, "ernie:laser:dotSeed", 1, "The seed the dots derive from.",                    \
      .group = "Pattern", .lower = AtLeast(0.0))                                                   \
    X(glm::dvec2, ScanFrequencies, "ernie:laser:scanFrequencies", glm::dvec2(3.0, 2.0),            \
      "A scan's Lissajous turns along x and y each period, swept faster than cameras see.",        \
      .group = "Pattern", .lower = AtLeast(0.0))                                                   \
    X(double, ScanPhase, "ernie:laser:scanPhase", 90.0, "A scan's phase along x, degrees.",        \
      .group = "Pattern", .finite = true)                                                          \
    X(double, Efficiency, "ernie:laser:efficiency", 1.0,                                           \
      "The fraction of the power that goes into the pattern.", .group = "Pattern",                 \
      .lower = AtLeast(0.0), .upper = AtMost(1.0))                                                 \
    X(double, ZeroOrder, "ernie:laser:zeroOrder", 0.0,                                             \
      "The fraction of the power left in a beam along the axis.", .group = "Pattern",              \
      .lower = AtLeast(0.0), .upper = AtMost(1.0))                                                 \
    X(std::string, PatternImage, "ernie:laser:pattern:image", "",                                  \
      "The image pattern's relative power (linear), its top towards +y.", .group = "Pattern",      \
      .asset = true)                                                                               \
    X(Tensor, PatternImageArray, "ernie:laser:pattern:imageArray", Tensor(),                       \
      "float32 (rows, columns) pattern image in place of the file, on the host or a device.",      \
      .group = "Pattern", .ranks = {2})                                                            \
    X(glm::dvec2, PatternFieldOfView, "ernie:laser:pattern:fieldOfView", glm::dvec2(40.0, 30.0),   \
      "Full angles an image, dots or a scan span along x and y, degrees.", .group = "Pattern",     \
      .lower = AtLeast(0.0), .upper = Below(170.0))                                                \
    X(int, SequenceCount, "ernie:laser:sequence:count", 1,                                         \
      "Patterns shown in turn; 1 for one pattern.", .group = "Time", .lower = AtLeast(1.0))        \
    X(double, SequencePeriod, "ernie:laser:sequence:period", 1.0 / 30.0,                           \
      "Seconds each pattern of a sequence shows.", .group = "Time", .lower = Above(0.0),           \
      .finite = true)                                                                              \
    X(double, SequenceDelay, "ernie:laser:sequence:delay", 0.0,                                    \
      "Seconds on the sensor clock the first pattern shows.", .group = "Time", .finite = true)
// clang-format on

namespace ernie {

ERNIE_SCHEMA_ACCESSORS(LaserLight, BoundableLightBase, ERNIE_LASER_LIGHT_ATTRIBUTES);

/// A laser along -z: a Gaussian beam and its pattern. Its power is `ernie:power`, its pulses
/// LightAPI's, and its extent the emitter, so framing it frames the device.
class LaserLight : public LaserLightAccessors<LaserLight, BoundableLightBase> {
    ERNIE_SCHEMA_BODY(LaserLight, BoundableLightBase)
    ERNIE_SCHEMA_DEFINE(LaserLight)
};

}  // namespace ernie
