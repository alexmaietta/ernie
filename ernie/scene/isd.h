#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// States sampled over time: ephemeris times (seconds past J2000, TDB) and a value at each.
template <typename T>
struct IsdSamples {
    std::vector<double> times;
    std::vector<T> values;
};

/// A rotation sampled over time between two NAIF frames, as ALE writes it: quaternions (x, y, z,
/// w, scalar last) of the time-dependent rotation from the reference frame to the last of
/// `timeDependentFrames`, and a constant rotation (row-major) on to the first of
/// `constantFrames`. Both are kept as the ISD states them.
struct IsdRotation {
    IsdSamples<glm::dquat> quaternions;
    std::vector<std::int64_t> timeDependentFrames;
    std::vector<std::int64_t> constantFrames;
    glm::dmat3 constantRotation{1.0};  ///< Identity when the ISD gives none.
    std::int64_t referenceFrame = 1;   ///< 1 is J2000.
};

/// A camera's image support data (ISD) in the Community Sensor Model's JSON form, as ALE (the USGS
/// Abstraction Layer for Ephemerides) writes it from SPICE kernels for planetary cameras such as
/// LRO's NAC and WAC, MRO's CTX and HiRISE: the image's size and timing, the focal plane's map to
/// detector pixels, the optics, and the spacecraft's and the body's states over the image. Lengths
/// are converted to metres; everything else is kept in the ISD's own frames and conventions.
///
/// References:
/// - USGS Astrogeology, "ALE: Abstraction Layer for Ephemerides",
///   https://github.com/DOI-USGS/ale
/// - USGS Astrogeology, "usgscsm: Community Sensor Model plugin",
///   https://github.com/DOI-USGS/usgscsm (the sensor models that read these ISDs).
/// - NGA, "Community Sensor Model (CSM) Technical Requirements Document", version 3.0.4.
struct IsdSensor {
    std::string model;  ///< `name_model`, such as USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL.
    std::string platform;
    std::string sensor;
    std::uint32_t lines = 0;
    std::uint32_t samples = 0;

    double focalLengthMm = 0.0;
    /// Detector line and sample from focal plane x and y in mm: a0 + a1 x + a2 y.
    std::array<double, 3> focalToLine{};
    std::array<double, 3> focalToSample{};
    double detectorCenterLine = 0.0;
    double detectorCenterSample = 0.0;
    double startingDetectorLine = 0.0;
    double startingDetectorSample = 0.0;
    double lineSumming = 1.0;
    double sampleSumming = 1.0;
    /// `optical_distortion`'s model name (such as "radial", "transverse", "lrolrocnac") and
    /// coefficients; for "transverse" the x coefficients then the y.
    std::string distortion;
    std::vector<double> distortionCoefficients;

    /// Line scanners' timing, `line_scan_rate`: from each start line on, (start line, its time
    /// relative to `centerTime`, seconds per line).
    std::vector<std::array<double, 3>> lineRates;
    double startingTime = 0.0;  ///< Ephemeris seconds.
    double centerTime = 0.0;

    /// Body radii, metres.
    double semiMajorAxis = 0.0;
    double semiMinorAxis = 0.0;

    /// The sensor's and the sun's positions and velocities relative to the body's centre, in
    /// `positionFrame` (1 is J2000), metres and metres per second.
    IsdSamples<glm::dvec3> positions;
    IsdSamples<glm::dvec3> velocities;
    std::int64_t positionFrame = 1;
    IsdSamples<glm::dvec3> sunPositions;
    IsdRotation pointing;      ///< `instrument_pointing`: to the sensor's frame.
    IsdRotation bodyRotation;  ///< `body_rotation`: to the body-fixed frame.

    /// Whether the ISD is a line scanner's (`model` USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL).
    bool IsLineScanner() const;
};

/// The image support data in JSON `json`.
/// @throws std::invalid_argument for malformed JSON, a missing or mistyped required field, or a
/// length unit other than "m" or "km".
IsdSensor ReadIsd(std::string_view json);

/// `ReadIsd` of the file at `path`.
/// @throws std::runtime_error if the file cannot be read, or as `ReadIsd`.
IsdSensor ReadIsdFile(const std::string& path);

}  // namespace ernie
