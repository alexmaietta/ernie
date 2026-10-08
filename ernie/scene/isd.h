#pragma once

#include "ernie/scene/camera.h"

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

/// A rotation sampled over time between two NAIF frames, as ALE reads it (`ale::Orientations`):
/// quaternions (w, x, y, z, scalar first) of the time-dependent rotation of vectors from the
/// reference frame into the first of `timeDependentFrames`, and a constant rotation (row-major,
/// applied to vectors after it) on into the first of `constantFrames`.
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

/// A line scanner (pushbroom camera) from its image support data, its geometry as USGS's usgscsm
/// line scanner model (`UsgsAstroLsSensorModel`) computes it, in the body-fixed frame in metres:
/// - Image coordinates are CSM's: (line, sample) with the first pixel's centre at (0.5, 0.5).
///   Line l is exposed around its time `ImageTime(l)`, a rate's start time plus its seconds per
///   line times (l - its start line + 0.5).
/// - The sensor's position is sampled as usgscsm samples it: at least 25 even times over the
///   states, each a cubic Hermite of the states around it (ALE's spline) rotated into the body's
///   frame, then interpolated eighth-order Lagrange (fewer points near the ends). Its orientation,
///   sensor to body = body rotation x inverse pointing, is taken at the union of their times
///   (each interpolated along the great arc), resampled evenly and interpolated the same way.
/// - Sample s of the detector, on line 0 of it, lies on the focal plane at the inverse of
///   `focal2pixel` of (starting detector line - detector centre line, s x summing + starting
///   detector sample - detector centre sample), in mm; the lens's distortion moves it; the camera
///   looks along (-x, -y, -f) in the sensor's frame.
///
/// Light-time and aberration corrections are left out, as usgscsm leaves them out.
class IsdLineScanner {
  public:
    /// @throws std::invalid_argument if `isd` is not a line scanner's, has no line rates or
    /// states, a singular focal plane map, or a distortion model other than none, "radial" and
    /// "lrolrocnac".
    explicit IsdLineScanner(IsdSensor isd);

    const IsdSensor& Isd() const {
        return isd_;
    }

    /// Ephemeris seconds at which line `line` (CSM image coordinates) is exposed.
    double ImageTime(double line) const;

    /// The sensor's body-fixed position at ephemeris time `time`, metres.
    glm::dvec3 Position(double time) const;

    /// The rotation from the sensor's frame to the body's at `time`.
    glm::dquat SensorToBody(double time) const;

    /// The unit body-fixed direction that pixel (`line`, `sample`) looks along.
    glm::dvec3 Look(double line, double sample) const;

    /// Where pixel (`line`, `sample`)'s ray meets the body's ellipsoid raised by `height` metres:
    /// the intersection nearer the sensor, or the ray's point nearest the ellipsoid when it
    /// misses.
    glm::dvec3 ImageToGround(double line, double sample, double height = 0.0) const;

    /// The image coordinates (line, sample) of body-fixed point `ground`: the line whose exposure
    /// puts it on the detector, by the secant method, and the sample there.
    glm::dvec2 GroundToImage(const glm::dvec3& ground) const;

    /// The sun's body-fixed position at `time`, metres, interpolated in the ISD's frame and
    /// rotated into the body's.
    glm::dvec3 SunPosition(double time) const;

    /// A strip of `lines` image lines from `firstLine` as Ernie renders it: a pushbroom camera
    /// (`Camera::pushbroom`) whose rows are those lines, posed `from` at the strip's first line's
    /// start and `to` at its end (camera to body-fixed world). Consecutive strips chain: each's
    /// `from` is the last one's `to`, so stepping the camera's transform through the strips'
    /// `to`s (after a first step at the first `from`) renders the image strip by strip.
    ///
    /// The pose between `from` and `to` is a straight line and a great-arc turn, which departs
    /// from the orbit's curve by about a T^2 / 8 for its acceleration a over the strip's time T:
    /// keep strips short (for LRO's NAC, 64 lines are 0.8 s and 0.1 m). The camera is a pinhole
    /// along the detector: the lens's distortion is not rendered.
    struct Strip {
        Camera camera;
        glm::dmat4 from{1.0};
        glm::dmat4 to{1.0};
        double fromTime = 0.0;
        double toTime = 0.0;
    };
    Strip StripOf(std::uint32_t firstLine, std::uint32_t lines) const;

  private:
    /// The rotation `rotation` describes at `time`: its constant after its time-dependent part.
    static glm::dquat RotationAt(const IsdRotation& rotation, double time);
    /// Undistorted focal plane coordinates of distorted `distorted`, and the inverse.
    glm::dvec2 Undistort(glm::dvec2 distorted) const;
    glm::dvec2 Distort(glm::dvec2 undistorted) const;
    /// Focal plane coordinates (distorted, mm) of detector sample `sample` on the detector line.
    glm::dvec2 FocalPlaneOf(double sample) const;
    /// Image (line offset from the detector line, sample) of distorted focal plane point `focal`.
    glm::dvec2 PixelOf(glm::dvec2 focal) const;
    /// Image coordinates of `ground` as the sensor sees it at `time`: (offset from the detector
    /// line, sample).
    glm::dvec2 DetectorView(const glm::dvec3& ground, double time) const;

    IsdSensor isd_;
    /// The body-fixed positions every `positionStep_` seconds from `positionStart_`.
    std::vector<glm::dvec3> positionNodes_;
    double positionStart_ = 0.0;
    double positionStep_ = 1.0;
    /// The sensor-to-body quaternions (x, y, z, w) every `quaternionStep_` seconds.
    std::vector<glm::dvec4> quaternionNodes_;
    double quaternionStart_ = 0.0;
    double quaternionStep_ = 1.0;
};

/// The image support data in JSON `json`.
/// @throws std::invalid_argument for malformed JSON, a missing or mistyped required field, or a
/// length unit other than "m" or "km".
IsdSensor ReadIsd(std::string_view json);

/// `ReadIsd` of the file at `path`.
/// @throws std::runtime_error if the file cannot be read, or as `ReadIsd`.
IsdSensor ReadIsdFile(const std::string& path);

}  // namespace ernie
