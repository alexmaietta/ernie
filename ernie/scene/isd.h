#pragma once

#include "ernie/scene/camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <span>
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
    /// coefficients; for "transverse" the x coefficients then the y (ten each: the cubic
    /// polynomial's terms 1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3).
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

    /// Push-frame cameras' framelets: `exposure_duration` and `interframe_delay` (seconds),
    /// `framelet_height` (detector lines, the overlap included), `num_lines_overlap` (lines shared
    /// with the next framelet, even), `framelets_flipped` and `framelet_order_reversed`.
    double exposureDuration = 0.0;
    double interframeDelay = 0.0;
    std::uint32_t frameletHeight = 0;
    std::uint32_t frameletOverlap = 0;
    bool frameletsFlipped = false;
    bool frameletOrderReversed = false;

    /// Whether the ISD is a line scanner's (`model` USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL).
    bool IsLineScanner() const;

    /// Whether the ISD is a push-frame camera's (USGS_ASTRO_PUSH_FRAME_SENSOR_MODEL).
    bool IsPushFrame() const;

    /// Whether the ISD is a frame camera's (USGS_ASTRO_FRAME_SENSOR_MODEL).
    bool IsFrame() const;
};

/// What usgscsm's camera models of image support data share, in the body-fixed frame in metres:
/// - Image coordinates are CSM's: (line, sample) with the first pixel's centre at (0.5, 0.5).
/// - The sensor's position is sampled as usgscsm samples it: at least 25 even times over the
///   states, each a cubic Hermite of the states around it (ALE's spline) rotated into the body's
///   frame, then interpolated eighth-order Lagrange (fewer points near the ends). Its orientation,
///   sensor to body = body rotation x inverse pointing, is taken at the union of their times
///   (each interpolated along the great arc), resampled evenly and interpolated the same way.
/// - Detector line d and sample s lie on the focal plane at the inverse of `focal2pixel` of
///   (d x line summing + starting detector line - detector centre line, s x sample summing +
///   starting detector sample - detector centre sample), in mm; the lens's distortion moves it;
///   the camera looks along (-x, -y, -f) in the sensor's frame.
///
/// Light-time and aberration corrections are left out, as usgscsm leaves them out.
class IsdCameraModel {
  public:
    const IsdSensor& Isd() const {
        return isd_;
    }

    /// The sensor's body-fixed position at ephemeris time `time`, metres.
    glm::dvec3 Position(double time) const;

    /// The rotation from the sensor's frame to the body's at `time`.
    glm::dquat SensorToBody(double time) const;

    /// The sun's body-fixed position at `time`, metres, interpolated in the ISD's frame and
    /// rotated into the body's.
    glm::dvec3 SunPosition(double time) const;

    /// A frame of the detector as Ernie renders it: a frame camera of the image's samples by its
    /// rows, posed (`camera.transform`, camera to body-fixed world) when the frame is exposed,
    /// the lens's distortion fitted as Brown-Conrady terms (and scale).
    struct FrameView {
        Camera camera;
        /// Whether the camera's rows run up the frame: the detector's map to the image is a
        /// mirror of Ernie's, so flip the rendered rows.
        bool rowsReversed = false;
        /// The camera's largest departure from the ISD's pixels over the frame (the fitted
        /// distortion, and any non-square or skewed pixels), in pixels.
        double distortionResidualPixels = 0.0;
    };

  protected:
    /// @throws std::invalid_argument if `isd` has no states or rotations, a singular focal plane
    /// map, or a distortion model other than none, "radial", "transverse", "radtan" (OpenCV's
    /// radial-tangential, coefficients k1, k2, p1, p2, k3 on coordinates over the focal length)
    /// and "lrolrocnac".
    explicit IsdCameraModel(IsdSensor isd);

    /// The rotation `rotation` describes at `time`: its constant after its time-dependent part.
    static glm::dquat RotationAt(const IsdRotation& rotation, double time);
    /// Undistorted focal plane coordinates of distorted `distorted`, and the inverse.
    glm::dvec2 Undistort(glm::dvec2 distorted) const;
    glm::dvec2 Distort(glm::dvec2 undistorted) const;
    /// Focal plane coordinates (distorted, mm) of detector line `line` and image sample `sample`.
    glm::dvec2 FocalPlaneOf(double line, double sample) const;
    /// Image (line on the detector, sample) of distorted focal plane point `focal`.
    glm::dvec2 PixelOf(glm::dvec2 focal) const;
    /// Undistorted focal plane coordinates of `ground` as the sensor sees it at `time`.
    glm::dvec2 FocalPlaneView(const glm::dvec3& ground, double time) const;
    /// The unit body-fixed direction of the ray at `time` through detector line `line` and
    /// sample `sample` (`IsdLineScanner::Look`).
    glm::dvec3 LookAt(double time, double line, double sample) const;
    /// Where the ray from `origin` along `look` meets the ellipsoid raised by `height`: the
    /// intersection nearer the origin (on either side, as usgscsm's line scanner takes it; or the
    /// first along `look` when not `eitherSide`, as its frame camera does), or the ray's point
    /// nearest the ellipsoid.
    glm::dvec3 EllipsoidIntersect(const glm::dvec3& origin, const glm::dvec3& look,
                                  double height, bool eitherSide = true) const;
    /// Ernie's camera axes (columns) in the sensor's frame for an image of the detector's samples
    /// across x: x along the samples as they increase, z back from the scene (`facing` -1 when
    /// the scene lies behind usgscsm's look, as for NAIF's +z boresights), y completing them.
    glm::dmat3 ErnieAxes(double line, double facing) const;
    /// The camera-to-body pose of Ernie's camera on `axes` at `time`.
    glm::dmat4 PoseAt(double time, const glm::dmat3& axes) const;
    /// Fits the lens's distortion at detector lines `lines` across the samples as Brown-Conrady
    /// terms on `camera` (and a scale into its focal length, from a radial model's constant
    /// term), with `facing` and `axes` as `ErnieAxes`; returns the fit's largest departure from
    /// the ISD's, in pixels. One line fits along it only (x) and radially, as a pushbroom's lens
    /// resamples; more fit both of the image's directions with the tangential terms too, which
    /// a radtan model's map onto exactly.
    double FitDistortion(Camera& camera, std::span<const double> lines, const glm::dmat3& axes,
                         double facing) const;
    /// The frame camera of `rows` image rows exposed at `time`, row coordinate r (0 the top edge,
    /// `rows` the bottom) on detector line `detectorLine(r)`, with `facing` as `ErnieAxes`.
    FrameView FrameViewOf(const std::function<double(double)>& detectorLine, std::uint32_t rows,
                          double time, double facing) const;

    IsdSensor isd_;

  private:
    /// The body-fixed positions every `positionStep_` seconds from `positionStart_`.
    std::vector<glm::dvec3> positionNodes_;
    double positionStart_ = 0.0;
    double positionStep_ = 1.0;
    /// The sensor-to-body quaternions (x, y, z, w) every `quaternionStep_` seconds.
    std::vector<glm::dvec4> quaternionNodes_;
    double quaternionStart_ = 0.0;
    double quaternionStep_ = 1.0;
};

/// A line scanner (pushbroom camera) from its image support data, its geometry as usgscsm's line
/// scanner model (`UsgsAstroLsSensorModel`) computes it (`IsdCameraModel`): line l is exposed
/// around its time `ImageTime(l)`, a rate's start time plus its seconds per line times (l - its
/// start line + 0.5), through detector line 0.
class IsdLineScanner : public IsdCameraModel {
  public:
    /// @throws std::invalid_argument if `isd` is not a line scanner's or has no line rates, or as
    /// `IsdCameraModel`.
    explicit IsdLineScanner(IsdSensor isd);

    /// Ephemeris seconds at which line `line` (CSM image coordinates) is exposed.
    double ImageTime(double line) const;

    /// The unit body-fixed direction of pixel (`line`, `sample`)'s ray as usgscsm takes it,
    /// (-x, -y, -f) in the sensor's frame: towards the scene, or away from it for ISDs whose
    /// boresight is the sensor's +z (ALE's real cameras), which usgscsm images the same, the ray
    /// meeting the ground on either side of the sensor.
    glm::dvec3 Look(double line, double sample) const;

    /// Where pixel (`line`, `sample`)'s ray meets the body's ellipsoid raised by `height` metres:
    /// the intersection nearer the sensor, or the ray's point nearest the ellipsoid when it
    /// misses.
    glm::dvec3 ImageToGround(double line, double sample, double height = 0.0) const;

    /// The image coordinates (line, sample) of body-fixed point `ground`: the line whose exposure
    /// puts it on the detector, by the secant method, and the sample there.
    glm::dvec2 GroundToImage(const glm::dvec3& ground) const;

    /// A strip of `lines` image lines from `firstLine` as Ernie renders it: a pushbroom camera
    /// (`Camera::pushbroom`) whose rows are those lines, posed `from` at the strip's first line's
    /// start and `to` at its end (camera to body-fixed world). Consecutive strips chain: each's
    /// `from` is the last one's `to`, so stepping the camera's transform through the strips'
    /// `to`s (after a first step at the first `from`) renders the image strip by strip.
    ///
    /// The pose between `from` and `to` is a straight line and a great-arc turn, which departs
    /// from the orbit's curve by about a T^2 / 8 for its acceleration a over the strip's time T:
    /// keep strips short (for LRO's NAC, 64 lines are 0.8 s and 0.1 m). The lens's distortion
    /// along the detector is fitted as the camera's Brown-Conrady k1 to k3, which the pushbroom's
    /// lens resamples through.
    struct Strip {
        Camera camera;
        glm::dmat4 from{1.0};
        glm::dmat4 to{1.0};
        double fromTime = 0.0;
        double toTime = 0.0;
        /// The fitted distortion's largest departure from the ISD's along the line, in pixels.
        double distortionResidualPixels = 0.0;
    };
    Strip StripOf(std::uint32_t firstLine, std::uint32_t lines) const;

  private:
    /// Image coordinates of `ground` as the sensor sees it at `time`: (offset from the detector
    /// line, sample).
    glm::dvec2 DetectorView(const glm::dvec3& ground, double time) const;
};

/// A push-frame camera (LROC's WAC, MRO's MARCI) from its image support data, its geometry as
/// usgscsm's push-frame model (`UsgsAstroPushFrameSensorModel`) computes it (`IsdCameraModel`):
/// the image is a stack of framelets, each a frame of `FrameletLines` lines exposed at once. The
/// framelet holding line l is f = l / framelet lines (reversed with `framelet_order_reversed`),
/// exposed at the image's start + half the exposure + f x the interframe delay; within it, line
/// l mod framelet lines (flipped with `framelets_flipped`, plus half the overlap) is the detector
/// line its rays come through.
///
/// Reference: usgscsm's `UsgsAstroPushFrameSensorModel` (`getImageTime`, `losToEcf`,
/// `groundToImage`).
class IsdPushFrame : public IsdCameraModel {
  public:
    /// @throws std::invalid_argument if `isd` is not a push-frame camera's (or misses its
    /// framelet fields), or as `IsdCameraModel`.
    explicit IsdPushFrame(IsdSensor isd);

    /// Lines of each framelet in the image (its height less the overlap, over the line summing).
    std::uint32_t FrameletLines() const;

    /// Framelets in the image.
    std::uint32_t Framelets() const;

    /// Ephemeris seconds at which line `line`'s framelet is exposed, the middle of its exposure.
    double ImageTime(double line) const;

    /// As `IsdLineScanner::Look`.
    glm::dvec3 Look(double line, double sample) const;

    /// As `IsdLineScanner::ImageToGround`.
    glm::dvec3 ImageToGround(double line, double sample, double height = 0.0) const;

    /// The image coordinates (line, sample) of body-fixed point `ground`, as usgscsm finds them:
    /// the framelet whose middle line comes nearest it, by the secant method over framelets, and
    /// its line and sample in that framelet.
    glm::dvec2 GroundToImage(const glm::dvec3& ground) const;

    /// Framelet `framelet` (in image order, from the top) as Ernie renders it: a frame camera of
    /// the image's samples by `FrameletLines` lines, posed when it is exposed (camera to
    /// body-fixed world), its exposure the ISD's, the lens's distortion fitted as Brown-Conrady
    /// terms (and scale). Render a whole image as a batch of these, one view per framelet, and
    /// stack the views from the top.
    struct Framelet {
        Camera camera;  ///< `transform` its pose.
        double time = 0.0;
        std::uint32_t firstLine = 0;
        /// Whether the camera's rows run up the framelet: the detector's map to the image is a
        /// mirror of Ernie's, so flip the rendered framelet's rows to stack it.
        bool rowsReversed = false;
        /// The camera's largest departure from the ISD's pixels over the framelet (the fitted
        /// distortion, and any non-square or skewed pixels), in pixels.
        double distortionResidualPixels = 0.0;
    };
    Framelet FrameletOf(std::uint32_t framelet) const;

  private:
    /// The framelet, counted in time, that image line `line` belongs to.
    int TimeOrder(double line) const;
    /// The detector line of image line `line` within its framelet (flipped, less the overlap).
    double DetectorLine(double line) const;
    /// usgscsm's `calcFrameDistance`: how many lines `ground` lies from framelet `framelet`'s
    /// middle, through the undistorted focal plane.
    double FrameDistance(int framelet, const glm::dvec3& ground) const;
};

/// A frame camera from its image support data, its geometry as usgscsm's frame model
/// (`UsgsAstroFrameSensorModel`) computes it (`IsdCameraModel`): the whole image is exposed at
/// once, at the ISD's centre time, from one position and orientation. Unlike the line scanner,
/// the camera looks along (x, y, f) in the sensor's frame, and a ray meets the ellipsoid at its
/// first intersection along it. Rolling-shutter jitter (`jitter`) is not read.
///
/// Reference: usgscsm's `UsgsAstroFrameSensorModel` (`imageToGround`, `groundToImage`,
/// `losEllipsoidIntersect`, `constructStateFromIsd`).
class IsdFrame : public IsdCameraModel {
  public:
    /// @throws std::invalid_argument if `isd` is not a frame camera's, or as `IsdCameraModel`.
    explicit IsdFrame(IsdSensor isd);

    /// Ephemeris seconds at which the image is exposed: the ISD's centre time.
    double ImageTime() const {
        return isd_.centerTime;
    }

    /// The unit body-fixed direction of pixel (`line`, `sample`)'s ray.
    glm::dvec3 Look(double line, double sample) const;

    /// Where pixel (`line`, `sample`)'s ray first meets the body's ellipsoid raised by `height`
    /// metres, or the ray's point nearest the ellipsoid when it misses.
    glm::dvec3 ImageToGround(double line, double sample, double height = 0.0) const;

    /// The image coordinates (line, sample) of body-fixed point `ground`.
    glm::dvec2 GroundToImage(const glm::dvec3& ground) const;

    /// The image as Ernie renders it: a frame camera of the image's samples by its lines, its
    /// exposure left as authored (frame ISDs carry none).
    FrameView View() const;
};

/// The image support data in JSON `json`.
/// @throws std::invalid_argument for malformed JSON, a missing or mistyped required field, or a
/// length unit other than "m" or "km".
IsdSensor ReadIsd(std::string_view json);

/// `ReadIsd` of the file at `path`.
/// @throws std::runtime_error if the file cannot be read, or as `ReadIsd`.
IsdSensor ReadIsdFile(const std::string& path);

}  // namespace ernie
