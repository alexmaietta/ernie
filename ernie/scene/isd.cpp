#include "ernie/scene/isd.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ernie {

namespace {

using Json = nlohmann::json;

const Json& Field(const Json& object, const char* name) {
    const auto found = object.find(name);
    if (found == object.end()) {
        throw std::invalid_argument(std::string("The ISD has no '") + name + "'");
    }
    return *found;
}

double Number(const Json& object, const char* name) {
    const Json& value = Field(object, name);
    if (!value.is_number()) {
        throw std::invalid_argument(std::string("The ISD's '") + name + "' is not a number");
    }
    return value.get<double>();
}

double NumberOr(const Json& object, const char* name, double fallback) {
    return object.contains(name) ? Number(object, name) : fallback;
}

/// Metres per unit of `object`'s "unit", kilometres when it has none (as ALE writes them).
double MetresPer(const Json& object) {
    const std::string unit = object.value("unit", std::string("km"));
    if (unit == "km") {
        return 1000.0;
    }
    if (unit == "m") {
        return 1.0;
    }
    throw std::invalid_argument("The ISD's length unit '" + unit + "' is neither m nor km");
}

std::vector<double> Numbers(const Json& value, const char* name) {
    if (!value.is_array()) {
        throw std::invalid_argument(std::string("The ISD's '") + name + "' is not an array");
    }
    std::vector<double> numbers;
    for (const Json& item : value) {
        numbers.push_back(item.get<double>());
    }
    return numbers;
}

glm::dvec3 Vector(const Json& value, double scale) {
    if (!value.is_array() || value.size() != 3) {
        throw std::invalid_argument("The ISD has a vector without three components");
    }
    return glm::dvec3(value[0].get<double>(), value[1].get<double>(), value[2].get<double>()) *
           scale;
}

IsdSamples<glm::dvec3> Vectors(const Json& block, const char* name, double scale) {
    IsdSamples<glm::dvec3> samples;
    samples.times = Numbers(Field(block, "ephemeris_times"), "ephemeris_times");
    for (const Json& item : Field(block, name)) {
        samples.values.push_back(Vector(item, scale));
    }
    if (samples.values.size() != samples.times.size()) {
        throw std::invalid_argument(std::string("The ISD's '") + name +
                                    "' and their times differ in number");
    }
    return samples;
}

std::vector<std::int64_t> Frames(const Json& block, const char* name) {
    std::vector<std::int64_t> frames;
    if (block.contains(name)) {
        for (const Json& item : block[name]) {
            frames.push_back(item.get<std::int64_t>());
        }
    }
    return frames;
}

IsdRotation Rotation(const Json& block) {
    IsdRotation rotation;
    rotation.quaternions.times = Numbers(Field(block, "ephemeris_times"), "ephemeris_times");
    for (const Json& item : Field(block, "quaternions")) {
        if (!item.is_array() || item.size() != 4) {
            throw std::invalid_argument("The ISD has a quaternion without four components");
        }
        // Scalar first, as ALE reads them (`ale::getJsonQuatArray`).
        rotation.quaternions.values.emplace_back(item[0].get<double>(), item[1].get<double>(),
                                                 item[2].get<double>(), item[3].get<double>());
    }
    if (rotation.quaternions.values.size() != rotation.quaternions.times.size()) {
        throw std::invalid_argument("The ISD's quaternions and their times differ in number");
    }
    rotation.timeDependentFrames = Frames(block, "time_dependent_frames");
    rotation.constantFrames = Frames(block, "constant_frames");
    if (block.contains("constant_rotation")) {
        const std::vector<double> m = Numbers(block["constant_rotation"], "constant_rotation");
        if (m.size() != 9) {
            throw std::invalid_argument("The ISD's constant rotation is not 3 x 3");
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                rotation.constantRotation[c][r] = m[static_cast<std::size_t>(r * 3 + c)];
            }
        }
    }
    rotation.referenceFrame = block.value("reference_frame", std::int64_t{1});
    return rotation;
}

std::array<double, 3> Triple(const Json& object, const char* name) {
    const std::vector<double> values = Numbers(Field(object, name), name);
    if (values.size() != 3) {
        throw std::invalid_argument(std::string("The ISD's '") + name + "' needs three numbers");
    }
    return {values[0], values[1], values[2]};
}

}  // namespace

namespace {

/// The index of the last of sorted `times` at or before `time`, clamped to [0, size - 2] so
/// that it and the next bracket `time` or extrapolate past the ends.
std::size_t BracketOf(const std::vector<double>& times, double time) {
    if (times.size() < 2) {
        return 0;
    }
    const auto after = std::upper_bound(times.begin(), times.end(), time);
    const std::size_t index = after == times.begin() ? 0 : std::size_t(after - times.begin()) - 1;
    return std::min(index, times.size() - 2);
}

/// Lagrange interpolation at `time` through up to eight of `samples` around it.
glm::dvec3 Lagrange(const IsdSamples<glm::dvec3>& samples, double time) {
    const std::size_t count = samples.times.size();
    if (count == 1) {
        return samples.values.front();
    }
    const std::size_t order = std::min<std::size_t>(8, count);
    const std::size_t at = BracketOf(samples.times, time);
    const std::size_t first =
        std::min(count - order, at + 1 >= order / 2 ? at + 1 - order / 2 : std::size_t{0});
    glm::dvec3 value(0.0);
    for (std::size_t i = first; i < first + order; ++i) {
        double weight = 1.0;
        for (std::size_t j = first; j < first + order; ++j) {
            if (j != i) {
                weight *= (time - samples.times[j]) / (samples.times[i] - samples.times[j]);
            }
        }
        value += weight * samples.values[i];
    }
    return value;
}

/// usgscsm's Lagrange interpolation (`lagrangeInterp`) at `time` of `values` sampled every
/// `step` from `start`: eight points around `time`, fewer near the ends.
template <typename T>
T UniformLagrange(const std::vector<T>& values, double start, double step, double time) {
    const int count = static_cast<int>(values.size());
    if (count == 1) {
        return values.front();
    }
    const double fraction = (time - start) / step;
    const int index = std::clamp(static_cast<int>(fraction), 0, count - 2);
    int order = 2;
    if (index >= 3 && index < count - 4) {
        order = 8;
    } else if (index >= 2 && index < count - 3) {
        order = 6;
    } else if (index >= 1 && index < count - 2) {
        order = 4;
    }
    const double t = fraction - index;
    const int first = index - order / 2 + 1;
    T value{};
    for (int i = 0; i < order; ++i) {
        double weight = 1.0;
        for (int j = 0; j < order; ++j) {
            if (j != i) {
                weight *= (t - (j - order / 2 + 1)) / static_cast<double>(i - j);
            }
        }
        value += weight * values[static_cast<std::size_t>(first + i)];
    }
    return value;
}

/// ALE's state interpolation (`ale::States::getState`, SPLINE) of `positions` with
/// `velocities`: a cubic Hermite between the samples around `time`, or Lagrange without
/// velocities.
glm::dvec3 Hermite(const IsdSamples<glm::dvec3>& positions,
                   const IsdSamples<glm::dvec3>& velocities, double time) {
    const std::vector<double>& times = positions.times;
    const auto exact = std::lower_bound(times.begin(), times.end(), time);
    if (exact != times.end() && *exact == time) {
        return positions.values[std::size_t(exact - times.begin())];
    }
    if (velocities.values.size() != positions.values.size() || times.size() < 2) {
        return Lagrange(positions, time);
    }
    const std::size_t at = BracketOf(times, time);
    const double h = times[at + 1] - times[at];
    const double t = (time - times[at]) / h;
    const double t2 = t * t;
    const double t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * positions.values[at] +
           (t3 - 2.0 * t2 + t) * h * velocities.values[at] +
           (-2.0 * t3 + 3.0 * t2) * positions.values[at + 1] +
           (t3 - t2) * h * velocities.values[at + 1];
}

/// Linear interpolation at `time`, holding the only sample of one.
glm::dvec3 Linear(const IsdSamples<glm::dvec3>& samples, double time) {
    if (samples.times.size() == 1) {
        return samples.values.front();
    }
    const std::size_t at = BracketOf(samples.times, time);
    const double t = (time - samples.times[at]) / (samples.times[at + 1] - samples.times[at]);
    return glm::mix(samples.values[at], samples.values[at + 1], t);
}

}  // namespace

bool IsdSensor::IsLineScanner() const {
    return model == "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL";
}

IsdSensor ReadIsd(std::string_view json) {
    Json isd;
    try {
        isd = Json::parse(json);
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("The ISD is not JSON: ") + error.what());
    }
    try {
        IsdSensor sensor;
        sensor.model = Field(isd, "name_model").get<std::string>();
        sensor.platform = isd.value("name_platform", std::string());
        sensor.sensor = isd.value("name_sensor", std::string());
        sensor.lines = static_cast<std::uint32_t>(Number(isd, "image_lines"));
        sensor.samples = static_cast<std::uint32_t>(Number(isd, "image_samples"));

        sensor.focalLengthMm = Number(Field(isd, "focal_length_model"), "focal_length");
        sensor.focalToLine = Triple(isd, "focal2pixel_lines");
        sensor.focalToSample = Triple(isd, "focal2pixel_samples");
        const Json& center = Field(isd, "detector_center");
        sensor.detectorCenterLine = Number(center, "line");
        sensor.detectorCenterSample = Number(center, "sample");
        sensor.startingDetectorLine = NumberOr(isd, "starting_detector_line", 0.0);
        sensor.startingDetectorSample = NumberOr(isd, "starting_detector_sample", 0.0);
        sensor.lineSumming = NumberOr(isd, "detector_line_summing", 1.0);
        sensor.sampleSumming = NumberOr(isd, "detector_sample_summing", 1.0);
        if (isd.contains("optical_distortion") && !isd["optical_distortion"].empty()) {
            const auto& [name, model] = *isd["optical_distortion"].items().begin();
            sensor.distortion = name;
            if (model.contains("coefficients")) {
                sensor.distortionCoefficients = Numbers(model["coefficients"], "coefficients");
            } else if (model.contains("x") && model.contains("y")) {
                sensor.distortionCoefficients = Numbers(model["x"], "x");
                const std::vector<double> y = Numbers(model["y"], "y");
                sensor.distortionCoefficients.insert(sensor.distortionCoefficients.end(),
                                                     y.begin(), y.end());
            }
        }

        if (isd.contains("line_scan_rate")) {
            for (const Json& rate : isd["line_scan_rate"]) {
                const std::vector<double> row = Numbers(rate, "line_scan_rate");
                if (row.size() != 3) {
                    throw std::invalid_argument("The ISD's line scan rates need three numbers");
                }
                sensor.lineRates.push_back({row[0], row[1], row[2]});
            }
        }
        sensor.startingTime = NumberOr(isd, "starting_ephemeris_time", 0.0);
        sensor.centerTime = Number(isd, "center_ephemeris_time");

        const Json& radii = Field(isd, "radii");
        sensor.semiMajorAxis = Number(radii, "semimajor") * MetresPer(radii);
        sensor.semiMinorAxis = Number(radii, "semiminor") * MetresPer(radii);

        const Json& position = Field(isd, "instrument_position");
        const double metres = MetresPer(position);
        sensor.positions = Vectors(position, "positions", metres);
        if (position.contains("velocities")) {
            sensor.velocities = Vectors(position, "velocities", metres);
        }
        sensor.positionFrame = position.value("reference_frame", std::int64_t{1});
        if (isd.contains("sun_position")) {
            const Json& sun = isd["sun_position"];
            sensor.sunPositions = Vectors(sun, "positions", MetresPer(sun));
        }
        sensor.pointing = Rotation(Field(isd, "instrument_pointing"));
        sensor.bodyRotation = Rotation(Field(isd, "body_rotation"));
        return sensor;
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("The ISD has a mistyped field: ") + error.what());
    }
}

IsdLineScanner::IsdLineScanner(IsdSensor isd) : isd_(std::move(isd)) {
    if (!isd_.IsLineScanner()) {
        throw std::invalid_argument("The ISD's model '" + isd_.model +
                                    "' is not a line scanner's");
    }
    if (isd_.lineRates.empty() || isd_.positions.values.empty() ||
        isd_.pointing.quaternions.values.empty() || isd_.bodyRotation.quaternions.values.empty()) {
        throw std::invalid_argument(
            "A line scanner's ISD needs line rates, positions and pointing and body rotations");
    }
    const double determinant = isd_.focalToLine[1] * isd_.focalToSample[2] -
                               isd_.focalToLine[2] * isd_.focalToSample[1];
    if (determinant == 0.0) {
        throw std::invalid_argument("The ISD's focal plane map to pixels is singular");
    }
    if (!isd_.distortion.empty() && isd_.distortion != "radial" &&
        isd_.distortion != "lrolrocnac") {
        throw std::invalid_argument("The ISD's distortion model '" + isd_.distortion +
                                    "' is not supported; use radial or lrolrocnac");
    }
    if (isd_.distortion == "lrolrocnac" && isd_.distortionCoefficients.size() != 1) {
        throw std::invalid_argument("LRO NAC distortion takes one coefficient");
    }
    if (isd_.distortion == "radial") {
        isd_.distortionCoefficients.resize(3, 0.0);
    }
    // As usgscsm samples the trajectory (`constructStateFromIsd`): at least 25 positions evenly
    // over the states' times, each interpolated in the ISD's frame and rotated into the body's.
    const std::vector<double>& times = isd_.positions.times;
    const std::size_t positions = std::max<std::size_t>(times.size(), 25);
    positionStart_ = times.front();
    positionStep_ = times.size() > 1 ? (times.back() - times.front()) / double(positions - 1) : 1.0;
    for (std::size_t i = 0; i < (times.size() > 1 ? positions : 1); ++i) {
        const double time = positionStart_ + double(i) * positionStep_;
        positionNodes_.push_back(RotationAt(isd_.bodyRotation, time) *
                                 Hermite(isd_.positions, isd_.velocities, time));
    }
    // And the sensor's orientation in the body: body rotation x inverse pointing at the union of
    // their times, interpolated along the great arc to as many even times, as components.
    std::vector<double> merged = isd_.pointing.quaternions.times;
    merged.insert(merged.end(), isd_.bodyRotation.quaternions.times.begin(),
                  isd_.bodyRotation.quaternions.times.end());
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    IsdRotation sensorToBody;
    sensorToBody.constantRotation = isd_.bodyRotation.constantRotation;
    const glm::dmat3 inverseConstant = glm::transpose(isd_.bodyRotation.constantRotation);
    for (const double time : merged) {
        sensorToBody.quaternions.times.push_back(time);
        sensorToBody.quaternions.values.push_back(glm::normalize(
            glm::quat_cast(inverseConstant * glm::mat3_cast(RotationAt(isd_.bodyRotation, time) *
                                                            glm::inverse(RotationAt(
                                                                isd_.pointing, time))))));
    }
    quaternionStart_ = merged.front();
    quaternionStep_ =
        merged.size() > 1 ? (merged.back() - merged.front()) / double(merged.size() - 1) : 1.0;
    for (std::size_t i = 0; i < merged.size(); ++i) {
        const glm::dquat q =
            RotationAt(sensorToBody, quaternionStart_ + double(i) * quaternionStep_);
        glm::dvec4 node(q.x, q.y, q.z, q.w);
        // Sign-continuous, so the components interpolate.
        if (!quaternionNodes_.empty() && glm::dot(node, quaternionNodes_.back()) < 0.0) {
            node = -node;
        }
        quaternionNodes_.push_back(node);
    }
}

glm::dquat IsdLineScanner::RotationAt(const IsdRotation& rotation, double time) {
    const IsdSamples<glm::dquat>& q = rotation.quaternions;
    glm::dquat timeDependent = q.values.front();
    if (q.values.size() > 1) {
        const std::size_t at = BracketOf(q.times, time);
        const double t = (time - q.times[at]) / (q.times[at + 1] - q.times[at]);
        glm::dquat next = q.values[at + 1];
        if (glm::dot(q.values[at], next) < 0.0) {
            next = -next;
        }
        timeDependent = glm::slerp(q.values[at], next, t);
    }
    return glm::normalize(glm::quat_cast(rotation.constantRotation *
                                         glm::mat3_cast(glm::normalize(timeDependent))));
}

double IsdLineScanner::ImageTime(double line) const {
    std::size_t at = 0;
    while (at + 1 < isd_.lineRates.size() && isd_.lineRates[at + 1][0] <= line) {
        ++at;
    }
    const std::array<double, 3>& rate = isd_.lineRates[at];
    return isd_.centerTime + rate[1] + rate[2] * (line - rate[0] + 0.5);
}

glm::dvec3 IsdLineScanner::Position(double time) const {
    return UniformLagrange(positionNodes_, positionStart_, positionStep_, time);
}

glm::dquat IsdLineScanner::SensorToBody(double time) const {
    // usgscsm interpolates eighth order, fourth with fewer than two quaternions.
    const glm::dvec4 q =
        glm::normalize(UniformLagrange(quaternionNodes_, quaternionStart_, quaternionStep_, time));
    return glm::dquat(q.w, q.x, q.y, q.z);
}

glm::dvec2 IsdLineScanner::Undistort(glm::dvec2 d) const {
    const std::vector<double>& k = isd_.distortionCoefficients;
    if (isd_.distortion == "radial") {
        const double rr = glm::dot(d, d);
        return d * (1.0 - (k[0] + rr * (k[1] + rr * k[2])));
    }
    if (isd_.distortion == "lrolrocnac") {
        return {d.x, d.y / (1.0 + k[0] * d.y * d.y)};
    }
    return d;
}

glm::dvec2 IsdLineScanner::Distort(glm::dvec2 undistorted) const {
    if (isd_.distortion.empty()) {
        return undistorted;
    }
    // Newton's method on `Undistort`, with its Jacobian by central differences.
    glm::dvec2 d = undistorted;
    for (int i = 0; i < 20; ++i) {
        const glm::dvec2 residual = Undistort(d) - undistorted;
        if (glm::length(residual) < 1e-14) {
            break;
        }
        constexpr double h = 1e-6;
        const glm::dvec2 dx =
            (Undistort(d + glm::dvec2(h, 0.0)) - Undistort(d - glm::dvec2(h, 0.0))) / (2.0 * h);
        const glm::dvec2 dy =
            (Undistort(d + glm::dvec2(0.0, h)) - Undistort(d - glm::dvec2(0.0, h))) / (2.0 * h);
        d -= glm::inverse(glm::dmat2(dx, dy)) * residual;
    }
    return d;
}

glm::dvec2 IsdLineScanner::FocalPlaneOf(double sample) const {
    const std::array<double, 3>& l = isd_.focalToLine;
    const std::array<double, 3>& s = isd_.focalToSample;
    // Line 0 of the detector, as usgscsm's line scanner takes it.
    const double t1 = isd_.startingDetectorLine - isd_.detectorCenterLine - l[0];
    const double t2 =
        sample * isd_.sampleSumming + isd_.startingDetectorSample - isd_.detectorCenterSample - s[0];
    const double determinant = l[1] * s[2] - l[2] * s[1];
    return {(s[2] * t1 - l[2] * t2) / determinant, (-s[1] * t1 + l[1] * t2) / determinant};
}

glm::dvec2 IsdLineScanner::PixelOf(glm::dvec2 focal) const {
    const std::array<double, 3>& l = isd_.focalToLine;
    const std::array<double, 3>& s = isd_.focalToSample;
    const double detectorSample = s[0] + s[1] * focal.x + s[2] * focal.y + isd_.detectorCenterSample;
    const double detectorLine = l[0] + l[1] * focal.x + l[2] * focal.y + isd_.detectorCenterLine;
    return {(detectorLine - isd_.startingDetectorLine) / isd_.lineSumming,
            (detectorSample - isd_.startingDetectorSample) / isd_.sampleSumming};
}

glm::dvec3 IsdLineScanner::Look(double line, double sample) const {
    const double time = ImageTime(line);
    const glm::dvec2 focal = Undistort(FocalPlaneOf(sample));
    const glm::dvec3 camera = glm::normalize(glm::dvec3(-focal.x, -focal.y, -isd_.focalLengthMm));
    return SensorToBody(time) * camera;
}

glm::dvec3 IsdLineScanner::ImageToGround(double line, double sample, double height) const {
    const glm::dvec3 c = Position(ImageTime(line));
    const glm::dvec3 l = Look(line, sample);
    // The ray against the ellipsoid of radii a + h, a + h, b + h: (x^2 + y^2 + k z^2) = (a + h)^2.
    const double a = isd_.semiMajorAxis + height;
    const double b = isd_.semiMinorAxis + height;
    const double k = a * a / (b * b);
    const double at = l.x * l.x + l.y * l.y + k * l.z * l.z;
    const double bt = 2.0 * (l.x * c.x + l.y * c.y + k * l.z * c.z);
    const double ct = c.x * c.x + c.y * c.y + k * c.z * c.z - a * a;
    const double root = std::sqrt(std::max(bt * bt - 4.0 * at * ct, 0.0));
    double scale = -bt - root;
    if (std::abs(-bt + root) < std::abs(scale)) {
        scale = -bt + root;
    }
    return c + scale / (2.0 * at) * l;
}

glm::dvec2 IsdLineScanner::DetectorView(const glm::dvec3& ground, double time) const {
    const glm::dvec3 camera = glm::inverse(SensorToBody(time)) * (ground - Position(time));
    // The look (-x, -y, -f) scaled to reach `camera`.
    const glm::dvec2 undistorted =
        glm::dvec2(camera.x, camera.y) * (isd_.focalLengthMm / camera.z);
    return PixelOf(Distort(undistorted));
}

glm::dvec2 IsdLineScanner::GroundToImage(const glm::dvec3& ground) const {
    // The line whose exposure puts `ground` on the detector line (an offset of 0 from it).
    double l0 = 0.5 * isd_.lines;
    double l1 = l0 + 1.0;
    double g0 = DetectorView(ground, ImageTime(l0)).x;
    double g1 = DetectorView(ground, ImageTime(l1)).x;
    for (int i = 0; i < 50 && std::abs(g1) > 1e-10; ++i) {
        const double slope = g1 - g0;
        if (slope == 0.0) {
            break;
        }
        const double next = l1 - g1 * (l1 - l0) / slope;
        l0 = l1;
        g0 = g1;
        l1 = next;
        g1 = DetectorView(ground, ImageTime(l1)).x;
    }
    return {l1, DetectorView(ground, ImageTime(l1)).y};
}

glm::dvec3 IsdLineScanner::SunPosition(double time) const {
    if (isd_.sunPositions.values.empty()) {
        throw std::invalid_argument("The ISD has no sun positions");
    }
    return RotationAt(isd_.bodyRotation, time) * Linear(isd_.sunPositions, time);
}

IsdLineScanner::Strip IsdLineScanner::StripOf(std::uint32_t firstLine,
                                              std::uint32_t lines) const {
    if (lines == 0) {
        throw std::invalid_argument("A strip needs at least one line");
    }
    // Which way the rays run: usgscsm's look (-x, -y, -f) meets the ellipsoid at its nearer
    // intersection on either side of the sensor, so ISDs whose boresight is the sensor's +z (as
    // NAIF's instrument frames have it, ALE's real cameras) and those whose is -z both image the
    // same. Ernie's camera must face the ground: `facing` is -1 when it lies behind the look.
    const double middle = ImageTime(firstLine + 0.5 * lines);
    const glm::dvec3 centre = ImageToGround(firstLine + 0.5 * lines, 0.5 * isd_.samples);
    const double facing =
        glm::dot(centre - Position(middle), Look(firstLine + 0.5 * lines, 0.5 * isd_.samples)) <
                0.0
            ? -1.0
            : 1.0;
    // Ernie's camera axes in the sensor's frame: x along the detector as samples increase
    // (the rays' change with the sample), z back from the scene, y completing them.
    const glm::dvec2 first = FocalPlaneOf(0.0);
    const glm::dvec2 along = FocalPlaneOf(1.0) - first;  // mm per image sample
    const glm::dvec3 x = glm::normalize(facing * glm::dvec3(-along.x, -along.y, 0.0));
    const glm::dvec3 z(0.0, 0.0, facing);
    const glm::dvec3 y = glm::cross(z, x);
    const glm::dmat3 ernieToSensor(x, y, z);
    // The ray of sample 0 in Ernie's axes, per unit of view depth: where the detector line lies
    // across the image (y) and where its principal point is (x).
    const double f = isd_.focalLengthMm;
    const glm::dvec3 ray0 =
        glm::transpose(ernieToSensor) * (facing * glm::dvec3(-first.x, -first.y, -f)) / f;
    const double pitch = glm::length(along);
    const double principalSample = -ray0.x * f / pitch;

    Strip strip;
    Camera& camera = strip.camera;
    camera.pushbroom = true;
    camera.width = isd_.samples;
    camera.height = lines;
    camera.focalLengthMm = f;
    camera.verticalApertureMm = pitch * lines;
    camera.horizontalApertureOffsetMm = (principalSample - 0.5 * isd_.samples) * pitch;
    camera.lineOffsetMm = ray0.y * f;
    // The lens's distortion along the detector as Brown-Conrady radial terms: distorted =
    // undistorted (1 + k1 r^2 + k2 r^4 + k3 r^6), fitted by least squares to the ISD's model at
    // samples along the line, in Ernie's normalised image coordinates.
    if (!isd_.distortion.empty()) {
        const auto normalised = [&](glm::dvec2 focal) {
            const glm::dvec3 ray =
                glm::transpose(ernieToSensor) * (facing * glm::dvec3(-focal.x, -focal.y, -f));
            return glm::dvec2(ray.x, ray.y) / -ray.z;
        };
        std::vector<std::pair<glm::dvec2, double>> samples;
        double largest = 0.0;  // r^2
        constexpr int kSamples = 128;
        for (int i = 0; i <= kSamples; ++i) {
            const double sample = isd_.samples * double(i) / kSamples;
            const glm::dvec2 distorted = FocalPlaneOf(sample);
            const glm::dvec2 u = normalised(Undistort(distorted));
            samples.emplace_back(u, normalised(distorted).x);
            largest = std::max(largest, glm::dot(u, u));
        }
        // distorted = undistorted (a + b1 r^2 + b2 r^4 + b3 r^6): a scale too, as radial models
        // with a constant term have, which the camera's focal length takes (f a), leaving
        // Brown-Conrady's k = b / a on its sensor's coordinates (distorted / a). In powers of
        // r^2 / largest, for a well-conditioned system, then scaled back.
        glm::dmat4 normal(0.0);
        glm::dvec4 right(0.0);
        for (const auto& [u, d] : samples) {
            const double s = glm::dot(u, u) / largest;
            const glm::dvec4 basis = u.x * glm::dvec4(1.0, s, s * s, s * s * s);
            normal += glm::outerProduct(basis, basis);
            right += basis * d;
        }
        glm::dvec4 b(1.0, 0.0, 0.0, 0.0);
        if (largest > 0.0 && glm::determinant(normal) != 0.0) {
            b = glm::inverse(normal) * right;
            b /= glm::dvec4(1.0, largest, largest * largest, largest * largest * largest);
        }
        const double scale = b.x;
        camera.focalLengthMm = f * scale;
        camera.lens.k1 = b.y / scale;
        camera.lens.k2 = b.z / scale;
        camera.lens.k3 = b.w / scale;
        for (const auto& [u, d] : samples) {
            const double r2 = glm::dot(u, u);
            const double fitted = u.x * (scale + r2 * (b.y + r2 * (b.z + r2 * b.w)));
            strip.distortionResidualPixels =
                std::max(strip.distortionResidualPixels, std::abs(fitted - d) * f / pitch);
        }
    }
    // Row r's centre is image line first + r + 0.5, exposed (r + 0.5) / lines of the way from
    // `fromTime` to `toTime` (the rate constant over the strip).
    strip.fromTime = ImageTime(firstLine);
    strip.toTime = ImageTime(static_cast<double>(firstLine) + lines);
    const auto pose = [&](double time) {
        glm::dmat4 pose(glm::mat3_cast(SensorToBody(time)) * ernieToSensor);
        pose[3] = glm::dvec4(Position(time), 1.0);
        return pose;
    };
    strip.from = pose(strip.fromTime);
    strip.to = pose(strip.toTime);
    // Clipping around the body: from half the height above it to past its far side.
    const double distance = glm::length(glm::dvec3(strip.to[3]));
    camera.nearPlane = std::max(0.5 * (distance - isd_.semiMajorAxis), 1e-3);
    camera.farPlane = distance + isd_.semiMajorAxis;
    return strip;
}

IsdSensor ReadIsdFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot read the ISD '" + path + "'");
    }
    std::ostringstream text;
    text << file.rdbuf();
    return ReadIsd(text.str());
}

}  // namespace ernie
