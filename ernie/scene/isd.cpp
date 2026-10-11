// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/isd.h"

#include "ernie/util/least_squares.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace ernie {

namespace {

using Json = nlohmann::json;

const Json& Field(const Json& object, const char* name) {
    const auto found = object.find(name);
    if (found == object.end()) {
        throw std::invalid_argument(std::string("The ISD has no '") +
                                    name + "'");
    }
    return *found;
}

double Number(const Json& object, const char* name) {
    const Json& value = Field(object, name);
    if (!value.is_number()) {
        throw std::invalid_argument(std::string("The ISD's '") +
                                    name + "' is not a number");
    }
    return value.get<double>();
}

double NumberOr(const Json& object, const char* name,
                double fallback) {
    return object.contains(name) ? Number(object, name) : fallback;
}

/// Metres per unit of `object`'s "unit", kilometres when it has none
/// (as ALE writes them).
double MetresPer(const Json& object) {
    const std::string unit = object.value("unit", std::string("km"));
    if (unit == "km") {
        return 1000.0;
    }
    if (unit == "m") {
        return 1.0;
    }
    throw std::invalid_argument("The ISD's length unit '" + unit +
                                "' is neither m nor km");
}

std::vector<double> Numbers(const Json& value, const char* name) {
    if (!value.is_array()) {
        throw std::invalid_argument(std::string("The ISD's '") +
                                    name + "' is not an array");
    }
    std::vector<double> numbers;
    for (const Json& item : value) {
        numbers.push_back(item.get<double>());
    }
    return numbers;
}

glm::dvec3 Vector(const Json& value, double scale) {
    if (!value.is_array() || value.size() != 3) {
        throw std::invalid_argument(
            "The ISD has a vector without three components");
    }
    return glm::dvec3(value[0].get<double>(), value[1].get<double>(),
                      value[2].get<double>()) *
           scale;
}

IsdSamples<glm::dvec3> Vectors(const Json& block, const char* name,
                               double scale) {
    IsdSamples<glm::dvec3> samples;
    samples.times =
        Numbers(Field(block, "ephemeris_times"), "ephemeris_times");
    for (const Json& item : Field(block, name)) {
        samples.values.push_back(Vector(item, scale));
    }
    if (samples.values.size() != samples.times.size()) {
        throw std::invalid_argument(
            std::string("The ISD's '") + name +
            "' and their times differ in number");
    }
    return samples;
}

std::vector<std::int64_t> Frames(const Json& block,
                                 const char* name) {
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
    rotation.quaternions.times =
        Numbers(Field(block, "ephemeris_times"), "ephemeris_times");
    for (const Json& item : Field(block, "quaternions")) {
        if (!item.is_array() || item.size() != 4) {
            throw std::invalid_argument(
                "The ISD has a quaternion without four components");
        }
        // Scalar first, as ALE reads them (`ale::getJsonQuatArray`).
        rotation.quaternions.values.emplace_back(
            item[0].get<double>(), item[1].get<double>(),
            item[2].get<double>(), item[3].get<double>());
    }
    if (rotation.quaternions.values.size() !=
        rotation.quaternions.times.size()) {
        throw std::invalid_argument(
            "The ISD's quaternions and their times differ in number");
    }
    rotation.timeDependentFrames =
        Frames(block, "time_dependent_frames");
    rotation.constantFrames = Frames(block, "constant_frames");
    if (block.contains("constant_rotation")) {
        const std::vector<double> m =
            Numbers(block["constant_rotation"], "constant_rotation");
        if (m.size() != 9) {
            throw std::invalid_argument(
                "The ISD's constant rotation is not 3 x 3");
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                rotation.constantRotation[c][r] =
                    m[static_cast<std::size_t>(r * 3 + c)];
            }
        }
    }
    rotation.referenceFrame =
        block.value("reference_frame", std::int64_t{1});
    return rotation;
}

std::array<double, 3> Triple(const Json& object, const char* name) {
    const std::vector<double> values =
        Numbers(Field(object, name), name);
    if (values.size() != 3) {
        throw std::invalid_argument(std::string("The ISD's '") +
                                    name + "' needs three numbers");
    }
    return {values[0], values[1], values[2]};
}

}  // namespace

namespace {

/// The index of the last of sorted `times` at or before `time`,
/// clamped to [0, size - 2] so that it and the next bracket `time` or
/// extrapolate past the ends.
std::size_t BracketOf(const std::vector<double>& times, double time) {
    if (times.size() < 2) {
        return 0;
    }
    const auto after =
        std::upper_bound(times.begin(), times.end(), time);
    const std::size_t index =
        after == times.begin()
            ? 0
            : std::size_t(after - times.begin()) - 1;
    return std::min(index, times.size() - 2);
}

/// Lagrange interpolation at `time` through up to eight of `samples`
/// around it.
glm::dvec3 Lagrange(const IsdSamples<glm::dvec3>& samples,
                    double time) {
    const std::size_t count = samples.times.size();
    if (count == 1) {
        return samples.values.front();
    }
    const std::size_t order = std::min<std::size_t>(8, count);
    const std::size_t at = BracketOf(samples.times, time);
    const std::size_t first = std::min(
        count - order,
        at + 1 >= order / 2 ? at + 1 - order / 2 : std::size_t{0});
    glm::dvec3 value(0.0);
    for (std::size_t i = first; i < first + order; ++i) {
        double weight = 1.0;
        for (std::size_t j = first; j < first + order; ++j) {
            if (j != i) {
                weight *= (time - samples.times[j]) /
                          (samples.times[i] - samples.times[j]);
            }
        }
        value += weight * samples.values[i];
    }
    return value;
}

/// usgscsm's Lagrange interpolation (`lagrangeInterp`) at `time` of
/// `values` sampled every `step` from `start`: eight points around
/// `time`, fewer near the ends.
template <typename T>
T UniformLagrange(const std::vector<T>& values, double start,
                  double step, double time) {
    const int count = static_cast<int>(values.size());
    if (count == 1) {
        return values.front();
    }
    const double fraction = (time - start) / step;
    const int index =
        std::clamp(static_cast<int>(fraction), 0, count - 2);
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
                weight *= (t - (j - order / 2 + 1)) /
                          static_cast<double>(i - j);
            }
        }
        value += weight * values[static_cast<std::size_t>(first + i)];
    }
    return value;
}

/// ALE's state interpolation (`ale::States::getState`, SPLINE) of
/// `positions` with `velocities`: a cubic Hermite between the samples
/// around `time`, or Lagrange without velocities.
glm::dvec3 Hermite(const IsdSamples<glm::dvec3>& positions,
                   const IsdSamples<glm::dvec3>& velocities,
                   double time) {
    const std::vector<double>& times = positions.times;
    const auto exact =
        std::lower_bound(times.begin(), times.end(), time);
    if (exact != times.end() && *exact == time) {
        return positions.values[std::size_t(exact - times.begin())];
    }
    if (velocities.values.size() != positions.values.size() ||
        times.size() < 2) {
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
glm::dvec3 Linear(const IsdSamples<glm::dvec3>& samples,
                  double time) {
    if (samples.times.size() == 1) {
        return samples.values.front();
    }
    const std::size_t at = BracketOf(samples.times, time);
    const double t = (time - samples.times[at]) /
                     (samples.times[at + 1] - samples.times[at]);
    return glm::mix(samples.values[at], samples.values[at + 1], t);
}

}  // namespace

bool IsdSensor::IsLineScanner() const {
    return model == "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL";
}

bool IsdSensor::IsPushFrame() const {
    return model == "USGS_ASTRO_PUSH_FRAME_SENSOR_MODEL";
}

bool IsdSensor::IsFrame() const {
    return model == "USGS_ASTRO_FRAME_SENSOR_MODEL";
}

IsdSensor ReadIsd(std::string_view json) {
    Json isd;
    try {
        isd = Json::parse(json);
    } catch (const Json::exception& error) {
        throw std::invalid_argument(
            std::string("The ISD is not JSON: ") + error.what());
    }
    try {
        IsdSensor sensor;
        sensor.model = Field(isd, "name_model").get<std::string>();
        sensor.platform = isd.value("name_platform", std::string());
        sensor.sensor = isd.value("name_sensor", std::string());
        sensor.lines =
            static_cast<std::uint32_t>(Number(isd, "image_lines"));
        sensor.samples =
            static_cast<std::uint32_t>(Number(isd, "image_samples"));

        sensor.focalLengthMm =
            Number(Field(isd, "focal_length_model"), "focal_length");
        sensor.focalToLine = Triple(isd, "focal2pixel_lines");
        sensor.focalToSample = Triple(isd, "focal2pixel_samples");
        const Json& center = Field(isd, "detector_center");
        sensor.detectorCenterLine = Number(center, "line");
        sensor.detectorCenterSample = Number(center, "sample");
        sensor.startingDetectorLine =
            NumberOr(isd, "starting_detector_line", 0.0);
        sensor.startingDetectorSample =
            NumberOr(isd, "starting_detector_sample", 0.0);
        sensor.lineSumming =
            NumberOr(isd, "detector_line_summing", 1.0);
        sensor.sampleSumming =
            NumberOr(isd, "detector_sample_summing", 1.0);
        if (isd.contains("optical_distortion") &&
            !isd["optical_distortion"].empty()) {
            // Its one model, by an iterator that outlives the reads
            // (items() makes a temporary).
            const auto first = isd["optical_distortion"].begin();
            const std::string name = first.key();
            const auto& model = first.value();
            sensor.distortion = name;
            if (model.contains("coefficients")) {
                sensor.distortionCoefficients =
                    Numbers(model["coefficients"], "coefficients");
            } else if (model.contains("x") && model.contains("y")) {
                // As usgscsm's `getDistortionCoeffs`: the x then the
                // y, padded with zeros to ten each for "transverse",
                // four each after their boresights for "kaguyalism".
                const bool kaguya = name == "kaguyalism";
                std::vector<double> x = Numbers(model["x"], "x");
                std::vector<double> y = Numbers(model["y"], "y");
                x.resize(kaguya ? 4 : 10, 0.0);
                y.resize(kaguya ? 4 : 10, 0.0);
                if (kaguya) {
                    x.insert(x.begin(), Number(model, "boresight_x"));
                    y.insert(y.begin(), Number(model, "boresight_y"));
                }
                sensor.distortionCoefficients = std::move(x);
                sensor.distortionCoefficients.insert(
                    sensor.distortionCoefficients.end(), y.begin(),
                    y.end());
            }
        }

        if (isd.contains("line_scan_rate")) {
            for (const Json& rate : isd["line_scan_rate"]) {
                const std::vector<double> row =
                    Numbers(rate, "line_scan_rate");
                if (row.size() != 3) {
                    throw std::invalid_argument(
                        "The ISD's line scan rates need three "
                        "numbers");
                }
                sensor.lineRates.push_back({row[0], row[1], row[2]});
            }
        }
        sensor.startingTime =
            NumberOr(isd, "starting_ephemeris_time", 0.0);
        sensor.centerTime = Number(isd, "center_ephemeris_time");
        sensor.exposureDuration =
            NumberOr(isd, "exposure_duration", 0.0);
        sensor.interframeDelay =
            NumberOr(isd, "interframe_delay", 0.0);
        sensor.frameletHeight = static_cast<std::uint32_t>(
            NumberOr(isd, "framelet_height", 0.0));
        sensor.frameletOverlap = static_cast<std::uint32_t>(
            NumberOr(isd, "num_lines_overlap", 0.0));
        sensor.frameletsFlipped =
            isd.value("framelets_flipped", false);
        sensor.frameletOrderReversed =
            isd.value("framelet_order_reversed", false);
        if (isd.contains("jitter")) {
            const Json& jitter = isd["jitter"];
            sensor.lineJitter =
                Numbers(Field(jitter, "lineJitterCoefficients"),
                        "lineJitterCoefficients");
            sensor.sampleJitter =
                Numbers(Field(jitter, "sampleJitterCoefficients"),
                        "sampleJitterCoefficients");
            sensor.jitterLineTimes =
                Numbers(Field(jitter, "lineExposureTimes"),
                        "lineExposureTimes");
        }

        const Json& radii = Field(isd, "radii");
        sensor.semiMajorAxis =
            Number(radii, "semimajor") * MetresPer(radii);
        sensor.semiMinorAxis =
            Number(radii, "semiminor") * MetresPer(radii);

        const Json& position = Field(isd, "instrument_position");
        const double metres = MetresPer(position);
        sensor.positions = Vectors(position, "positions", metres);
        if (position.contains("velocities")) {
            sensor.velocities =
                Vectors(position, "velocities", metres);
        }
        sensor.positionFrame =
            position.value("reference_frame", std::int64_t{1});
        if (isd.contains("sun_position")) {
            const Json& sun = isd["sun_position"];
            sensor.sunPositions =
                Vectors(sun, "positions", MetresPer(sun));
        }
        sensor.pointing = Rotation(Field(isd, "instrument_pointing"));
        sensor.bodyRotation = Rotation(Field(isd, "body_rotation"));
        return sensor;
    } catch (const Json::exception& error) {
        throw std::invalid_argument(
            std::string("The ISD has a mistyped field: ") +
            error.what());
    }
}

IsdCameraModel::IsdCameraModel(IsdSensor isd) : isd_(std::move(isd)) {
    if (isd_.positions.values.empty() ||
        isd_.pointing.quaternions.values.empty() ||
        isd_.bodyRotation.quaternions.values.empty()) {
        throw std::invalid_argument(
            "A camera's ISD needs positions and pointing and body "
            "rotations");
    }
    const double determinant =
        isd_.focalToLine[1] * isd_.focalToSample[2] -
        isd_.focalToLine[2] * isd_.focalToSample[1];
    if (determinant == 0.0) {
        throw std::invalid_argument(
            "The ISD's focal plane map to pixels is singular");
    }
    // Each model's coefficient count, as usgscsm's `removeDistortion`
    // checks them (CAHVOR's sixth, which usgscsm's default carries,
    // is unused).
    static const std::map<
        std::string, std::pair<std::size_t, std::size_t>, std::less<>>
        kCoefficients = {
            {"radial", {0, 3}},       {"transverse", {20, 20}},
            {"radtan", {5, 5}},       {"lrolrocnac", {1, 1}},
            {"kaguyalism", {10, 10}}, {"dawnfc", {1, 1}},
            {"cahvor", {5, 6}},       {"kplo_shadowcam", {1, 1}},
            {"cassis", {36, 36}}};
    if (!isd_.distortion.empty()) {
        const auto model = kCoefficients.find(isd_.distortion);
        if (model == kCoefficients.end()) {
            throw std::invalid_argument(
                "The ISD's distortion model '" + isd_.distortion +
                "' is not supported; use radial, transverse, radtan, "
                "lrolrocnac, kaguyalism, dawnfc, cahvor, "
                "kplo_shadowcam "
                "or cassis");
        }
        const auto [least, most] = model->second;
        const std::size_t count = isd_.distortionCoefficients.size();
        if (count < least || count > most) {
            throw std::invalid_argument(
                "The ISD's " + isd_.distortion +
                " distortion takes " +
                (least == most ? std::to_string(least)
                               : std::to_string(least) + " to " +
                                     std::to_string(most)) +
                " coefficients, not " + std::to_string(count));
        }
    }
    if (isd_.distortion == "radial") {
        isd_.distortionCoefficients.resize(3, 0.0);
    }
    // As usgscsm samples the trajectory (`constructStateFromIsd`): at
    // least 25 positions evenly over the states' times, each
    // interpolated in the ISD's frame and rotated into the body's.
    const std::vector<double>& times = isd_.positions.times;
    const std::size_t positions =
        std::max<std::size_t>(times.size(), 25);
    positionStart_ = times.front();
    positionStep_ =
        times.size() > 1
            ? (times.back() - times.front()) / double(positions - 1)
            : 1.0;
    for (std::size_t i = 0; i < (times.size() > 1 ? positions : 1);
         ++i) {
        const double time =
            positionStart_ + double(i) * positionStep_;
        positionNodes_.push_back(
            RotationAt(isd_.bodyRotation, time) *
            Hermite(isd_.positions, isd_.velocities, time));
    }
    // And the sensor's orientation in the body: body rotation x
    // inverse pointing at the union of their times, interpolated
    // along the great arc to as many even times, as components.
    std::vector<double> merged = isd_.pointing.quaternions.times;
    merged.insert(merged.end(),
                  isd_.bodyRotation.quaternions.times.begin(),
                  isd_.bodyRotation.quaternions.times.end());
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()),
                 merged.end());
    IsdRotation sensorToBody;
    sensorToBody.constantRotation =
        isd_.bodyRotation.constantRotation;
    const glm::dmat3 inverseConstant =
        glm::transpose(isd_.bodyRotation.constantRotation);
    for (const double time : merged) {
        sensorToBody.quaternions.times.push_back(time);
        sensorToBody.quaternions.values.push_back(
            glm::normalize(glm::quat_cast(
                inverseConstant *
                glm::mat3_cast(
                    RotationAt(isd_.bodyRotation, time) *
                    glm::inverse(RotationAt(isd_.pointing, time))))));
    }
    quaternionStart_ = merged.front();
    quaternionStep_ = merged.size() > 1
                          ? (merged.back() - merged.front()) /
                                double(merged.size() - 1)
                          : 1.0;
    for (std::size_t i = 0; i < merged.size(); ++i) {
        const glm::dquat q =
            RotationAt(sensorToBody, quaternionStart_ +
                                         double(i) * quaternionStep_);
        glm::dvec4 node(q.x, q.y, q.z, q.w);
        // Sign-continuous, so the components interpolate.
        if (!quaternionNodes_.empty() &&
            glm::dot(node, quaternionNodes_.back()) < 0.0) {
            node = -node;
        }
        quaternionNodes_.push_back(node);
    }
}

glm::dquat IsdCameraModel::RotationAt(const IsdRotation& rotation,
                                      double time) {
    const IsdSamples<glm::dquat>& q = rotation.quaternions;
    glm::dquat timeDependent = q.values.front();
    if (q.values.size() > 1) {
        const std::size_t at = BracketOf(q.times, time);
        const double t =
            (time - q.times[at]) / (q.times[at + 1] - q.times[at]);
        glm::dquat next = q.values[at + 1];
        if (glm::dot(q.values[at], next) < 0.0) {
            next = -next;
        }
        timeDependent = glm::slerp(q.values[at], next, t);
    }
    return glm::normalize(glm::quat_cast(
        rotation.constantRotation *
        glm::mat3_cast(glm::normalize(timeDependent))));
}

glm::dvec3 IsdCameraModel::Position(double time) const {
    return UniformLagrange(positionNodes_, positionStart_,
                           positionStep_, time);
}

glm::dquat IsdCameraModel::SensorToBody(double time) const {
    // usgscsm interpolates eighth order, fourth with fewer than two
    // quaternions.
    const glm::dvec4 q = glm::normalize(UniformLagrange(
        quaternionNodes_, quaternionStart_, quaternionStep_, time));
    return glm::dquat(q.w, q.x, q.y, q.z);
}

namespace {

// The point p with f(p) = `target`, by Newton's method from `target`,
// f's Jacobian by central differences.
template <typename F>
glm::dvec2 NewtonInverse(const F& f, glm::dvec2 target) {
    glm::dvec2 p = target;
    for (int i = 0; i < 20; ++i) {
        const glm::dvec2 residual = f(p) - target;
        if (glm::length(residual) < 1e-14) {
            break;
        }
        constexpr double h = 1e-6;
        const glm::dvec2 dx =
            (f(p + glm::dvec2(h, 0.0)) - f(p - glm::dvec2(h, 0.0))) /
            (2.0 * h);
        const glm::dvec2 dy =
            (f(p + glm::dvec2(0.0, h)) - f(p - glm::dvec2(0.0, h))) /
            (2.0 * h);
        p -= glm::inverse(glm::dmat2(dx, dy)) * residual;
    }
    return p;
}

// usgscsm's `computeTransverseDistortion`: distorted x and y as cubic
// polynomials of undistorted `u`, coefficients `k` the x's ten then
// the y's.
glm::dvec2 TransverseDistort(const std::vector<double>& k,
                             glm::dvec2 u) {
    const double terms[10] = {1.0,
                              u.x,
                              u.y,
                              u.x * u.x,
                              u.x * u.y,
                              u.y * u.y,
                              u.x * u.x * u.x,
                              u.x * u.x * u.y,
                              u.x * u.y * u.y,
                              u.y * u.y * u.y};
    glm::dvec2 d(0.0);
    for (int i = 0; i < 10; ++i) {
        d += terms[i] *
             glm::dvec2(k[static_cast<std::size_t>(i)],
                        k[static_cast<std::size_t>(i + 10)]);
    }
    return d;
}

// `transverseDistortionJacobian`: d(distorted) / d(undistorted),
// columns x and y.
glm::dmat2 TransverseJacobian(const std::vector<double>& k,
                              glm::dvec2 u) {
    const double dX[10] = {0.0,
                           1.0,
                           0.0,
                           2.0 * u.x,
                           u.y,
                           0.0,
                           3.0 * u.x * u.x,
                           2.0 * u.x * u.y,
                           u.y * u.y,
                           0.0};
    const double dY[10] = {0.0,
                           0.0,
                           1.0,
                           0.0,
                           u.x,
                           2.0 * u.y,
                           0.0,
                           u.x * u.x,
                           2.0 * u.x * u.y,
                           3.0 * u.y * u.y};
    glm::dmat2 jacobian(0.0);
    for (std::size_t i = 0; i < 10; ++i) {
        jacobian[0] += dX[i] * glm::dvec2(k[i], k[i + 10]);
        jacobian[1] += dY[i] * glm::dvec2(k[i], k[i + 10]);
    }
    return jacobian;
}

// usgscsm's `computeRadTanDistortion`: OpenCV's radial-tangential
// model on normalised coordinates `u` (focal plane over focal
// length), coefficients (k1, k2, p1, p2, k3).
glm::dvec2 RadTanDistort(const std::vector<double>& k, glm::dvec2 u) {
    const double r2 = glm::dot(u, u);
    const double radial = 1.0 + r2 * (k[0] + r2 * (k[1] + r2 * k[4]));
    return {u.x * radial + 2.0 * k[2] * u.x * u.y +
                k[3] * (r2 + 2.0 * u.x * u.x),
            u.y * radial + k[2] * (r2 + 2.0 * u.y * u.y) +
                2.0 * k[3] * u.x * u.y};
}

// `radTanDistortionJacobian`.
glm::dmat2 RadTanJacobian(const std::vector<double>& k,
                          glm::dvec2 u) {
    const double r2 = glm::dot(u, u);
    const double radial = 1.0 + r2 * (k[0] + r2 * (k[1] + r2 * k[4]));
    // d(radial) / d(r^2)
    const double slope = k[0] + r2 * (2.0 * k[1] + 3.0 * r2 * k[4]);
    const double xx = radial + 2.0 * u.x * u.x * slope +
                      2.0 * k[2] * u.y + 6.0 * k[3] * u.x;
    const double xy =
        2.0 * u.x * u.y * slope + 2.0 * k[2] * u.x + 2.0 * k[3] * u.y;
    const double yx =
        2.0 * u.x * u.y * slope + 2.0 * k[2] * u.x + 2.0 * k[3] * u.y;
    const double yy = radial + 2.0 * u.y * u.y * slope +
                      6.0 * k[2] * u.y + 2.0 * k[3] * u.x;
    return glm::dmat2(glm::dvec2(xx, yx), glm::dvec2(xy, yy));
}

// TGO CaSSIS's rational model (usgscsm's `cassisChiDotA`, from ISIS's
// `TgoCassisDistortionMap`): a point's (A1 . chi, A2 . chi) / (A3 .
// chi), chi = (x^2, xy, y^2, x, y, 1), `a` the three six-vectors.
// Valid on the 2048-pixel, 10 um detector and 0.2 mm beyond, the
// point unchanged outside.
glm::dvec2 CassisMap(const double* a, glm::dvec2 p) {
    constexpr double kBound = 0.5 * 0.01 * 2048.0 + 0.2;
    if (std::abs(p.x) > kBound || std::abs(p.y) > kBound) {
        return p;
    }
    const auto dot = [&](const double* c) {
        return c[0] * p.x * p.x + c[1] * p.x * p.y +
               c[2] * p.y * p.y + c[3] * p.x + c[4] * p.y + c[5];
    };
    return glm::dvec2(dot(a), dot(a + 6)) / dot(a + 12);
}

// usgscsm's `newtonRaphson`, as its `removeDistortion` calls it: the
// undistorted point whose `distort` is `d`, from `d`, to a tolerance
// of 1e-6 in 20 tries, stopping where the Jacobian's determinant
// falls under 1e-6 (so a singular polynomial, as usgscsm's simple
// framer has, leaves the point as it is).
template <typename Distort, typename Jacobian>
glm::dvec2 UsgscsmUndistort(glm::dvec2 d, const Distort& distort,
                            const Jacobian& jacobian) {
    constexpr double kTolerance = 1e-6;
    glm::dvec2 u = d;
    glm::dvec2 f = distort(u);
    for (int count = 1;
         std::abs(f.x) + std::abs(f.y) > kTolerance && count < 20;
         ++count) {
        f = d - distort(u);
        const glm::dmat2 j = jacobian(u);
        const double determinant = glm::determinant(j);
        if (std::abs(determinant) < 1e-6) {
            break;
        }
        u += glm::inverse(j) * f;
    }
    return u;
}

}  // namespace

glm::dvec2 IsdCameraModel::Undistort(glm::dvec2 d) const {
    const std::vector<double>& k = isd_.distortionCoefficients;
    if (isd_.distortion == "radial") {
        const double rr = glm::dot(d, d);
        return d * (1.0 - (k[0] + rr * (k[1] + rr * k[2])));
    }
    if (isd_.distortion == "lrolrocnac") {
        return {d.x, d.y / (1.0 + k[0] * d.y * d.y)};
    }
    if (isd_.distortion == "transverse") {
        return UsgscsmUndistort(
            d, [&](glm::dvec2 u) { return TransverseDistort(k, u); },
            [&](glm::dvec2 u) { return TransverseJacobian(k, u); });
    }
    if (isd_.distortion == "radtan") {
        // On coordinates normalised by the focal length, as usgscsm's
        // `removeDistortion`.
        const double f = isd_.focalLengthMm;
        return f *
               UsgscsmUndistort(
                   d / f,
                   [&](glm::dvec2 u) { return RadTanDistort(k, u); },
                   [&](glm::dvec2 u) {
                       return RadTanJacobian(k, u);
                   });
    }
    if (isd_.distortion == "kaguyalism") {
        // Kaguya TC and MI (SELENE's SEL_TC_V01.TI and
        // SEL_MI_V01.TI): each axis offset by its boresight and a
        // cubic in the distorted point's radius.
        const double r = glm::length(d);
        const auto cubic = [r](const double* c) {
            return c[0] + r * (c[1] + r * (c[2] + r * c[3]));
        };
        return d +
               glm::dvec2(k[0] + cubic(&k[1]), k[5] + cubic(&k[6]));
    }
    if (isd_.distortion == "dawnfc") {
        // Dawn FC's model distorts in closed form (see `Distort`);
        // undone by Newton's method.
        return NewtonInverse([&](glm::dvec2 u) { return Distort(u); },
                             d);
    }
    if (isd_.distortion == "cahvor") {
        // The radial model about the optical centre (k[3], k[4]).
        const glm::dvec2 centre(k[3], k[4]);
        const glm::dvec2 shifted = d - centre;
        const double rr = glm::dot(shifted, shifted);
        return centre +
               shifted * (1.0 - (k[0] + rr * (k[1] + rr * k[2])));
    }
    if (isd_.distortion == "kplo_shadowcam") {
        // KPLO ShadowCam (ISIS's `ShadowCamDistortionMap`): a cubic
        // across the line only.
        return {d.x, d.y * (1.0 + k[0] * d.y * d.y)};
    }
    if (isd_.distortion == "cassis") {
        return CassisMap(k.data(), d);
    }
    return d;
}

glm::dvec2 IsdCameraModel::Distort(glm::dvec2 undistorted) const {
    if (isd_.distortion.empty()) {
        return undistorted;
    }
    if (isd_.distortion == "transverse") {
        return TransverseDistort(isd_.distortionCoefficients,
                                 undistorted);
    }
    if (isd_.distortion == "radtan") {
        const double f = isd_.focalLengthMm;
        return f * RadTanDistort(isd_.distortionCoefficients,
                                 undistorted / f);
    }
    if (isd_.distortion == "dawnfc") {
        // usgscsm's `applyDistortion`: Dawn FC's radial term on the
        // undistorted radius.
        return undistorted *
               (1.0 + isd_.distortionCoefficients[0] *
                          glm::dot(undistorted, undistorted));
    }
    if (isd_.distortion == "cassis") {
        // CaSSIS's own distorting fit, not the correcting one's
        // inverse.
        return CassisMap(isd_.distortionCoefficients.data() + 18,
                         undistorted);
    }
    return NewtonInverse([&](glm::dvec2 d) { return Undistort(d); },
                         undistorted);
}

glm::dvec2 IsdCameraModel::FocalPlaneOf(double line,
                                        double sample) const {
    const std::array<double, 3>& l = isd_.focalToLine;
    const std::array<double, 3>& s = isd_.focalToSample;
    // usgscsm's `computeDistortedFocalPlaneCoordinates`.
    const double t1 = line * isd_.lineSumming +
                      isd_.startingDetectorLine -
                      isd_.detectorCenterLine - l[0];
    const double t2 = sample * isd_.sampleSumming +
                      isd_.startingDetectorSample -
                      isd_.detectorCenterSample - s[0];
    const double determinant = l[1] * s[2] - l[2] * s[1];
    return {(s[2] * t1 - l[2] * t2) / determinant,
            (-s[1] * t1 + l[1] * t2) / determinant};
}

glm::dvec2 IsdCameraModel::PixelOf(glm::dvec2 focal) const {
    const std::array<double, 3>& l = isd_.focalToLine;
    const std::array<double, 3>& s = isd_.focalToSample;
    const double detectorSample = s[0] + s[1] * focal.x +
                                  s[2] * focal.y +
                                  isd_.detectorCenterSample;
    const double detectorLine = l[0] + l[1] * focal.x +
                                l[2] * focal.y +
                                isd_.detectorCenterLine;
    return {(detectorLine - isd_.startingDetectorLine) /
                isd_.lineSumming,
            (detectorSample - isd_.startingDetectorSample) /
                isd_.sampleSumming};
}

glm::dvec2 IsdCameraModel::FocalPlaneView(const glm::dvec3& ground,
                                          double time) const {
    const glm::dvec3 camera =
        glm::inverse(SensorToBody(time)) * (ground - Position(time));
    // The look (-x, -y, -f) scaled to reach `camera`.
    return glm::dvec2(camera.x, camera.y) *
           (isd_.focalLengthMm / camera.z);
}

glm::dvec3 IsdCameraModel::LookAt(double time, double line,
                                  double sample) const {
    const glm::dvec2 focal = Undistort(FocalPlaneOf(line, sample));
    const glm::dvec3 camera = glm::normalize(
        glm::dvec3(-focal.x, -focal.y, -isd_.focalLengthMm));
    return SensorToBody(time) * camera;
}

glm::dvec3 IsdCameraModel::EllipsoidIntersect(const glm::dvec3& c,
                                              const glm::dvec3& l,
                                              double height,
                                              bool eitherSide) const {
    // The ray against the ellipsoid of radii a + h, a + h, b + h:
    // (x^2 + y^2 + k z^2) = (a + h)^2.
    const double a = isd_.semiMajorAxis + height;
    const double b = isd_.semiMinorAxis + height;
    const double k = a * a / (b * b);
    const double at = l.x * l.x + l.y * l.y + k * l.z * l.z;
    const double bt = 2.0 * (l.x * c.x + l.y * c.y + k * l.z * c.z);
    const double ct = c.x * c.x + c.y * c.y + k * c.z * c.z - a * a;
    const double root =
        std::sqrt(std::max(bt * bt - 4.0 * at * ct, 0.0));
    double scale = -bt - root;
    if (eitherSide && std::abs(-bt + root) < std::abs(scale)) {
        scale = -bt + root;
    }
    return c + scale / (2.0 * at) * l;
}

glm::dvec3 IsdCameraModel::SunPosition(double time) const {
    if (isd_.sunPositions.values.empty()) {
        throw std::invalid_argument("The ISD has no sun positions");
    }
    return RotationAt(isd_.bodyRotation, time) *
           Linear(isd_.sunPositions, time);
}

glm::dmat3 IsdCameraModel::ErnieAxes(double line,
                                     double facing) const {
    // x along the detector as samples increase (the rays' change with
    // the sample), z back from the scene, y completing them.
    const glm::dvec2 along =
        FocalPlaneOf(line, 1.0) - FocalPlaneOf(line, 0.0);
    const glm::dvec3 x =
        glm::normalize(facing * glm::dvec3(-along.x, -along.y, 0.0));
    const glm::dvec3 z(0.0, 0.0, facing);
    return glm::dmat3(x, glm::cross(z, x), z);
}

glm::dmat4 IsdCameraModel::PoseAt(double time,
                                  const glm::dmat3& axes) const {
    glm::dmat4 pose(glm::mat3_cast(SensorToBody(time)) * axes);
    pose[3] = glm::dvec4(Position(time), 1.0);
    return pose;
}

double IsdCameraModel::FitDistortion(CameraParams& camera,
                                     std::span<const double> lines,
                                     const glm::dmat3& axes,
                                     double facing) const {
    if (isd_.distortion.empty()) {
        return 0.0;
    }
    // The lens's distortion as Brown-Conrady terms: distorted =
    // undistorted (1 + k1 r^2 + k2 r^4 + k3 r^6) + decentring of p1
    // and p2, fitted by least squares to the ISD's model at samples
    // along the lines, in Ernie's normalised image coordinates.
    const double f = isd_.focalLengthMm;
    const double pitch =
        glm::length(FocalPlaneOf(lines.front(), 1.0) -
                    FocalPlaneOf(lines.front(), 0.0));
    const auto normalised = [&](glm::dvec2 focal) {
        const glm::dvec3 ray =
            glm::transpose(axes) *
            (facing * glm::dvec3(-focal.x, -focal.y, -f));
        return glm::dvec2(ray.x, ray.y) / -ray.z;
    };
    const bool across = lines.size() > 1;
    std::vector<std::pair<glm::dvec2, glm::dvec2>>
        samples;           // undistorted, distorted
    double largest = 0.0;  // r^2
    constexpr int kSamples = 128;
    for (const double line : lines) {
        for (int i = 0; i <= kSamples; ++i) {
            const double sample = isd_.samples * double(i) / kSamples;
            const glm::dvec2 distorted = FocalPlaneOf(line, sample);
            const glm::dvec2 u = normalised(Undistort(distorted));
            samples.emplace_back(u, normalised(distorted));
            largest = std::max(largest, glm::dot(u, u));
        }
    }
    // distorted = undistorted (a + b1 r^2 + b2 r^4 + b3 r^6) + the
    // tangential terms of t1 and t2: a scale too, as radial models
    // with a constant term have, which the camera's focal length
    // takes (f a), leaving Brown-Conrady's k = b / a and p = t / a on
    // its sensor's coordinates (distorted / a).
    const auto basis = [](glm::dvec2 u, int axis) {
        const double r2 = glm::dot(u, u);
        const double v = u[axis];
        const double cross = 2.0 * u.x * u.y;
        const double own = r2 + 2.0 * v * v;
        return std::array<double, 6>{v,
                                     v * r2,
                                     v * r2 * r2,
                                     v * r2 * r2 * r2,
                                     axis == 0 ? cross : own,
                                     axis == 0 ? own : cross};
    };
    std::array<double, 6> b{};
    if (across) {
        NormalEquations<6> fit;
        for (const auto& [u, d] : samples) {
            for (int axis = 0; axis < 2; ++axis) {
                fit.Add(basis(u, axis), d[axis]);
            }
        }
        b = fit.Solve(largest > 0.0 ? 6 : 0);
    } else {
        // One line, a pushbroom's: along it (x) the scale, the radial
        // terms and t2; t1 moves points along it only as the scale
        // does (2 t1 x y, y about the line's), so it comes from their
        // displacement off it (y), which changes the time that sees
        // them. Alternately, until they settle.
        double t1 = 0.0;
        for (int round = 0; round < 4; ++round) {
            NormalEquations<6> along;
            for (const auto& [u, d] : samples) {
                const std::array<double, 6> row = basis(u, 0);
                along.Add(
                    {row[0], row[1], row[2], row[3], row[5], 0.0},
                    d.x - t1 * row[4], 5);
            }
            const std::array<double, 6> x =
                along.Solve(largest > 0.0 ? 5 : 0);
            b = {x[0], x[1], x[2], x[3], t1, x[4]};
            double moved = 0.0;
            double weight = 0.0;
            for (const auto& [u, d] : samples) {
                const std::array<double, 6> row = basis(u, 1);
                double fitted = 0.0;
                for (std::size_t i = 0; i < 6; ++i) {
                    fitted += i == 4 ? 0.0 : row[i] * b[i];
                }
                moved += row[4] * (d.y - fitted);
                weight += row[4] * row[4];
            }
            t1 = weight > 0.0 ? moved / weight : 0.0;
        }
        b[4] = t1;
    }
    if (b[0] == 0.0) {
        b = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    const double scale = b[0];
    camera.focalLengthMm = f * scale;
    camera.lens.k1 = b[1] / scale;
    camera.lens.k2 = b[2] / scale;
    camera.lens.k3 = b[3] / scale;
    camera.lens.p1 = b[4] / scale;
    camera.lens.p2 = b[5] / scale;
    double residual = 0.0;
    for (const auto& [u, d] : samples) {
        for (int axis = 0; axis < (across ? 2 : 1); ++axis) {
            const std::array<double, 6> row = basis(u, axis);
            double fitted = 0.0;
            for (std::size_t i = 0; i < 6; ++i) {
                fitted += row[i] * b[i];
            }
            residual = std::max(residual, std::abs(fitted - d[axis]) *
                                              (f / pitch));
        }
    }
    return residual;
}

IsdLineScanner::IsdLineScanner(IsdSensor isd)
    : IsdCameraModel(std::move(isd)) {
    if (!isd_.IsLineScanner()) {
        throw std::invalid_argument("The ISD's model '" + isd_.model +
                                    "' is not a line scanner's");
    }
    if (isd_.lineRates.empty()) {
        throw std::invalid_argument(
            "A line scanner's ISD needs line rates");
    }
}

double IsdLineScanner::ImageTime(double line) const {
    std::size_t at = 0;
    while (at + 1 < isd_.lineRates.size() &&
           isd_.lineRates[at + 1][0] <= line) {
        ++at;
    }
    const std::array<double, 3>& rate = isd_.lineRates[at];
    return isd_.centerTime + rate[1] +
           rate[2] * (line - rate[0] + 0.5);
}

// Detector line 0, as usgscsm's line scanner takes it.
glm::dvec3 IsdLineScanner::Look(double line, double sample) const {
    return LookAt(ImageTime(line), 0.0, sample);
}

glm::dvec3 IsdLineScanner::ImageToGround(double line, double sample,
                                         double height) const {
    return EllipsoidIntersect(Position(ImageTime(line)),
                              Look(line, sample), height);
}

glm::dvec2 IsdLineScanner::DetectorView(const glm::dvec3& ground,
                                        double time) const {
    return PixelOf(Distort(FocalPlaneView(ground, time)));
}

glm::dvec2
IsdLineScanner::GroundToImage(const glm::dvec3& ground) const {
    // The line whose exposure puts `ground` on the detector line (an
    // offset of 0 from it).
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

IsdLineScanner::Strip
IsdLineScanner::StripOf(std::uint32_t firstLine,
                        std::uint32_t lines) const {
    if (lines == 0) {
        throw std::invalid_argument(
            "A strip needs at least one line");
    }
    // Which way the rays run: usgscsm's look (-x, -y, -f) meets the
    // ellipsoid at its nearer intersection on either side of the
    // sensor, so ISDs whose boresight is the sensor's +z (as NAIF's
    // instrument frames have it, ALE's real cameras) and those whose
    // is -z both image the same. Ernie's camera must face the ground:
    // `facing` is -1 when it lies behind the look.
    const double middle = ImageTime(firstLine + 0.5 * lines);
    const glm::dvec3 centre =
        ImageToGround(firstLine + 0.5 * lines, 0.5 * isd_.samples);
    const double facing = glm::dot(centre - Position(middle),
                                   Look(firstLine + 0.5 * lines,
                                        0.5 * isd_.samples)) < 0.0
                              ? -1.0
                              : 1.0;
    const glm::dmat3 ernieToSensor = ErnieAxes(0.0, facing);
    const glm::dvec2 first = FocalPlaneOf(0.0, 0.0);
    const glm::dvec2 along =
        FocalPlaneOf(0.0, 1.0) - first;  // mm per image sample
    // The ray of sample 0 in Ernie's axes, per unit of view depth:
    // where the detector line lies across the image (y) and where its
    // principal point is (x).
    const double f = isd_.focalLengthMm;
    const glm::dvec3 ray0 =
        glm::transpose(ernieToSensor) *
        (facing * glm::dvec3(-first.x, -first.y, -f)) / f;
    const double pitch = glm::length(along);
    const double principalSample = -ray0.x * f / pitch;

    Strip strip;
    CameraParams& camera = strip.camera;
    camera.pushbroom = true;
    camera.width = isd_.samples;
    camera.height = lines;
    camera.focalLengthMm = f;
    camera.verticalApertureMm = pitch * lines;
    camera.horizontalApertureOffsetMm =
        (principalSample - 0.5 * isd_.samples) * pitch;
    camera.lineOffsetMm = ray0.y * f;
    // Each line integrates over its line time (the ISD's seconds per
    // line where the strip starts).
    camera.exposureTime =
        ImageTime(firstLine + 1.0) - ImageTime(firstLine);
    // The lens's distortion along the detector line.
    const double detectorLine = 0.0;
    strip.distortionResidualPixels = FitDistortion(
        camera, std::span(&detectorLine, 1), ernieToSensor, facing);
    // Row r's centre is image line first + r + 0.5, exposed (r + 0.5)
    // / lines of the way from `fromTime` to `toTime` (the rate
    // constant over the strip).
    strip.fromTime = ImageTime(firstLine);
    strip.toTime = ImageTime(static_cast<double>(firstLine) + lines);
    strip.from = PoseAt(strip.fromTime, ernieToSensor);
    strip.to = PoseAt(strip.toTime, ernieToSensor);
    // Clipping around the body: from half the height above it to past
    // its far side.
    const double distance = glm::length(glm::dvec3(strip.to[3]));
    camera.nearPlane =
        std::max(0.5 * (distance - isd_.semiMajorAxis), 1e-3);
    camera.farPlane = distance + isd_.semiMajorAxis;
    return strip;
}

IsdPushFrame::IsdPushFrame(IsdSensor isd)
    : IsdCameraModel(std::move(isd)) {
    if (!isd_.IsPushFrame()) {
        throw std::invalid_argument("The ISD's model '" + isd_.model +
                                    "' is not a push-frame camera's");
    }
    if (isd_.frameletHeight == 0 || isd_.interframeDelay <= 0.0) {
        throw std::invalid_argument(
            "A push-frame camera's ISD needs a framelet height and "
            "an interframe delay");
    }
    // usgscsm removes half the overlap from each end of a framelet,
    // so it must be even.
    if (isd_.frameletOverlap % 2 != 0 ||
        isd_.frameletOverlap >= isd_.frameletHeight) {
        throw std::invalid_argument(
            "A push-frame camera's framelet overlap must be even and "
            "less than its height");
    }
    if (FrameletLines() == 0 || Framelets() == 0) {
        throw std::invalid_argument(
            "The push-frame camera's image has no whole framelet");
    }
}

std::uint32_t IsdPushFrame::FrameletLines() const {
    return static_cast<std::uint32_t>(
        double(isd_.frameletHeight - isd_.frameletOverlap) /
        isd_.lineSumming);
}

std::uint32_t IsdPushFrame::Framelets() const {
    // usgscsm's reduced lines: whole framelets of the image, each
    // without its overlap.
    return isd_.lines / isd_.frameletHeight *
           (isd_.frameletHeight - isd_.frameletOverlap) /
           FrameletLines();
}

int IsdPushFrame::TimeOrder(double line) const {
    // Truncated, as usgscsm's `getImageTime` takes it.
    const int framelet = static_cast<int>(line / FrameletLines());
    return isd_.frameletOrderReversed
               ? static_cast<int>(Framelets()) - framelet - 1
               : framelet;
}

double IsdPushFrame::DetectorLine(double line) const {
    const double height = FrameletLines();
    double frameletLine = std::fmod(line, height);
    if (isd_.frameletsFlipped) {
        frameletLine = height - frameletLine;
    }
    return frameletLine + isd_.frameletOverlap / 2;
}

double IsdPushFrame::ImageTime(double line) const {
    return isd_.startingTime + 0.5 * isd_.exposureDuration +
           TimeOrder(line) * isd_.interframeDelay;
}

glm::dvec3 IsdPushFrame::Look(double line, double sample) const {
    return LookAt(ImageTime(line), DetectorLine(line), sample);
}

glm::dvec3 IsdPushFrame::ImageToGround(double line, double sample,
                                       double height) const {
    return EllipsoidIntersect(Position(ImageTime(line)),
                              Look(line, sample), height);
}

double IsdPushFrame::FrameDistance(int framelet,
                                   const glm::dvec3& ground) const {
    const double height = FrameletLines();
    const double time = ImageTime((framelet + 0.5) * height);
    // The undistorted focal plane straight to the detector, as
    // usgscsm takes it.
    const double line = PixelOf(FocalPlaneView(ground, time)).x -
                        isd_.frameletOverlap / 2;
    return line - 0.5 * height;
}

glm::dvec2
IsdPushFrame::GroundToImage(const glm::dvec3& ground) const {
    // The secant method over whole framelets, from the first and the
    // last.
    const int last = static_cast<int>(Framelets()) - 1;
    int start = 0;
    double startDistance = FrameDistance(start, ground);
    int end = last;
    double endDistance = FrameDistance(end, ground);
    for (int i = 0;
         i < 20 && start != end && startDistance != endDistance;
         ++i) {
        const double offset = endDistance * (end - start) /
                              (endDistance - startDistance);
        start = end;
        startDistance = endDistance;
        // usgscsm clips to the framelet count, one past the last;
        // Ernie to the last.
        end = std::clamp(static_cast<int>(std::round(end - offset)),
                         0, last);
        endDistance = FrameDistance(end, ground);
    }
    const double height = FrameletLines();
    const double time = ImageTime(0.5 + height * end);
    const glm::dvec2 pixel =
        PixelOf(Distort(FocalPlaneView(ground, time)));
    return {pixel.x + end * height - isd_.frameletOverlap / 2,
            pixel.y};
}

IsdPushFrame::Framelet
IsdPushFrame::FrameletOf(std::uint32_t index) const {
    if (index >= Framelets()) {
        throw std::invalid_argument(
            "Framelet " + std::to_string(index) +
            " is past the image's " + std::to_string(Framelets()));
    }
    const std::uint32_t height = FrameletLines();
    const double first = double(index) * height;
    const double middle = first + 0.5 * height;
    Framelet framelet;
    framelet.firstLine = index * height;
    framelet.time = ImageTime(middle);
    // Which way the rays run, as `IsdLineScanner::StripOf` finds it.
    const glm::dvec3 centre =
        ImageToGround(middle, 0.5 * isd_.samples);
    const double facing =
        glm::dot(centre - Position(framelet.time),
                 Look(middle, 0.5 * isd_.samples)) < 0.0
            ? -1.0
            : 1.0;
    // The detector line of framelet row coordinate `row` (0 its top
    // edge, `height` its bottom).
    const auto detectorLine = [&](double row) {
        return (isd_.frameletsFlipped ? height - row : row) +
               isd_.frameletOverlap / 2;
    };
    FrameView view =
        FrameViewOf(detectorLine, height, framelet.time, facing);
    framelet.camera = view.camera;
    framelet.rowsReversed = view.rowsReversed;
    framelet.distortionResidualPixels = view.distortionResidualPixels;
    if (isd_.exposureDuration > 0.0) {
        framelet.camera.exposureTime = isd_.exposureDuration;
    }
    return framelet;
}

IsdCameraModel::FrameView IsdCameraModel::FrameViewOf(
    const std::function<double(double)>& detectorLine,
    std::uint32_t rows, double time, double facing) const {
    const double height = rows;
    const glm::dmat3 axes =
        ErnieAxes(detectorLine(0.5 * height), facing);
    const double f = isd_.focalLengthMm;
    const double pitch =
        glm::length(FocalPlaneOf(detectorLine(0.5 * height), 1.0) -
                    FocalPlaneOf(detectorLine(0.5 * height), 0.0));
    const double pixels = f / pitch;  // the focal length in pixels
    // Pixel (row, sample)'s distorted ray in Ernie's normalised image
    // coordinates.
    const auto normalised = [&](double row, double sample) {
        const glm::dvec2 focal =
            FocalPlaneOf(detectorLine(row), sample);
        const glm::dvec3 ray =
            glm::transpose(axes) *
            (facing * glm::dvec3(-focal.x, -focal.y, -f));
        return glm::dvec2(ray.x, ray.y) / -ray.z;
    };
    FrameView view;
    // Ernie's rows run down its y; the frame's may run up it, a
    // mirror no pose undoes.
    const double across = 0.5 * isd_.samples;
    view.rowsReversed = normalised(height - 0.5, across).y >
                        normalised(0.5, across).y;
    const auto rowOf = [&](double row) {
        return view.rowsReversed ? height - row : row;
    };
    // The principal point, from the top left pixel's centre.
    const glm::dvec2 topLeft = normalised(rowOf(0.5), 0.5);
    const glm::dvec2 principal(0.5 - topLeft.x * pixels,
                               0.5 + topLeft.y * pixels);

    CameraParams& camera = view.camera;
    camera.width = isd_.samples;
    camera.height = rows;
    camera.focalLengthMm = f;
    camera.verticalApertureMm = pitch * height;
    camera.horizontalApertureOffsetMm =
        (principal.x - 0.5 * isd_.samples) * pitch;
    camera.verticalApertureOffsetMm =
        (0.5 * height - principal.y) * pitch;
    std::vector<double> lines;
    for (std::uint32_t row = 0; row <= rows; ++row) {
        lines.push_back(detectorLine(row));
    }
    view.distortionResidualPixels =
        FitDistortion(camera, lines, axes, facing);
    // And how far Ernie's pixel grid departs from the detector's
    // (non-square or skewed pixels).
    for (const double row : {0.5, height - 0.5}) {
        for (const double sample : {0.5, isd_.samples - 0.5}) {
            const glm::dvec2 n = normalised(row, sample);
            const glm::dvec2 ernie =
                principal + glm::dvec2(n.x, -n.y) * pixels;
            const glm::dvec2 error =
                glm::abs(ernie - glm::dvec2(sample, rowOf(row)));
            view.distortionResidualPixels = std::max(
                {view.distortionResidualPixels, error.x, error.y});
        }
    }
    camera.transform = PoseAt(time, axes);
    const double distance = glm::length(Position(time));
    camera.nearPlane =
        std::max(0.5 * (distance - isd_.semiMajorAxis), 1e-3);
    camera.farPlane = distance + isd_.semiMajorAxis;
    return view;
}

IsdFrame::IsdFrame(IsdSensor isd) : IsdCameraModel(std::move(isd)) {
    if (!isd_.IsFrame()) {
        throw std::invalid_argument("The ISD's model '" + isd_.model +
                                    "' is not a frame camera's");
    }
    if (!isd_.jitterLineTimes.empty() &&
        isd_.lineJitter.size() != isd_.sampleJitter.size()) {
        throw std::invalid_argument(
            "The ISD's line and sample jitter need as many "
            "coefficients");
    }
}

glm::dvec2 IsdFrame::Jitter(double line) const {
    const std::vector<double>& times = isd_.jitterLineTimes;
    if (times.empty()) {
        return glm::dvec2(0.0);
    }
    // As usgscsm's `removeJitter`: line l's time at index round(l) -
    // 1, within the list.
    const double index =
        std::clamp(std::round(line), 1.0,
                   static_cast<double>(times.size())) -
        1.0;
    const double time = times[static_cast<std::size_t>(index)];
    glm::dvec2 jitter(0.0);
    const std::size_t terms = isd_.lineJitter.size();
    for (std::size_t n = 0; n < terms; ++n) {
        const double power =
            std::pow(time, static_cast<double>(terms - n));
        jitter +=
            glm::dvec2(isd_.lineJitter[n], isd_.sampleJitter[n]) *
            power;
    }
    return jitter;
}

glm::dvec3 IsdFrame::Look(double line, double sample) const {
    const glm::dvec2 unjittered =
        glm::dvec2(line, sample) - Jitter(line);
    // usgscsm's frame camera looks along (x, y, f), against the line
    // scanner's (-x, -y, -f).
    return -LookAt(ImageTime(), unjittered.x, unjittered.y);
}

glm::dvec3 IsdFrame::ImageToGround(double line, double sample,
                                   double height) const {
    return EllipsoidIntersect(Position(ImageTime()),
                              Look(line, sample), height, false);
}

glm::dvec2 IsdFrame::GroundToImage(const glm::dvec3& ground) const {
    const glm::dvec2 unjittered =
        PixelOf(Distort(FocalPlaneView(ground, ImageTime())));
    // usgscsm's `addJitter`: the image point whose jitter removed
    // gives the unjittered one.
    glm::dvec2 image = unjittered;
    for (int iteration = 0;
         iteration < 50 && !isd_.jitterLineTimes.empty();
         ++iteration) {
        const glm::dvec2 error = image - Jitter(image.x) - unjittered;
        if (std::abs(error.x) < 1e-10 && std::abs(error.y) < 1e-10) {
            break;
        }
        image -= error;
    }
    return image;
}

IsdCameraModel::FrameView IsdFrame::View() const {
    const double time = ImageTime();
    const double middle = 0.5 * isd_.lines;
    const glm::dvec3 centre =
        ImageToGround(middle, 0.5 * isd_.samples);
    // `facing` against usgscsm's line-scanner look, (-x, -y, -f), as
    // `ErnieAxes` takes it.
    const double facing =
        glm::dot(centre - Position(time),
                 LookAt(time, middle, 0.5 * isd_.samples)) < 0.0
            ? -1.0
            : 1.0;
    FrameView view = FrameViewOf([](double row) { return row; },
                                 isd_.lines, time, facing);
    if (isd_.jitterLineTimes.empty()) {
        return view;
    }
    // The jitter's range over the image's lines (each line's pixels
    // round to it).
    glm::dvec2 least(std::numeric_limits<double>::max());
    glm::dvec2 most(std::numeric_limits<double>::lowest());
    for (std::uint32_t line = 1; line <= isd_.lines; ++line) {
        least = glm::min(least, Jitter(line));
        most = glm::max(most, Jitter(line));
    }
    // Pixel (line, sample) sees the unjittered camera's (line,
    // sample) - jitter: moving the principal point by the midrange
    // shifts the whole image by it, along the samples (Ernie's x) and
    // the lines (Ernie's rows, down or, reversed, up).
    const glm::dvec2 midrange = 0.5 * (least + most);
    const double pitch =
        view.camera.verticalApertureMm / view.camera.height;
    view.camera.horizontalApertureOffsetMm += midrange.y * pitch;
    view.camera.verticalApertureOffsetMm +=
        (view.rowsReversed ? midrange.x : -midrange.x) * pitch;
    const glm::dvec2 half = 0.5 * (most - least);
    view.distortionResidualPixels += std::max(half.x, half.y);
    return view;
}

IsdSensor ReadIsdFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot read the ISD '" + path +
                                 "'");
    }
    std::ostringstream text;
    text << file.rdbuf();
    return ReadIsd(text.str());
}

}  // namespace ernie
