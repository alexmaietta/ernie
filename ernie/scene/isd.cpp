#include "ernie/scene/isd.h"

#include <nlohmann/json.hpp>

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
        // Scalar last, as ALE writes them.
        rotation.quaternions.values.emplace_back(item[3].get<double>(), item[0].get<double>(),
                                                 item[1].get<double>(), item[2].get<double>());
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
