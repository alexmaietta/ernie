// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/dem.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace ernie {

namespace {

constexpr double kDegree = std::numbers::pi / 180.0;

std::string Trim(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end &&
           std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

// A value's text without its units (`<km>`) and outer quotes.
std::string Clean(std::string value) {
    if (const std::size_t unit = value.find('<');
        unit != std::string::npos && value.front() != '"' &&
        value.front() != '(') {
        value = Trim(value.substr(0, unit));
    }
    if (value.size() >= 2 &&
        (value.front() == '"' || value.front() == '\'') &&
        value.back() == value.front()) {
        value = value.substr(1, value.size() - 2);
    }
    return Trim(value);
}

// Whether `value` still opens a quote or a parenthesis it does not
// close.
bool Open(const std::string& value) {
    int depth = 0;
    bool quoted = false;
    for (const char c : value) {
        if (c == '"') {
            quoted = !quoted;
        } else if (!quoted && c == '(') {
            ++depth;
        } else if (!quoted && c == ')') {
            --depth;
        }
    }
    return quoted || depth > 0;
}

}  // namespace

std::optional<std::string>
Pds3Label::Get(std::string_view key) const {
    const auto it = values.find(key);
    if (it == values.end()) {
        return std::nullopt;
    }
    return it->second;
}

double Pds3Label::Number(std::string_view key) const {
    const std::string text = Text(key);
    try {
        std::size_t used = 0;
        const double value = std::stod(text, &used);
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("PDS3 keyword " +
                                    std::string(key) + " = '" + text +
                                    "' is not a number");
    }
}

std::string Pds3Label::Text(std::string_view key) const {
    if (const std::optional<std::string> value = Get(key)) {
        return *value;
    }
    throw std::invalid_argument("The PDS3 label has no " +
                                std::string(key));
}

Pds3Label ParsePds3Label(std::string_view text) {
    // Comments out first; they may span lines.
    std::string plain;
    for (std::size_t i = 0; i < text.size();) {
        if (text.compare(i, 2, "/*") == 0) {
            const std::size_t end = text.find("*/", i + 2);
            i = end == std::string_view::npos ? text.size() : end + 2;
        } else {
            plain.push_back(text[i++]);
        }
    }
    Pds3Label label;
    std::vector<std::string> objects;
    std::istringstream lines(plain);
    std::string line;
    while (std::getline(lines, line)) {
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) {
            if (Trim(line) == "END") {
                break;
            }
            continue;
        }
        const std::string key = Trim(line.substr(0, equals));
        std::string value = Trim(line.substr(equals + 1));
        while (Open(value) && std::getline(lines, line)) {
            value += " " + Trim(line);
        }
        if (key == "OBJECT") {
            objects.push_back(Clean(value));
            continue;
        }
        if (key == "END_OBJECT") {
            if (!objects.empty()) {
                objects.pop_back();
            }
            continue;
        }
        std::string name;
        for (const std::string& object : objects) {
            name += object + ".";
        }
        label.values[name + key] = Clean(value);
    }
    return label;
}

double CylindricalDem::HeightAt(double latitudeDeg,
                                double longitudeDeg) const {
    if (lines == 0 || samples == 0) {
        return 0.0;
    }
    double east =
        std::fmod(longitudeDeg - westLongitudeDeg, 360.0) + 360.0;
    east = std::fmod(east, 360.0);
    const double line =
        (maxLatitudeDeg - latitudeDeg) * pixelsPerDegree - 0.5;
    const double sample = east * pixelsPerDegree - 0.5;
    const double y = std::clamp(line, 0.0, double(lines - 1));
    const double x = std::clamp(sample, 0.0, double(samples - 1));
    const auto l0 = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(y), lines > 1 ? lines - 2 : 0);
    const auto s0 = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(x), samples > 1 ? samples - 2 : 0);
    const std::uint32_t l1 = std::min(l0 + 1, lines - 1);
    const std::uint32_t s1 = std::min(s0 + 1, samples - 1);
    const double ty = y - l0;
    const double tx = x - s0;
    const auto at = [&](std::uint32_t l, std::uint32_t s) {
        return static_cast<double>(
            heights[std::size_t(l) * samples + s]);
    };
    const double top = at(l0, s0) + tx * (at(l0, s1) - at(l0, s0));
    const double bottom = at(l1, s0) + tx * (at(l1, s1) - at(l1, s0));
    return top + ty * (bottom - top);
}

CylindricalDem ReadCylindricalDem(const std::string& labelPath) {
    std::ifstream labelFile(labelPath);
    if (!labelFile) {
        throw std::runtime_error("Cannot read the PDS3 label '" +
                                 labelPath + "'");
    }
    std::stringstream text;
    text << labelFile.rdbuf();
    const Pds3Label label = ParsePds3Label(text.str());

    const std::string projection =
        label.Text("IMAGE_MAP_PROJECTION.MAP_PROJECTION_TYPE");
    if (projection != "SIMPLE CYLINDRICAL" &&
        projection != "EQUIRECTANGULAR") {
        throw std::invalid_argument("The DEM's projection '" +
                                    projection +
                                    "' is not simple cylindrical");
    }
    CylindricalDem dem;
    dem.lines =
        static_cast<std::uint32_t>(label.Number("IMAGE.LINES"));
    dem.samples = static_cast<std::uint32_t>(
        label.Number("IMAGE.LINE_SAMPLES"));
    dem.pixelsPerDegree =
        label.Number("IMAGE_MAP_PROJECTION.MAP_RESOLUTION");
    dem.maxLatitudeDeg =
        label.Number("IMAGE_MAP_PROJECTION.MAXIMUM_LATITUDE");
    dem.westLongitudeDeg =
        label.Number("IMAGE_MAP_PROJECTION.WESTERNMOST_LONGITUDE");
    const double scale = label.Get("IMAGE.SCALING_FACTOR")
                             ? label.Number("IMAGE.SCALING_FACTOR")
                             : 1.0;
    const double offset = label.Get("IMAGE.OFFSET")
                              ? label.Number("IMAGE.OFFSET")
                              : 0.0;
    // LOLA's OFFSET is the reference sphere's radius, the heights
    // above it; the projection's A axis is the same sphere in km.
    dem.referenceRadius =
        offset > 1e5
            ? offset
            : 1e3 *
                  label.Number("IMAGE_MAP_PROJECTION.A_AXIS_RADIUS");
    const double addend = offset > 1e5 ? 0.0 : offset;

    const std::string type = label.Text("IMAGE.SAMPLE_TYPE");
    const int bits =
        static_cast<int>(label.Number("IMAGE.SAMPLE_BITS"));
    const bool real = type.find("REAL") != std::string::npos;
    if (type.rfind("LSB", 0) != 0 && type != "PC_REAL" &&
        type != "PC_INTEGER") {
        throw std::invalid_argument("The DEM's samples are " + type +
                                    ", not least significant byte "
                                    "first");
    }
    if (!(real ? bits == 32 : bits == 16 || bits == 32)) {
        throw std::invalid_argument("The DEM's samples are " +
                                    std::to_string(bits) + "-bit " +
                                    type);
    }

    // ^IMAGE names the data file beside the label, possibly with a
    // starting record: ("FILE.IMG", 2) or 2 (in this file).
    std::string pointer = label.Text("^IMAGE");
    std::filesystem::path data = labelPath;
    std::uint64_t start = 0;
    const std::uint64_t recordBytes =
        label.Get("RECORD_BYTES")
            ? static_cast<std::uint64_t>(label.Number("RECORD_BYTES"))
            : 0;
    if (!pointer.empty() && pointer.front() == '(') {
        pointer = Trim(pointer.substr(1, pointer.size() - 2));
        const std::size_t comma = pointer.find(',');
        std::string file = Clean(Trim(pointer.substr(0, comma)));
        if (comma != std::string::npos) {
            start =
                (std::stoull(Trim(pointer.substr(comma + 1))) - 1) *
                recordBytes;
        }
        data = std::filesystem::path(labelPath).parent_path() / file;
    } else if (!pointer.empty() &&
               std::isdigit(static_cast<unsigned char>(pointer[0]))) {
        start = (std::stoull(pointer) - 1) * recordBytes;
    } else {
        data = std::filesystem::path(labelPath).parent_path() /
               Clean(pointer);
    }
    // Labels name files in upper case; archives keep them in either.
    if (!std::filesystem::exists(data)) {
        std::string lower = data.filename().string();
        std::transform(
            lower.begin(), lower.end(), lower.begin(),
            [](unsigned char c) { return std::tolower(c); });
        data = data.parent_path() / lower;
    }
    std::ifstream file(data, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot read the DEM's data '" +
                                 data.string() + "'");
    }
    const std::size_t count = std::size_t(dem.lines) * dem.samples;
    const std::size_t bytes =
        count * static_cast<std::size_t>(bits / 8);
    std::vector<char> raw(bytes);
    file.seekg(static_cast<std::streamoff>(start));
    file.read(raw.data(), static_cast<std::streamsize>(bytes));
    if (static_cast<std::size_t>(file.gcount()) != bytes) {
        throw std::runtime_error("The DEM's data '" + data.string() +
                                 "' is shorter than its label says");
    }
    // Little-endian samples, as x86 and ARM hosts hold them.
    dem.heights.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (real) {
            float f = 0.0f;
            std::memcpy(&f, raw.data() + 4 * i, 4);
            value = f;
        } else if (bits == 16) {
            std::int16_t v = 0;
            std::memcpy(&v, raw.data() + 2 * i, 2);
            value = v;
        } else {
            std::int32_t v = 0;
            std::memcpy(&v, raw.data() + 4 * i, 4);
            value = v;
        }
        dem.heights[i] = static_cast<float>(value * scale + addend);
    }
    return dem;
}

glm::dvec3 BodyDirection(double latitudeDeg, double longitudeDeg) {
    const double lat = latitudeDeg * kDegree;
    const double lon = longitudeDeg * kDegree;
    return {std::cos(lat) * std::cos(lon),
            std::cos(lat) * std::sin(lon), std::sin(lat)};
}

LocalTerrain LocalTerrainOf(const CylindricalDem& dem,
                            double latitudeDeg, double longitudeDeg,
                            std::uint32_t cells, double cellSize) {
    if (cells < 2 || !(cellSize > 0.0)) {
        throw std::invalid_argument(
            "A local terrain needs at least 2 cells a positive size "
            "apart");
    }
    const double radius = dem.referenceRadius;
    // The local frame: up along the point's direction, east and north
    // along its parallel and meridian.
    const glm::dvec3 up = BodyDirection(latitudeDeg, longitudeDeg);
    const double lon = longitudeDeg * kDegree;
    const glm::dvec3 east(-std::sin(lon), std::cos(lon), 0.0);
    const glm::dvec3 north = glm::cross(up, east);
    const glm::dvec3 centre = radius * up;
    LocalTerrain terrain;
    glm::dmat4 localToBody(1.0);
    localToBody[0] = glm::dvec4(east, 0.0);
    localToBody[1] = glm::dvec4(north, 0.0);
    localToBody[2] = glm::dvec4(up, 0.0);
    localToBody[3] = glm::dvec4(centre, 1.0);
    terrain.bodyToLocal = glm::inverse(localToBody);
    terrain.cellSize = cellSize;
    const double half = 0.5 * (cells - 1) * cellSize;
    terrain.origin = glm::dvec2(-half, half);

    std::vector<float> heights(std::size_t(cells) * cells);
    for (std::uint32_t row = 0; row < cells; ++row) {
        for (std::uint32_t col = 0; col < cells; ++col) {
            const double x = -half + col * cellSize;
            const double y = half - row * cellSize;
            // The surface point whose local x and y are these: on the
            // sphere of radius R + h through them, h the DEM's height
            // where it lands, settled in a few rounds.
            double h = 0.0;
            double z = 0.0;
            for (int i = 0; i < 4; ++i) {
                const double r = radius + h;
                const double inside = r * r - x * x - y * y;
                z = std::sqrt(std::max(inside, 0.0)) - radius;
                const glm::dvec3 point =
                    centre + x * east + y * north + z * up;
                const glm::dvec3 d = glm::normalize(point);
                const double lat =
                    std::asin(std::clamp(d.z, -1.0, 1.0)) / kDegree;
                const double along = std::atan2(d.y, d.x) / kDegree;
                h = dem.HeightAt(lat, along);
            }
            const double r = radius + h;
            z = std::sqrt(std::max(r * r - x * x - y * y, 0.0)) -
                radius;
            heights[std::size_t(row) * cells + col] =
                static_cast<float>(z);
        }
    }
    terrain.heights = Tensor::Copy<float>(
        heights, {std::int64_t(cells), std::int64_t(cells)});
    return terrain;
}

}  // namespace ernie
