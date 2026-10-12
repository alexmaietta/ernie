// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/dem.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace ernie {

namespace {

constexpr double kPi = std::numbers::pi;
constexpr double kDegree = kPi / 180.0;

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

namespace {

/// `heights` (`lines` rows of `samples`) at fractional line `line`
/// and sample `sample` (cell centres at whole numbers), bilinear,
/// clamped to the grid's edges; 0 for an empty grid.
double Bilinear(const std::vector<float>& heights,
                std::uint32_t lines, std::uint32_t samples,
                double line, double sample) {
    if (lines == 0 || samples == 0) {
        return 0.0;
    }
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
    // The corners weighted, those of no weight left out, so a cell
    // without data (NaN) reaches only the points beside it.
    const double weights[4] = {(1.0 - tx) * (1.0 - ty),
                               tx * (1.0 - ty), (1.0 - tx) * ty,
                               tx * ty};
    const double values[4] = {at(l0, s0), at(l0, s1), at(l1, s0),
                              at(l1, s1)};
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (weights[i] > 0.0) {
            sum += weights[i] * values[i];
        }
    }
    return sum;
}

/// The label at `labelPath`, parsed.
Pds3Label ReadLabel(const std::string& labelPath) {
    std::ifstream labelFile(labelPath);
    if (!labelFile) {
        throw std::runtime_error("Cannot read the PDS3 label '" +
                                 labelPath + "'");
    }
    std::stringstream text;
    text << labelFile.rdbuf();
    return ParsePds3Label(text.str());
}

/// A DEM's grid as its label describes it: its size, its heights
/// (metres above the reference sphere) and that sphere's radius.
struct DemGrid {
    std::uint32_t lines = 0;
    std::uint32_t samples = 0;
    double referenceRadius = 0.0;
    std::vector<float> heights;
};

/// Reads the IMAGE object of `label` (at `labelPath`): 16- or 32-bit
/// LSB integers, or 32-bit LSB reals, each times SCALING_FACTOR, plus
/// the label's OFFSET where the label says the heights are radii, in
/// the file its ^IMAGE pointer names beside the label.
DemGrid ReadGrid(const Pds3Label& label,
                 const std::string& labelPath) {
    DemGrid dem;
    dem.lines =
        static_cast<std::uint32_t>(label.Number("IMAGE.LINES"));
    dem.samples = static_cast<std::uint32_t>(
        label.Number("IMAGE.LINE_SAMPLES"));
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
        // PDS's null for reals, -3.4028227e38, and non-finite values
        // are no data.
        dem.heights[i] =
            real && !(value > -1e30 && std::isfinite(value))
                ? std::numeric_limits<float>::quiet_NaN()
                : static_cast<float>(value * scale + addend);
    }
    return dem;
}

/// Local terrain about a point of a sphere of `radius`, its surface
/// `heightAt(latitude, longitude)` above it (`LocalTerrainOf`).
template <typename HeightAt>
LocalTerrain LocalTerrainOn(double radius, const HeightAt& heightAt,
                            double latitudeDeg, double longitudeDeg,
                            std::uint32_t cells, double cellSize);

}  // namespace

double CylindricalDem::HeightAt(double latitudeDeg,
                                double longitudeDeg) const {
    double east =
        std::fmod(longitudeDeg - westLongitudeDeg, 360.0) + 360.0;
    east = std::fmod(east, 360.0);
    return Bilinear(heights, lines, samples,
                    (maxLatitudeDeg - latitudeDeg) * pixelsPerDegree -
                        0.5,
                    east * pixelsPerDegree *
                            std::cos(standardParallelDeg * kDegree) -
                        0.5);
}

CylindricalDem ReadCylindricalDem(const std::string& labelPath) {
    const Pds3Label label = ReadLabel(labelPath);
    const std::string projection =
        label.Text("IMAGE_MAP_PROJECTION.MAP_PROJECTION_TYPE");
    if (projection != "SIMPLE CYLINDRICAL" &&
        projection != "EQUIRECTANGULAR") {
        throw std::invalid_argument("The DEM's projection '" +
                                    projection +
                                    "' is not simple cylindrical");
    }
    DemGrid grid = ReadGrid(label, labelPath);
    CylindricalDem dem;
    dem.lines = grid.lines;
    dem.samples = grid.samples;
    dem.referenceRadius = grid.referenceRadius;
    dem.heights = std::move(grid.heights);
    dem.pixelsPerDegree =
        label.Number("IMAGE_MAP_PROJECTION.MAP_RESOLUTION");
    dem.maxLatitudeDeg =
        label.Number("IMAGE_MAP_PROJECTION.MAXIMUM_LATITUDE");
    dem.westLongitudeDeg =
        label.Number("IMAGE_MAP_PROJECTION.WESTERNMOST_LONGITUDE");
    dem.standardParallelDeg =
        projection == "EQUIRECTANGULAR" &&
                label.Get("IMAGE_MAP_PROJECTION.CENTER_LATITUDE")
            ? label.Number("IMAGE_MAP_PROJECTION.CENTER_LATITUDE")
            : 0.0;
    return dem;
}

glm::dvec2 PolarStereographicDem::ToGrid(double latitudeDeg,
                                         double longitudeDeg) const {
    const double lat = latitudeDeg * kDegree;
    const double lon = (longitudeDeg - centerLongitudeDeg) * kDegree;
    const bool south = centerLatitudeDeg < 0.0;
    // Snyder 1987, eq. 21-33 and 21-34 for the sphere (k0 = 1, true
    // at the pole): rho = 2 R tan(pi/4 -+ phi/2).
    const double rho =
        2.0 * referenceRadius *
        std::tan(0.25 * kPi + (south ? 0.5 : -0.5) * lat);
    const double x = rho * std::sin(lon);
    const double y = (south ? 1.0 : -1.0) * rho * std::cos(lon);
    return {lineProjectionOffset - y / mapScale,
            sampleProjectionOffset + x / mapScale};
}

double PolarStereographicDem::HeightAt(double latitudeDeg,
                                       double longitudeDeg) const {
    const glm::dvec2 at = ToGrid(latitudeDeg, longitudeDeg);
    return Bilinear(heights, lines, samples, at.x, at.y);
}

PolarStereographicDem
ReadPolarStereographicDem(const std::string& labelPath) {
    const Pds3Label label = ReadLabel(labelPath);
    const std::string projection =
        label.Text("IMAGE_MAP_PROJECTION.MAP_PROJECTION_TYPE");
    if (projection != "POLAR STEREOGRAPHIC") {
        throw std::invalid_argument("The DEM's projection '" +
                                    projection +
                                    "' is not polar stereographic");
    }
    DemGrid grid = ReadGrid(label, labelPath);
    PolarStereographicDem dem;
    dem.lines = grid.lines;
    dem.samples = grid.samples;
    dem.referenceRadius = grid.referenceRadius;
    dem.heights = std::move(grid.heights);
    dem.centerLatitudeDeg =
        label.Number("IMAGE_MAP_PROJECTION.CENTER_LATITUDE");
    if (std::abs(std::abs(dem.centerLatitudeDeg) - 90.0) > 1e-9) {
        throw std::invalid_argument(
            "The DEM's polar stereographic projection is not centred "
            "on a pole");
    }
    dem.centerLongitudeDeg =
        label.Get("IMAGE_MAP_PROJECTION.CENTER_LONGITUDE")
            ? label.Number("IMAGE_MAP_PROJECTION.CENTER_LONGITUDE")
            : 0.0;
    dem.mapScale = label.Number("IMAGE_MAP_PROJECTION.MAP_SCALE");
    if (label.Text("IMAGE_MAP_PROJECTION.MAP_SCALE").find("KM") !=
        std::string::npos) {
        dem.mapScale *= 1e3;
    }
    dem.lineProjectionOffset =
        label.Number("IMAGE_MAP_PROJECTION.LINE_PROJECTION_OFFSET");
    dem.sampleProjectionOffset =
        label.Number("IMAGE_MAP_PROJECTION.SAMPLE_PROJECTION_OFFSET");
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
    return LocalTerrainOn(
        dem.referenceRadius,
        [&dem](double lat, double lon) {
            return dem.HeightAt(lat, lon);
        },
        latitudeDeg, longitudeDeg, cells, cellSize);
}

LocalTerrain LocalTerrainOf(const CylindricalDem& dem,
                            const CylindricalDem& fallback,
                            double latitudeDeg, double longitudeDeg,
                            std::uint32_t cells, double cellSize) {
    return LocalTerrainOn(
        dem.referenceRadius,
        [&dem, &fallback](double lat, double lon) {
            const double h = dem.HeightAt(lat, lon);
            return std::isnan(h) ? fallback.HeightAt(lat, lon) : h;
        },
        latitudeDeg, longitudeDeg, cells, cellSize);
}

LocalTerrain LocalTerrainOf(const PolarStereographicDem& dem,
                            double latitudeDeg, double longitudeDeg,
                            std::uint32_t cells, double cellSize) {
    return LocalTerrainOn(
        dem.referenceRadius,
        [&dem](double lat, double lon) {
            return dem.HeightAt(lat, lon);
        },
        latitudeDeg, longitudeDeg, cells, cellSize);
}

namespace {

template <typename HeightAt>
LocalTerrain LocalTerrainOn(double radius, const HeightAt& heightAt,
                            double latitudeDeg, double longitudeDeg,
                            std::uint32_t cells, double cellSize) {
    if (cells < 2 || !(cellSize > 0.0)) {
        throw std::invalid_argument(
            "A local terrain needs at least 2 cells a positive size "
            "apart");
    }
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
                h = heightAt(lat, along);
                if (std::isnan(h)) {
                    h = 0.0;
                }
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

}  // namespace

}  // namespace ernie
