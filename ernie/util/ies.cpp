// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/ies.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

namespace {

/// Index of the interval of increasing `angles` holding `x` and the
/// fraction across it.
std::pair<std::size_t, double>
Bracket(const std::vector<double>& angles, double x) {
    const auto upper =
        std::upper_bound(angles.begin(), angles.end(), x);
    const std::size_t i = std::min<std::size_t>(
        static_cast<std::size_t>(
            std::max<std::ptrdiff_t>(upper - angles.begin() - 1, 0)),
        angles.size() - 2);
    const double span = angles[i + 1] - angles[i];
    return {i, span > 0.0
                   ? std::clamp((x - angles[i]) / span, 0.0, 1.0)
                   : 0.0};
}

bool Near(double a, double b) {
    return std::abs(a - b) < 1e-3;
}

/// Reads blank- or comma-separated numbers.
class Numbers {
  public:
    explicit Numbers(std::string text) {
        std::replace(text.begin(), text.end(), ',', ' ');
        stream_.str(std::move(text));
    }

    double Next() {
        double value = 0.0;
        if (!(stream_ >> value)) {
            throw std::invalid_argument("IES LM-63 file ends early");
        }
        return value;
    }

    std::size_t Count() {
        const double value = Next();
        if (!(value >= 0.0) || value != std::floor(value)) {
            throw std::invalid_argument(
                "IES LM-63 file has a bad count");
        }
        return static_cast<std::size_t>(value);
    }

  private:
    std::istringstream stream_;
};

/// Tilt data: the lamp-to-luminaire geometry, the number of pairs,
/// the angles, the factors.
IesTilt ReadTilt(Numbers& numbers) {
    IesTilt tilt;
    tilt.geometry = static_cast<int>(numbers.Next());
    const std::size_t pairs = numbers.Count();
    tilt.angles.resize(pairs);
    tilt.factors.resize(pairs);
    for (double& angle : tilt.angles) {
        angle = numbers.Next();
    }
    for (double& factor : tilt.factors) {
        factor = numbers.Next();
    }
    if (tilt.geometry < 1 || tilt.geometry > 3 || pairs == 0 ||
        !std::is_sorted(tilt.angles.begin(), tilt.angles.end())) {
        throw std::invalid_argument(
            "IES LM-63 file has bad tilt data");
    }
    return tilt;
}

}  // namespace

double IesTilt::AngleOf(const glm::dvec3& x, const glm::dvec3& y,
                        const glm::dvec3& z) const {
    const auto degrees = [](double radians) {
        return radians * 180.0 / M_PI;
    };
    switch (geometry) {
    case 1:
        return degrees(std::acos(std::clamp(z.z, -1.0, 1.0)));
    case 2:
        return degrees(
            std::asin(std::clamp(std::abs(x.z), 0.0, 1.0)));
    case 3:
        return degrees(
            std::asin(std::clamp(std::abs(y.z), 0.0, 1.0)));
    default:
        return 0.0;
    }
}

double IesTilt::Factor(double degrees) const {
    if (angles.empty()) {
        return 1.0;
    }
    if (angles.back() <= 90.0 + 1e-9 && degrees > 90.0) {
        degrees =
            180.0 - degrees;  // data to 90 degrees mirrors about it
    }
    if (degrees <= angles.front()) {
        return factors.front();
    }
    if (degrees >= angles.back()) {
        return factors.back();
    }
    const auto [i, t] = Bracket(angles, degrees);
    return (1.0 - t) * factors[i] + t * factors[i + 1];
}

double IesProfile::Sample(double theta, double phi) const {
    if (type == IesType::C) {
        return Table(theta, phi);
    }
    // The direction in the luminaire's frame, then the file's angles
    // for it.
    const double t = theta * M_PI / 180.0;
    const double p = phi * M_PI / 180.0;
    const glm::dvec3 e(std::sin(t) * std::cos(p),
                       std::sin(t) * std::sin(p), -std::cos(t));
    const auto degrees = [](double radians) {
        return radians * 180.0 / M_PI;
    };
    if (type == IesType::B) {
        return Table(degrees(std::atan2(e.x, -e.z)),
                     degrees(std::asin(std::clamp(-e.y, -1.0, 1.0))));
    }
    return Table(degrees(std::atan2(e.y, -e.z)),
                 degrees(std::asin(std::clamp(e.x, -1.0, 1.0))));
}

double IesProfile::Table(double v, double h) const {
    if (vertical.empty() || horizontal.empty()) {
        return 0.0;
    }
    if (type != IesType::C) {
        // Types A and B give one side of a symmetric luminaire from 0
        // degrees.
        v = Near(vertical.front(), 0.0) ? std::abs(v) : v;
        h = Near(horizontal.front(), 0.0) ? std::abs(h) : h;
        if (horizontal.size() > 1 && (h < horizontal.front() - 1e-9 ||
                                      h > horizontal.back() + 1e-9)) {
            return 0.0;
        }
    }
    if (v < vertical.front() - 1e-9 || v > vertical.back() + 1e-9) {
        return 0.0;
    }
    const std::size_t columns = vertical.size();
    const auto row = [&](std::size_t plane) {
        if (columns == 1) {
            return candela[plane];
        }
        const auto [i, t] = Bracket(vertical, v);
        return (1.0 - t) * candela[plane * columns + i] +
               t * candela[plane * columns + i + 1];
    };
    if (horizontal.size() == 1) {
        return row(0);  // rotationally symmetric, or one plane
    }
    if (type != IesType::C) {
        const auto [i, t] = Bracket(horizontal, h);
        return (1.0 - t) * row(i) + t * row(i + 1);
    }
    double phi = h;
    // Mirror phi into the planes the file gives.
    phi = std::fmod(phi, 360.0);
    if (phi < 0.0) {
        phi += 360.0;
    }
    const double first = horizontal.front();
    const double last = horizontal.back();
    if (Near(first, 0.0) && Near(last, 90.0)) {
        phi = phi > 180.0 ? 360.0 - phi : phi;  // quadrants
        phi = phi > 90.0 ? 180.0 - phi : phi;
    } else if (Near(first, 0.0) && Near(last, 180.0)) {
        phi = phi > 180.0 ? 360.0 - phi
                          : phi;  // halves about the 0-180 plane
    } else if (Near(first, 90.0) && Near(last, 270.0)) {
        phi = phi < 90.0    ? 180.0 - phi
              : phi > 270.0 ? 540.0 - phi
                            : phi;  // about 90-270
    } else if (phi > last) {
        // Whole: between the last plane and the first, around again.
        const double span = first + 360.0 - last;
        const double t = span > 0.0 ? (phi - last) / span : 0.0;
        return (1.0 - t) * row(horizontal.size() - 1) + t * row(0);
    } else if (phi < first) {
        phi += 360.0;
        const double span = first + 360.0 - last;
        const double t = span > 0.0 ? (phi - last) / span : 0.0;
        return (1.0 - t) * row(horizontal.size() - 1) + t * row(0);
    }
    const auto [plane, t] = Bracket(horizontal, phi);
    return (1.0 - t) * row(plane) + t * row(plane + 1);
}

double IesProfile::Flux() const {
    constexpr int kTheta = 720;
    constexpr int kPhi = 360;
    const double dTheta = 180.0 / kTheta;
    const double dPhi = 360.0 / kPhi;
    double sum = 0.0;
    for (int i = 0; i < kTheta; ++i) {
        const double theta = (i + 0.5) * dTheta;
        double ring = 0.0;
        for (int j = 0; j < kPhi; ++j) {
            ring += Sample(theta, (j + 0.5) * dPhi);
        }
        sum += ring * std::sin(theta * M_PI / 180.0);
    }
    return sum * (dTheta * M_PI / 180.0) * (dPhi * M_PI / 180.0);
}

namespace {

std::string ReadText(const std::filesystem::path& path,
                     std::string_view what) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot read " + std::string(what) +
                                 " '" + path.string() + "'");
    }
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

}  // namespace

IesProfile ParseIes(std::string_view text,
                    const std::filesystem::path& directory) {
    std::istringstream lines{std::string(text)};
    std::string line;
    bool found = false;
    std::string tiltValue;
    while (std::getline(lines, line)) {
        const std::size_t at = line.find("TILT=");
        if (at != std::string::npos) {
            found = true;
            tiltValue = line.substr(at + 5);
            break;
        }
    }
    if (!found) {
        throw std::invalid_argument(
            "Not an IES LM-63 file: no TILT= line");
    }
    const auto blank = [](unsigned char c) {
        return std::isspace(c) != 0;
    };
    tiltValue.erase(
        std::find_if_not(tiltValue.rbegin(), tiltValue.rend(), blank)
            .base(),
        tiltValue.end());
    tiltValue.erase(
        tiltValue.begin(),
        std::find_if_not(tiltValue.begin(), tiltValue.end(), blank));
    // The rest is numbers, separated by blanks, line breaks or
    // commas.
    Numbers numbers(
        std::string((std::istreambuf_iterator<char>(lines)),
                    std::istreambuf_iterator<char>()));
    IesProfile profile;
    if (tiltValue == "INCLUDE") {
        profile.tilt = ReadTilt(numbers);
    } else if (tiltValue != "NONE" && !tiltValue.empty()) {
        Numbers file(
            ReadText(directory / tiltValue, "IES tilt file"));
        profile.tilt = ReadTilt(file);
    }
    numbers.Next();  // number of lamps
    numbers.Next();  // lumens per lamp (-1: absolute photometry)
    const double multiplier = numbers.Next();
    const std::size_t verticals = numbers.Count();
    const std::size_t horizontals = numbers.Count();
    const double type = numbers.Next();
    numbers.Next();  // units
    numbers.Next();  // width
    numbers.Next();  // length
    numbers.Next();  // height
    const double ballast = numbers.Next();
    numbers.Next();  // file generation type (formerly the
                     // ballast-lamp factor)
    numbers.Next();  // input watts
    if (type != 1.0 && type != 2.0 && type != 3.0) {
        throw std::invalid_argument(
            "IES photometric type must be 1 (C), 2 (B) or 3 (A)");
    }
    profile.type = static_cast<IesType>(static_cast<int>(type));
    if (verticals == 0 || horizontals == 0) {
        throw std::invalid_argument("IES LM-63 file has no angles");
    }
    profile.vertical.resize(verticals);
    profile.horizontal.resize(horizontals);
    for (double& angle : profile.vertical) {
        angle = numbers.Next();
    }
    for (double& angle : profile.horizontal) {
        angle = numbers.Next();
    }
    const auto increasing = [](const std::vector<double>& angles) {
        return std::is_sorted(angles.begin(), angles.end()) &&
               std::adjacent_find(angles.begin(), angles.end()) ==
                   angles.end();
    };
    // Type C's vertical angles run from the nadir to the zenith;
    // types A and B's both run from -90 to 90 degrees about the
    // beam's axis.
    const bool c = profile.type == IesType::C;
    const double low = c ? 0.0 : -90.0;
    const double high = c ? 180.0 : 90.0;
    if (!increasing(profile.vertical) ||
        !increasing(profile.horizontal) ||
        profile.vertical.front() < low - 1e-9 ||
        profile.vertical.back() > high + 1e-9 ||
        (!c && (profile.horizontal.front() < -90.0 - 1e-9 ||
                profile.horizontal.back() > 90.0 + 1e-9))) {
        throw std::invalid_argument(
            "IES LM-63 file's angles are out of order or range");
    }
    profile.candela.resize(verticals * horizontals);
    for (double& value : profile.candela) {
        value = std::max(numbers.Next(), 0.0) * multiplier * ballast;
    }
    return profile;
}

IesProfile ReadIes(const std::filesystem::path& path) {
    return ParseIes(ReadText(path, "IES file"), path.parent_path());
}

}  // namespace ernie
