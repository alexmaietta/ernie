// SPDX-License-Identifier: BSD-3-Clause
/// @file ies.cpp
/// @brief IES LM-63 parsing, sampling and tilt.
#include "ernie/io/ies.h"

#include "ernie/util/file.h"
#include "ernie/util/math.h"
#include "ernie/util/text.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

constexpr double kAngleSlack = 1e-9;

bool Near(double a, double b) {
    return std::abs(a - b) < 1e-3;
}

/// Whether `x` lies within the angles `angles` span.
bool Within(const std::vector<double>& angles, double x) {
    return x >= angles.front() - kAngleSlack && x <= angles.back() + kAngleSlack;
}

/// Whether `angles` strictly increase, within [`low`, `high`].
bool Increasing(const std::vector<double>& angles, double low, double high) {
    return std::adjacent_find(angles.begin(), angles.end(), std::greater_equal<>()) ==
               angles.end() &&
           angles.front() >= low - kAngleSlack && angles.back() <= high + kAngleSlack;
}

/// Type C's horizontal angle `phi` in the planes a file gives: mirrored into a quadrant or a half
/// of a symmetric luminaire; else a whole turn, past the last plane across the gap to the first.
double FoldPlane(const std::vector<double>& planes, double phi) {
    phi = Mod(phi, 360.0);
    const double first = planes.front();
    const double last = planes.back();
    if (Near(first, 0.0) && Near(last, 90.0)) {
        phi = phi > 180.0 ? 360.0 - phi : phi;
        return phi > 90.0 ? 180.0 - phi : phi;
    }
    if (Near(first, 0.0) && Near(last, 180.0)) {
        return phi > 180.0 ? 360.0 - phi : phi;
    }
    if (Near(first, 90.0) && Near(last, 270.0)) {
        return phi < 90.0 ? 180.0 - phi : phi > 270.0 ? 540.0 - phi : phi;
    }
    return phi < first ? phi + 360.0 : phi;
}

constexpr const char* kFormat = "IES LM-63 file";

/// Tilt data: the lamp-to-luminaire geometry, the number of pairs, the angles, the factors.
IesTilt ReadTilt(NumberReader& numbers) {
    IesTilt tilt;
    tilt.geometry = static_cast<int>(numbers.Next());
    const std::size_t pairs = numbers.Count();
    tilt.angles = numbers.Values(pairs);
    tilt.factors = numbers.Values(pairs);
    if (tilt.geometry < 1 || tilt.geometry > 3 || pairs == 0 ||
        !std::is_sorted(tilt.angles.begin(), tilt.angles.end())) {
        throw std::invalid_argument("IES LM-63 file has bad tilt data");
    }
    return tilt;
}

IesType ReadType(double type) {
    if (type != 1.0 && type != 2.0 && type != 3.0) {
        throw std::invalid_argument("IES photometric type must be 1 (C), 2 (B) or 3 (A)");
    }
    return static_cast<IesType>(static_cast<int>(type));
}

}  // namespace

double IesTilt::Angle(const glm::dvec3& x, const glm::dvec3& y, const glm::dvec3& z) const {
    switch (geometry) {
    case 1:
        return glm::degrees(std::acos(std::clamp(z.z, -1.0, 1.0)));
    case 2:
        return glm::degrees(std::asin(std::min(std::abs(x.z), 1.0)));
    case 3:
        return glm::degrees(std::asin(std::min(std::abs(y.z), 1.0)));
    default:
        return 0.0;
    }
}

double IesTilt::Factor(double degrees) const {
    if (angles.empty()) {
        return 1.0;
    }
    // Data to 90 degrees mirrors about it.
    if (angles.back() <= 90.0 + kAngleSlack && degrees > 90.0) {
        degrees = 180.0 - degrees;
    }
    return Interpolate<double>(angles, factors, degrees);
}

double IesProfile::Sample(double theta, double phi) const {
    if (type == IesType::C) {
        return Table(theta, phi);
    }
    // The direction in the luminaire's frame, then the file's angles for it.
    const double t = glm::radians(theta);
    const double p = glm::radians(phi);
    const glm::dvec3 e(std::sin(t) * std::cos(p), std::sin(t) * std::sin(p), -std::cos(t));
    if (type == IesType::B) {
        return Table(glm::degrees(std::atan2(e.x, -e.z)),
                     glm::degrees(std::asin(std::clamp(-e.y, -1.0, 1.0))));
    }
    return Table(glm::degrees(std::atan2(e.y, -e.z)),
                 glm::degrees(std::asin(std::clamp(e.x, -1.0, 1.0))));
}

double IesProfile::Table(double v, double h) const {
    if (vertical.empty() || horizontal.empty()) {
        return 0.0;
    }
    if (type != IesType::C) {
        // Types A and B give one side of a symmetric luminaire from 0 degrees.
        v = Near(vertical.front(), 0.0) ? std::abs(v) : v;
        h = Near(horizontal.front(), 0.0) ? std::abs(h) : h;
        if (horizontal.size() > 1 && !Within(horizontal, h)) {
            return 0.0;
        }
    }
    if (!Within(vertical, v)) {
        return 0.0;
    }
    const std::size_t columns = vertical.size();
    const auto row = [&](std::size_t plane) {
        if (columns == 1) {
            return candela[plane];
        }
        const auto [i, t] = Bracket(vertical, v);
        return std::lerp(candela[plane * columns + i], candela[plane * columns + i + 1], t);
    };
    // One plane: rotationally symmetric, or a single plane of data.
    if (horizontal.size() == 1) {
        return row(0);
    }
    const double phi = type == IesType::C ? FoldPlane(horizontal, h) : h;
    const double last = horizontal.back();
    if (phi > last) {
        const double gap = horizontal.front() + 360.0 - last;
        return std::lerp(row(horizontal.size() - 1), row(0), gap > 0.0 ? (phi - last) / gap : 0.0);
    }
    const auto [plane, t] = Bracket(horizontal, phi);
    return std::lerp(row(plane), row(plane + 1), t);
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
        sum += ring * std::sin(glm::radians(theta));
    }
    return sum * glm::radians(dTheta) * glm::radians(dPhi);
}

IesProfile ParseIes(std::string_view text, const std::filesystem::path& directory) {
    // Keywords up to TILT=; numbers after its line, separated by blanks, line breaks or commas.
    const std::size_t tiltAt = text.find("TILT=");
    if (tiltAt == std::string_view::npos) {
        throw std::invalid_argument("Not an IES LM-63 file: no TILT= line");
    }
    const std::size_t lineEnd = std::min(text.find('\n', tiltAt), text.size());
    const std::string tilt = Trim(text.substr(tiltAt + 5, lineEnd - tiltAt - 5));
    NumberReader numbers(text.substr(lineEnd), kFormat);

    IesProfile profile;
    if (tilt == "INCLUDE") {
        profile.tilt = ReadTilt(numbers);
    } else if (tilt != "NONE" && !tilt.empty()) {
        NumberReader file(ReadTextFile(directory / tilt), "IES tilt file");
        profile.tilt = ReadTilt(file);
    }

    numbers.Skip(2);  // lamps, lumens per lamp (-1: absolute photometry)
    const double multiplier = numbers.Next();
    const std::size_t verticals = numbers.Count();
    const std::size_t horizontals = numbers.Count();
    profile.type = ReadType(numbers.Next());
    numbers.Skip(4);  // units, width, length, height
    const double ballast = numbers.Next();
    numbers.Skip(2);  // file generation type (formerly the ballast-lamp factor), input watts
    if (verticals == 0 || horizontals == 0) {
        throw std::invalid_argument("IES LM-63 file has no angles");
    }
    profile.vertical = numbers.Values(verticals);
    profile.horizontal = numbers.Values(horizontals);

    // Type C's vertical angles run from the nadir to the zenith; types A and B's both run from -90
    // to 90 degrees about the beam's axis.
    constexpr double kAny = std::numeric_limits<double>::infinity();
    const bool c = profile.type == IesType::C;
    if (!Increasing(profile.vertical, c ? 0.0 : -90.0, c ? 180.0 : 90.0) ||
        !Increasing(profile.horizontal, c ? -kAny : -90.0, c ? kAny : 90.0)) {
        throw std::invalid_argument("IES LM-63 file's angles are out of order or range");
    }

    profile.candela = numbers.Values(verticals * horizontals);
    for (double& value : profile.candela) {
        value = std::max(value, 0.0) * multiplier * ballast;
    }
    return profile;
}

IesProfile ReadIes(const std::filesystem::path& path) {
    return ParseIes(ReadTextFile(path), path.parent_path());
}

}  // namespace ernie
