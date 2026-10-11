// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/ies.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

namespace {

/// Index of the interval of increasing `angles` holding `x` and the fraction across it.
std::pair<std::size_t, double> Bracket(const std::vector<double>& angles, double x) {
    const auto upper = std::upper_bound(angles.begin(), angles.end(), x);
    const std::size_t i = std::min<std::size_t>(
        static_cast<std::size_t>(std::max<std::ptrdiff_t>(upper - angles.begin() - 1, 0)),
        angles.size() - 2);
    const double span = angles[i + 1] - angles[i];
    return {i, span > 0.0 ? std::clamp((x - angles[i]) / span, 0.0, 1.0) : 0.0};
}

bool Near(double a, double b) {
    return std::abs(a - b) < 1e-3;
}

}  // namespace

double IesProfile::Sample(double theta, double phi) const {
    if (vertical.empty() || horizontal.empty() || theta < vertical.front() - 1e-9 ||
        theta > vertical.back() + 1e-9) {
        return 0.0;
    }
    const std::size_t columns = vertical.size();
    const auto row = [&](std::size_t h) {
        if (columns == 1) {
            return candela[h];
        }
        const auto [i, t] = Bracket(vertical, theta);
        return (1.0 - t) * candela[h * columns + i] + t * candela[h * columns + i + 1];
    };
    if (horizontal.size() == 1) {
        return row(0);  // rotationally symmetric
    }
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
        phi = phi > 180.0 ? 360.0 - phi : phi;  // halves about the 0-180 plane
    } else if (Near(first, 90.0) && Near(last, 270.0)) {
        phi = phi < 90.0 ? 180.0 - phi : phi > 270.0 ? 540.0 - phi : phi;  // about 90-270
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
    const auto [h, t] = Bracket(horizontal, phi);
    return (1.0 - t) * row(h) + t * row(h + 1);
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

IesProfile ParseIes(std::string_view text) {
    std::istringstream lines{std::string(text)};
    std::string line;
    bool tilt = false;
    std::string tiltValue;
    while (std::getline(lines, line)) {
        const std::size_t at = line.find("TILT=");
        if (at != std::string::npos) {
            tilt = true;
            tiltValue = line.substr(at + 5);
            break;
        }
    }
    if (!tilt) {
        throw std::invalid_argument("Not an IES LM-63 file: no TILT= line");
    }
    // The rest is numbers, separated by blanks, line breaks or commas.
    std::string rest((std::istreambuf_iterator<char>(lines)), std::istreambuf_iterator<char>());
    std::replace(rest.begin(), rest.end(), ',', ' ');
    std::istringstream numbers(rest);
    const auto next = [&]() {
        double value = 0.0;
        if (!(numbers >> value)) {
            throw std::invalid_argument("IES LM-63 file ends early");
        }
        return value;
    };
    const auto count = [&]() {
        const double value = next();
        if (!(value >= 0.0) || value != std::floor(value)) {
            throw std::invalid_argument("IES LM-63 file has a bad count");
        }
        return static_cast<std::size_t>(value);
    };
    if (tiltValue.rfind("INCLUDE", 0) == 0) {
        next();  // lamp-to-luminaire geometry
        const std::size_t pairs = count();
        for (std::size_t i = 0; i < 2 * pairs; ++i) {
            next();  // angles, then multiplying factors
        }
    }
    next();  // number of lamps
    next();  // lumens per lamp (-1: absolute photometry)
    const double multiplier = next();
    const std::size_t verticals = count();
    const std::size_t horizontals = count();
    const double type = next();
    next();  // units
    next();  // width
    next();  // length
    next();  // height
    const double ballast = next();
    next();  // file generation type (formerly the ballast-lamp factor)
    next();  // input watts
    if (type != 1.0) {
        throw std::invalid_argument("Only type C IES photometry is read");
    }
    if (verticals == 0 || horizontals == 0) {
        throw std::invalid_argument("IES LM-63 file has no angles");
    }
    IesProfile profile;
    profile.vertical.resize(verticals);
    profile.horizontal.resize(horizontals);
    for (double& angle : profile.vertical) {
        angle = next();
    }
    for (double& angle : profile.horizontal) {
        angle = next();
    }
    const auto increasing = [](const std::vector<double>& angles) {
        return std::is_sorted(angles.begin(), angles.end()) &&
               std::adjacent_find(angles.begin(), angles.end()) == angles.end();
    };
    if (!increasing(profile.vertical) || !increasing(profile.horizontal) ||
        profile.vertical.front() < 0.0 || profile.vertical.back() > 180.0) {
        throw std::invalid_argument("IES LM-63 file's angles are out of order or range");
    }
    profile.candela.resize(verticals * horizontals);
    for (double& value : profile.candela) {
        value = std::max(next(), 0.0) * multiplier * ballast;
    }
    return profile;
}

IesProfile ReadIes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot read IES file '" + path.string() + "'");
    }
    std::ostringstream text;
    text << file.rdbuf();
    return ParseIes(text.str());
}

}  // namespace ernie
