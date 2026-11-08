// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/io/ies.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <string>
#include <vector>

#include "ernie/util/error.h"
#include "ernie/util/text.h"

namespace ernie {
namespace {

constexpr uint32_t kTypeC = 1;

class Numbers {
  public:
    explicit Numbers(std::string_view text) : m_text(text) {}

    double Next() {
        while (m_at < m_text.size() &&
               (std::isspace(static_cast<unsigned char>(m_text[m_at])) || m_text[m_at] == ',')) {
            ++m_at;
        }
        double value = 0;
        const auto [end, error] =
            std::from_chars(m_text.data() + m_at, m_text.data() + m_text.size(), value);
        if (error != std::errc()) {
            throw Error("an IES profile ends early or holds {:.20}",
                        m_text.substr(m_at, m_text.find('\n', m_at) - m_at));
        }
        m_at = size_t(end - m_text.data());
        return value;
    }

    std::vector<double> Take(size_t count) {
        std::vector<double> values(count);
        for (double& value : values) {
            value = Next();
        }
        return values;
    }

  private:
    std::string_view m_text;
    size_t m_at = 0;
};

double Interpolate(const std::vector<double>& angles, const std::vector<double>& values,
                   double angle) {
    if (angles.size() == 1) {
        return values[0];
    }
    if (angle < angles.front() || angle > angles.back()) {
        return 0;
    }
    const auto upper = std::upper_bound(angles.begin(), angles.end(), angle);
    const size_t high = std::min(size_t(upper - angles.begin()), angles.size() - 1);
    const size_t low = high - 1;
    const double span = angles[high] - angles[low];
    const double t = span > 0 ? (angle - angles[low]) / span : 0;
    return values[low] + t * (values[high] - values[low]);
}

double Fold(double horizontal, double first, double last) {
    if (last <= 0) {
        return 0;
    }
    if (first == 90 && last == 270) {
        return horizontal < 90    ? 180 - horizontal
               : horizontal > 270 ? 540 - horizontal
                                  : horizontal;
    }
    if (last <= 90) {
        horizontal = std::fmod(horizontal, 180.0);
        return horizontal > 90 ? 180 - horizontal : horizontal;
    }
    if (last <= 180) {
        return horizontal > 180 ? 360 - horizontal : horizontal;
    }
    return horizontal;
}

}

bool IsIes(const std::filesystem::path& path) {
    return Lowercase(path.extension().string()) == ".ies";
}

Image ReadIes(std::string_view text) {
    const size_t tilt = text.find("TILT=");
    if (tilt == std::string_view::npos) {
        throw Error("an IES profile has no TILT line");
    }
    const size_t line = text.find('\n', tilt);
    const std::string_view mode = text.substr(tilt + 5, line - tilt - 5);
    Numbers numbers(line == std::string_view::npos ? std::string_view() : text.substr(line + 1));
    if (mode.starts_with("INCLUDE")) {
        numbers.Next();
        const size_t pairs = size_t(numbers.Next());
        numbers.Take(2 * pairs);
    }
    numbers.Take(2);
    const double multiplier = numbers.Next();
    const size_t verticalCount = size_t(numbers.Next());
    const size_t horizontalCount = size_t(numbers.Next());
    const uint32_t type = uint32_t(numbers.Next());
    numbers.Take(7);
    if (type != kTypeC) {
        throw Error("an IES profile has photometric type {}, and only type C is supported", type);
    }
    if (verticalCount == 0 || horizontalCount == 0) {
        throw Error("an IES profile has no angles");
    }
    const std::vector<double> vertical = numbers.Take(verticalCount);
    std::vector<double> horizontal = numbers.Take(horizontalCount);
    std::vector<std::vector<double>> candela(horizontalCount);
    double peak = 0;
    for (std::vector<double>& plane : candela) {
        plane = numbers.Take(verticalCount);
        for (double& value : plane) {
            value = std::max(value * multiplier, 0.0);
            peak = std::max(peak, value);
        }
    }

    if (horizontal.front() == 0 && horizontal.back() > 180 && horizontal.back() < 360) {
        horizontal.push_back(360);
        candela.push_back(candela.front());
    }
    Image image(kIesColumns, kIesRows, {0, 0, 0, 1});
    if (peak <= 0) {
        return image;
    }
    std::vector<double> column(verticalCount);
    for (uint32_t x = 0; x < kIesColumns; ++x) {
        const double around =
            Fold(360.0 * x / (kIesColumns - 1), horizontal.front(), horizontal.back());
        for (size_t v = 0; v < verticalCount; ++v) {
            std::vector<double> across(horizontal.size());
            for (size_t h = 0; h < horizontal.size(); ++h) {
                across[h] = candela[h][v];
            }
            column[v] = Interpolate(horizontal, across, around);
        }
        for (uint32_t y = 0; y < kIesRows; ++y) {
            const float value =
                float(Interpolate(vertical, column, 180.0 * y / (kIesRows - 1)) / peak);
            image.Pixel(x, y) = {value, value, value, 1};
        }
    }
    return image;
}

}
