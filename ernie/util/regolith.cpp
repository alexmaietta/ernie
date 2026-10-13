// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/regolith.h"

#include "ernie/util/profile.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

namespace ernie {

namespace {

constexpr double kStefanBoltzmann = 5.670374419e-8;

double HeatCapacity(const RegolithProperties& regolith, double kelvin) {
    const auto& c = regolith.heatCapacity;
    return c[0] +
           kelvin * (c[1] + kelvin * (c[2] + kelvin * (c[3] +
                                                       kelvin * c[4])));
}

/// Bond albedo at incidence `incidence` (radians).
double Albedo(const RegolithProperties& regolith, double incidence) {
    const double a = incidence / (0.25 * M_PI);
    const double b = incidence / (0.5 * M_PI);
    return regolith.albedo + regolith.albedoA * a * a * a +
           regolith.albedoB * std::pow(b, 8.0);
}

/// The model's nodes: the surface, then deeper ones each 1.2 times
/// further apart, the first a tenth of the surface's diurnal skin
/// depth down, the last ten skin depths of the deep regolith's.
struct Grid {
    std::vector<double> depth;
    std::vector<double> density;
    std::vector<double> contact;  ///< Contact conductivity.
    std::vector<double> volume;   ///< Per unit area, about each node.
};

Grid MakeGrid(const RegolithProperties& regolith) {
    const auto skinDepth = [&](double conductivity, double density) {
        const double diffusivity =
            conductivity / (density * HeatCapacity(regolith, 250.0));
        return std::sqrt(diffusivity * regolith.dayLength / M_PI);
    };
    const double bottom = 10.0 * skinDepth(regolith.deepConductivity,
                                           regolith.deepDensity);
    double step = 0.1 * skinDepth(regolith.surfaceConductivity,
                                  regolith.surfaceDensity);
    Grid grid;
    for (double z = 0.0;; z += step, step *= 1.2) {
        grid.depth.push_back(z);
        if (z >= bottom) {
            break;
        }
    }
    const std::size_t n = grid.depth.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double density =
            regolith.deepDensity -
            (regolith.deepDensity - regolith.surfaceDensity) *
                std::exp(-grid.depth[i] / regolith.densityScale);
        grid.density.push_back(density);
        grid.contact.push_back(
            regolith.deepConductivity -
            (regolith.deepConductivity - regolith.surfaceConductivity) *
                (regolith.deepDensity - density) /
                (regolith.deepDensity - regolith.surfaceDensity));
        const double above = i > 0 ? grid.depth[i] - grid.depth[i - 1] : 0.0;
        const double below =
            i + 1 < n ? grid.depth[i + 1] - grid.depth[i] : 0.0;
        grid.volume.push_back(0.5 * (above + below));
    }
    return grid;
}

/// One backward Euler step of `dt` seconds with `absorbed` W/m^2 of
/// sunlight: conductivity and heat capacity at the step's start, the
/// surface's emission linearised about it.
void Step(const RegolithProperties& regolith, const Grid& grid,
          std::vector<double>& kelvin, double dt, double absorbed) {
    const std::size_t n = kelvin.size();
    std::vector<double> conductance(n, 0.0);  // node i to i + 1
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const auto conductivity = [&](std::size_t j) {
            const double ratio = kelvin[j] / 350.0;
            return grid.contact[j] *
                   (1.0 + regolith.radiativeChi * ratio * ratio * ratio);
        };
        conductance[i] = 0.5 * (conductivity(i) + conductivity(i + 1)) /
                         (grid.depth[i + 1] - grid.depth[i]);
    }
    std::vector<double> lower(n), diagonal(n), upper(n), right(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double capacity = grid.density[i] *
                                HeatCapacity(regolith, kelvin[i]) *
                                grid.volume[i] / dt;
        const double above = i > 0 ? conductance[i - 1] : 0.0;
        lower[i] = -above;
        upper[i] = -conductance[i];
        diagonal[i] = capacity + above + conductance[i];
        right[i] = capacity * kelvin[i];
    }
    const double emitting = regolith.emissivity * kStefanBoltzmann;
    const double t3 = kelvin[0] * kelvin[0] * kelvin[0];
    diagonal[0] += 4.0 * emitting * t3;
    right[0] += absorbed + 3.0 * emitting * t3 * kelvin[0];
    right[n - 1] += regolith.geothermalFlux;
    // Thomas's algorithm.
    for (std::size_t i = 1; i < n; ++i) {
        const double factor = lower[i] / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        right[i] -= factor * right[i - 1];
    }
    kelvin[n - 1] = right[n - 1] / diagonal[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) {
        kelvin[i] = (right[i] - upper[i] * kelvin[i + 1]) / diagonal[i];
    }
}

}  // namespace

std::vector<double>
DiurnalSurfaceTemperatures(const RegolithProperties& regolith,
                           double latitude, int samples) {
    if (samples < 2) {
        throw std::invalid_argument(
            "Diurnal temperatures need at least 2 samples");
    }
    const Grid grid = MakeGrid(regolith);
    const double cosLatitude = std::max(std::cos(latitude), 0.0);
    // Sunlight absorbed at `phase` of the day from local midnight.
    const auto absorbedAt = [&](double phase) {
        const double cosIncidence =
            cosLatitude * std::cos(2.0 * M_PI * phase - M_PI);
        if (cosIncidence <= 0.0) {
            return 0.0;
        }
        // A bright regolith's albedo would pass 1 at grazing sun.
        return (1.0 -
                std::min(Albedo(regolith, std::acos(cosIncidence)),
                         1.0)) *
               regolith.solarIrradiance * cosIncidence;
    };
    // From about the deep regolith's temperature, a hundred days in
    // coarse steps settle the layers below the daily swing, then two
    // in fine steps the swing itself; the last is recorded.
    const double noon =
        std::pow((1.0 - regolith.albedo) * regolith.solarIrradiance *
                     cosLatitude /
                     (regolith.emissivity * kStefanBoltzmann),
                 0.25);
    std::vector<double> kelvin(grid.depth.size(),
                               std::max(0.65 * noon, 40.0));
    const auto run = [&](int days, int stepsPerDay, auto&& record) {
        const double dt = regolith.dayLength / stepsPerDay;
        for (int day = 0; day < days; ++day) {
            for (int s = 0; s < stepsPerDay; ++s) {
                const double phase =
                    static_cast<double>(s + 1) / stepsPerDay;
                Step(regolith, grid, kelvin, dt, absorbedAt(phase));
                record(day, s + 1);
            }
        }
    };
    run(100, 96, [](int, int) {});
    const int stride = (2880 + samples - 1) / samples;
    std::vector<double> surface(static_cast<std::size_t>(samples));
    run(2, stride * samples, [&](int day, int step) {
        if (day == 1 && step % stride == 0) {
            surface[static_cast<std::size_t>((step / stride) % samples)] =
                kelvin[0];
        }
    });
    return surface;
}

double ThermalInertia(const RegolithProperties& regolith) {
    return std::sqrt(regolith.deepConductivity *
                     regolith.deepDensity *
                     HeatCapacity(regolith, 250.0));
}

RegolithProperties WithThermalInertia(RegolithProperties regolith,
                                      double inertia) {
    if (!(inertia > 0.0)) {
        throw std::invalid_argument(
            "A thermal inertia must be above 0, not " +
            std::to_string(inertia));
    }
    const double ratio = inertia / ThermalInertia(regolith);
    regolith.surfaceConductivity *= ratio * ratio;
    regolith.deepConductivity *= ratio * ratio;
    return regolith;
}

std::size_t DiurnalTable::At(int latitude, int time, int inertia,
                             int albedo) const {
    const auto index = [](int value) {
        return static_cast<std::size_t>(value);
    };
    return ((index(latitude) * index(times) + index(time)) *
                inertias.size() +
            index(inertia)) *
               albedos.size() +
           index(albedo);
}

namespace {

/// Whether `values` is not empty and strictly ascending.
bool Ascending(const std::vector<float>& values) {
    return !values.empty() &&
           std::adjacent_find(values.begin(), values.end(),
                              std::greater_equal<>()) == values.end();
}

}  // namespace

void DiurnalTable::Check() const {
    const auto bad = [](const std::string& why) {
        return std::invalid_argument("Diurnal table: " + why);
    };
    if (latitudes < 2 || times < 2) {
        throw bad("needs at least 2 latitudes and times");
    }
    if (!Ascending(inertias) || !Ascending(albedos)) {
        throw bad("inertias and albedos must be given ascending");
    }
    if (!(inertias.front() > 0.0f)) {
        throw bad("inertias must be above 0");
    }
    if (kelvin.size() != At(latitudes, 0)) {
        throw bad("has " + std::to_string(kelvin.size()) +
                  " temperatures for " +
                  std::to_string(At(latitudes, 0)) + " entries");
    }
}

std::vector<float> DiurnalTable::ModelData() const {
    Check();
    std::vector<float> data{kModelDataTag,
                            static_cast<float>(latitudes),
                            static_cast<float>(times),
                            static_cast<float>(inertias.size()),
                            static_cast<float>(albedos.size())};
    data.insert(data.end(), inertias.begin(), inertias.end());
    data.insert(data.end(), albedos.begin(), albedos.end());
    data.insert(data.end(), kelvin.begin(), kelvin.end());
    return data;
}

DiurnalTable
DiurnalTable::FromModelData(const std::vector<float>& data) {
    constexpr std::size_t kHeader = 5;
    if (data.size() < kHeader || data[0] != kModelDataTag) {
        throw std::invalid_argument(
            "Diurnal table data must start with its layout tag (-2), "
            "as DiurnalTable::ModelData writes it");
    }
    const auto count = [&](std::size_t i) {
        const float value = data[i];
        if (!(value >= 0.0f && value < 1e7f) ||
            value != std::floor(value)) {
            throw std::invalid_argument(
                "Diurnal table data has a bad count: " +
                std::to_string(value));
        }
        return static_cast<std::size_t>(value);
    };
    DiurnalTable table;
    table.latitudes = static_cast<int>(count(1));
    table.times = static_cast<int>(count(2));
    const std::size_t inertias = count(3);
    const std::size_t albedos = count(4);
    if (data.size() < kHeader + inertias + albedos) {
        throw std::invalid_argument(
            "Diurnal table data is cut short");
    }
    const auto first = data.begin() + kHeader;
    table.inertias.assign(first, first + static_cast<long>(inertias));
    table.albedos.assign(first + static_cast<long>(inertias),
                         first +
                             static_cast<long>(inertias + albedos));
    table.kelvin.assign(first + static_cast<long>(inertias + albedos),
                        data.end());
    table.Check();
    return table;
}

DiurnalTable DiurnalTemperatureTable(
    const RegolithProperties& regolith, int latitudes, int times,
    std::vector<float> inertias, std::vector<float> albedos) {
    ERNIE_ZONE("Diurnal temperature table");
    DiurnalTable table;
    table.latitudes = latitudes;
    table.times = times;
    table.inertias = inertias.empty()
                         ? std::vector<float>{static_cast<float>(
                               ThermalInertia(regolith))}
                         : std::move(inertias);
    table.albedos =
        albedos.empty()
            ? std::vector<float>{static_cast<float>(regolith.albedo)}
            : std::move(albedos);
    if (!(table.albedos.front() >= 0.0f &&
          table.albedos.back() < 1.0f)) {
        throw std::invalid_argument(
            "Diurnal table: albedos must lie in [0, 1)");
    }
    table.kelvin.assign(
        latitudes >= 2 && times >= 2 ? table.At(latitudes, 0) : 0,
        0.0f);
    table.Check();
    // A row a latitude, inertia and albedo, run across threads.
    const std::size_t inertiaCount = table.inertias.size();
    const std::size_t albedoCount = table.albedos.size();
    const std::size_t rows = static_cast<std::size_t>(latitudes) *
                             inertiaCount * albedoCount;
    std::atomic<std::size_t> next{0};
    const auto work = [&] {
        for (std::size_t r; (r = next.fetch_add(1)) < rows;) {
            const auto albedo = static_cast<int>(r % albedoCount);
            const auto inertia =
                static_cast<int>(r / albedoCount % inertiaCount);
            const auto row =
                static_cast<int>(r / (albedoCount * inertiaCount));
            RegolithProperties properties = WithThermalInertia(
                regolith,
                table.inertias[static_cast<std::size_t>(inertia)]);
            properties.albedo =
                table.albedos[static_cast<std::size_t>(albedo)];
            const std::vector<double> kelvin =
                DiurnalSurfaceTemperatures(
                    properties, 0.5 * M_PI * row / (latitudes - 1),
                    times);
            for (int time = 0; time < times; ++time) {
                table.kelvin[table.At(row, time, inertia, albedo)] =
                    static_cast<float>(
                        kelvin[static_cast<std::size_t>(time)]);
            }
        }
    };
    const std::size_t threads = std::clamp<std::size_t>(
        std::thread::hardware_concurrency() / 2, 1, rows);
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < threads; ++t) {
        pool.emplace_back(work);
    }
    work();
    for (std::thread& thread : pool) {
        thread.join();
    }
    return table;
}

const DiurnalTable& LunarDiurnalTable() {
    static const DiurnalTable table =
        DiurnalTemperatureTable(RegolithProperties{}, 19, 96);
    return table;
}

}  // namespace ernie
