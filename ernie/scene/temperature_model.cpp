// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/temperature_model.h"

#include "ernie/util/regolith.h"
#include "ernie/util/solar.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

bool IsIdentifier(std::string_view text) {
    return !text.empty() &&
           std::isalpha(static_cast<unsigned char>(text[0])) != 0 &&
           std::all_of(text.begin(), text.end(), [](char ch) {
               return std::isalnum(static_cast<unsigned char>(ch)) !=
                          0 ||
                      ch == '_';
           });
}

constexpr const char* kTemperatureModelSlang = R"(
// A surface's temperature computed as it renders (ernie/scene/temperature_model.h): what a model
// is evaluated at, its parameters with the material's data, and the interface it implements.

// Where a model is evaluated: the point and its outward surface normal (world, relative to the
// frame's origin); the brightest distant light's direction towards it (0 for none), its
// irradiance over all wavelengths facing it (W/m^2) and the share of it reaching the point past
// shadows; the sunlight on the surface from every distant light (W/m^2, shadowed); the material's
// solar absorptivity and hemispherical thermal emissivity; its own temperature (K, 0 for none);
// and the frame's time (s).
public struct TemperatureInput {
    public float3 position;
    public float3 normal;
    public float3 sunDirection;
    public float sunIrradiance;
    public float sunVisibility;
    public float irradiance;
    public float absorptivity;
    public float emissivity;
    public float kelvin;
    public float time;
}

// A material's parameters for its model, and its data.
public struct TemperatureParams {
    public float values[16];
    public float* table;
    public uint count;

    public float get(uint i) {
        return i < 16 ? values[i] : 0.0;
    }
    public float data(uint i) {
        return i < count ? table[i] : 0.0;
    }
    public uint dataCount() {
        return count;
    }
}

public interface ITemperatureModel {
    // The temperature at `input`, K.
    public float kelvin(TemperatureInput input, TemperatureParams p);
}

public static const float kStefanBoltzmann = 5.670374419e-8;
)";

constexpr const char* kBalanceSlang = R"(
import temperature_model;

// Zero-capacitance energy balance: absorbed sunlight and internal heat radiated away at once,
// eps sigma faces (T^4 - Tsink^4) = alpha E + q.
public struct BalanceTemperature : ITemperatureModel {
    public float kelvin(TemperatureInput input, TemperatureParams p) {
        let absorptivity = p.get(0) >= 0.0 ? p.get(0) : input.absorptivity;
        let emissivity = max(p.get(1) >= 0.0 ? p.get(1) : input.emissivity, 1e-3);
        let sink = max(p.get(2), 0.0);
        let faces = max(p.get(4), 1.0);
        let absorbed = max(absorptivity * input.irradiance + p.get(3), 0.0);
        let t4 = absorbed / (faces * emissivity * kStefanBoltzmann) + sink * sink * sink * sink;
        return sqrt(sqrt(t4));
    }
}
)";

constexpr const char* kDiurnalSlang = R"(
import temperature_model;

// A table of latitude (rows, 0 to 90 degrees) and local solar time (columns from midnight), after
// its two sizes, read at the surface normal's latitude and hour angle about the body's pole.
public struct DiurnalTemperature : ITemperatureModel {
    public float kelvin(TemperatureInput input, TemperatureParams p) {
        let rows = uint(p.data(0));
        let columns = uint(p.data(1));
        if (rows < 2 || columns < 2 || p.dataCount() < 2 + rows * columns) {
            return input.kelvin;
        }
        var pole = float3(p.get(0), p.get(1), p.get(2));
        pole = dot(pole, pole) > 0.0 ? normalize(pole) : float3(0.0, 0.0, 1.0);
        let n = input.normal;
        let latitude = asin(saturate(abs(dot(n, pole))));
        // The sun's hour angle, westward from the meridian: the body turns towards its east,
        // pole x n, so the sun moves west and the afternoon has it there.
        let up = n - dot(n, pole) * pole;
        let east = cross(pole, n);
        let sun = input.sunDirection - dot(input.sunDirection, pole) * pole;
        let hour = atan2(-dot(sun, east), dot(sun, up));
        let phase = frac(hour / (2.0 * 3.14159265) + 0.5);
        let row = latitude / (0.5 * 3.14159265) * float(rows - 1);
        let column = phase * float(columns);
        let r0 = min(uint(row), rows - 2);
        let c0 = uint(column) % columns;
        let c1 = (c0 + 1) % columns;
        let fr = saturate(row - float(r0));
        let fc = frac(column);
        let first = 2 + r0 * columns;
        let next = first + columns;
        let low = lerp(p.data(first + c0), p.data(first + c1), fc);
        let high = lerp(p.data(next + c0), p.data(next + c1), fc);
        return lerp(low, high, fr);
    }
}
)";

/// The built-in models' ids.
bool IsBuiltinId(std::string_view id) {
    return id == "balance" || id == "diurnal";
}

}  // namespace

std::vector<float> TemperatureModel::Defaults() const {
    std::vector<float> values;
    for (const TemperatureModelParam& param : params) {
        values.push_back(param.defaultValue);
    }
    return values;
}

TemperatureModelRegistry TemperatureModelRegistry::WithBuiltins() {
    TemperatureModelRegistry registry;
    TemperatureModel balance;
    balance.id = "balance";
    balance.name = "Zero-capacitance balance";
    balance.description =
        "Each point at once at the temperature where the sunlight it "
        "absorbs and its internal heat equal what it radiates to its "
        "surroundings: a spacecraft's thin panels.";
    balance.references = {
        "D. G. Gilmore (ed.), \"Spacecraft Thermal Control Handbook\", "
        "2nd ed., The Aerospace Press, 2002, chapter 2.",
        "M. Dunstan et al., \"PANGU: Planet and Asteroid Natural Scene "
        "Generation Utility\", University of Dundee."};
    balance.params = {
        {"absorptivity", -1.0f,
         "Solar absorptivity; below 0, the material's (1 minus its "
         "sunlight-weighted reflectance)."},
        {"emissivity", -1.0f,
         "Hemispherical emissivity; below 0, the material's thermal "
         "emissivity weighted by Planck's law at 300 K."},
        {"sink", 3.0f, "The surroundings' temperature, K."},
        {"internal", 0.0f, "Internal heat per unit area, W/m^2."},
        {"faces", 1.0f,
         "Faces radiating: 1, or 2 for a panel open behind."}};
    balance.typeName = "BalanceTemperature";
    balance.slang = kBalanceSlang;
    registry.models_.push_back(std::move(balance));

    TemperatureModel diurnal;
    diurnal.id = "diurnal";
    diurnal.name = "Diurnal table";
    diurnal.description =
        "The temperature of a table of latitude and local solar time, "
        "baked offline by a regolith model whose thermal inertia keeps "
        "the night warm and lags the afternoon: the Moon's by default.";
    diurnal.references = {
        "P. O. Hayne et al., \"Global regolith thermophysical "
        "properties of the Moon from the Diviner Lunar Radiometer "
        "Experiment\", JGR Planets 122(12), 2017. "
        "https://doi.org/10.1002/2017JE005387"};
    diurnal.params = {{"poleX", 0.0f, "The body's pole, world x."},
                      {"poleY", 0.0f, "The body's pole, world y."},
                      {"poleZ", 1.0f, "The body's pole, world z."}};
    diurnal.typeName = "DiurnalTemperature";
    diurnal.slang = kDiurnalSlang;
    diurnal.defaultData = [] {
        const DiurnalTable& table = LunarDiurnalTable();
        std::vector<float> data{static_cast<float>(table.latitudes),
                                static_cast<float>(table.times)};
        data.insert(data.end(), table.kelvin.begin(), table.kelvin.end());
        return data;
    };
    registry.models_.push_back(std::move(diurnal));
    return registry;
}

void TemperatureModelRegistry::Add(TemperatureModel model) {
    if (!IsIdentifier(model.id)) {
        throw std::invalid_argument(
            "Temperature model id must be an identifier: '" + model.id +
            "'");
    }
    if (IsBuiltinId(model.id)) {
        throw std::invalid_argument("Temperature model id '" + model.id +
                                    "' is a built-in model's");
    }
    if (Find(model.id) != nullptr) {
        throw std::invalid_argument(
            "Temperature model id already registered: " + model.id);
    }
    if (model.params.size() > kMaxParams) {
        throw std::invalid_argument("Temperature model " + model.id +
                                    " has more than 16 parameters");
    }
    if (!IsIdentifier(model.typeName) ||
        model.slang.find(model.typeName) == std::string::npos) {
        throw std::invalid_argument(
            "Temperature model " + model.id +
            " Slang source does not define '" + model.typeName + "'");
    }
    models_.push_back(std::move(model));
}

const TemperatureModel*
TemperatureModelRegistry::Find(std::string_view id) const {
    const int index = IndexOf(id);
    return index < 0 ? nullptr
                     : &models_[static_cast<std::size_t>(index)];
}

int TemperatureModelRegistry::IndexOf(std::string_view id) const {
    for (std::size_t i = 0; i < models_.size(); ++i) {
        if (models_[i].id == id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::string_view TemperatureModelSlang() {
    return kTemperatureModelSlang;
}

std::vector<ShaderModuleSource>
TemperatureModelModules(const TemperatureModelRegistry& registry) {
    std::vector<ShaderModuleSource> modules{
        {"temperature_model", std::string(TemperatureModelSlang())}};
    std::string dispatch = "import temperature_model;\n";
    std::string cases;
    for (std::size_t i = 0; i < registry.Models().size(); ++i) {
        const TemperatureModel& model = registry.Models()[i];
        const std::string module = "temperature_model_" + model.id;
        modules.push_back({module, model.slang});
        dispatch += "import " + module + ";\n";
        cases += "    case " + std::to_string(i) + ": {\n        " +
                 model.typeName +
                 " model;\n        return model.kelvin(input, p);\n"
                 "    }\n";
    }
    dispatch += "public float temperatureModel(uint index, "
                "TemperatureInput input, TemperatureParams p) {\n"
                "    switch (index) {\n" +
                cases +
                "    default:\n        return 0.0;\n    }\n}\n";
    modules.push_back({"temperature_models", dispatch});
    return modules;
}

double SolarAbsorptivity(const Spectrum& reflectance) {
    const Spectrum& sun = SolarSpectralIrradiance();
    double total = 0.0;
    double reflected = 0.0;
    for (double nm = 250.5; nm < 2500.0; nm += 1.0) {
        const double irradiance = sun(nm);
        total += irradiance;
        reflected += irradiance * std::clamp(reflectance(nm), 0.0, 1.0);
    }
    return total > 0.0 ? 1.0 - reflected / total : 0.0;
}

double ThermalEmissivity(const std::function<double(double)>& emissivity,
                         double kelvin) {
    // Planck's radiance up to its constants, per nm.
    constexpr double kC2 = 1.438776877e7;  // nm K
    double total = 0.0;
    double emitted = 0.0;
    for (double nm = 2025.0; nm < 100000.0; nm += 50.0) {
        const double r = 1.0 / nm;
        const double planck =
            r * r * r * r * r / std::expm1(kC2 / (nm * kelvin));
        total += planck;
        emitted += planck * std::clamp(emissivity(nm), 0.0, 1.0);
    }
    return total > 0.0 ? emitted / total : 0.0;
}

}  // namespace ernie
