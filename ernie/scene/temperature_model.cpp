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

// The most distant lights a model sees (the renderer's lights a frame).
public static const uint kTemperatureLights = 16;

// Where a model is evaluated: the point and its outward surface normal (world, relative to the
// frame's origin); every distant light that lights it (none singled out; light linking leaves
// out those not linked to it), `lightCount` of them: its direction towards it, its irradiance
// over all wavelengths facing it (W/m^2) and the share of it reaching the point past shadows;
// the sunlight on the surface from them all (W/m^2, shadowed); the pixel's solar absorptivity
// (the material's scaled as its albedo textures scale its reflectance) and the material's
// hemispherical thermal emissivity; its own temperature (K, 0 for none); the frame's time (s);
// the world +z axis of a prim bound to the material (0 for none); and the pixel's thermal inertia
// (J m^-2 K^-1 s^-1/2, its texture's else the material's, 0 for none).
public struct TemperatureInput {
    public float3 position;
    public float3 normal;
    public uint lightCount;
    public float3 lightDirection[kTemperatureLights];
    public float lightIrradiance[kTemperatureLights];
    public float lightVisibility[kTemperatureLights];
    public float irradiance;
    public float absorptivity;
    public float emissivity;
    public float kelvin;
    public float time;
    public float3 axis;
    public float inertia;
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

// Where `value` lies along an ascending axis of `count` values from data `first`, clamped to its
// ends: the index of the value below and the weight of the one above, by its logarithm when
// `logarithmic` (thermal inertia spans decades).
float2 axisWeight(TemperatureParams p, uint first, uint count, float value, bool logarithmic) {
    if (count < 2) {
        return float2(0.0);
    }
    uint i = 0;
    while (i + 2 < count && p.data(first + i + 1) < value) {
        ++i;
    }
    var low = p.data(first + i);
    var high = p.data(first + i + 1);
    var x = value;
    if (logarithmic) {
        low = log(max(low, 1e-6));
        high = log(max(high, 1e-6));
        x = log(max(x, 1e-6));
    }
    return float2(float(i), saturate((x - low) / max(high - low, 1e-12)));
}

// A table of latitude (0 to 90 degrees), local solar time (from midnight), thermal inertia and
// albedo (DiurnalTable::ModelData), read at the surface normal's latitude and hour angle about the
// body's pole, the pixel's thermal inertia (the table's first without one) and its albedo (1 minus
// its solar absorptivity).
public struct DiurnalTemperature : ITemperatureModel {
    public float kelvin(TemperatureInput input, TemperatureParams p) {
        if (p.dataCount() < 5 || p.data(0) != -2.0) {
            return input.kelvin;
        }
        let rows = uint(p.data(1));
        let columns = uint(p.data(2));
        let inertias = uint(p.data(3));
        let albedos = uint(p.data(4));
        let first = 5 + inertias + albedos;
        if (rows < 2 || columns < 2 || inertias < 1 || albedos < 1 ||
            p.dataCount() < first + rows * columns * inertias * albedos) {
            return input.kelvin;
        }
        // The pole given, else the bound prim's +z, else world +z.
        var pole = float3(p.get(0), p.get(1), p.get(2));
        if (dot(pole, pole) == 0.0) {
            pole = input.axis;
        }
        pole = dot(pole, pole) > 0.0 ? normalize(pole) : float3(0.0, 0.0, 1.0);
        let n = input.normal;
        let latitude = asin(saturate(abs(dot(n, pole))));
        let row = latitude / (0.5 * 3.14159265) * float(rows - 1);
        let inertia = axisWeight(p, 5, inertias,
                                 input.inertia > 0.0 ? input.inertia : p.data(5), true);
        let albedo = axisWeight(p, 5 + inertias, albedos, 1.0 - input.absorptivity, false);
        // Each light's day, its table temperature at its own hour angle, weighted by its
        // irradiance over the table's: T^4 summed, as radiated power adds (exact for a surface
        // that keeps no heat; the table's stored heat counted once per light).
        let up = n - dot(n, pole) * pole;
        let east = cross(pole, n);
        let reference = p.get(3) > 0.0 ? p.get(3) : 1361.0;
        var power = 0.0;
        for (uint k = 0; k < min(input.lightCount, kTemperatureLights); ++k) {
            if (input.lightIrradiance[k] <= 0.0) {
                continue;
            }
            // Its hour angle, westward from the meridian: the body turns towards its east,
            // pole x n, so a light moves west and the afternoon has it there.
            let towards = input.lightDirection[k];
            let along = towards - dot(towards, pole) * pole;
            let hour = atan2(-dot(along, east), dot(along, up));
            let phase = frac(hour / (2.0 * 3.14159265) + 0.5);
            let kelvin = tableKelvin(p, first, rows, columns, inertias, albedos, row,
                                     phase * float(columns), inertia, albedo);
            power += input.lightIrradiance[k] / reference * kelvin * kelvin * kelvin * kelvin;
        }
        return power > 0.0 ? sqrt(sqrt(power)) : input.kelvin;
    }

    // The table at fractional row `row` and column `column` (wrapping over the day), inertia and
    // albedo `inertia` and `albedo` (`axisWeight`'s), quadrilinear.
    float tableKelvin(TemperatureParams p, uint first, uint rows, uint columns, uint inertias,
                      uint albedos, float row, float column, float2 inertia, float2 albedo) {
        let r0 = min(uint(row), rows - 2);
        let c0 = uint(column) % columns;
        let fr = saturate(row - float(r0));
        let fc = frac(column);
        var sum = 0.0;
        for (uint corner = 0; corner < 16; ++corner) {
            let r = corner & 1u;
            let c = (corner >> 1) & 1u;
            let i = (corner >> 2) & 1u;
            let a = (corner >> 3) & 1u;
            let weight = (r != 0 ? fr : 1.0 - fr) * (c != 0 ? fc : 1.0 - fc) *
                         (i != 0 ? inertia.y : 1.0 - inertia.y) *
                         (a != 0 ? albedo.y : 1.0 - albedo.y);
            if (weight == 0.0) {
                continue;
            }
            let ii = min(uint(inertia.x) + i, inertias - 1);
            let aa = min(uint(albedo.x) + a, albedos - 1);
            let index = (((r0 + r) * columns + (c0 + c) % columns) * inertias + ii) * albedos + aa;
            sum += weight * p.data(first + index);
        }
        return sum;
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
        "D. G. Gilmore (ed.), \"Spacecraft Thermal Control "
        "Handbook\", "
        "2nd ed., The Aerospace Press, 2002, chapter 2.",
        "M. Dunstan et al., \"PANGU: Planet and Asteroid Natural "
        "Scene "
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
        "The temperature of a table of latitude, local solar time, "
        "thermal inertia and albedo, baked offline by a regolith "
        "model "
        "whose thermal inertia keeps the night warm and lags the "
        "afternoon: the Moon's by default. Each pixel reads it at "
        "its "
        "own thermal inertia and albedo.";
    diurnal.references = {
        "P. O. Hayne et al., \"Global regolith thermophysical "
        "properties of the Moon from the Diviner Lunar Radiometer "
        "Experiment\", JGR Planets 122(12), 2017. "
        "https://doi.org/10.1002/2017JE005387"};
    const std::string pole =
        "; all 0, the +z axis of the prim the material is bound to.";
    diurnal.params = {
        {"poleX", 0.0f, "The body's pole, world x" + pole},
        {"poleY", 0.0f, "The body's pole, world y" + pole},
        {"poleZ", 0.0f, "The body's pole, world z" + pole},
        {"irradiance", 1361.0f,
         "The sunlight the table was made for, W/m^2: each distant "
         "light counts as its irradiance over this, their T^4 "
         "summed."}};
    diurnal.typeName = "DiurnalTemperature";
    diurnal.slang = kDiurnalSlang;
    diurnal.defaultData = [] {
        return LunarDiurnalTable().ModelData();
    };
    registry.models_.push_back(std::move(diurnal));
    return registry;
}

void TemperatureModelRegistry::Add(TemperatureModel model) {
    if (!IsIdentifier(model.id)) {
        throw std::invalid_argument(
            "Temperature model id must be an identifier: '" +
            model.id + "'");
    }
    if (IsBuiltinId(model.id)) {
        throw std::invalid_argument("Temperature model id '" +
                                    model.id +
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
        reflected +=
            irradiance * std::clamp(reflectance(nm), 0.0, 1.0);
    }
    return total > 0.0 ? 1.0 - reflected / total : 0.0;
}

double
ThermalEmissivity(const std::function<double(double)>& emissivity,
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
