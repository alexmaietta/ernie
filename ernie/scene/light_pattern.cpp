// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/light_pattern.h"

#include "ernie/scene/tokens.h"

#include <algorithm>
#include <cctype>
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

constexpr const char* kLightPatternSlang = R"(
// A laser's pattern (ernie/scene/light_pattern.h): sixteen parameters
// and the interface a pattern implements.
public struct LightPatternParams {
    public float4 values[4];

    public float get(uint i) {
        return values[i / 4][i % 4];
    }
}

public interface ILightPattern {
    // Relative power per unit area of the tangent plane at t (x / z,
    // y / z), averaged over `footprint` of it, at `wavelength` nm and
    // `time` s; any scale.
    float emission(float2 t, float2 footprint, float wavelength,
                   float time, LightPatternParams p);
}
)";

}  // namespace

std::vector<float> LightPatternModel::Defaults() const {
    std::vector<float> values;
    for (const LightPatternParam& param : params) {
        values.push_back(param.defaultValue);
    }
    return values;
}

void LightPatternRegistry::Add(LightPatternModel model) {
    if (!IsIdentifier(model.id)) {
        throw std::invalid_argument(
            "Light pattern id must be an identifier: '" + model.id +
            "'");
    }
    if (model.id == tokens::kSpot || model.id == tokens::kLine ||
        model.id == tokens::kGrid) {
        throw std::invalid_argument("Light pattern id '" + model.id +
                                    "' is a built-in pattern's");
    }
    if (Find(model.id) != nullptr) {
        throw std::invalid_argument(
            "Light pattern id already registered: " + model.id);
    }
    if (model.params.size() > kMaxParams) {
        throw std::invalid_argument("Light pattern " + model.id +
                                    " has more than 16 parameters");
    }
    if (!(model.extent > 0.0)) {
        throw std::invalid_argument("Light pattern " + model.id +
                                    " needs a positive extent");
    }
    if (!IsIdentifier(model.typeName) ||
        model.slang.find(model.typeName) == std::string::npos) {
        throw std::invalid_argument(
            "Light pattern " + model.id +
            " Slang source does not define '" + model.typeName + "'");
    }
    models_.push_back(std::move(model));
}

const LightPatternModel*
LightPatternRegistry::Find(std::string_view id) const {
    const int index = IndexOf(id);
    return index < 0 ? nullptr
                     : &models_[static_cast<std::size_t>(index)];
}

int LightPatternRegistry::IndexOf(std::string_view id) const {
    for (std::size_t i = 0; i < models_.size(); ++i) {
        if (models_[i].id == id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::string_view LightPatternSlang() {
    return kLightPatternSlang;
}

std::vector<ShaderModuleSource>
LightPatternModules(const LightPatternRegistry& registry) {
    std::vector<ShaderModuleSource> modules{
        {"light_pattern", std::string(LightPatternSlang())}};
    std::string dispatch = "import light_pattern;\n";
    std::string cases;
    for (std::size_t i = 0; i < registry.Models().size(); ++i) {
        const LightPatternModel& model = registry.Models()[i];
        const std::string module = "light_pattern_" + model.id;
        modules.push_back({module, model.slang});
        dispatch += "import " + module + ";\n";
        cases += "    case " + std::to_string(i) + ": {\n        " +
                 model.typeName +
                 " pattern;\n        return pattern.emission(t, "
                 "footprint, wavelength, time, p);\n    }\n";
    }
    dispatch +=
        "public float customPattern(uint index, float2 t, float2 "
        "footprint, float wavelength, float time, LightPatternParams "
        "p) {\n    switch (index) {\n" +
        cases + "    default:\n        return 0.0;\n    }\n}\n";
    modules.push_back({"light_patterns", dispatch});
    return modules;
}

}  // namespace ernie
