#include "ernie/raster/sensor_plan.h"

#include <algorithm>

namespace ernie {

namespace {

/// The Slang type of a built-in filter (sensor.slang).
std::string_view TypeOf(SensorFilter filter) {
    switch (filter) {
    case SensorFilter::Lens:
        return "Lens";
    case SensorFilter::Qe:
        return "Qe";
    case SensorFilter::Noise:
        return "Noise";
    case SensorFilter::Adc:
        return "Adc";
    case SensorFilter::Cie:
        return "Cie";
    case SensorFilter::Exposure:
        return "Exposure";
    }
    return "";
}

std::uint32_t BitOf(std::string_view var) {
    const auto it = std::find(kRenderVarNames.begin(), kRenderVarNames.end(), var);
    return it == kRenderVarNames.end()
               ? 0u
               : 1u << static_cast<std::uint32_t>(it - kRenderVarNames.begin());
}

/// The Slang module name of custom filter `id`.
std::string ModuleOf(std::string_view id) {
    return "filter_" + std::string(id);
}

}  // namespace

SensorPlan PlanSensor(std::span<const SensorFilterStep> chain,
                      std::span<const std::string_view> held,
                      const SensorFilterRegistry& registry) {
    SensorPlan plan;
    for (const std::string_view var : held) {
        plan.holds |= BitOf(var);
    }
    const auto holds = [&plan](std::string_view var) { return (plan.holds & BitOf(var)) != 0; };
    const bool electrons = holds(render_vars::kElectrons) || holds(render_vars::kRawDn);
    const bool colour = holds(render_vars::kHdrColor) || holds(render_vars::kLdrColor);
    const bool bands = electrons || colour || holds(render_vars::kSpectralRadiance);
    for (const SensorFilterStep& step : CheckFilterChain(chain, registry)) {
        SensorPlan::Stage stage;
        bool needed = false;
        if (const std::optional<SensorFilter> builtin = step.Builtin()) {
            stage.builtin = builtin;
            switch (*builtin) {
            case SensorFilter::Lens:
                needed = true;
                break;
            case SensorFilter::Qe:
            case SensorFilter::Noise:
                needed = electrons;
                break;
            case SensorFilter::Adc:
                needed = holds(render_vars::kRawDn);
                break;
            case SensorFilter::Cie:
                needed = colour;
                break;
            case SensorFilter::Exposure:
                needed = holds(render_vars::kLdrColor);
                break;
            }
        } else {
            const SensorFilterModel& model = *registry.Find(step.type);
            switch (model.domain) {
            case FilterDomain::Spectrum:
                needed = bands;
                break;
            case FilterDomain::Electrons:
                needed = electrons;
                break;
            case FilterDomain::Digital:
                needed = holds(render_vars::kRawDn);
                break;
            case FilterDomain::Rgb:
                needed = colour;
                break;
            case FilterDomain::Display:
                needed = holds(render_vars::kLdrColor);
                break;
            }
            if (needed) {
                stage.firstParam = static_cast<std::uint32_t>(plan.params.size());
                for (const SensorFilterParam& param : model.params) {
                    const auto input = step.inputs.find(param.name);
                    plan.params.push_back(input != step.inputs.end() ? input->second
                                                                     : param.defaultValue);
                }
                stage.custom = model;
            }
        }
        if (needed) {
            if (stage.builtin) {
                plan.filters.push_back(*stage.builtin);
            }
            plan.stages.push_back(std::move(stage));
        }
    }
    return plan;
}

SensorPlan PlanSensor(std::span<const SensorFilter> chain, std::span<const std::string_view> held) {
    return PlanSensor(StepsOf(chain), held, SensorFilterRegistry{});
}

std::string SensorPlan::Key() const {
    std::string key = std::to_string(holds);
    for (const Stage& stage : stages) {
        key += ':';
        key += stage.builtin ? std::string(NameOf(*stage.builtin)) : stage.custom->id;
    }
    return key;
}

std::vector<std::pair<std::string, std::string>> SensorPlan::Modules() const {
    std::vector<std::pair<std::string, std::string>> modules;
    for (const Stage& stage : stages) {
        if (stage.custom && std::none_of(modules.begin(), modules.end(), [&](const auto& module) {
                return module.first == ModuleOf(stage.custom->id);
            })) {
            modules.emplace_back(ModuleOf(stage.custom->id), stage.custom->slang);
        }
    }
    return modules;
}

std::string SensorPlan::Source() const {
    const std::string mask = std::to_string(holds) + "u";
    std::string imports;
    std::string params;
    std::string bandFilters;
    std::string integrate;
    std::string pixels;
    for (std::size_t s = 0; s < stages.size(); ++s) {
        const Stage& stage = stages[s];
        const std::string type =
            stage.builtin ? std::string(TypeOf(*stage.builtin)) : stage.custom->typeName;
        const FilterKind kind = stage.builtin ? KindOf(*stage.builtin) : stage.custom->kind;
        const std::string var = "params" + std::to_string(s);
        // A custom filter's parameters from the buffer, field by field; built-ins take none.
        if (stage.custom) {
            imports += "import " + ModuleOf(stage.custom->id) + ";\n";
            params += "    " + type + ".Params " + var + ";\n";
            for (std::size_t p = 0; p < stage.custom->params.size(); ++p) {
                params += "    " + var + "." + stage.custom->params[p].name + " = filterParams[" +
                          std::to_string(stage.firstParam + p) + "];\n";
            }
        }
        const std::string args = stage.custom ? ", " + var : "";
        switch (kind) {
        case FilterKind::Spatial:
            break;  // the head: the lens
        case FilterKind::Band:
            bandFilters +=
                "        radiance = " + type + ".apply(radiance, band, pixel" + args + ");\n";
            break;
        case FilterKind::Integrator:
            integrate += "        " + type + ".accumulate(value, band, radiance, pixel);\n";
            pixels += "    " + type + ".finish(value, pixel);\n";
            break;
        case FilterKind::Pixel:
            pixels += "    " + type + ".apply(value, pixel" + args + ");\n";
            break;
        }
    }
    const bool spectral = (holds & 1u) != 0;  // SpectralRadiance, kRenderVarNames' first
    return "// Generated by PlanSensor for plan " + Key() +
           ".\n"
           "import sensor;\n" +
           imports +
           "\n"
           "[shader(\"compute\")]\n"
           "[numthreads(8, 8, 1)]\n"
           "void sensorMain(uint3 id : SV_DispatchThreadID) {\n"
           "    PixelContext pixel;\n"
           "    if (!beginPixel(id, pixel)) {\n"
           "        return;\n"
           "    }\n" +
           params +
           "    let lens = Lens.at(pixel);\n"
           "    SensorPixel value = {};\n"
           "    lens.finish(value, pixel, " +
           mask +
           ");\n"
           "    for (uint band = 0; band < pixel.bands(); ++band) {\n"
           "        var radiance = lens.band(band, pixel);\n" +
           bandFilters + (spectral ? "        writeSpectral(pixel, band, radiance);\n" : "") +
           integrate + "    }\n" + pixels + "    writeOutputs(value, pixel, " + mask +
           ");\n"
           "}\n";
}

}  // namespace ernie
