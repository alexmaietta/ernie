// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/sensor_plan.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

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

constexpr std::size_t kDomains = 5;

std::size_t IndexOf(FilterDomain domain) {
    return static_cast<std::size_t>(domain);
}

/// The domain a stage reads.
FilterDomain InputOf(const SensorPlan::Stage& stage) {
    if (stage.custom) {
        // An integrator reads the bands and writes its domain.
        return stage.custom->kind == FilterKind::Integrator ? FilterDomain::Spectrum
                                                            : stage.custom->domain;
    }
    switch (*stage.builtin) {
    case SensorFilter::Lens:
    case SensorFilter::Qe:
    case SensorFilter::Cie:
        return FilterDomain::Spectrum;
    case SensorFilter::Noise:
    case SensorFilter::Adc:
        return FilterDomain::Electrons;
    case SensorFilter::Exposure:
        return FilterDomain::Rgb;
    }
    return FilterDomain::Spectrum;
}

/// The domain a stage writes.
FilterDomain OutputOf(const SensorPlan::Stage& stage) {
    if (stage.custom) {
        return stage.custom->domain;
    }
    switch (*stage.builtin) {
    case SensorFilter::Lens:
        return FilterDomain::Spectrum;
    case SensorFilter::Qe:
    case SensorFilter::Noise:
        return FilterDomain::Electrons;
    case SensorFilter::Adc:
        return FilterDomain::Digital;
    case SensorFilter::Cie:
        return FilterDomain::Rgb;
    case SensorFilter::Exposure:
        return FilterDomain::Display;
    }
    return FilterDomain::Spectrum;
}

/// Whether `domain` is made from `from`, or is it: everything from the spectrum, digital numbers
/// from electrons, display colour from linear colour.
bool DerivesFrom(FilterDomain domain, FilterDomain from) {
    switch (from) {
    case FilterDomain::Spectrum:
        return true;
    case FilterDomain::Electrons:
        return domain == FilterDomain::Electrons || domain == FilterDomain::Digital;
    case FilterDomain::Rgb:
        return domain == FilterDomain::Rgb || domain == FilterDomain::Display;
    default:
        return domain == from;
    }
}

bool IsSpatialHead(const SensorPlan::Stage& stage) {
    return stage.custom && stage.custom->kind == FilterKind::Spatial;
}

/// sensor.slang's `kChannel*` layer of a domain after the spectrum.
std::string_view ChannelOf(FilterDomain domain) {
    switch (domain) {
    case FilterDomain::Electrons:
        return "kChannelElectrons";
    case FilterDomain::Digital:
        return "kChannelDigital";
    case FilterDomain::Rgb:
        return "kChannelRgb";
    default:
        return "kChannelDisplay";
    }
}

}  // namespace

SensorPlan PlanSensor(std::span<const SensorFilterStep> chain,
                      std::span<const std::string_view> held, const SensorFilterRegistry& registry,
                      std::uint32_t aovOffset) {
    SensorPlan plan;
    for (const std::string_view var : held) {
        plan.holds |= BitOf(var);
        for (std::size_t slot = 0; slot < kAovSlotNames.size(); ++slot) {
            if (kAovSlotNames[slot] == var) {
                plan.aovs |= 1u << slot;
            }
        }
    }
    const auto holds = [&plan](std::string_view var) { return (plan.holds & BitOf(var)) != 0; };
    const std::vector<SensorFilterStep> steps = CheckFilterChain(chain, registry);
    // Each custom step's first AOV slot, in chain order, and whether one of its slots is held: a
    // held AOV runs its filter and the branch of the chain it is on.
    std::vector<std::uint32_t> bases(steps.size(), 0);
    std::vector<bool> forced(steps.size(), false);
    std::array<bool, kDomains> forcedDomains{};
    std::uint32_t slot = aovOffset;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        if (steps[i].Builtin()) {
            continue;
        }
        const SensorFilterModel& model = *registry.Find(steps[i].type);
        bases[i] = slot;
        for (std::size_t k = 0; k < model.outputs.size(); ++k, ++slot) {
            if (slot < kAovSlotNames.size() && ((plan.aovs >> slot) & 1u) != 0) {
                forced[i] = true;
                forcedDomains[IndexOf(model.domain)] = true;
            }
        }
    }
    const auto forcedDomain = [&](FilterDomain domain) { return forcedDomains[IndexOf(domain)]; };
    const bool digital = holds(render_vars::kRawDn) || forcedDomain(FilterDomain::Digital);
    const bool electrons =
        holds(render_vars::kElectrons) || digital || forcedDomain(FilterDomain::Electrons);
    const bool display = holds(render_vars::kLdrColor) || forcedDomain(FilterDomain::Display);
    const bool colour = holds(render_vars::kHdrColor) || display || forcedDomain(FilterDomain::Rgb);
    const bool bands = electrons || colour || holds(render_vars::kSpectralRadiance) ||
                       forcedDomain(FilterDomain::Spectrum);
    for (std::size_t i = 0; i < steps.size(); ++i) {
        const SensorFilterStep& step = steps[i];
        SensorPlan::Stage stage;
        stage.outputBase = bases[i];
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
                needed = digital;
                break;
            case SensorFilter::Cie:
                needed = colour;
                break;
            case SensorFilter::Exposure:
                needed = display;
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
                needed = digital;
                break;
            case FilterDomain::Rgb:
                needed = colour;
                break;
            case FilterDomain::Display:
                needed = display;
                break;
            }
            needed = needed || forced[i];
            if (needed) {
                stage.firstParam = static_cast<std::uint32_t>(plan.params.size());
                for (const SensorFilterParam& param : model.params) {
                    const auto input = step.inputs.find(param.name);
                    if (param.size == 1) {
                        plan.params.push_back(input != step.inputs.end() ? input->second
                                                                         : param.defaultValue);
                        continue;
                    }
                    // An array: the step's array, else its number, else the defaults,
                    // resampled to the parameter's size.
                    const auto array = step.arrays.find(param.name);
                    std::vector<float> values =
                        array != step.arrays.end()   ? array->second
                        : input != step.inputs.end() ? std::vector<float>{input->second}
                        : !param.defaults.empty()    ? param.defaults
                                                     : std::vector<float>{param.defaultValue};
                    const std::vector<float> resampled = ResampleCurve(values, param.size);
                    plan.params.insert(plan.params.end(), resampled.begin(), resampled.end());
                }
                stage.custom = model;
                stage.stateBase = plan.states;
                plan.states += model.states;
                if (plan.states > kMaxStates) {
                    throw std::invalid_argument("The sensor filter chain keeps more than " +
                                                std::to_string(kMaxStates) +
                                                " state slots per pixel");
                }
            }
        }
        if (needed) {
            if (stage.builtin) {
                plan.filters.push_back(*stage.builtin);
            }
            plan.stages.push_back(std::move(stage));
        }
    }
    // A stage's kernel: one past each spatial filter before it on what its input derives from,
    // and one more for a spatial filter itself.
    for (std::size_t s = 0; s < plan.stages.size(); ++s) {
        SensorPlan::Stage& stage = plan.stages[s];
        const FilterDomain input = InputOf(stage);
        std::uint32_t segment = IsSpatialHead(stage) ? 1u : 0u;
        for (std::size_t before = 0; before < s; ++before) {
            const SensorPlan::Stage& earlier = plan.stages[before];
            if (IsSpatialHead(earlier) && DerivesFrom(input, InputOf(earlier))) {
                ++segment;
            }
        }
        stage.segment = segment;
        plan.segments = std::max(plan.segments, segment + 1);
        if (IsSpatialHead(stage)) {
            (input == FilterDomain::Spectrum ? plan.spectrumImages : plan.channelImages) = true;
            if (stage.custom->pyramid) {
                plan.pyramids |= 1u << segment;
            }
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
        // Filters with outputs write them from their slot, part of the kernel.
        if (stage.custom && !stage.custom->outputs.empty()) {
            key += '@' + std::to_string(stage.outputBase);
        }
        if (stage.custom && stage.custom->states > 0) {
            key += '#' + std::to_string(stage.stateBase);
        }
    }
    // Kernels read the pyramid's mips through the same bindings: no difference in source, but
    // plans build different images.
    if (pyramids != 0) {
        key += "^" + std::to_string(pyramids);
    }
    return key + "~" + lens + (pushbroom ? "~pushbroom" : "") + (cube ? "~cube" : "") +
           (table ? "~table" : "") + (chromatic ? "~chromatic" : "");
}

std::string LensKernelOf(const LensModel& lens, bool pushbroom) {
    if (pushbroom) {
        return "Pushbroom";
    }
    if (lens.projection == LensProjection::Fisheye) {
        return "Fisheye";
    }
    return lens.HasDistortion() ? "BrownConrady" : "Pinhole";
}

std::string SensorPlan::LensSource() const {
    return "// Generated for the lens projection " + lens +
           ".\n"
           "import lens;\n"
           "public static const bool kLensCube = " +
           (cube ? "true" : "false") +
           ";\n"
           "public static const bool kLensChromatic = " +
           (chromatic ? "true" : "false") +
           ";\n"
           "public static const bool kLensTable = " +
           (table ? "true" : "false") +
           ";\n"
           "public float2 lensPinholeOf(float2 sensor, float4 k, float p2, float overscan) {\n"
           "    return " +
           lens +
           (lens == "Pushbroom" ? ".pinholeOf(sensor, k, overscan);\n"
                                : ".pinholeOf(sensor, k, p2);\n") +
           "}\n"
           "public float3 lensRayOf(float2 sensor, float4 k, float p2) {\n"
           "    return " +
           lens +
           ".rayOf(sensor, k, p2);\n"
           "}\n"
           "public float lensFalloff(float2 sensor, float2 pinhole, float4 k) {\n"
           "    return " +
           lens +
           ".falloff(sensor, pinhole, k);\n"
           "}\n";
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

std::string SensorPlan::Source(std::uint32_t segment) const {
    // Where each domain's values are final: the last segment writing them, where its var is
    // written; and the domains the next segment's spatial filters read, which this one stores.
    std::array<std::uint32_t, kDomains> final{};
    std::array<bool, kDomains> stores{};
    for (const Stage& stage : stages) {
        final[IndexOf(OutputOf(stage))] = std::max(final[IndexOf(OutputOf(stage))], stage.segment);
        if (IsSpatialHead(stage) && stage.segment == segment + 1) {
            stores[IndexOf(InputOf(stage))] = true;
        }
    }
    std::uint32_t mask = 0;
    const auto writes = [&](std::string_view var, FilterDomain domain) {
        if (final[IndexOf(domain)] == segment) {
            mask |= holds & BitOf(var);
        }
    };
    writes(render_vars::kElectrons, FilterDomain::Electrons);
    writes(render_vars::kRawDn, FilterDomain::Digital);
    writes(render_vars::kHdrColor, FilterDomain::Rgb);
    writes(render_vars::kLdrColor, FilterDomain::Display);
    const bool spectral = (holds & 1u) != 0 && final[IndexOf(FilterDomain::Spectrum)] == segment;
    // Geometry vars come from the lens, in the first segment.
    const std::uint32_t geometry =
        segment == 0 ? holds & ~(BitOf(render_vars::kSpectralRadiance) |
                                 BitOf(render_vars::kElectrons) | BitOf(render_vars::kRawDn) |
                                 BitOf(render_vars::kHdrColor) | BitOf(render_vars::kLdrColor))
                     : 0u;

    std::string imports;
    std::string params;
    std::string source = "lens.band(band, pixel)";  // the spectrum's source in this segment
    std::string sourceBase;  // its outputs' slot, set before it in each band's iteration
    std::string bandFilters;
    std::string integrate;
    std::string pixels;
    bool usesBands = spectral || stores[IndexOf(FilterDomain::Spectrum)];
    for (std::size_t s = 0; s < stages.size(); ++s) {
        const Stage& stage = stages[s];
        if (stage.segment != segment) {
            continue;
        }
        const std::string type =
            stage.builtin ? std::string(TypeOf(*stage.builtin)) : stage.custom->typeName;
        const FilterKind kind = stage.builtin ? KindOf(*stage.builtin) : stage.custom->kind;
        const std::string var = "params" + std::to_string(s);
        // A custom filter's parameters from the buffer, field by field; built-ins take none.
        if (stage.custom) {
            imports += "import " + ModuleOf(stage.custom->id) + ";\n";
            params += "    " + type + ".Params " + var + ";\n";
            std::uint32_t offset = stage.firstParam;
            for (const SensorFilterParam& param : stage.custom->params) {
                if (param.size == 1) {
                    params += "    " + var + "." + param.name + " = filterParams[" +
                              std::to_string(offset) + "];\n";
                } else {
                    params += "    for (uint i = 0; i < " + std::to_string(param.size) +
                              "; ++i) {\n        " + var + "." + param.name +
                              "[i] = filterParams[" + std::to_string(offset) + " + i];\n    }\n";
                }
                offset += param.size;
            }
        }
        const std::string args = stage.custom ? ", " + var : "";
        // Where the filter's outputs go (`writeOutput`), set before each call.
        std::string base;
        if (stage.custom && !stage.custom->outputs.empty()) {
            base += "pixel.outputBase = " + std::to_string(stage.outputBase) + "u; ";
        }
        // And its state slots (`readState`, `writeState`).
        if (stage.custom && stage.custom->states > 0) {
            base += "pixel.stateBase = " + std::to_string(stage.stateBase) + "u; ";
        }
        switch (kind) {
        case FilterKind::Spatial:
            if (stage.builtin) {
                break;  // the lens heads the first segment
            }
            if (stage.custom->domain == FilterDomain::Spectrum) {
                params += "    SpatialImage input" + std::to_string(s) + " = {0};\n";
                source = type + ".band(band, pixel, input" + std::to_string(s) + args + ")";
                sourceBase = base;
                usesBands = true;
            } else {
                params += "    SpatialImage input" + std::to_string(s) + " = {" +
                          std::string(ChannelOf(stage.custom->domain)) + "};\n";
                pixels += "    " + base + type + ".apply(value, pixel, input" + std::to_string(s) +
                          args + ");\n";
            }
            break;
        case FilterKind::Band:
            bandFilters += "        " + base + "radiance = " + type +
                           ".apply(radiance, band, pixel" + args + ");\n";
            usesBands = true;
            break;
        case FilterKind::Integrator:
            // Built-in integrators take no parameters; custom ones (`ICustomIntegrator`) do.
            integrate += "        " + base + type + ".accumulate(value, band, radiance, pixel" +
                         args + ");\n";
            pixels += "    " + base + type + ".finish(value, pixel" + args + ");\n";
            usesBands = true;
            break;
        case FilterKind::Pixel:
            pixels += "    " + base + type + ".apply(value, pixel" + args + ");\n";
            break;
        }
    }
    std::string stored;
    for (const FilterDomain domain : {FilterDomain::Electrons, FilterDomain::Digital,
                                      FilterDomain::Rgb, FilterDomain::Display}) {
        if (stores[IndexOf(domain)]) {
            stored += "    storeChannel(value, pixel, " + std::string(ChannelOf(domain)) + ");\n";
        }
    }
    std::string loop;
    if (usesBands) {
        loop = "    for (uint band = 0; band < pixel.bands(); ++band) {\n" +
               (sourceBase.empty() ? std::string() : "        " + sourceBase + "\n") +
               "        var radiance = " + source + ";\n" + bandFilters +
               (spectral ? "        writeSpectral(pixel, band, radiance);\n" : "") +
               (stores[IndexOf(FilterDomain::Spectrum)]
                    ? "        storeSpectrum(pixel, band, radiance);\n"
                    : "") +
               integrate + "    }\n";
    }
    return "// Generated by PlanSensor for plan " + Key() + ", segment " + std::to_string(segment) +
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
           std::to_string(geometry) + "u);\n" + loop + pixels + stored +
           "    writeOutputs(value, pixel, " + std::to_string(mask) +
           "u);\n"
           "}\n";
}

}  // namespace ernie
