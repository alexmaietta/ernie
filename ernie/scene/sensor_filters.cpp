#include "ernie/scene/sensor_filters.h"

#include "ernie/kernel/embedded_shaders.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace ernie {

std::string_view NameOf(SensorFilter filter) {
    switch (filter) {
    case SensorFilter::Lens:
        return "lens";
    case SensorFilter::Qe:
        return "qe";
    case SensorFilter::Noise:
        return "noise";
    case SensorFilter::Adc:
        return "adc";
    case SensorFilter::Cie:
        return "cie";
    case SensorFilter::Exposure:
        return "exposure";
    }
    return "";
}

std::optional<SensorFilter> SensorFilterNamed(std::string_view name) {
    for (const SensorFilter filter : kSensorFilters) {
        if (NameOf(filter) == name) {
            return filter;
        }
    }
    return std::nullopt;
}

std::vector<SensorFilter> DefaultFilterChain() {
    return {kSensorFilters.begin(), kSensorFilters.end()};
}

std::vector<SensorFilter> CheckFilterChain(std::span<const SensorFilter> chain) {
    std::vector<SensorFilter> checked;
    if (chain.empty() || chain.front() != SensorFilter::Lens) {
        checked.push_back(SensorFilter::Lens);
    }
    const auto seen = [&checked](SensorFilter filter) {
        return std::find(checked.begin(), checked.end(), filter) != checked.end();
    };
    for (const SensorFilter filter : chain) {
        if (seen(filter)) {
            throw std::invalid_argument("Sensor filter '" + std::string(NameOf(filter)) +
                                        "' appears twice");
        }
        if ((filter == SensorFilter::Noise || filter == SensorFilter::Adc) &&
            !seen(SensorFilter::Qe)) {
            throw std::invalid_argument("Sensor filter '" + std::string(NameOf(filter)) +
                                        "' needs 'qe' before it");
        }
        if (filter == SensorFilter::Exposure && !seen(SensorFilter::Cie)) {
            throw std::invalid_argument("Sensor filter 'exposure' needs 'cie' before it");
        }
        checked.push_back(filter);
    }
    return checked;
}

FilterKind KindOf(SensorFilter filter) {
    switch (filter) {
    case SensorFilter::Lens:
        return FilterKind::Spatial;
    case SensorFilter::Qe:
    case SensorFilter::Cie:
        return FilterKind::Integrator;
    case SensorFilter::Noise:
    case SensorFilter::Adc:
    case SensorFilter::Exposure:
        return FilterKind::Pixel;
    }
    return FilterKind::Pixel;
}

namespace {

bool IsIdentifier(std::string_view name) {
    return !name.empty() &&
           (std::isalpha(static_cast<unsigned char>(name.front())) != 0 || name.front() == '_') &&
           std::all_of(name.begin(), name.end(), [](char ch) {
               return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
           });
}

}  // namespace

void SensorFilterRegistry::Add(SensorFilterModel model) {
    const std::string what = "Sensor filter '" + model.id + "'";
    if (!IsIdentifier(model.id)) {
        throw std::invalid_argument(what + ": its id must be an identifier");
    }
    if (SensorFilterNamed(model.id) || Find(model.id) != nullptr) {
        throw std::invalid_argument(what + " is already defined");
    }
    const bool suits = model.kind == FilterKind::Band    ? model.domain == FilterDomain::Spectrum
                       : model.kind == FilterKind::Pixel ? model.domain != FilterDomain::Spectrum
                                                         : model.kind == FilterKind::Spatial;
    if (!suits) {
        throw std::invalid_argument(what + ": a custom filter is a band filter on the spectrum, "
                                           "a pixel filter on another domain or a spatial filter");
    }
    if (model.params.size() > kMaxParams) {
        throw std::invalid_argument(what + " has more than " + std::to_string(kMaxParams) +
                                    " parameters");
    }
    for (std::size_t i = 0; i < model.params.size(); ++i) {
        const std::string& name = model.params[i].name;
        if (!IsIdentifier(name)) {
            throw std::invalid_argument(what + ": parameter '" + name + "' is not an identifier");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (model.params[j].name == name) {
                throw std::invalid_argument(what + ": parameter '" + name + "' repeats");
            }
        }
    }
    if (!IsIdentifier(model.typeName) || model.slang.find(model.typeName) == std::string::npos) {
        throw std::invalid_argument(what + ": its Slang must define type '" + model.typeName + "'");
    }
    models_.push_back(std::move(model));
}

const SensorFilterModel* SensorFilterRegistry::Find(std::string_view id) const {
    for (const SensorFilterModel& model : models_) {
        if (model.id == id) {
            return &model;
        }
    }
    return nullptr;
}

std::vector<SensorFilterModel> StandardSensorFilters() {
    using Kind = FilterKind;
    using Domain = FilterDomain;
    return {
        {"psf", "Point spread",
         "The lens's point-spread function: diffraction for the camera's f-number per band "
         "(a Gaussian fit to the Airy disc, sigma = 0.42 lambda N) with a Gaussian blur for "
         "aberrations; light is kept.",
         {"G. B. Airy, \"On the Diffraction of an Object-glass with Circular Aperture\", 1835",
          "B. Zhang, J. Zerubia and J.-C. Olivo-Marin, \"Gaussian approximations of fluorescence "
          "microscope point-spread function models\", Applied Optics 46(10), 2007"},
         Kind::Spatial, Domain::Spectrum,
         {{"blur", 0.0f, "Gaussian sigma in pixels added to diffraction's"},
          {"maxRadius", 12.0f, "The widest kernel in pixels, bounding the cost"}},
         "Psf", shaders::kFilterPsfSlang},
        {"defocus", "Depth of field",
         "Thin-lens defocus from the camera's fStop and focusDistance: each point a disc of the "
         "circle of confusion's size; pinhole cameras (fStop 0) are untouched.",
         {"M. Potmesil and I. Chakravarty, \"A lens and aperture camera model for synthetic "
          "image generation\", SIGGRAPH 1981",
          "J. Jimenez, \"Next Generation Post Processing in Call of Duty: Advanced Warfare\", "
          "SIGGRAPH 2014 course"},
         Kind::Spatial, Domain::Spectrum,
         {{"maxRadius", 8.0f, "The widest blur disc's radius in pixels, bounding the cost"}},
         "Defocus", shaders::kFilterDefocusSlang},
        {"bayer", "Bayer mosaic",
         "A Bayer colour filter array: each pixel of an RGB sensor keeps one channel's "
         "electrons.",
         {"B. E. Bayer, \"Color imaging array\", US patent 3,971,065, 1976"},
         Kind::Pixel, Domain::Electrons,
         {{"pattern", 0.0f, "0 RGGB, 1 BGGR, 2 GRBG, 3 GBRG"}},
         "Bayer", shaders::kFilterBayerSlang},
        {"demosaic", "Demosaic",
         "Full RGB digital numbers interpolated from a Bayer mosaic.",
         {"H. S. Malvar, L. He and R. Cutler, \"High-quality linear interpolation for "
          "demosaicing of Bayer-patterned color images\", IEEE ICASSP 2004"},
         Kind::Spatial, Domain::Digital,
         {{"pattern", 0.0f, "As the bayer filter's"},
          {"method", 1.0f, "0 bilinear, 1 Malvar-He-Cutler"}},
         "Demosaic", shaders::kFilterDemosaicSlang},
        {"whiteBalance", "White balance", "Gains on linear colour's red, green and blue.",
         {"J. von Kries, 1905, in D. L. MacAdam (ed.), \"Sources of Color Science\", 1970"},
         Kind::Pixel, Domain::Rgb,
         {{"red", 1.0f, "Red gain"}, {"green", 1.0f, "Green gain"}, {"blue", 1.0f, "Blue gain"}},
         "WhiteBalance", shaders::kFilterWhiteBalanceSlang},
        {"colorMatrix", "Colour matrix",
         "A 3 x 3 colour correction matrix on linear colour, row by row.",
         {"M. S. Brown, \"Understanding the In-Camera Image Processing Pipeline for Computer "
          "Vision\", CVPR 2016 tutorial"},
         Kind::Pixel, Domain::Rgb,
         {{"m00", 1.0f, ""}, {"m01", 0.0f, ""}, {"m02", 0.0f, ""},
          {"m10", 0.0f, ""}, {"m11", 1.0f, ""}, {"m12", 0.0f, ""},
          {"m20", 0.0f, ""}, {"m21", 0.0f, ""}, {"m22", 1.0f, ""}},
         "ColorMatrix", shaders::kFilterColorMatrixSlang},
        {"compand", "Square-root companding",
         "Digital numbers compressed as the square root of their fraction of full scale, as "
         "LROC's cameras compress 12 bits to 8.",
         {"M. S. Robinson et al., \"Lunar Reconnaissance Orbiter Camera (LROC) Instrument "
          "Overview\", Space Science Reviews 150, 2010"},
         Kind::Pixel, Domain::Digital,
         {{"inputBits", 12.0f, "Bits of the ADC's numbers"},
          {"outputBits", 8.0f, "Bits after companding"}},
         "Compand", shaders::kFilterCompandSlang},
    };
}

SensorFilterRegistry SensorFilterRegistry::WithStandard() {
    SensorFilterRegistry registry;
    for (SensorFilterModel& model : StandardSensorFilters()) {
        registry.Add(std::move(model));
    }
    return registry;
}

std::vector<SensorFilterStep> StepsOf(std::span<const SensorFilter> chain) {
    std::vector<SensorFilterStep> steps;
    for (const SensorFilter filter : chain) {
        steps.push_back({std::string(NameOf(filter)), {}});
    }
    return steps;
}

std::vector<SensorFilterStep> CheckFilterChain(std::span<const SensorFilterStep> chain,
                                               const SensorFilterRegistry& registry) {
    // The built-in filters keep their own rules; custom ones need their domain to exist.
    std::vector<SensorFilter> builtins;
    for (const SensorFilterStep& step : chain) {
        if (const std::optional<SensorFilter> builtin = step.Builtin()) {
            builtins.push_back(*builtin);
        } else if (registry.Find(step.type) == nullptr) {
            throw std::invalid_argument("Unknown sensor filter type '" + step.type + "'");
        }
    }
    CheckFilterChain(builtins);
    std::vector<SensorFilterStep> checked;
    if (chain.empty() || chain.front().Builtin() != SensorFilter::Lens) {
        checked.push_back({std::string(NameOf(SensorFilter::Lens)), {}});
    }
    const auto seen = [&checked](SensorFilter filter) {
        return std::any_of(checked.begin(), checked.end(), [filter](const SensorFilterStep& step) {
            return step.Builtin() == filter;
        });
    };
    for (const SensorFilterStep& step : chain) {
        if (step.Builtin() == SensorFilter::Lens && !checked.empty()) {
            throw std::invalid_argument("Sensor filter 'lens' must come first");
        }
        if (const SensorFilterModel* model = registry.Find(step.type)) {
            bool ready = true;
            switch (model->domain) {
            case FilterDomain::Spectrum:
                ready = !seen(SensorFilter::Qe) && !seen(SensorFilter::Cie);
                break;
            case FilterDomain::Electrons:
                ready = seen(SensorFilter::Qe);
                break;
            case FilterDomain::Digital:
                ready = seen(SensorFilter::Adc);
                break;
            case FilterDomain::Rgb:
                ready = seen(SensorFilter::Cie);
                break;
            case FilterDomain::Display:
                ready = seen(SensorFilter::Exposure);
                break;
            }
            if (!ready) {
                throw std::invalid_argument(
                    "Sensor filter '" + step.type +
                    (model->domain == FilterDomain::Spectrum
                         ? "' works on the bands, so it must come before 'qe' and 'cie'"
                         : "' runs before the filter producing what it works on"));
            }
        }
        checked.push_back(step);
    }
    return checked;
}

std::vector<std::string_view> VarsWrittenBy(std::span<const SensorFilter> chain) {
    std::vector<std::string_view> vars;
    for (const SensorFilter filter : chain) {
        switch (filter) {
        case SensorFilter::Lens:
            vars.insert(vars.end(),
                        {render_vars::kSpectralRadiance, render_vars::kDistanceToImagePlane,
                         render_vars::kDistanceToCamera, render_vars::kNormal, render_vars::kPrimId,
                         render_vars::kInstanceId});
            break;
        case SensorFilter::Qe:
            vars.push_back(render_vars::kElectrons);
            break;
        case SensorFilter::Noise:
            break;
        case SensorFilter::Adc:
            vars.push_back(render_vars::kRawDn);
            break;
        case SensorFilter::Cie:
            vars.push_back(render_vars::kHdrColor);
            break;
        case SensorFilter::Exposure:
            vars.push_back(render_vars::kLdrColor);
            break;
        }
    }
    return vars;
}

}  // namespace ernie

namespace ernie {

std::vector<std::string_view> RequiredVars(std::span<const std::string_view> requested) {
    if (requested.empty()) {
        return {kRenderVarNames.begin(), kRenderVarNames.end()};
    }
    // The sensor computes every var from the raster radiance in one pass, keeping what a var is
    // made from in registers, so a frame holds just the requested vars, in binding order.
    std::vector<std::string_view> vars;
    for (const std::string_view name : kRenderVarNames) {
        if (std::find(requested.begin(), requested.end(), name) != requested.end()) {
            vars.push_back(name);
        }
    }
    return vars;
}

}  // namespace ernie
