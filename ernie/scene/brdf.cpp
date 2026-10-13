// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/brdf.h"

#include "ernie/kernel/embedded_shaders.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

namespace {

const char* const kHapke2012 =
    "B. Hapke, \"Theory of Reflectance and Emittance Spectroscopy\", "
    "2nd ed., Cambridge University "
    "Press, 2012.";

BrdfModel Lambert() {
    return {"lambert",
            "Lambert",
            "Ideal diffuse reflector: I/F = A cos(i). The albedo "
            "input is the Lambert albedo A.",
            {"J. H. Lambert, \"Photometria\", 1760."},
            {},
            "Lambert",
            shaders::kBrdfLambertSlang};
}

BrdfModel LommelSeeliger() {
    return {"lommel_seeliger",
            "Lommel-Seeliger",
            "Single scattering by a particulate medium: I/F = (w / "
            "4) mu0 / (mu0 + mu) p(g). The "
            "albedo input is the single-scattering albedo w.",
            {kHapke2012},
            {{"xi", "Asymmetry", "", 0.0f, -0.95f, 0.95f, false,
              "Henyey-Greenstein asymmetry; negative values "
              "backscatter."}},
            "LommelSeeliger",
            shaders::kBrdfLommelSeeligerSlang};
}

BrdfModel LunarLambert() {
    return {"lunar_lambert",
            "Lunar-Lambert (McEwen)",
            "Blend of Lommel-Seeliger and Lambert with "
            "phase-dependent weight L(g) = 1 + a1 g + "
            "a2 g^2 + a3 g^3 (g in degrees). The albedo input is the "
            "I/F at normal incidence and "
            "emission. The default coefficients are a widely quoted "
            "lunar fit; check them against "
            "your data.",
            {"A. S. McEwen, \"Photometric functions for "
             "photoclinometry and other applications\", "
             "Icarus 92(2):298-311, 1991. "
             "https://doi.org/10.1016/0019-1035(91)90053-V"},
            {{"a1", "L(g) a1", "1/deg", -0.019f, -0.05f, 0.05f, false,
              "Linear coefficient."},
             {"a2", "L(g) a2", "1/deg^2", 0.000242f, -0.001f, 0.001f,
              false, "Quadratic coefficient."},
             {"a3", "L(g) a3", "1/deg^3", -0.00000146f, -0.00001f,
              0.00001f, false, "Cubic coefficient."}},
            "LunarLambert",
            shaders::kBrdfLunarLambertSlang};
}

BrdfModel OrenNayar() {
    return {"oren_nayar",
            "Oren-Nayar",
            "Rough diffuse surface of Lambertian V-cavities "
            "(qualitative model). The albedo input "
            "is the facet albedo rho; sigma = 0 reduces to Lambert.",
            {"M. Oren and S. K. Nayar, \"Generalization of Lambert's "
             "Reflectance Model\", "
             "SIGGRAPH '94, 239-246, 1994. "
             "https://doi.org/10.1145/192161.192213"},
            {{"sigma", "Roughness", "deg", 20.0f, 0.0f, 60.0f, false,
              "Standard deviation of facet slope angles."}},
            "OrenNayar",
            shaders::kBrdfOrenNayarSlang};
}

BrdfModel Hapke() {
    return {
        "hapke",
        "Hapke (Sato 2014)",
        "Hapke 2012 model with porosity, shadow-hiding opposition "
        "effect, double "
        "Henyey-Greenstein phase function and macroscopic roughness, "
        "as parameterised for the "
        "LROC WAC Hapke maps. The albedo input is the "
        "single-scattering albedo w. The default "
        "parameters are illustrative mid-range lunar values, not "
        "fitted ones.",
        {kHapke2012,
         "H. Sato, M. S. Robinson, B. Hapke, B. W. Denevi and A. K. "
         "Boyd, \"Resolved Hapke "
         "parameter maps of the Moon\", J. Geophys. Res. Planets "
         "119(8):1775-1805, 2014. "
         "https://doi.org/10.1002/2013JE004580",
         "B. Hapke, \"Bidirectional reflectance spectroscopy 3. "
         "Correction for macroscopic "
         "roughness\", Icarus 59(1):41-59, 1984. "
         "https://doi.org/10.1016/0019-1035(84)90054-X",
         "B. Hapke, \"Bidirectional reflectance spectroscopy 5. The "
         "coherent backscatter "
         "opposition effect and anisotropic scattering\", Icarus "
         "157(2):523-534, 2002.",
         "B. Hapke, \"Bidirectional reflectance spectroscopy 6. "
         "Effects of porosity\", Icarus "
         "195(2):918-926, 2008."},
        {{"b", "Phase width b", "", 0.25f, 0.0f, 0.99f, false,
          "Double Henyey-Greenstein lobe width."},
         {"c", "Phase weight c", "", 0.5f, -1.0f, 1.0f, false,
          "Double Henyey-Greenstein weight; positive values favour "
          "backscatter."},
         {"bs0", "SHOE amplitude B_S0", "", 1.0f, 0.0f, 6.0f, false,
          "Shadow-hiding opposition effect amplitude."},
         {"hs", "SHOE width h_S", "", 0.07f, 0.001f, 0.5f, true,
          "Shadow-hiding opposition effect angular width."},
         {"theta_bar", "Roughness theta-bar", "deg", 20.0f, 0.0f,
          45.0f, false, "Mean slope angle of unresolved roughness."},
         {"phi", "Filling factor phi", "", 0.3f, 0.0f, 0.74f, false,
          "Volume fraction occupied by particles; sets the porosity "
          "coefficient K."}},
        "Hapke",
        shaders::kBrdfHapkeSlang};
}

BrdfModel PreviewSurface() {
    return {
        "preview_surface",
        "UsdPreviewSurface",
        "UsdPreviewSurface's reflectance as glTF 2.0's "
        "metallic-roughness BRDF defines it: "
        "Lambert diffuse and GGX specular (height-correlated Smith "
        "visibility, Schlick "
        "Fresnel), the specular workflow's specularColor as f0, and "
        "a clearcoat lobe. The "
        "albedo input is the diffuseColor. Materials whose surface "
        "is a UsdPreviewSurface "
        "use it unless they name another model.",
        {"Pixar, \"UsdPreviewSurface Specification\", "
         "https://openusd.org/release/spec_usdpreviewsurface.html",
         "The Khronos Group, \"glTF 2.0 Specification\", Appendix B: "
         "BRDF Implementation, "
         "https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html",
         "B. Walter, S. R. Marschner, H. Li and K. E. Torrance, "
         "\"Microfacet Models for "
         "Refraction through Rough Surfaces\", EGSR 2007.",
         "E. Heitz, \"Understanding the Masking-Shadowing Function "
         "in Microfacet-Based "
         "BRDFs\", Journal of Computer Graphics Techniques 3(2), "
         "2014.",
         "C. Schlick, \"An Inexpensive BRDF Model for "
         "Physically-based Rendering\", Computer "
         "Graphics Forum 13(3):233-246, 1994."},
        {{"roughness", "Roughness", "", 0.5f, 0.0f, 1.0f, false,
          "Perceptual roughness; the GGX alpha is its square."},
         {"metallic", "Metallic", "", 0.0f, 0.0f, 1.0f, false,
          "0 dielectric, 1 metal tinted by the diffuse colour."},
         {"ior", "Index of refraction", "", 1.5f, 1.0f, 3.0f, false,
          "Sets a dielectric's reflectance at normal incidence."},
         {"useSpecularWorkflow", "Specular workflow", "", 0.0f, 0.0f,
          1.0f, false,
          "1 takes the reflectance at normal incidence from "
          "specularColor."},
         {"specularColorR", "Specular colour R", "", 0.0f, 0.0f, 1.0f,
          false,
          "inputs:specularColor's red; the model uses the colour's "
          "mean."},
         {"specularColorG", "Specular colour G", "", 0.0f, 0.0f, 1.0f,
          false, "inputs:specularColor's green."},
         {"specularColorB", "Specular colour B", "", 0.0f, 0.0f, 1.0f,
          false, "inputs:specularColor's blue."},
         {"clearcoat", "Clearcoat", "", 0.0f, 0.0f, 1.0f, false,
          "Weight of a second, colourless specular layer."},
         {"clearcoatRoughness", "Clearcoat roughness", "", 0.01f,
          0.0f, 1.0f, false,
          "The clearcoat's perceptual roughness."}},
        "PreviewSurface",
        shaders::kBrdfPreviewSurfaceSlang};
}

// OpenPBR's layers, as open_pbr.slang's kOpenPbrCoat, kOpenPbrFuzz
// and kOpenPbrThinFilm.
constexpr std::uint32_t kOpenPbrCoat = 1;
constexpr std::uint32_t kOpenPbrFuzz = 2;
constexpr std::uint32_t kOpenPbrThinFilm = 4;
constexpr std::uint32_t kOpenPbrTransmission = 8;
constexpr std::uint32_t kOpenPbrSubsurface = 16;

BrdfModel OpenPbr() {
    BrdfModel model{
        "open_pbr",
        "OpenPBR Surface",
        "OpenPBR Surface's reflectance: a dielectric base (an "
        "energy-preserving Oren-Nayar diffuse under a GGX lobe of "
        "exact Fresnel) mixed with an F82-tint metal, an optional "
        "thin film's interference at each band's wavelength, an "
        "optional coat and an optional fuzz, layered by albedo "
        "scaling. The albedo input is base_color. Layers no "
        "material of a scene has are compiled out, and a material "
        "without one skips it.",
        {"J. Andersson, A. Portsmouth et al., \"OpenPBR Surface "
         "Specification\", v1.1, Academy Software Foundation, 2024. "
         "https://academysoftwarefoundation.github.io/OpenPBR/",
         "J. Portsmouth, P. Kutz and S. Hill, \"EON: A practical "
         "energy-preserving rough diffuse BRDF\", Journal of "
         "Computer Graphics Techniques 14(1), 2025.",
         "N. Hoffman, \"Generalization of Adobe's Fresnel Model\", "
         "2023; P. Kutz, M. Hasan and P. Edmondson, \"Novel aspects "
         "of the Adobe Standard Material\", 2021.",
         "T. Zeltner, B. Burley and M. J.-Y. Chiang, \"Practical "
         "Multiple-Scattering Sheen Using Linearly Transformed "
         "Cosines\", SIGGRAPH 2022 Talks.",
         "L. Belcour and P. Barla, \"A Practical Extension to "
         "Microfacet Theory for the Modeling of Varying "
         "Iridescence\", ACM Transactions on Graphics 36(4), 2017.",
         "C. Kulla and A. Conty, \"Revisiting Physically Based "
         "Shading at Imageworks\", SIGGRAPH 2017 course.",
         "W. Jakob and J. Hanika, \"A Low-Dimensional Function "
         "Space for Efficient Spectral Upsampling\", Computer "
         "Graphics Forum 38(2), 2019 (its colours' spectra).",
         "B. Burley, \"Extending the Disney BRDF to a BSDF with "
         "Integrated Subsurface Scattering\", SIGGRAPH 2015 course, "
         "and P. H. Christensen and B. Burley, \"Approximate "
         "Reflectance Profiles for Efficient Subsurface "
         "Scattering\", Pixar, 2015 (the profile its subsurface "
         "light spreads by).",
         "J. Jimenez et al., \"Separable Subsurface Scattering\", "
         "Computer Graphics Forum 34(6), 2015 (spread in screen "
         "space)."},
        {{"base_weight", "Base weight", "", 1.0f, 0.0f, 1.0f, false,
          "Scales the base colour."},
         {"base_metalness", "Metalness", "", 0.0f, 0.0f, 1.0f, false,
          "0 dielectric, 1 metal of the base colour."},
         {"base_diffuse_roughness", "Diffuse roughness", "", 0.0f,
          0.0f, 1.0f, false,
          "The diffuse's Oren-Nayar roughness; 0 is Lambert."},
         {"specular_weight", "Specular weight", "", 1.0f, 0.0f, 1.0f,
          false, "Scales the specular lobe."},
         {"specular_roughness", "Specular roughness", "", 0.3f, 0.0f,
          1.0f, false,
          "Perceptual roughness; the GGX alpha is its square."},
         {"specular_ior", "Specular IOR", "", 1.5f, 1.0f, 3.0f, false,
          "The dielectric's index of refraction."},
         {"specular_color", "Specular colour", "", 1.0f, 0.0f, 1.0f,
          false,
          "Tints the dielectric's reflection, and the metal's near "
          "grazing (F82 tint).",
          true},
         {"specular_color_g", "Specular colour G", "", 1.0f, 0.0f,
          1.0f, false, "specular_color's green."},
         {"specular_color_b", "Specular colour B", "", 1.0f, 0.0f,
          1.0f, false, "specular_color's blue."},
         {"coat_weight", "Coat", "", 0.0f, 0.0f, 1.0f, false,
          "Coverage of a clear coat over the base."},
         {"fuzz_weight", "Fuzz", "", 0.0f, 0.0f, 1.0f, false,
          "Coverage of a fuzz (sheen) over everything."},
         {"thin_film_weight", "Thin film", "", 0.0f, 0.0f, 1.0f,
          false, "Coverage of a thin film on the base's specular."},
         {"coat_roughness", "Coat roughness", "", 0.0f, 0.0f, 1.0f,
          false, "The coat's perceptual roughness."},
         {"coat_ior", "Coat IOR", "", 1.6f, 1.0f, 3.0f, false,
          "The coat's index of refraction."},
         {"coat_darkening", "Coat darkening", "", 1.0f, 0.0f, 1.0f,
          false,
          "How much the light the coat reflects back down darkens "
          "the base."},
         {"fuzz_roughness", "Fuzz roughness", "", 0.5f, 0.0f, 1.0f,
          false, "The fuzz's roughness."},
         {"coat_color", "Coat colour", "", 1.0f, 0.0f, 1.0f, false,
          "What the coat lets through to the base and back.", true,
          kOpenPbrCoat},
         {"coat_color_g", "Coat colour G", "", 1.0f, 0.0f, 1.0f,
          false, "coat_color's green."},
         {"coat_color_b", "Coat colour B", "", 1.0f, 0.0f, 1.0f,
          false, "coat_color's blue."},
         {"fuzz_color", "Fuzz colour", "", 1.0f, 0.0f, 1.0f, false,
          "The fuzz's colour.", true, kOpenPbrFuzz},
         {"fuzz_color_g", "Fuzz colour G", "", 1.0f, 0.0f, 1.0f,
          false, "fuzz_color's green."},
         {"fuzz_color_b", "Fuzz colour B", "", 1.0f, 0.0f, 1.0f,
          false, "fuzz_color's blue."},
         {"thin_film_thickness", "Thin film thickness", "um", 0.5f,
          0.0f, 2.0f, false, "The film's thickness."},
         {"thin_film_ior", "Thin film IOR", "", 1.4f, 1.0f, 3.0f,
          false, "The film's index of refraction."},
         {"specular_roughness_anisotropy", "Specular anisotropy", "",
          0.0f, 0.0f, 1.0f, false,
          "Stretches the specular lobe along the surface's tangent, "
          "along which its texture coordinates' u grows."},
         {"transmission_weight", "Transmission", "", 0.0f, 0.0f, 1.0f,
          false,
          "How much of the dielectric base lets light through in "
          "place of its diffuse."},
         {"transmission_color", "Transmission colour", "", 1.0f, 0.0f,
          1.0f, false, "Tints the light let through.", true,
          kOpenPbrTransmission},
         {"transmission_color_g", "Transmission colour G", "", 1.0f,
          0.0f, 1.0f, false, "transmission_color's green."},
         {"transmission_color_b", "Transmission colour B", "", 1.0f,
          0.0f, 1.0f, false, "transmission_color's blue."},
         {"geometry_thin_walled", "Thin walled", "", 0.0f, 0.0f, 1.0f,
          false,
          "1 for a thin shell (a pane, a bubble), which lets light "
          "through unbent; 0 for a solid body, which refracts it."},
         {"subsurface_weight", "Subsurface", "", 0.0f, 0.0f, 1.0f,
          false,
          "How much of the dielectric base's diffuse light scatters "
          "beneath it, to leave it further on."},
         {"subsurface_color", "Subsurface colour", "", 0.8f, 0.0f,
          1.0f, false,
          "The albedo of the light scattered beneath it.", true,
          kOpenPbrSubsurface},
         {"subsurface_color_g", "Subsurface colour G", "", 0.8f, 0.0f,
          1.0f, false, "subsurface_color's green."},
         {"subsurface_color_b", "Subsurface colour B", "", 0.8f, 0.0f,
          1.0f, false, "subsurface_color's blue."},
         {"subsurface_radius", "Subsurface radius", "", 0.01f, 1e-5f,
          1000.0f, true,
          "How far light travels beneath it, in scene units (a "
          "centimetre in metres)."},
         {"subsurface_radius_scale_r", "Subsurface radius scale R",
          "", 1.0f, 0.0f, 1.0f, false,
          "The radius's share for red light: how much further red "
          "travels than blue in skin."},
         {"subsurface_radius_scale_g", "Subsurface radius scale G",
          "", 0.5f, 0.0f, 1.0f, false,
          "The radius's share for green."},
         {"subsurface_radius_scale_b", "Subsurface radius scale B",
          "", 0.25f, 0.0f, 1.0f, false,
          "The radius's share for blue."}},
        "OpenPbr",
        shaders::kBrdfOpenPbrSlang};
    // A layer where its weight is above 0.
    model.features = [](std::span<const float> p) {
        std::uint32_t bits = 0;
        bits |= p.size() > 9 && p[9] > 0.0f ? kOpenPbrCoat : 0u;
        bits |= p.size() > 10 && p[10] > 0.0f ? kOpenPbrFuzz : 0u;
        bits |= p.size() > 11 && p[11] > 0.0f ? kOpenPbrThinFilm : 0u;
        // Transmission draws it in front of what it shows through.
        bits |= p.size() > 25 && p[25] > 0.0f
                    ? kOpenPbrTransmission |
                          BrdfRegistry::kTransmissionFeature
                    : 0u;
        // Subsurface light is spread across the image.
        bits |= p.size() > 30 && p[30] > 0.0f
                    ? kOpenPbrSubsurface |
                          BrdfRegistry::kSubsurfaceFeature
                    : 0u;
        // Anisotropy reads the surface's tangents.
        bits |= p.size() > 24 && p[24] > 0.0f
                    ? BrdfRegistry::kTangentsFeature
                    : 0u;
        return bits;
    };
    // subsurface_radius times its scale per channel.
    model.subsurfaceRadius = [](std::span<const float> p) {
        return p.size() > 37 ? p[34] * glm::vec3(p[35], p[36], p[37])
                             : glm::vec3(0.0f);
    };
    return model;
}

}  // namespace

std::vector<float> BrdfModel::Defaults() const {
    std::vector<float> values;
    values.reserve(params.size());
    for (const BrdfParam& param : params) {
        values.push_back(param.defaultValue);
    }
    return values;
}

std::vector<std::size_t> BrdfModel::Colours() const {
    std::vector<std::size_t> colours;
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (params[i].colour) {
            colours.push_back(i);
        }
    }
    return colours;
}

BrdfRegistry BrdfRegistry::WithBuiltins() {
    BrdfRegistry registry;
    registry.Add(Lambert());
    registry.Add(LommelSeeliger());
    registry.Add(LunarLambert());
    registry.Add(OrenNayar());
    registry.Add(Hapke());
    registry.Add(PreviewSurface());
    registry.Add(OpenPbr());
    return registry;
}

void BrdfRegistry::Add(BrdfModel model) {
    const bool identifier =
        !model.id.empty() &&
        std::isalpha(static_cast<unsigned char>(model.id[0])) != 0 &&
        std::all_of(model.id.begin(), model.id.end(), [](char ch) {
            return std::isalnum(static_cast<unsigned char>(ch)) !=
                       0 ||
                   ch == '_';
        });
    if (!identifier) {
        throw std::invalid_argument(
            "BRDF id must be an identifier: '" + model.id + "'");
    }
    if (Find(model.id) != nullptr) {
        throw std::invalid_argument("BRDF id already registered: " +
                                    model.id);
    }
    if (model.params.size() > kMaxParams) {
        throw std::invalid_argument(
            "BRDF " + model.id + " has more than " +
            std::to_string(kMaxParams) + " parameters");
    }
    const std::vector<std::size_t> colours = model.Colours();
    if (colours.size() > kMaxColours) {
        throw std::invalid_argument(
            "BRDF " + model.id + " has more than " +
            std::to_string(kMaxColours) + " colours");
    }
    for (const std::size_t colour : colours) {
        if (colour + 2 >= model.params.size() ||
            model.params[colour + 1].colour ||
            model.params[colour + 2].colour) {
            throw std::invalid_argument(
                "BRDF " + model.id + " colour " +
                model.params[colour].name +
                " lacks its green and blue parameters");
        }
    }
    for (const BrdfParam& param : model.params) {
        if (!(param.minValue < param.maxValue) ||
            param.defaultValue < param.minValue ||
            param.defaultValue > param.maxValue ||
            (param.logarithmic && !(param.minValue > 0.0f))) {
            throw std::invalid_argument(
                "BRDF " + model.id + " parameter " + param.name +
                " has an invalid range or default");
        }
    }
    if (model.slang.empty() || model.typeName.empty() ||
        model.slang.find(model.typeName) == std::string::npos) {
        throw std::invalid_argument(
            "BRDF " + model.id + " Slang source does not define '" +
            model.typeName + "'");
    }
    models_.push_back(std::move(model));
}

const BrdfModel* BrdfRegistry::Find(std::string_view id) const {
    const auto it = std::find_if(
        models_.begin(), models_.end(),
        [id](const BrdfModel& model) { return model.id == id; });
    return it == models_.end() ? nullptr : &*it;
}

std::string_view BrdfSlang() {
    return shaders::kBrdfSlang;
}

std::string BrdfModuleName(const BrdfModel& model) {
    return "brdf_" + model.id;
}

std::vector<ShaderModuleSource> BrdfModules(const BrdfModel& model) {
    return {{"brdf", std::string(BrdfSlang())},
            {BrdfModuleName(model), model.slang}};
}

std::vector<float> ResolveBrdfParams(const BrdfModel& model,
                                     std::span<const float> values) {
    std::vector<float> resolved = model.Defaults();
    for (std::size_t i = 0; i < resolved.size() && i < values.size();
         ++i) {
        resolved[i] = std::clamp(values[i], model.params[i].minValue,
                                 model.params[i].maxValue);
    }
    return resolved;
}

}  // namespace ernie
