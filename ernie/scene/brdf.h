// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/shader_source.h"

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// One scalar parameter of a reflectance model, with what a UI needs
/// to edit it.
struct BrdfParam {
    std::string name;   ///< Short symbol, e.g. "b" or "theta_bar".
    std::string label;  ///< UI label.
    std::string units;  ///< Empty when dimensionless.
    float defaultValue = 0.0f;
    float minValue = 0.0f;
    float maxValue = 1.0f;
    bool logarithmic = false;  ///< Edit on a logarithmic scale.
    std::string description;
    /// The first of three parameters (red, green, blue) of one colour
    /// input, `inputs:<name>`, which the shaders uplift to a spectrum
    /// at each band on the GPU (`BrdfInput::colours`, in the order of
    /// the model's colours). The next two are its green and blue.
    bool colour = false;
    /// The feature bit (`BrdfModel::features`) without which the
    /// model never reads this colour, so a scene none of whose
    /// materials have it does not uplift it; 0 for always.
    std::uint32_t feature = 0;
};

/// A reflectance model: shader source plus parameter metadata.
///
/// `slang` is a Slang module that may `import brdf;` (`BrdfSlang`)
/// and defines a struct named `typeName` implementing `IBrdf`, whose
/// `radianceFactor(BrdfInput s, BrdfParams p)` returns the radiance
/// factor I/F (pi times the bidirectional reflectance) for unit
/// incident irradiance. Parameter `i`, in the order of `params`, is
/// `p.get(i)`. `BrdfInput` carries cos(i), cos(e), cos(g), the cosine
/// of the azimuth between the incidence and emission planes, an rgb
/// albedo input whose meaning each model documents, and the variance
/// per axis of surface slopes finer than the renderer resolves (a
/// filtered normal map's), which models with a roughness add to it
/// (`widenedMeanSlope` for Hapke-like ones) and others may ignore.
///
/// A model only defines `radianceFactor`; every render backend
/// evaluates and samples it through the generic helpers of the `brdf`
/// module (`evaluateBrdf`, `sampleBrdf`, `brdfPdf`), so a model works
/// unchanged in the rasterizer and in ray and path tracers.
struct BrdfModel {
    std::string
        id;  ///< Stable identifier used in settings, e.g. "hapke".
    std::string name;  ///< Display name.
    std::string description;
    std::vector<std::string> references;
    std::vector<BrdfParam> params;
    std::string typeName;  ///< Slang struct implementing `IBrdf`,
                           ///< e.g. "Hapke".
    std::string slang;
    /// The layers a material of these parameters uses, as bits, for
    /// models whose layers cost nothing where no material has them:
    /// the renderer compiles the model with `BRDF_FEATURES` defined
    /// as the union over the scene's materials of it (so code under
    /// a bit none has falls away), and gives each material's own to
    /// `BrdfParams::features`. Empty for models without layers.
    std::function<std::uint32_t(std::span<const float>)> features;
    /// For models that scatter light beneath their surfaces
    /// (`BrdfRegistry::kSubsurfaceFeature`): how far, per linear RGB
    /// channel in scene units, for a material of these parameters;
    /// the renderer spreads its subsurface light over this radius,
    /// per band.
    std::function<glm::vec3(std::span<const float>)> subsurfaceRadius;

    /// Default value of every parameter, in order.
    std::vector<float> Defaults() const;
    /// The indices of its colour parameters' red ones, in order.
    std::vector<std::size_t> Colours() const;
};

/// Reflectance models available to a renderer, keyed by
/// `BrdfModel::id`.
class BrdfRegistry {
  public:
    /// Most parameters a model may declare: the first
    /// `kCoreParams`, packed with each material, and the rest in a
    /// buffer the shaders read only where the model asks for them
    /// (`BrdfParams::extended`).
    static constexpr std::size_t kMaxParams = 48;
    static constexpr std::size_t kCoreParams = 16;
    /// Most colour parameters (`BrdfParam::colour`) a model may
    /// declare.
    static constexpr std::size_t kMaxColours = 6;
    /// The feature bit (`BrdfModel::features`) of a material that
    /// reads the surface's tangent (`BrdfInput`'s frame), such as an
    /// anisotropic one: the renderer keeps the G-buffer's tangents
    /// only while a material has it.
    static constexpr std::uint32_t kTangentsFeature = 1u << 31;
    /// The feature bit of a material that lets light from behind it
    /// through (`IBrdf::transmission`): drawn after the opaque
    /// surfaces, in front of what it shows through.
    static constexpr std::uint32_t kTransmissionFeature = 1u << 30;
    /// The feature bit of a material that scatters light beneath its
    /// surface (`IBrdf::subsurface`): its subsurface light is spread
    /// across the image by `BrdfModel::subsurfaceRadius`.
    static constexpr std::uint32_t kSubsurfaceFeature = 1u << 29;

    /// An empty registry.
    BrdfRegistry() = default;

    /// A registry with the built-in models: "lambert",
    /// "lommel_seeliger", "lunar_lambert", "oren_nayar", "hapke",
    /// "preview_surface" and "open_pbr".
    static BrdfRegistry WithBuiltins();

    /// Adds `model`. Its shader source is only checked superficially
    /// here; it is compiled when a renderer first uses it
    /// (`RasterRenderer::AddBrdf` compiles eagerly).
    /// @throws std::invalid_argument if the id is empty, not an
    /// identifier or taken, there are more than `kMaxParams`
    /// parameters or `kMaxColours` colours, a colour lacks its green
    /// and blue, a parameter's range is empty or excludes its
    /// default, or its Slang source is empty or does not mention
    /// `typeName`.
    void Add(BrdfModel model);

    /// The model with `id`, or nullptr.
    const BrdfModel* Find(std::string_view id) const;

    std::span<const BrdfModel> Models() const {
        return models_;
    }

  private:
    std::vector<BrdfModel> models_;
};

/// The shared Slang module `brdf`: `BrdfInput`, `BrdfParams`,
/// `IBrdf`, `PI`, phase functions, and the backend-neutral helpers:
/// - `brdfInputOf(n, l, v, albedo)`: the `BrdfInput` of unit light
///   and view directions about a normal, in any frame;
/// - `evaluateBrdf(model, n, l, v, albedo, p)`: f_r cos(i),
///   `radianceFactor / pi`, the radiance reflected towards v per unit
///   radiance from l per steradian;
/// - `sampleBrdf(model, n, v, albedo, p, u)`: a cosine-weighted
///   incident direction with its density and weight (evaluation over
///   density) from uniform `u` in [0, 1)^2;
/// - `brdfPdf(n, l)`: that density, for multiple importance sampling;
/// - `cosineHemisphere` and `orthonormalBasis`, the sampling building
///   blocks.
std::string_view BrdfSlang();

/// Slang modules to compile before a shader that uses `model`:
/// `brdf`, then the model's module, named `BrdfModuleName(model)`.
std::vector<ShaderModuleSource> BrdfModules(const BrdfModel& model);

/// Name under which `BrdfModules` loads `model`'s module, e.g.
/// "brdf_hapke".
std::string BrdfModuleName(const BrdfModel& model);

/// Parameter values for `model`: `values` where present and in order,
/// the model's defaults for the rest, each clamped to its parameter's
/// range.
std::vector<float> ResolveBrdfParams(const BrdfModel& model,
                                     std::span<const float> values);

}  // namespace ernie
