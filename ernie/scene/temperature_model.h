// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/shader_source.h"
#include "ernie/util/spectrum.h"

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// One parameter of a temperature model, which a material gives in
/// `ernie:thermal:model:params` in this order.
struct TemperatureModelParam {
    std::string name;
    float defaultValue = 0.0f;
    std::string description;
};

/// A surface's temperature computed as it renders, written in Slang:
/// what a thermal camera sees of a body whose temperatures follow its
/// lighting, with no solver running beside the renderer (PANGU's
/// approach). A material names one in `ernie:thermal:model`; its
/// pixels take the model's temperature in place of the material's
/// own, though a mesh's solver temperatures, a prim's and a
/// temperature texture still win.
///
/// `slang` is a Slang module that may `import temperature_model;`
/// (`TemperatureModelSlang`) and defines a struct named `typeName`
/// implementing `ITemperatureModel`, whose
///
///     float kelvin(TemperatureInput input, TemperatureParams p);
///
/// gives the temperature at a pixel from `input`: its point and
/// outward surface normal (world, relative to the frame's origin);
/// the brightest distant light's direction, its irradiance over all
/// wavelengths facing it (W/m^2) and how much of it reaches the point
/// past shadows; the sunlight on the surface from every distant light
/// (W/m^2, shadowed); the material's solar absorptivity and thermal
/// emissivity; its own temperature (0 for none); and the frame's time
/// (s). Parameter `i` is `p.get(i)`, the material's
/// `ernie:thermal:model:data` `p.data(i)` (`p.dataCount()` of them).
///
///     import temperature_model;
///     public struct Warm : ITemperatureModel {
///         public float kelvin(TemperatureInput input,
///                             TemperatureParams p) {
///             return p.get(0) + p.get(1) * input.irradiance;
///         }
///     }
struct TemperatureModel {
    std::string id;  ///< The `ernie:thermal:model` naming it.
    std::string name;
    std::string description;
    std::vector<std::string> references;
    std::vector<TemperatureModelParam> params;
    std::string typeName;  ///< The Slang struct implementing it.
    std::string slang;     ///< Its module's source.
    /// The data of materials that give none, made when one first
    /// needs it (the built-in `diurnal`'s lunar table); none without.
    std::function<std::vector<float>()> defaultData = {};

    /// Default value of every parameter, in order.
    std::vector<float> Defaults() const;
};

/// Temperature models a renderer can run, by `TemperatureModel::id`.
class TemperatureModelRegistry {
  public:
    /// Most parameters a model may declare.
    static constexpr std::size_t kMaxParams = 16;

    /// An empty registry.
    TemperatureModelRegistry() = default;

    /// A registry with the built-in models:
    /// - "balance": zero-capacitance energy balance, each point at the
    ///   temperature where what it absorbs of the sunlight on it, and
    ///   its internal heat, equals what it radiates to its
    ///   surroundings, at once (a spacecraft's thin panels; PANGU's
    ///   spacecraft model). Parameters: absorptivity and emissivity
    ///   (below 0, the material's), the surroundings' temperature
    ///   (K), internal heat (W/m^2), and the faces radiating (1, or 2
    ///   for a panel open behind).
    /// - "diurnal": the temperature a table of latitude and local time
    ///   gives (PANGU's planetary LUTs), whose thermal inertia, from a
    ///   regolith model run offline, keeps the night warm and lags the
    ///   afternoon; latitude and local solar time from the surface
    ///   normal, the body's pole and the brightest distant light, so
    ///   slopes take the temperatures of the latitude and hour that
    ///   face the sun as they do. Parameters: the pole's x, y and z
    ///   (world). Data: latitudes, local times, then the kelvin row
    ///   by row (`DiurnalTable`); without, the Moon's
    ///   (`LunarDiurnalTable`).
    static TemperatureModelRegistry WithBuiltins();

    /// Adds `model`; its Slang is compiled when the renderer next
    /// builds its shading (`RasterRenderer::AddTemperatureModel`
    /// compiles eagerly).
    /// @throws std::invalid_argument if the id is not an identifier,
    /// names a built-in model or is taken; there are more than
    /// `kMaxParams` parameters; or the type name is not an identifier
    /// or the source does not mention it.
    void Add(TemperatureModel model);

    /// The model with id `id`, or null.
    const TemperatureModel* Find(std::string_view id) const;

    /// The index of the model with id `id`, or -1.
    int IndexOf(std::string_view id) const;

    std::span<const TemperatureModel> Models() const {
        return models_;
    }

  private:
    std::vector<TemperatureModel> models_;
};

/// The Slang module `temperature_model`: `TemperatureInput`,
/// `TemperatureParams` and `ITemperatureModel`.
std::string_view TemperatureModelSlang();

/// The Slang modules a shader running `registry`'s models compiles
/// with: `temperature_model`, each model's module, and
/// `temperature_models`, whose `temperatureModel(index, input, p)`
/// calls model `index`'s `kelvin` (0 for none).
std::vector<ShaderModuleSource>
TemperatureModelModules(const TemperatureModelRegistry& registry);

/// The share of sunlight a surface of `reflectance` absorbs: 1 minus
/// its reflectance weighted by the Sun's spectrum
/// (`SolarSpectralIrradiance`) from 250 to 2500 nm.
double SolarAbsorptivity(const Spectrum& reflectance);

/// A surface's hemispherical thermal emissivity at `kelvin`: its
/// spectral `emissivity` weighted by Planck's radiance from 2 to 100
/// um.
double ThermalEmissivity(const std::function<double(double)>& emissivity,
                         double kelvin);

}  // namespace ernie
