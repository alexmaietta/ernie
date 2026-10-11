// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/shader_source.h"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// One parameter of a light pattern, which a LaserLight gives in
/// `ernie:laser:pattern:params` in this order.
struct LightPatternParam {
    std::string name;
    float defaultValue = 0.0f;
    std::string description;
};

/// A laser's pattern written in Slang: the relative power a laser
/// sends towards each direction, its watts, beam and shadows staying
/// Ernie's.
///
/// `slang` is a Slang module that may `import light_pattern;`
/// (`LightPatternSlang`) and defines a struct named `typeName`
/// implementing `ILightPattern`, whose
///
///     float emission(float2 t, float2 footprint, float wavelength,
///                    float time, LightPatternParams p);
///
/// gives the relative power per unit area of the tangent plane
/// (x / z, y / z along the laser's x and y, z along its beam) at `t`,
/// averaged over a footprint `footprint` wide in it (a pixel's, so
/// features finer than a pixel can keep their power), at the laser's
/// `wavelength` (nm) and the frame's `time` (s; 0 until frames carry
/// one). Its scale does not matter: Ernie integrates it over
/// |t| < `extent` on the GPU, so the laser emits its power's
/// `ernie:laser:efficiency` into it. Parameter `i` is `p.get(i)`.
///
///     import light_pattern;
///     struct Stripes : ILightPattern {
///         float emission(float2 t, float2 footprint,
///                        float wavelength, float time,
///                        LightPatternParams p) {
///             return 0.5 + 0.5 * cos(2.0 * 3.14159265 * t.x /
///                                    p.get(0));
///         }
///     }
struct LightPatternModel {
    std::string id;  ///< The `ernie:laser:pattern` naming it.
    std::string name;
    std::string description;
    std::vector<LightPatternParam> params;
    std::string typeName;  ///< The Slang struct implementing it.
    std::string slang;     ///< Its module's source.
    /// The largest |x / z| and |y / z| it lights: it emits nothing
    /// beyond, and its shadow view covers that much.
    double extent = 1.0;

    /// Default value of every parameter, in order.
    std::vector<float> Defaults() const;
};

/// Light patterns a renderer can draw, by `LightPatternModel::id`.
class LightPatternRegistry {
  public:
    /// Most parameters a pattern may declare (four vec4s).
    static constexpr std::size_t kMaxParams = 16;

    /// Adds `model`; its Slang is compiled when the renderer next
    /// builds its shading (`RasterRenderer::AddLightPattern` compiles
    /// eagerly).
    /// @throws std::invalid_argument if the id is not an identifier,
    /// names a built-in pattern ("spot", "line", "grid") or is taken;
    /// there are more than `kMaxParams` parameters; the extent is not
    /// positive; or the type name is not an identifier or the source
    /// does not mention it.
    void Add(LightPatternModel model);

    /// The pattern with id `id`, or null.
    const LightPatternModel* Find(std::string_view id) const;

    /// The index of the pattern with id `id` (its `customPattern`
    /// case), or -1.
    int IndexOf(std::string_view id) const;

    std::span<const LightPatternModel> Models() const {
        return models_;
    }

  private:
    std::vector<LightPatternModel> models_;
};

/// The Slang module `light_pattern`: `LightPatternParams` (sixteen
/// floats, `get(i)`) and `ILightPattern`.
std::string_view LightPatternSlang();

/// The Slang modules a shader drawing `registry`'s patterns compiles
/// with: `light_pattern`, each pattern's module, and
/// `light_patterns`, whose `customPattern(index, t, footprint,
/// wavelength, time, p)` calls pattern `index`'s `emission` (0 for
/// none).
std::vector<ShaderModuleSource>
LightPatternModules(const LightPatternRegistry& registry);

}  // namespace ernie
