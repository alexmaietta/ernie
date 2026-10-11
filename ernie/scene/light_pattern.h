// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/shader_source.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
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
///     float emission(LightPatternInput input, LightPatternParams p);
///
/// gives the relative power per unit area of the tangent plane
/// (x / z, y / z along the laser's x and y, z along its beam) at
/// `input.t`, averaged over a pixel's footprint `input.footprint`
/// wide in it (so features finer than a pixel keep their power), at
/// the laser's `input.wavelength` (nm) and the frame's `input.time`
/// on the sensor clock (s). `input.beam` is the beam's own 1/e^2
/// radius there, w(z) / z, which a pattern of beamlets draws each one
/// with (`beamlet`); a continuous pattern can ignore it. A laser with
/// a sequence (`ernie:laser:sequence:count`) shows its frame
/// `input.frame` of `input.frames` (0 of 1 without one), the frames
/// its exposure spans weighted by their time in it.
///
/// Its scale does not matter: Ernie integrates it over |t| < `extent`
/// on the GPU, averaged over its sequence's frames, so the laser
/// emits its power's
/// `ernie:laser:efficiency` into it (its `ernie:laser:zeroOrder`
/// going into an undiffracted spot), unless `normalized` says it
/// integrates to 1 already. Parameter `i` is `p.get(i)`; the laser's
/// `ernie:laser:pattern:data` is `p.data(i)` (`p.dataCount()` of
/// them) and its `ernie:laser:pattern:image` `p.image(uv,
/// footprint)`.
///
///     import light_pattern;
///     public struct Stripes : ILightPattern {
///         public float emission(LightPatternInput input,
///                               LightPatternParams p) {
///             return 0.5 + 0.5 * cos(2.0 * 3.14159265 * input.t.x /
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
    /// beyond, and its shadow view covers that much. A built-in
    /// pattern's comes with each laser (`LightParams::Beam::extent`).
    double extent = 1.0;
    /// Its emission integrates to 1 over the whole tangent plane,
    /// whatever its parameters and footprint, so Ernie need not
    /// integrate it (the built-in spot, line and grid).
    bool normalized = false;

    /// Default value of every parameter, in order.
    std::vector<float> Defaults() const;
};

/// Light patterns a renderer can draw, by `LightPatternModel::id`.
class LightPatternRegistry {
  public:
    /// Most parameters a pattern may declare (four vec4s).
    static constexpr std::size_t kMaxParams = 16;

    /// An empty registry.
    LightPatternRegistry() = default;

    /// A registry with the built-in patterns, as `ILightPattern`s:
    /// - "spot": the beam itself, a Gaussian of `input.beam`;
    /// - "line": parameters half fan angle (radians) and profile (1
    ///   Gaussian, else flat in angle), the beam's Gaussian across;
    /// - "grid": parameters columns, rows and pitch along x and y
    ///   (radians), a beamlet at each;
    /// - "image": parameters tan of half the field of view along x
    ///   and y, the laser's image over it;
    /// - "dots": parameters dots per tile along x and y, tiles along
    ///   x and y, the jitter (cells), tan of half the field of view
    ///   along x and y, and the seed; a beamlet at each dot.
    static LightPatternRegistry WithBuiltins();

    /// Adds `model`; its Slang is compiled when the renderer next
    /// builds its shading (`RasterRenderer::AddLightPattern` compiles
    /// eagerly).
    /// @throws std::invalid_argument if the id is not an identifier,
    /// names a built-in pattern ("spot", "line", "grid", "image",
    /// "dots") or is taken; there are more than `kMaxParams`
    /// parameters; the extent is not positive; or the type name is
    /// not an identifier or the source does not mention it.
    void Add(LightPatternModel model);

    /// The pattern with id `id`, or null.
    const LightPatternModel* Find(std::string_view id) const;

    /// The index of the pattern with id `id` (its `lightPattern`
    /// case), or -1.
    int IndexOf(std::string_view id) const;

    std::span<const LightPatternModel> Models() const {
        return models_;
    }

  private:
    std::vector<LightPatternModel> models_;
};

/// The Slang module `light_pattern`: `LightPatternInput`,
/// `LightPatternParams` (sixteen floats `get(i)`, the laser's data
/// `data(i)` and image `image(uv, footprint)`), `ILightPattern`, and
/// helpers for patterns: `beamAverage`, `beamlet` and `normalBelow`.
std::string_view LightPatternSlang();

/// The Slang modules a shader drawing `registry`'s patterns compiles
/// with: `light_pattern`, each pattern's module, and
/// `light_patterns`, whose `lightPattern(index, input, p)` calls
/// pattern `index`'s `emission` (0 for none).
std::vector<ShaderModuleSource>
LightPatternModules(const LightPatternRegistry& registry);

/// A laser's pattern image (`ernie:laser:pattern:image`) as patterns
/// read it: one linear channel, the mean of its colour channels, its
/// rows top to bottom, and its mip levels after it down to 1 x 1,
/// each `max(1, size >> n)` on a side and averaging the texels above.
struct LightPatternImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> texels;  ///< Every level's, level 0 first.
};

/// Reads `path` (PNG, JPEG or OpenEXR; PNG and JPEG values taken as
/// linear, as a pattern's relative power) as a `LightPatternImage`.
/// @throws std::runtime_error if it cannot be read, or
/// std::invalid_argument if it is larger than 65535 on a side.
LightPatternImage
LoadLightPatternImage(const std::filesystem::path& path);

}  // namespace ernie
