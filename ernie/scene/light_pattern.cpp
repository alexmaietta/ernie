// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/light_pattern.h"

#include "ernie/scene/tokens.h"
#include "ernie/util/image.h"

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
// A laser's pattern (ernie/scene/light_pattern.h): what a pattern is evaluated at, its parameters
// with the laser's data and image, the interface it implements, and helpers for patterns.

// Every laser's pattern data and image texels, at the offsets its LightPatternParams hold: first
// the images written as arrays on the GPU, which applications may write in place, and their mip
// levels, then what the host writes each frame.
[[vk::binding(29, 0)]] StructuredBuffer<float> lightPatternData;

// Where a pattern is evaluated: t = (x / z, y / z) on the tangent plane, a pixel's footprint
// there, the beam's own 1/e^2 radius there (w(z) / z), the laser's wavelength (nm), the frame's
// time on the sensor clock (s, as a float), and which of the `frames` patterns of the laser's
// sequence (ernie:laser:sequence:count) shows: 0 of 1 without one.
public struct LightPatternInput {
    public float2 t;
    public float2 footprint;
    public float2 beam;
    public float wavelength;
    public float time;
    public uint frame;
    public uint frames;
}

public struct LightPatternParams {
    public float4 values[4];
    // x: the offset of the laser's data in lightPatternData, y: their count, z: the offset of its
    // image, w: the image's width | height << 16 (0 without one).
    public uint4 layout;

    public float get(uint i) {
        return values[i / 4][i % 4];
    }

    public uint dataCount() {
        return layout.y;
    }

    // The laser's datum i (ernie:laser:pattern:data), 0 past its end.
    public float data(uint i) {
        return i < layout.y ? lightPatternData[layout.x + i] : 0.0;
    }

    public uint2 imageSize() {
        return uint2(layout.w & 0xffffu, layout.w >> 16);
    }

    // The laser's image (ernie:laser:pattern:image) at uv, (0, 0) its top left corner and (1, 1)
    // its bottom right, averaged over `footprint` of uv: bilinear in the mip level whose texels
    // are that wide, linear between levels; 0 outside it or without one.
    public float image(float2 uv, float2 footprint) {
        let size = imageSize();
        if (size.x == 0 || any(uv < 0.0) || any(uv > 1.0)) {
            return 0.0;
        }
        let levels = uint(firstbithigh(max(size.x, size.y))) + 1;
        let texels = max(footprint * float2(size), float2(1.0));
        let level = clamp(log2(max(texels.x, texels.y)), 0.0, float(levels - 1));
        let below = uint(level);
        let above = min(below + 1, levels - 1);
        return lerp(imageLevel(below, uv), imageLevel(above, uv), level - float(below));
    }

    float texel(uint index) {
        return lightPatternData[index];
    }

    float imageLevel(uint level, float2 uv) {
        var offset = layout.z;
        var size = imageSize();
        for (uint i = 0; i < level; ++i) {
            offset += size.x * size.y;
            size = max(size / 2, uint2(1));
        }
        let at = uv * float2(size) - 0.5;
        let corner = floor(at);
        let f = at - corner;
        let last = int2(size) - 1;
        let low = uint2(clamp(int2(corner), int2(0), last));
        let high = uint2(clamp(int2(corner) + 1, int2(0), last));
        let top = lerp(texel(offset + low.y * size.x + low.x),
                       texel(offset + low.y * size.x + high.x), f.x);
        let bottom = lerp(texel(offset + high.y * size.x + low.x),
                          texel(offset + high.y * size.x + high.x), f.x);
        return lerp(top, bottom, f.y);
    }
}

public interface ILightPattern {
    // Relative power per unit area of the tangent plane at input.t, averaged over
    // input.footprint of it; any scale.
    float emission(LightPatternInput input, LightPatternParams p);
}

// The error function, to 1.5e-7 (Abramowitz and Stegun 7.1.26).
//
// References:
// - M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions, 1964, 7.1.26.
public float erfApprox(float x) {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let poly =
        t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * exp(-x * x);
    return x < 0.0 ? -y : y;
}

// The standard normal distribution below x.
public float normalBelow(float x) {
    return 0.5 * (1.0 + erfApprox(x * 0.70710678));
}

// A Gaussian beam's 1D profile across one axis, 1/e^2 radius `w`, normalised to 1 over the line,
// averaged over `width` centred at `u`: exactly, from erf, so a beam narrower than the width
// keeps its power; the profile itself at `u` for a width of 0, a box for a radius of 0.
public float beamAverage(float u, float w, float width) {
    let radius = max(w, 1e-12);
    if (width < 1e-4 * radius) {
        return sqrt(2.0 / 3.14159265358979) / radius * exp(-2.0 * u * u / (radius * radius));
    }
    let k = sqrt(2.0) / radius;
    return 0.5 * (erfApprox(k * (u + 0.5 * width)) - erfApprox(k * (u - 0.5 * width))) / width;
}

// A beamlet of power 1 centred at `centre` on the tangent plane: the beam's Gaussian, averaged
// over the pixel's footprint.
public float beamlet(float2 centre, LightPatternInput input) {
    return beamAverage(input.t.x - centre.x, input.beam.x, input.footprint.x) *
           beamAverage(input.t.y - centre.y, input.beam.y, input.footprint.y);
}
)";

constexpr const char* kSpotSlang = R"(
import light_pattern;

// The beam itself.
public struct LaserSpot : ILightPattern {
    public float emission(LightPatternInput input, LightPatternParams p) {
        return beamlet(float2(0.0), input);
    }
}
)";

constexpr const char* kLineSlang = R"(
import light_pattern;

// A line fanned along x, half fan angle p0 (radians): flat in angle, or Gaussian with 1/e^2 at
// the fan's edges for p1 = 1; across it the beam's Gaussian.
public struct LaserLine : ILightPattern {
    public float emission(LightPatternInput input, LightPatternParams p) {
        // Along the line, in angle a: the pixel spans footprint cos^2 a of it, and the beam's own
        // width softens the fan's edges (Gaussian, as it is the 1/e^2 radius, sigma half that).
        let a = atan(input.t.x);
        let c2 = 1.0 / (1.0 + input.t.x * input.t.x);
        let pixel = input.footprint.x * c2;
        let sigma2 = 0.25 * input.beam.x * input.beam.x + pixel * pixel / 12.0;
        let halfFan = p.get(0);
        var density = 0.0;
        if (p.get(1) > 0.5) {
            let s = sqrt(0.25 * halfFan * halfFan + sigma2);
            density = exp(-0.5 * a * a / (s * s)) / (sqrt(2.0 * 3.14159265358979) * s);
        } else {
            let s = sqrt(max(sigma2, 1e-20));
            density = (normalBelow((a + halfFan) / s) - normalBelow((a - halfFan) / s)) /
                      (2.0 * halfFan);
        }
        // Per unit of t.x, da / dt = cos^2 a.
        return density * c2 * beamAverage(input.t.y, input.beam.y, input.footprint.y);
    }
}
)";

constexpr const char* kGridSlang = R"(
import light_pattern;

// p0 x p1 beamlets p2 and p3 radians apart along x and y, sharing the power.
public struct LaserGrid : ILightPattern {
    public float emission(LightPatternInput input, LightPatternParams p) {
        // The nearest beamlet along each axis and its neighbours, which a pixel can straddle.
        let cells = float2(p.get(0), p.get(1));
        let pitch = float2(p.get(2), p.get(3));
        let nearest = round(atan(input.t) / pitch + 0.5 * (cells - 1.0));
        var sum = 0.0;
        for (int i = -1; i <= 1; ++i) {
            let column = nearest.x + float(i);
            if (column < 0.0 || column > cells.x - 1.0) {
                continue;
            }
            let cx = tan((column - 0.5 * (cells.x - 1.0)) * pitch.x);
            let across = beamAverage(input.t.x - cx, input.beam.x, input.footprint.x);
            for (int j = -1; j <= 1; ++j) {
                let row = nearest.y + float(j);
                if (row < 0.0 || row > cells.y - 1.0) {
                    continue;
                }
                let cy = tan((row - 0.5 * (cells.y - 1.0)) * pitch.y);
                sum += across * beamAverage(input.t.y - cy, input.beam.y, input.footprint.y);
            }
        }
        return sum / (cells.x * cells.y);
    }
}
)";

constexpr const char* kDotsSlang = R"(
import light_pattern;

// A 32-bit integer hash, Wellons' lowbias32.
//
// References:
// - C. Wellons, "Prospecting for Hash Functions", 2018,
//   https://nullprogram.com/blog/2018/07/31/
uint dotHash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Where in its cell, from 0 to 1 along each axis, the dot of cell `cell` of a tile lies.
float2 dotPlace(uint2 cell, uint seed) {
    let h = dotHash(cell.x + dotHash(cell.y + dotHash(seed)));
    return float2(h & 0xffffu, h >> 16) / 65536.0;
}

// A seeded field of dots, as a depth camera's diffractive projector makes (the original Kinect
// repeats one pattern over 3 x 3 tiles): p0 x p1 dots per tile, one in each cell of a grid,
// strayed from its centre by up to p4 of a cell; p2 x p3 tiles of the same dots over
// |t.x| < p5, |t.y| < p6; seed p7. Each dot is a beamlet with its share of the power.
public struct LaserDots : ILightPattern {
    public float emission(LightPatternInput input, LightPatternParams p) {
        let count = float2(p.get(0), p.get(1));
        let cells = count * float2(p.get(2), p.get(3));
        let jitter = p.get(4);
        let half = float2(p.get(5), p.get(6));
        let seed = uint(p.get(7));
        let cell = 2.0 * half / cells;
        let share = 1.0 / (cells.x * cells.y);
        // A pixel or beam wider than a cell sees the dots' mean over the field.
        if (any(input.footprint + 2.0 * input.beam > cell)) {
            let cover = saturate((half - abs(input.t)) / max(input.footprint, 1e-9) + 0.5);
            return cover.x * cover.y / (4.0 * half.x * half.y);
        }
        // The dots of the cell holding t and of its neighbours, which a dot can stray into.
        let nearest = floor((input.t + half) / cell);
        var sum = 0.0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                let at = nearest + float2(i, j);
                if (any(at < 0.0) || any(at > cells - 1.0)) {
                    continue;
                }
                let local = uint2(at - count * floor(at / count));
                let offset = (dotPlace(local, seed) - 0.5) * jitter;
                sum += beamlet((at + 0.5 + offset) * cell - half, input);
            }
        }
        return sum * share;
    }
}
)";

constexpr const char* kImageSlang = R"(
import light_pattern;

// The laser's image over |t.x| < p0 and |t.y| < p1 (tangents of half its field of view), its top
// towards +y, averaged over the pixel's footprint and the beam's width. A sequence's frames are
// the image's equal bands, top to bottom.
public struct LaserImage : ILightPattern {
    public float emission(LightPatternInput input, LightPatternParams p) {
        let half = float2(p.get(0), p.get(1));
        let frames = float(max(input.frames, 1));
        // Within its frame's band, half a texel from the next, so its edges do not blend in
        // another frame's.
        let edge = 0.5 / float(max(p.imageSize().y, 1));
        let uv = float2(0.5 + 0.5 * input.t.x / half.x,
                        clamp((float(input.frame) + 0.5 - 0.5 * input.t.y / half.y) / frames,
                              float(input.frame) / frames + edge,
                              float(input.frame + 1) / frames - edge));
        let blur = sqrt(input.footprint * input.footprint + input.beam * input.beam);
        return p.image(uv, 0.5 * blur / half / float2(1.0, frames));
    }
}
)";

/// The ids built-in patterns hold, which `Add` refuses.
bool IsBuiltinId(std::string_view id) {
    return id == tokens::kSpot || id == tokens::kLine ||
           id == tokens::kGrid || id == tokens::kImage ||
           id == tokens::kDots;
}

}  // namespace

std::vector<float> LightPatternModel::Defaults() const {
    std::vector<float> values;
    for (const LightPatternParam& param : params) {
        values.push_back(param.defaultValue);
    }
    return values;
}

LightPatternRegistry LightPatternRegistry::WithBuiltins() {
    LightPatternRegistry registry;
    const auto add = [&](std::string_view id, std::string name,
                         std::string typeName, const char* slang,
                         std::vector<LightPatternParam> params,
                         bool normalized) {
        LightPatternModel model;
        model.id = std::string(id);
        model.name = std::move(name);
        model.typeName = std::move(typeName);
        model.slang = slang;
        model.params = std::move(params);
        model.normalized = normalized;
        registry.models_.push_back(std::move(model));
    };
    add(tokens::kSpot, "Spot", "LaserSpot", kSpotSlang, {}, true);
    add(tokens::kLine, "Line", "LaserLine", kLineSlang,
        {{"halfFan", 0.5f, "Half the fan angle, radians"},
         {"gaussian", 0.0f, "1 Gaussian along the line, else flat"}},
        true);
    add(tokens::kGrid, "Grid", "LaserGrid", kGridSlang,
        {{"columns", 1.0f, "Beamlets along x"},
         {"rows", 1.0f, "Beamlets along y"},
         {"pitchX", 0.0349f, "Radians between columns"},
         {"pitchY", 0.0349f, "Radians between rows"}},
        true);
    add(tokens::kImage, "Image", "LaserImage", kImageSlang,
        {{"halfX", 0.364f, "tan of half the field of view along x"},
         {"halfY", 0.268f, "tan of half the field of view along y"}},
        false);
    add(tokens::kDots, "Dots", "LaserDots", kDotsSlang,
        {{"columns", 64.0f, "Dots per tile along x"},
         {"rows", 48.0f, "Dots per tile along y"},
         {"tileColumns", 1.0f, "Tiles along x"},
         {"tileRows", 1.0f, "Tiles along y"},
         {"jitter", 0.9f, "How far a dot strays, in cells"},
         {"halfX", 0.364f, "tan of half the field of view along x"},
         {"halfY", 0.268f, "tan of half the field of view along y"},
         {"seed", 1.0f, "The seed the dots derive from"}},
        true);
    return registry;
}

void LightPatternRegistry::Add(LightPatternModel model) {
    if (!IsIdentifier(model.id)) {
        throw std::invalid_argument(
            "Light pattern id must be an identifier: '" + model.id +
            "'");
    }
    if (IsBuiltinId(model.id)) {
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
                 " pattern;\n        return pattern.emission(input, "
                 "p);\n    }\n";
    }
    dispatch += "public float lightPattern(uint index, "
                "LightPatternInput input, LightPatternParams p) {\n"
                "    switch (index) {\n" +
                cases +
                "    default:\n        return 0.0;\n    }\n}\n";
    modules.push_back({"light_patterns", dispatch});
    return modules;
}

LightPatternImage
LoadLightPatternImage(const std::filesystem::path& path) {
    const Image image = LoadImageFile(path, ImageEncoding::Linear);
    if (image.width > 0xffffu || image.height > 0xffffu) {
        throw std::invalid_argument(
            "Light pattern image is over 65535 texels on a side: " +
            path.string());
    }
    LightPatternImage pattern;
    pattern.width = image.width;
    pattern.height = image.height;
    for (const Image& level : MipChain(image, MipFilter::Average)) {
        for (const glm::vec4& texel : level.pixels) {
            pattern.texels.push_back((texel.r + texel.g + texel.b) /
                                     3.0f);
        }
    }
    return pattern;
}

LightPatternImage LightPatternImageOf(const Tensor& array) {
    if (IsGpu(array.Location().kind) ||
        array.Type() != DType::Float32 || array.Rank() != 2 ||
        array.Shape()[0] < 1 || array.Shape()[1] < 1 ||
        array.Shape()[0] > 0xffff || array.Shape()[1] > 0xffff) {
        throw std::invalid_argument(
            "A light pattern image array is a float32 (rows, cols) "
            "array on the host, at most 65535 on a side");
    }
    const Tensor values = array.Contiguous();
    Image image(static_cast<std::uint32_t>(array.Shape()[1]),
                static_cast<std::uint32_t>(array.Shape()[0]));
    const float* data = values.As<float>();
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        image.pixels[i] = glm::vec4(glm::vec3(data[i]), 1.0f);
    }
    LightPatternImage pattern;
    pattern.width = image.width;
    pattern.height = image.height;
    for (const Image& level : MipChain(image, MipFilter::Average)) {
        for (const glm::vec4& texel : level.pixels) {
            pattern.texels.push_back(texel.r);
        }
    }
    return pattern;
}

}  // namespace ernie
