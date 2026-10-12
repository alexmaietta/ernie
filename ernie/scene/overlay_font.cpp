// SPDX-License-Identifier: BSD-3-Clause
// Overlay text's font: Inter, embedded, baked once into signed
// distance fields with stb_truetype, with its pair kerning read from
// the font's GPOS table (stb_truetype reads none of a variable
// font's, whose values carry device offsets).
//
// References:
// - C. Green, "Improved Alpha-Tested Magnification for Vector
// Textures and Special Effects",
//   SIGGRAPH 2007 course notes: text drawn from distance fields, and
//   outlines from the same field.
// - OpenType specification, "GPOS — Glyph Positioning Table", lookup
// types 2 (pair adjustment) and
//   9 (extension),
//   https://learn.microsoft.com/typography/opentype/spec/gpos.
#include "ernie/scene/overlay.h"

#include <stb_truetype.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

extern "C" {
extern const unsigned char kErnieFont_inter[];
extern const std::size_t kErnieFont_inter_size;
}

namespace ernie {

namespace {

/// Whether the font draws Latin-1 code `c`.
bool Drawn(std::uint32_t c) {
    return (c >= 32 && c < 127) || (c >= 160 && c < 256);
}

/// A big-endian table of the font, read with bounds checks: past its
/// end reads 0.
class Table {
  public:
    Table(const unsigned char* data, std::size_t size, std::size_t at)
        : data_(data), size_(size), at_(at) {}

    std::uint16_t U16(std::size_t offset) const {
        const std::size_t p = at_ + offset;
        return p + 2 <= size_ ? static_cast<std::uint16_t>(
                                    (data_[p] << 8) | data_[p + 1])
                              : 0;
    }

    std::int16_t S16(std::size_t offset) const {
        return static_cast<std::int16_t>(U16(offset));
    }

    std::uint32_t U32(std::size_t offset) const {
        return (std::uint32_t{U16(offset)} << 16) | U16(offset + 2);
    }

    Table At(std::size_t offset) const {
        return {data_, size_, at_ + offset};
    }

  private:
    const unsigned char* data_;
    std::size_t size_;
    std::size_t at_;
};

/// Glyph `glyph`'s index in coverage table `coverage`, if covered.
std::optional<std::uint16_t> CoverageIndex(const Table& coverage,
                                           std::uint16_t glyph) {
    const std::uint16_t count = coverage.U16(2);
    if (coverage.U16(0) == 1) {
        for (std::uint16_t i = 0; i < count; ++i) {
            if (coverage.U16(4 + 2u * i) == glyph) {
                return i;
            }
        }
    } else if (coverage.U16(0) == 2) {
        for (std::uint16_t i = 0; i < count; ++i) {
            const Table range = coverage.At(4 + 6u * i);
            if (glyph >= range.U16(0) && glyph <= range.U16(2)) {
                return static_cast<std::uint16_t>(
                    range.U16(4) + glyph - range.U16(0));
            }
        }
    }
    return std::nullopt;
}

/// Glyph `glyph`'s class in class definition table `classes`.
std::uint16_t ClassOf(const Table& classes, std::uint16_t glyph) {
    if (classes.U16(0) == 1) {
        const std::uint16_t first = classes.U16(2);
        if (glyph >= first && glyph - first < classes.U16(4)) {
            return classes.U16(6 + 2u * (glyph - first));
        }
    } else if (classes.U16(0) == 2) {
        const std::uint16_t count = classes.U16(2);
        for (std::uint16_t i = 0; i < count; ++i) {
            const Table range = classes.At(4 + 6u * i);
            if (glyph >= range.U16(0) && glyph <= range.U16(2)) {
                return range.U16(4);
            }
        }
    }
    return 0;
}

/// The X advance of a value record of `format`, 0 without one: the
/// fields it has are 16 bits each, in order of their flags.
std::int16_t XAdvance(const Table& record, std::uint16_t format) {
    return (format & 4) != 0
               ? record.S16(2 *
                            static_cast<std::size_t>(std::popcount(
                                static_cast<unsigned>(format & 3))))
               : 0;
}

std::size_t RecordSize(std::uint16_t format) {
    return 2 * static_cast<std::size_t>(std::popcount(
                   static_cast<unsigned>(format & 0xff)));
}

/// A pair adjustment subtable (lookup type 2) as it applies to the
/// codes the font kerns: each first code's coverage index (-1 for
/// none) and class, each second code's class.
struct PairTable {
    Table sub;
    std::vector<std::int32_t> index;
    std::vector<std::uint16_t> class1;
    std::vector<std::uint16_t> class2;
};

PairTable MakePairTable(const Table& sub,
                        std::span<const std::uint16_t> glyphs) {
    PairTable table{sub, {}, {}, {}};
    const Table coverage = sub.At(sub.U16(2));
    for (const std::uint16_t glyph : glyphs) {
        const auto index = CoverageIndex(coverage, glyph);
        table.index.push_back(index ? *index : -1);
        if (sub.U16(0) == 2) {
            table.class1.push_back(
                ClassOf(sub.At(sub.U16(8)), glyph));
            table.class2.push_back(
                ClassOf(sub.At(sub.U16(10)), glyph));
        }
    }
    return table;
}

/// The adjustment of `pairs` to the advance of the first of codes
/// `a` and `b` (from kOverlayKernFirst) before the second, glyph
/// `second`, in font units; none when it does not cover the pair.
std::optional<std::int16_t> PairAdjustment(const PairTable& pairs,
                                           std::size_t a,
                                           std::size_t b,
                                           std::uint16_t second) {
    if (pairs.index[a] < 0) {
        return std::nullopt;
    }
    const Table& sub = pairs.sub;
    const auto index = static_cast<std::uint16_t>(pairs.index[a]);
    const std::uint16_t format1 = sub.U16(4);
    const std::uint16_t format2 = sub.U16(6);
    const std::size_t size1 = RecordSize(format1);
    const std::size_t size2 = RecordSize(format2);
    if (sub.U16(0) == 1) {
        if (index >= sub.U16(8)) {
            return std::nullopt;
        }
        const Table set = sub.At(sub.U16(10 + 2u * index));
        const std::uint16_t count = set.U16(0);
        const std::size_t stride = 2 + size1 + size2;
        // Records by second glyph, ascending.
        std::size_t low = 0;
        std::size_t high = count;
        while (low < high) {
            const std::size_t mid = (low + high) / 2;
            const std::uint16_t glyph = set.U16(2 + mid * stride);
            if (glyph == second) {
                return XAdvance(set.At(4 + mid * stride), format1);
            }
            if (glyph < second) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return std::nullopt;
    }
    if (sub.U16(0) == 2) {
        const std::uint16_t class1 = pairs.class1[a];
        const std::uint16_t class2 = pairs.class2[b];
        const std::uint16_t count1 = sub.U16(12);
        const std::uint16_t count2 = sub.U16(14);
        if (class1 >= count1 || class2 >= count2) {
            return std::nullopt;
        }
        return XAdvance(
            sub.At(16 + (std::size_t{class1} * count2 + class2) *
                            (size1 + size2)),
            format1);
    }
    return std::nullopt;
}

/// The pair adjustment subtables of the GPOS table's `kern` feature,
/// each lookup's in order, through extension lookups.
std::vector<std::vector<Table>>
KernLookups(const unsigned char* data, std::size_t size, int gpos) {
    std::vector<std::vector<Table>> lookups;
    if (gpos <= 0) {
        return lookups;
    }
    const Table table(data, size, static_cast<std::size_t>(gpos));
    const Table features = table.At(table.U16(6));
    const Table list = table.At(table.U16(8));
    std::vector<std::uint16_t> indices;
    for (std::uint16_t f = 0; f < features.U16(0); ++f) {
        const Table record = features.At(2 + 6u * f);
        if (record.U32(0) != 0x6b65726e) {  // 'kern'
            continue;
        }
        const Table feature = features.At(record.U16(4));
        for (std::uint16_t i = 0; i < feature.U16(2); ++i) {
            indices.push_back(feature.U16(4 + 2u * i));
        }
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()),
                  indices.end());
    for (const std::uint16_t index : indices) {
        if (index >= list.U16(0)) {
            continue;
        }
        const Table lookup = list.At(list.U16(2 + 2u * index));
        std::vector<Table> subtables;
        for (std::uint16_t s = 0; s < lookup.U16(4); ++s) {
            Table sub = lookup.At(lookup.U16(6 + 2u * s));
            std::uint16_t type = lookup.U16(0);
            if (type == 9) {
                type = sub.U16(2);
                sub = sub.At(sub.U32(4));
            }
            if (type == 2) {
                subtables.push_back(sub);
            }
        }
        lookups.push_back(std::move(subtables));
    }
    return lookups;
}

OverlayFont Bake() {
    stbtt_fontinfo info;
    if (stbtt_InitFont(&info, kErnieFont_inter, 0) == 0) {
        throw std::runtime_error("Overlay text's font fails to load");
    }
    int ascent = 0;
    int descent = 0;
    int gap = 0;
    stbtt_GetFontVMetrics(&info, &ascent, &descent, &gap);
    // Units per font unit, a line from the ascender to the descender;
    // the pen's y is the top of the capitals.
    const float units =
        kOverlayLineHeight / static_cast<float>(ascent - descent);
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int capHeight = 0;
    stbtt_GetCodepointBox(&info, 'H', &x0, &y0, &x1, &capHeight);
    const float cap = static_cast<float>(capHeight) * units;

    OverlayFont font;
    font.textHeight = cap - static_cast<float>(descent) * units;
    std::array<int, 256> glyphIndex{};
    for (std::uint32_t c = 0; c < 256; ++c) {
        const int glyph =
            Drawn(c)
                ? stbtt_FindGlyphIndex(&info, static_cast<int>(c))
                : 0;
        if (glyph == 0) {
            continue;
        }
        glyphIndex[c] = glyph;
        OverlayGlyph& g = font.glyphs[c];
        int advance = 0;
        int bearing = 0;
        stbtt_GetGlyphHMetrics(&info, glyph, &advance, &bearing);
        g.advance = static_cast<float>(advance) * units;
        int width = 0;
        int height = 0;
        int xoff = 0;
        int yoff = 0;
        unsigned char* field = stbtt_GetGlyphSDF(
            &info, kOverlayFieldTexels * units, glyph,
            kOverlayFieldPadding, 128, kOverlayFieldLevels, &width,
            &height, &xoff, &yoff);
        if (field == nullptr) {
            continue;
        }
        g.texel = static_cast<std::uint32_t>(font.field.size());
        g.width = static_cast<std::uint16_t>(width);
        g.height = static_cast<std::uint16_t>(height);
        g.box = glm::vec4(static_cast<float>(xoff),
                          static_cast<float>(yoff),
                          static_cast<float>(xoff + width),
                          static_cast<float>(yoff + height)) /
                    kOverlayFieldTexels +
                glm::vec4(0.0f, cap, 0.0f, cap);
        font.field.insert(font.field.end(), field,
                          field + std::size_t(width) *
                                      std::size_t(height));
        stbtt_FreeSDF(field, nullptr);
    }
    // What the font lacks is '?'.
    for (std::uint32_t c = 0; c < 256; ++c) {
        if (glyphIndex[c] == 0) {
            font.glyphs[c] = font.glyphs['?'];
            glyphIndex[c] = glyphIndex['?'];
        }
    }
    // Kerning: each lookup's first subtable covering a pair adjusts
    // it.
    std::vector<std::uint16_t> kerned;
    for (std::uint32_t c = 0; c < kOverlayKernCodes; ++c) {
        kerned.push_back(static_cast<std::uint16_t>(
            glyphIndex[c + kOverlayKernFirst]));
    }
    std::vector<std::vector<PairTable>> lookups;
    for (const std::vector<Table>& subtables : KernLookups(
             kErnieFont_inter, kErnieFont_inter_size, info.gpos)) {
        lookups.emplace_back();
        for (const Table& sub : subtables) {
            lookups.back().push_back(MakePairTable(sub, kerned));
        }
    }
    font.kerning.assign(
        std::size_t{kOverlayKernCodes} * kOverlayKernCodes, 0);
    for (std::uint32_t a = 0; a < kOverlayKernCodes; ++a) {
        for (std::uint32_t b = 0; b < kOverlayKernCodes; ++b) {
            int adjust = 0;
            for (const std::vector<PairTable>& subtables : lookups) {
                for (const PairTable& pairs : subtables) {
                    if (const auto value =
                            PairAdjustment(pairs, a, b, kerned[b])) {
                        adjust += *value;
                        break;
                    }
                }
            }
            font.kerning[std::size_t{a} * kOverlayKernCodes + b] =
                static_cast<std::int8_t>(std::clamp(
                    std::lround(static_cast<float>(adjust) * units *
                                16.0f),
                    -127L, 127L));
        }
    }
    return font;
}

}  // namespace

float OverlayFont::Kerning(std::uint8_t a, std::uint8_t b) const {
    if (a < kOverlayKernFirst || b < kOverlayKernFirst) {
        return 0.0f;
    }
    return static_cast<float>(
               kerning[std::size_t{a - kOverlayKernFirst} *
                           kOverlayKernCodes +
                       (b - kOverlayKernFirst)]) /
           16.0f;
}

const OverlayFont& OverlayTextFont() {
    static const OverlayFont font = Bake();
    return font;
}

std::vector<std::uint8_t> OverlayCodes(std::string_view text) {
    std::vector<std::uint8_t> codes;
    codes.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        // The sequence's length and its code point, '?' if malformed.
        const std::size_t length = lead < 0x80         ? 1
                                   : (lead >> 5) == 6  ? 2
                                   : (lead >> 4) == 14 ? 3
                                   : (lead >> 3) == 30 ? 4
                                                       : 0;
        std::uint32_t code = '?';
        if (length == 0 || i + length > text.size()) {
            ++i;
        } else {
            code = length == 1 ? lead : lead & (0x7f >> length);
            bool valid = true;
            for (std::size_t k = 1; k < length; ++k) {
                const auto next =
                    static_cast<unsigned char>(text[i + k]);
                valid = valid && (next >> 6) == 2;
                code = (code << 6) | (next & 0x3f);
            }
            i += valid ? length : 1;
            if (!valid) {
                code = '?';
            }
        }
        codes.push_back(code == '\n' || Drawn(code)
                            ? static_cast<std::uint8_t>(code)
                            : std::uint8_t{'?'});
    }
    return codes;
}

}  // namespace ernie
