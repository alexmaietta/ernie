// SPDX-License-Identifier: BSD-3-Clause
#include <stb_truetype.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>

#include "ernie/render/overlay/overlay_impl.h"
#include "ernie/util/error.h"

extern "C" {
extern const unsigned char kErnieFont_inter[];
extern const std::size_t kErnieFont_inter_size;
}

namespace ernie {
namespace {

constexpr uint32_t kGposKern = 0x6b65726e;

bool Printable(uint32_t code) {
    return (code >= 32 && code < 127) || (code >= 160 && code < 256);
}

int BitCount(uint32_t bits) {
    int count = 0;
    for (; bits != 0; bits &= bits - 1) {
        ++count;
    }
    return count;
}

class Table {
  public:
    Table(const unsigned char* data, size_t size, size_t at)
        : m_data(data), m_size(size), m_at(at) {}

    uint16_t U16(size_t offset) const {
        const size_t at = m_at + offset;
        return at + 2 <= m_size ? uint16_t((m_data[at] << 8) | m_data[at + 1]) : 0;
    }

    int16_t S16(size_t offset) const {
        return int16_t(U16(offset));
    }

    uint32_t U32(size_t offset) const {
        return (uint32_t{U16(offset)} << 16) | U16(offset + 2);
    }

    Table Subtable(size_t offset) const {
        return {m_data, m_size, m_at + offset};
    }

  private:
    const unsigned char* m_data;
    size_t m_size;
    size_t m_at;
};

std::optional<uint16_t> CoverageIndex(const Table& coverage, uint16_t glyph) {
    const uint16_t count = coverage.U16(2);
    if (coverage.U16(0) == 1) {
        for (uint16_t i = 0; i < count; ++i) {
            if (coverage.U16(4 + 2u * i) == glyph) {
                return i;
            }
        }
    } else if (coverage.U16(0) == 2) {
        for (uint16_t i = 0; i < count; ++i) {
            const Table range = coverage.Subtable(4 + 6u * i);
            if (glyph >= range.U16(0) && glyph <= range.U16(2)) {
                return uint16_t(range.U16(4) + glyph - range.U16(0));
            }
        }
    }
    return std::nullopt;
}

uint16_t GlyphClass(const Table& classes, uint16_t glyph) {
    if (classes.U16(0) == 1) {
        const uint16_t first = classes.U16(2);
        if (glyph >= first && glyph - first < classes.U16(4)) {
            return classes.U16(6 + 2u * (glyph - first));
        }
    } else if (classes.U16(0) == 2) {
        const uint16_t count = classes.U16(2);
        for (uint16_t i = 0; i < count; ++i) {
            const Table range = classes.Subtable(4 + 6u * i);
            if (glyph >= range.U16(0) && glyph <= range.U16(2)) {
                return range.U16(4);
            }
        }
    }
    return 0;
}

int16_t XAdvance(const Table& record, uint16_t format) {
    return (format & 4) != 0 ? record.S16(2 * size_t(BitCount(format & 3u))) : 0;
}

size_t RecordSize(uint16_t format) {
    return 2 * size_t(BitCount(format & 0xFFu));
}

struct PairTable {
    Table sub;
    std::vector<int32_t> index;
    std::vector<uint16_t> firstClasses;
    std::vector<uint16_t> secondClasses;
};

PairTable BuildPairTable(const Table& sub, const std::vector<uint16_t>& glyphs) {
    PairTable table{sub, {}, {}, {}};
    const Table coverage = sub.Subtable(sub.U16(2));
    for (const uint16_t glyph : glyphs) {
        const std::optional<uint16_t> index = CoverageIndex(coverage, glyph);
        table.index.push_back(index ? *index : -1);
        if (sub.U16(0) == 2) {
            table.firstClasses.push_back(GlyphClass(sub.Subtable(sub.U16(8)), glyph));
            table.secondClasses.push_back(GlyphClass(sub.Subtable(sub.U16(10)), glyph));
        }
    }
    return table;
}

std::optional<int16_t> PairSetAdjustment(const Table& set, uint16_t second, uint16_t format,
                                         size_t stride) {
    size_t low = 0;
    size_t high = set.U16(0);
    while (low < high) {
        const size_t middle = (low + high) / 2;
        const uint16_t glyph = set.U16(2 + middle * stride);
        if (glyph == second) {
            return XAdvance(set.Subtable(4 + middle * stride), format);
        }
        if (glyph < second) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return std::nullopt;
}

std::optional<int16_t> PairAdjustment(const PairTable& pairs, size_t a, size_t b, uint16_t second) {
    if (pairs.index[a] < 0) {
        return std::nullopt;
    }
    const Table& sub = pairs.sub;
    const auto index = uint16_t(pairs.index[a]);
    const uint16_t format = sub.U16(4);
    const size_t firstSize = RecordSize(format);
    const size_t secondSize = RecordSize(sub.U16(6));
    if (sub.U16(0) == 1) {
        if (index >= sub.U16(8)) {
            return std::nullopt;
        }
        return PairSetAdjustment(sub.Subtable(sub.U16(10 + 2u * index)), second, format,
                                 2 + firstSize + secondSize);
    }
    if (sub.U16(0) == 2) {
        const uint16_t first = pairs.firstClasses[a];
        const uint16_t next = pairs.secondClasses[b];
        const uint16_t firstCount = sub.U16(12);
        const uint16_t secondCount = sub.U16(14);
        if (first >= firstCount || next >= secondCount) {
            return std::nullopt;
        }
        return XAdvance(
            sub.Subtable(16 + (size_t{first} * secondCount + next) * (firstSize + secondSize)),
            format);
    }
    return std::nullopt;
}

std::vector<std::vector<Table>> FindKernLookups(const unsigned char* data, size_t size, int gpos) {
    std::vector<std::vector<Table>> lookups;
    if (gpos <= 0) {
        return lookups;
    }
    const Table table(data, size, size_t(gpos));
    const Table features = table.Subtable(table.U16(6));
    const Table list = table.Subtable(table.U16(8));
    std::vector<uint16_t> indices;
    for (uint16_t f = 0; f < features.U16(0); ++f) {
        const Table record = features.Subtable(2 + 6u * f);
        if (record.U32(0) != kGposKern) {
            continue;
        }
        const Table feature = features.Subtable(record.U16(4));
        for (uint16_t i = 0; i < feature.U16(2); ++i) {
            indices.push_back(feature.U16(4 + 2u * i));
        }
    }
    std::ranges::sort(indices);
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    for (const uint16_t index : indices) {
        if (index >= list.U16(0)) {
            continue;
        }
        const Table lookup = list.Subtable(list.U16(2 + 2u * index));
        std::vector<Table> subtables;
        for (uint16_t s = 0; s < lookup.U16(4); ++s) {
            Table sub = lookup.Subtable(lookup.U16(6 + 2u * s));
            uint16_t type = lookup.U16(0);
            if (type == 9) {
                type = sub.U16(2);
                sub = sub.Subtable(sub.U32(4));
            }
            if (type == 2) {
                subtables.push_back(sub);
            }
        }
        lookups.push_back(std::move(subtables));
    }
    return lookups;
}

void BakeGlyph(const stbtt_fontinfo& info, int glyph, float units, float cap, OverlayGlyph& out,
               std::vector<uint8_t>& field) {
    int advance = 0;
    int bearing = 0;
    stbtt_GetGlyphHMetrics(&info, glyph, &advance, &bearing);
    out.advance = float(advance) * units;
    int width = 0;
    int height = 0;
    int left = 0;
    int top = 0;
    unsigned char* baked =
        stbtt_GetGlyphSDF(&info, kOverlayFieldTexelsPerUnit * units, glyph,
                          int(kOverlayFieldPadding), static_cast<unsigned char>(kOverlayFieldEdge),
                          kOverlayFieldLevels, &width, &height, &left, &top);
    if (baked == nullptr) {
        return;
    }
    out.texel = uint32_t(field.size());
    out.width = uint32_t(width);
    out.height = uint32_t(height);
    out.x0 = float(left) / kOverlayFieldTexelsPerUnit;
    out.y0 = float(top) / kOverlayFieldTexelsPerUnit + cap;
    out.x1 = float(left + width) / kOverlayFieldTexelsPerUnit;
    out.y1 = float(top + height) / kOverlayFieldTexelsPerUnit + cap;
    field.insert(field.end(), baked, baked + size_t(width) * size_t(height));
    stbtt_FreeSDF(baked, nullptr);
}

std::vector<int8_t> BakeKerning(const stbtt_fontinfo& info, const std::array<int, 256>& glyphs,
                                float units) {
    std::vector<uint16_t> kerned;
    for (uint32_t code = 0; code < kOverlayKernCodeCount; ++code) {
        kerned.push_back(uint16_t(glyphs[code + kOverlayKernFirstCode]));
    }
    std::vector<std::vector<PairTable>> lookups;
    for (const std::vector<Table>& subtables :
         FindKernLookups(kErnieFont_inter, kErnieFont_inter_size, info.gpos)) {
        lookups.emplace_back();
        for (const Table& sub : subtables) {
            lookups.back().push_back(BuildPairTable(sub, kerned));
        }
    }
    std::vector<int8_t> kerning(size_t{kOverlayKernCodeCount} * kOverlayKernCodeCount, 0);
    for (uint32_t a = 0; a < kOverlayKernCodeCount; ++a) {
        for (uint32_t b = 0; b < kOverlayKernCodeCount; ++b) {
            int adjust = 0;
            for (const std::vector<PairTable>& subtables : lookups) {
                for (const PairTable& pairs : subtables) {
                    if (const std::optional<int16_t> value =
                            PairAdjustment(pairs, a, b, kerned[b])) {
                        adjust += *value;
                        break;
                    }
                }
            }
            kerning[size_t{a} * kOverlayKernCodeCount + b] =
                int8_t(std::clamp(std::lround(float(adjust) * units * 16.0f), -127L, 127L));
        }
    }
    return kerning;
}

OverlayFont BakeFont() {
    stbtt_fontinfo info;
    if (stbtt_InitFont(&info, kErnieFont_inter, 0) == 0) {
        throw Error("Overlay text's font failed to load");
    }
    int ascent = 0;
    int descent = 0;
    int gap = 0;
    stbtt_GetFontVMetrics(&info, &ascent, &descent, &gap);
    const float units = kOverlayLineUnits / float(ascent - descent);
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int capHeight = 0;
    stbtt_GetCodepointBox(&info, 'H', &x0, &y0, &x1, &capHeight);
    const float cap = float(capHeight) * units;
    OverlayFont font;
    font.textHeight = cap - float(descent) * units;
    std::array<int, 256> glyphs{};
    for (uint32_t code = 0; code < 256; ++code) {
        const int glyph = Printable(code) ? stbtt_FindGlyphIndex(&info, int(code)) : 0;
        if (glyph != 0) {
            glyphs[code] = glyph;
            BakeGlyph(info, glyph, units, cap, font.glyphs[code], font.field);
        }
    }
    for (uint32_t code = 0; code < 256; ++code) {
        if (glyphs[code] == 0) {
            font.glyphs[code] = font.glyphs['?'];
            glyphs[code] = glyphs['?'];
        }
    }
    font.kerning = BakeKerning(info, glyphs, units);
    return font;
}

}

const OverlayFont& OverlayTextFont() {
    static const OverlayFont font = BakeFont();
    return font;
}

std::vector<uint8_t> OverlayCodes(std::string_view text) {
    std::vector<uint8_t> codes;
    codes.reserve(text.size());
    for (size_t at = 0; at < text.size();) {
        const auto lead = static_cast<unsigned char>(text[at]);
        const size_t length = lead < 0x80         ? 1
                              : (lead >> 5) == 6  ? 2
                              : (lead >> 4) == 14 ? 3
                              : (lead >> 3) == 30 ? 4
                                                  : 0;
        uint32_t code = '?';
        if (length == 0 || at + length > text.size()) {
            ++at;
        } else {
            code = length == 1 ? lead : lead & (0x7Fu >> length);
            bool valid = true;
            for (size_t k = 1; k < length; ++k) {
                const auto next = static_cast<unsigned char>(text[at + k]);
                valid = valid && (next >> 6) == 2;
                code = (code << 6) | (next & 0x3Fu);
            }
            at += valid ? length : 1;
            if (!valid) {
                code = '?';
            }
        }
        codes.push_back(code == kOverlayNewline || Printable(code) ? uint8_t(code) : uint8_t('?'));
    }
    return codes;
}

}
