// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.overlay.kernels.common;

using namespace ernie;

uint AffixCode(uint words[4], uint index) {
    return (words[index / 4] >> ((index % 4) * 8)) & 0xFFu;
}

static const uint kMinus = 45;
static const uint kPoint = 46;
static const uint kZero = 48;

uint FormatNumber(OverlayBatch batch, float value, out uint codes[kOverlayTextChars]) {
    for (uint i = 0; i < kOverlayTextChars; ++i) {
        codes[i] = 0;
    }
    uint count = 0;
    for (uint i = 0; i < batch.prefixLength; ++i) {
        codes[count++] = AffixCode(batch.prefix, i);
    }
    uint64_t scale = 1;
    for (uint i = 0; i < batch.decimals; ++i) {
        scale *= 10;
    }
    const float scaled = abs(value) * float(scale) + 0.5f;
    if (isnan(value) || isinf(value) || scaled >= 1.8e19f) {
        if (value < 0) {
            codes[count++] = kMinus;
        }
        const bool undefined = isnan(value);
        codes[count++] = undefined ? 110u : 105u;
        codes[count++] = undefined ? 97u : 110u;
        codes[count++] = undefined ? 110u : 102u;
    } else {
        const uint64_t whole = uint64_t(scaled);
        uint64_t integer = whole / max(scale, uint64_t(1));
        const uint64_t fraction = whole % max(scale, uint64_t(1));
        if (value < 0 && whole > 0) {
            codes[count++] = kMinus;
        }
        uint digits[20];
        uint written = 0;
        do {
            digits[written++] = uint(integer % 10);
            integer /= 10;
        } while (integer > 0 && written < 20);
        while (written > 0) {
            codes[count++] = kZero + digits[--written];
        }
        if (batch.decimals > 0) {
            codes[count++] = kPoint;
            uint64_t place = scale / 10;
            for (uint i = 0; i < batch.decimals; ++i) {
                codes[count++] = kZero + uint((fraction / max(place, uint64_t(1))) % 10);
                place /= 10;
            }
        }
    }
    for (uint i = 0; i < batch.suffixLength; ++i) {
        codes[count++] = AffixCode(batch.suffix, i);
    }
    return count;
}

uint GatherCodes(ShapeRow shape, out uint codes[kOverlayTextChars]) {
    const OverlayBatch batch = shape.batch;
    if (batch.kind == uint(OverlayKind::Numbers)) {
        return FormatNumber(batch, shape.Scalar(), codes);
    }
    for (uint i = 0; i < kOverlayTextChars; ++i) {
        codes[i] = 0;
    }
    uint start = shape.row * batch.charStride;
    uint length = min(batch.maxChars, kOverlayTextChars);
    if ((batch.flags & kOverlayPackedText) != 0) {
        const uint packed = shape.Extra(0);
        start = packed & 0xFFFFFFu;
        length = min(packed >> 24, kOverlayTextChars);
    }
    uint count = 0;
    for (uint i = 0; i < length; ++i) {
        const uint code = ReadByte(batch.chars, start + i);
        if (code == 0) {
            break;
        }
        codes[count++] = code;
    }
    return count;
}

float2 PenPosition(OverlayParams p, uint codes[kOverlayTextChars], uint count, uint index,
                   out float2 size, out float lineWidth) {
    float2 pen = float2(0);
    float2 found = float2(0);
    float widest = 0;
    bool inLine = false;
    lineWidth = 0;
    for (uint i = 0; i < count; ++i) {
        if (i == index) {
            found = pen;
            inLine = true;
        }
        if (codes[i] == kOverlayNewline) {
            widest = max(widest, pen.x);
            if (inLine) {
                lineWidth = pen.x;
                inLine = false;
            }
            pen = float2(0, pen.y + kOverlayLineUnits);
        } else {
            const uint next = i + 1 < count ? codes[i + 1] : 0;
            pen.x += p.glyphs[codes[i]].advance;
            if (codes[i] >= kOverlayKernFirstCode && next >= kOverlayKernFirstCode) {
                const uint entry = (codes[i] - kOverlayKernFirstCode) * kOverlayKernCodeCount +
                                   (next - kOverlayKernFirstCode);
                pen.x += float(int(ReadByte(p.kerning, entry) << 24) >> 24) / 16;
            }
        }
    }
    if (inLine) {
        lineWidth = pen.x;
    }
    size = float2(max(widest, pen.x), pen.y + p.textHeight);
    return found;
}

bool ProjectText(OverlayParams p, ShapeRow shape, OverlayView view, OverlayCamera camera,
                 uint viewWorld, uint item, uint piece, float4 color, float size,
                 out OverlayRecord record) {
    record = {};
    const OverlayBatch batch = shape.batch;
    uint codes[kOverlayTextChars];
    const uint count = GatherCodes(shape, codes);
    const float4 haloColor = DecodeOverlayColor(batch.halo);
    const bool haloed = haloColor.w > 0;
    const uint perPass = max((batch.pieces - 1) / (haloed ? 2u : 1u), 1u);
    const bool haloPiece = haloed && piece >= 1 && piece <= perPass;
    const uint index = (piece - 1) % perPass;
    if (piece > 0 && (index >= count || codes[index] == kOverlayNewline)) {
        return false;
    }
    float2 block;
    float lineWidth;
    float2 pen = PenPosition(p, codes, count, piece > 0 ? index : count, block, lineWidth);
    const float scale = size / kOverlayLineUnits;
    const float pad = 0.25f * size;
    const float2 blockPixels = block * scale;
    float2 corner;
    float depth = 0;
    uint mode = batch.depth;
    float2 align = float2(batch.alignX, batch.alignY);
    if (batch.kind == uint(OverlayKind::Texts) || (batch.flags & kOverlayScreen) != 0) {
        float2 at = float2(batch.atX, batch.atY);
        float2 offset = float2(batch.offsetX, batch.offsetY + float(item) * 1.25f * size);
        if (batch.kind == uint(OverlayKind::Texts)) {
            at = shape.Point().xy;
            offset = shape.End().xy;
            align = shape.Rotation().xy;
        }
        const float2 anchor = at * float2(view.width, view.height) + offset;
        corner = anchor - align * (blockPixels + 2 * pad) + pad;
        mode = uint(OverlayDepth::Always);
    } else {
        const float3 point = RenderSpace(p, batch, viewWorld, shape.Point());
        if (any(isnan(point)) || any(isinf(point))) {
            return false;
        }
        depth = camera.model.ImagePlaneDistance(point);
        if (depth < camera.near) {
            return false;
        }
        const float2 anchor = Pixel(camera.model.Project(point).film, view);
        corner = anchor - align * blockPixels + (1 - 2 * align) * 0.6f * size;
    }
    pen.x += batch.justify * (block.x - lineWidth);
    record.depthA = depth;
    record.depthB = depth;
    if (piece == 0) {
        const float4 background = DecodeOverlayColor(batch.background);
        if (!(background.w > 0) || count == 0) {
            return false;
        }
        record.ax = corner.x - pad;
        record.ay = corner.y - pad;
        record.bx = corner.x + blockPixels.x + pad;
        record.by = corner.y + blockPixels.y + pad;
        record.color = EncodeOverlayColor(background);
        record.flags = mode | (uint(OverlayPiece::Box) << kOverlayKindShift);
        return true;
    }
    const OverlayGlyph glyph = p.glyphs[codes[index]];
    if (glyph.width == 0) {
        return false;
    }
    record.ax = corner.x + (pen.x + glyph.x0) * scale;
    record.ay = corner.y + (pen.y + glyph.y0) * scale;
    record.bx = corner.x + (pen.x + glyph.x1) * scale;
    record.by = corner.y + (pen.y + glyph.y1) * scale;
    record.depthB = asfloat(glyph.texel);
    record.color = EncodeOverlayColor(haloPiece ? float4(haloColor.xyz, haloColor.w * color.w) : color);
    record.flags = mode | (uint(OverlayPiece::Glyph) << kOverlayKindShift) |
                   (glyph.width << kOverlayFieldShift) | (glyph.height << 24);
    if (haloPiece) {
        const uint quarters = uint(clamp(batch.length * 4 + 0.5f, 0.0f, float(kOverlayHaloMask)));
        record.flags |= kOverlayHalo | (quarters << kOverlayHaloShift);
    }
    return true;
}
