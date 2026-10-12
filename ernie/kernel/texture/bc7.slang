// SPDX-License-Identifier: BSD-3-Clause
// BC7 encoding on the GPU, a thread a block (`GpuBlockEncoder`), in modes 1, 6 and 7 as bc7enc
// encodes them. Mode 6 has one subset of RGBA endpoints at 7 bits with a p-bit each and 4-bit
// indices. Modes 1 and 7 have two subsets in one of 64 partitions: mode 1 RGB endpoints at 6 bits
// with a p-bit shared by each subset's two and 3-bit indices, mode 7 RGBA endpoints at 5 bits with
// a p-bit each and 2-bit indices. Each block takes whichever of mode 6 and mode 1 (opaque blocks)
// or 7 (the rest) errs least, the two-subset mode fit in the few partitions whose subsets'
// bounding boxes alone fit them best (bc7enc's estimate_partition over every partition, keeping
// four where it keeps one).
//
// Each subset is fit as bc7enc's color_cell_compression at its uber level 1: endpoints spanning
// the texels along their principal axis (the covariance's dominant eigenvector, by power
// iteration as Castaño's CUDA encoder finds it), quantized trying both p-bits, each texel taking
// the nearer palette entry about its projection onto the endpoints' line; then the endpoints
// refit by least squares to those indices, and to them nudged inward, keeping whichever fit errs
// least.
//
// References:
// - R. Geldreich, "bc7enc_rdo" (bc7enc.cpp: color_cell_compression, find_optimal_solution,
//   evaluate_solution, compute_least_squares_endpoints_rgba, estimate_partition,
//   encode_bc7_block, and the partition and anchor tables), MIT or public domain.
//   https://github.com/richgel999/bc7enc_rdo
// - I. Castaño, "High Quality DXT Compression using CUDA", NVIDIA, 2007 (the principal axis by
//   power iteration).
// - Khronos, "Data Format Specification" 1.3, section 21.4 "BC7" (modes 1, 6 and 7's bit layouts,
//   the interpolation weights, anchor indices' implied high bits).

struct Constants {
    uint width;        // the image's texels across
    uint height;       // and down
    uint blocksAcross; // 4 x 4 blocks across
    uint blocks;       // blocks in all
    uint firstBlock;   // the dispatch's first block
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::binding(0)]] StructuredBuffer<uint> source; // RGBA8 texels, red in the low byte
[[vk::binding(1)]] RWStructuredBuffer<uint4> encoded;
[[vk::binding(2)]] StructuredBuffer<Constants> constants;

// BC7's 4-bit (mode 6), 3-bit (mode 1) and 2-bit (mode 7) interpolation weights, in 64ths.
static const uint kWeights4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };
static const uint kWeights3[8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
static const uint kWeights2[4] = { 0, 21, 43, 64 };

// The two-subset partitions: bit i set where texel i is in the second subset.
static const uint kPartitions[64] = {
    0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80, 0xC800, 0xFFEC, 0xFE80, 0xE800,
    0xFFE8, 0xFF00, 0xFFF0, 0xF000, 0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
    0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C, 0xAAAA, 0xF0F0, 0x5A5A, 0x33CC,
    0x3C3C, 0x55AA, 0x9696, 0xA55A, 0x73CE, 0x13C8, 0x324C, 0x3BDC, 0x6996, 0xC33C, 0x9966, 0x0660,
    0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6, 0x639C, 0x9336, 0x9CC6, 0x817E, 0xE718,
    0xCCF0, 0x0FCC, 0x7744, 0xEE22,
};
// The second subset's anchor texel in each partition.
static const uint kAnchors[64] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
};

// Two-subset partitions fit in full, the best by estimate: bc7enc fits one.
static const uint kTried = 4;

// A subset's fit in mode 1, 6 or 7: its quantized endpoints (mode 1: 6 bits a channel, RGB; mode
// 6: 7, RGBA; mode 7: 5, RGBA), their p-bits (both equal in mode 1, which shares one), and the
// squared error of its texels.
struct Fit {
    uint4 low;
    uint4 high;
    uint lowP;
    uint highP;
    uint error;
};

// The mode's endpoint bits, before its p-bit.
uint endpointBits(uint mode) {
    return mode == 6 ? 7 : mode == 1 ? 6 : 5;
}

// The mode's indices' weights.
uint indexCount(uint mode) {
    return mode == 6 ? 16 : mode == 1 ? 8 : 4;
}

uint weight(uint mode, uint index) {
    return mode == 6 ? kWeights4[index] : mode == 1 ? kWeights3[index] : kWeights2[index];
}

// An endpoint of `bits` bits and its p-bit expanded to 8 bits as decoders do: shifted to the top,
// its top bits repeated below (bc7enc's scale_color). Mode 1 has no alpha: 255.
uint4 expand(uint mode, uint4 endpoint, uint p) {
    const uint n = endpointBits(mode) + 1;
    uint4 v = ((endpoint << 1) | p) << (8 - n);
    v |= v >> n;
    return mode == 1 ? uint4(v.rgb, 255) : v;
}

// The endpoints nearest `low` and `high` (0 to 255) of either p-bit, as find_optimal_solution:
// each endpoint's p-bit chosen apart (modes 6 and 7), or the one both share (mode 1), by the
// squared error of the expanded endpoints against the unquantized.
void quantize(uint mode, float4 low, float4 high, inout Fit fit) {
    const float scale = float((2u << endpointBits(mode)) - 1) / 255.0;
    const int top = (1 << endpointBits(mode)) - 1;
    const uint channels = mode == 1 ? 3 : 4;
    float bestLow = 1e30;
    float bestHigh = 1e30;
    float bestBoth = 1e30;
    for (uint p = 0; p < 2; ++p) {
        const uint4 ql = uint4(clamp(int4((low * scale - float(p)) * 0.5 + 0.5), 0, top));
        const uint4 qh = uint4(clamp(int4((high * scale - float(p)) * 0.5 + 0.5), 0, top));
        float errorLow = 0.0;
        float errorHigh = 0.0;
        const float4 dl = float4(expand(mode, ql, p)) - low;
        const float4 dh = float4(expand(mode, qh, p)) - high;
        for (uint c = 0; c < channels; ++c) {
            errorLow += dl[c] * dl[c];
            errorHigh += dh[c] * dh[c];
        }
        if (mode == 1) {
            if (errorLow + errorHigh < bestBoth) {
                bestBoth = errorLow + errorHigh;
                fit.low = ql;
                fit.high = qh;
                fit.lowP = p;
                fit.highP = p;
            }
            continue;
        }
        if (errorLow < bestLow) {
            bestLow = errorLow;
            fit.low = ql;
            fit.lowP = p;
        }
        if (errorHigh < bestHigh) {
            bestHigh = errorHigh;
            fit.high = qh;
            fit.highP = p;
        }
    }
}

// The indices and error over the texels in `subset` (a bit a texel) of the fit's endpoints, as
// evaluate_solution: each texel projected onto the endpoints' line, then the nearer of the two
// palette entries about it, by squared RGBA error.
void evaluate(uint mode, inout Fit fit, int4 texels[16], uint subset, inout uint indices[16]) {
    const uint4 low = expand(mode, fit.low, fit.lowP);
    const uint4 high = expand(mode, fit.high, fit.highP);
    const uint count = indexCount(mode);
    int4 palette[16];
    for (uint i = 0; i < count; ++i) {
        const uint w = weight(mode, i);
        palette[i] = int4((low * (64 - w) + high * w + 32) >> 6);
    }
    const int4 d = int4(high) - int4(low);
    const float scale = float(count - 1) / (float(dot(d, d)) + 1.25e-6);
    fit.error = 0;
    for (uint i = 0; i < 16; ++i) {
        if ((subset & (1u << i)) == 0) {
            continue;
        }
        const int at = clamp(int(float(dot(texels[i] - int4(low), d)) * scale + 0.5), 1, int(count) - 1);
        const int4 e0 = palette[at - 1] - texels[i];
        const int4 e1 = palette[at] - texels[i];
        const uint error0 = uint(dot(e0, e0));
        const uint error1 = uint(dot(e1, e1));
        indices[i] = error0 < error1 ? uint(at - 1) : uint(at);
        fit.error += min(error0, error1);
    }
}

// The fit of float endpoints `low` and `high` (0 to 255): quantized, then evaluated.
Fit fitEndpoints(uint mode, float4 low, float4 high, int4 texels[16], uint subset, inout uint indices[16]) {
    Fit fit;
    quantize(mode, clamp(low, 0.0, 255.0), clamp(high, 0.0, 255.0), fit);
    evaluate(mode, fit, texels, subset, indices);
    return fit;
}

// The endpoints that best reproduce the subset's texels by least squares with `indices` held, as
// compute_least_squares_endpoints_rgba: the normal equations of texel = (1 - w) low + w high.
// False when every texel takes the same weight, leaving them undetermined.
bool leastSquares(uint mode, uint indices[16], int4 texels[16], uint subset, out float4 low, out float4 high) {
    float a = 0.0, b = 0.0, c = 0.0;
    float4 x = 0.0, y = 0.0;
    for (uint i = 0; i < 16; ++i) {
        if ((subset & (1u << i)) == 0) {
            continue;
        }
        const float w = float(weight(mode, indices[i])) / 64.0;
        a += (1.0 - w) * (1.0 - w);
        b += (1.0 - w) * w;
        c += w * w;
        x += (1.0 - w) * float4(texels[i]);
        y += w * float4(texels[i]);
    }
    const float det = a * c - b * b;
    low = 0.0;
    high = 0.0;
    if (abs(det) < 1e-6) {
        return false;
    }
    low = (c * x - b * y) / det;
    high = (a * y - b * x) / det;
    return true;
}

// The best fit of the texels in `subset` (a bit a texel), writing their indices.
Fit fitSubset(uint mode, int4 texels[16], uint subset, inout uint indices[16]) {
    float4 mean = 0.0;
    float n = 0.0;
    for (uint i = 0; i < 16; ++i) {
        if ((subset & (1u << i)) != 0) {
            mean += float4(texels[i]);
            n += 1.0;
        }
    }
    mean /= n;

    // The principal axis: the covariance's dominant eigenvector by power iteration, from
    // bc7enc's starting vector.
    float4x4 covariance = 0.0;
    for (uint i = 0; i < 16; ++i) {
        if ((subset & (1u << i)) != 0) {
            const float4 d = float4(texels[i]) - mean;
            covariance += float4x4(d * d.x, d * d.y, d * d.z, d * d.w);
        }
    }
    float4 axis = float4(0.9, 1.0, 0.7, 0.8);
    for (uint i = 0; i < 4; ++i) {
        axis = mul(covariance, axis);
        const float largest = max(max(abs(axis.x), abs(axis.y)), max(abs(axis.z), abs(axis.w)));
        axis = largest > 0.0 ? axis / largest : float4(0.0);
    }
    const float length2 = dot(axis, axis);
    axis = length2 > 0.0 ? axis * rsqrt(length2) : float4(0.0);

    // The texels' extent along it.
    float lowest = 1e30;
    float highest = -1e30;
    for (uint i = 0; i < 16; ++i) {
        if ((subset & (1u << i)) != 0) {
            const float t = dot(float4(texels[i]) - mean, axis);
            lowest = min(lowest, t);
            highest = max(highest, t);
        }
    }
    Fit best = fitEndpoints(mode, mean + axis * lowest, mean + axis * highest, texels, subset, indices);

    // Refit to the indices chosen, as long as it helps.
    for (uint pass = 0; pass < 2; ++pass) {
        float4 low, high;
        if (!leastSquares(mode, indices, texels, subset, low, high)) {
            break;
        }
        uint trial[16] = indices;
        const Fit refit = fitEndpoints(mode, low, high, texels, subset, trial);
        if (refit.error >= best.error) {
            break;
        }
        best = refit;
        indices = trial;
    }

    // bc7enc's uber level 1, somewhat like a cluster fit: refit with the lowest indices raised by
    // one, then the highest lowered, then both.
    uint lowestIndex = 15;
    uint highestIndex = 0;
    for (uint i = 0; i < 16; ++i) {
        if ((subset & (1u << i)) != 0) {
            lowestIndex = min(lowestIndex, indices[i]);
            highestIndex = max(highestIndex, indices[i]);
        }
    }
    const uint top = indexCount(mode) - 1;
    for (uint variant = 1; variant <= 3; ++variant) {
        uint nudged[16] = indices;
        for (uint i = 0; i < 16; ++i) {
            if ((variant & 1) != 0 && nudged[i] == lowestIndex && lowestIndex < top) {
                ++nudged[i];
            } else if ((variant & 2) != 0 && nudged[i] == highestIndex && highestIndex > 0) {
                --nudged[i];
            }
        }
        float4 low, high;
        if (!leastSquares(mode, nudged, texels, subset, low, high)) {
            continue;
        }
        uint trial[16] = indices;
        const Fit refit = fitEndpoints(mode, low, high, texels, subset, trial);
        if (refit.error < best.error) {
            best = refit;
            indices = trial;
        }
    }
    return best;
}

// The squared error of the subset's texels against a palette spanning their bounding box at the
// mode's weights, as color_cell_compression_est_mode1 and _mode7 estimate a subset.
uint estimate(uint mode, int4 texels[16], uint subset) {
    int4 low = 255;
    int4 high = 0;
    for (uint i = 0; i < 16; ++i) {
        if ((subset & (1u << i)) != 0) {
            low = min(low, texels[i]);
            high = max(high, texels[i]);
        }
    }
    const uint count = indexCount(mode);
    const int4 d = high - low;
    const float scale = float(count - 1) / (float(dot(d, d)) + 1.25e-6);
    uint error = 0;
    for (uint i = 0; i < 16; ++i) {
        if ((subset & (1u << i)) != 0) {
            const uint at = uint(clamp(int(float(dot(texels[i] - low, d)) * scale + 0.5), 0, int(count) - 1));
            const int w = int(weight(mode, at));
            const int4 e = ((low * (64 - w) + high * w + 32) >> 6) - texels[i];
            error += uint(dot(e, e));
        }
    }
    return error;
}

// Writes the low `count` bits of `value` at bit `offset` of `bits`.
void put(inout uint bits[4], inout uint offset, uint value, uint count) {
    const uint word = offset >> 5;
    const uint shift = offset & 31;
    bits[word] |= value << shift;
    if (shift + count > 32) {
        bits[word + 1] |= value >> (32 - shift);
    }
    offset += count;
}

// Mode 6's block: the mode's bit, each channel's two 7-bit endpoints, the p-bits, a 3-bit anchor
// index and fifteen 4-bit indices. An anchor index's high bit is implied 0, so the indices flip
// and the endpoints swap when it is set.
uint4 pack6(Fit fit, uint indices[16]) {
    if (indices[0] >= 8) {
        for (uint i = 0; i < 16; ++i) {
            indices[i] = 15 - indices[i];
        }
        const uint4 low = fit.low;
        fit.low = fit.high;
        fit.high = low;
        const uint lowP = fit.lowP;
        fit.lowP = fit.highP;
        fit.highP = lowP;
    }
    uint bits[4] = { 0, 0, 0, 0 };
    uint offset = 0;
    put(bits, offset, 1u << 6, 7);
    for (uint c = 0; c < 4; ++c) {
        put(bits, offset, fit.low[c], 7);
        put(bits, offset, fit.high[c], 7);
    }
    put(bits, offset, fit.lowP, 1);
    put(bits, offset, fit.highP, 1);
    for (uint i = 0; i < 16; ++i) {
        put(bits, offset, indices[i], i == 0 ? 3 : 4);
    }
    return uint4(bits[0], bits[1], bits[2], bits[3]);
}

// A two-subset block (mode 1 or 7): the mode's bits, the partition, each channel's four endpoints
// (the first subset's two, then the second's; mode 1 has no alpha), the p-bits (mode 1: a
// subset's; mode 7: an endpoint's), and the indices less the anchors' implied high bits (texel 0's
// and the second subset's anchor's).
uint4 packTwo(uint mode, uint partition, Fit fits[2], uint indices[16]) {
    const uint count = indexCount(mode);
    const uint anchors[2] = { 0, kAnchors[partition] };
    for (uint s = 0; s < 2; ++s) {
        if (indices[anchors[s]] >= count / 2) {
            for (uint i = 0; i < 16; ++i) {
                if (((kPartitions[partition] >> i) & 1) == s) {
                    indices[i] = count - 1 - indices[i];
                }
            }
            const uint4 low = fits[s].low;
            fits[s].low = fits[s].high;
            fits[s].high = low;
            const uint lowP = fits[s].lowP;
            fits[s].lowP = fits[s].highP;
            fits[s].highP = lowP;
        }
    }
    const uint bitsEach = endpointBits(mode);
    const uint indexBits = mode == 1 ? 3 : 2;
    uint bits[4] = { 0, 0, 0, 0 };
    uint offset = 0;
    put(bits, offset, 1u << mode, mode + 1);
    put(bits, offset, partition, 6);
    for (uint c = 0; c < (mode == 1 ? 3 : 4); ++c) {
        for (uint s = 0; s < 2; ++s) {
            put(bits, offset, fits[s].low[c], bitsEach);
            put(bits, offset, fits[s].high[c], bitsEach);
        }
    }
    for (uint s = 0; s < 2; ++s) {
        put(bits, offset, fits[s].lowP, 1);
        if (mode == 7) {
            put(bits, offset, fits[s].highP, 1);
        }
    }
    for (uint i = 0; i < 16; ++i) {
        put(bits, offset, indices[i], i == anchors[0] || i == anchors[1] ? indexBits - 1 : indexBits);
    }
    return uint4(bits[0], bits[1], bits[2], bits[3]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const Constants k = constants[0];
    const uint block = k.firstBlock + id.x;
    if (block >= k.blocks) {
        return;
    }
    const uint bx = block % k.blocksAcross;
    const uint by = block / k.blocksAcross;

    // The block's texels, clamped at the image's right and bottom edges.
    int4 texels[16];
    bool opaque = true;
    for (uint i = 0; i < 16; ++i) {
        const uint x = min(bx * 4 + (i & 3), k.width - 1);
        const uint y = min(by * 4 + (i >> 2), k.height - 1);
        const uint packed = source[y * k.width + x];
        texels[i] = int4(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, packed >> 24);
        opaque = opaque && texels[i].a == 255;
    }

    uint indices6[16];
    const Fit fit6 = fitSubset(6, texels, 0xFFFF, indices6);
    if (fit6.error == 0) {
        encoded[block] = pack6(fit6, indices6);
        return;
    }

    // Mode 1 (opaque) or 7, fit in the kTried partitions whose subsets' bounding boxes fit them
    // best.
    const uint mode = opaque ? 1 : 7;
    uint candidates[kTried];
    uint estimates[kTried];
    for (uint t = 0; t < kTried; ++t) {
        candidates[t] = 0;
        estimates[t] = 0xFFFFFFFF;
    }
    for (uint p = 0; p < 64; ++p) {
        uint error = estimate(mode, texels, ~kPartitions[p] & 0xFFFF) + estimate(mode, texels, kPartitions[p]);
        uint partition = p;
        for (uint t = 0; t < kTried; ++t) {
            if (error < estimates[t]) {
                const uint e = estimates[t];
                const uint c = candidates[t];
                estimates[t] = error;
                candidates[t] = partition;
                error = e;
                partition = c;
            }
        }
    }
    uint best = 0xFFFFFFFF;
    uint4 bestBlock = 0;
    for (uint t = 0; t < kTried; ++t) {
        const uint partition = candidates[t];
        uint indices[16];
        Fit fits[2];
        fits[0] = fitSubset(mode, texels, ~kPartitions[partition] & 0xFFFF, indices);
        fits[1] = fitSubset(mode, texels, kPartitions[partition], indices);
        const uint error = fits[0].error + fits[1].error;
        if (error < best) {
            best = error;
            bestBlock = packTwo(mode, partition, fits, indices);
        }
    }
    encoded[block] = best < fit6.error ? bestBlock : pack6(fit6, indices6);
}
