// SPDX-License-Identifier: BSD-3-Clause

import ernie.pipeline;

using namespace ernie;

static const int kPsfRadius = 3;
static const int kDenoiseRadius = 2;

struct DemosaicNode<let method : uint> {
    static void Evaluate(PointContext context, Neighborhood<Digital> raw, out Digital full) {
        const uint channels = context.Detector().channels;
        Channels result = Channels(max(channels, 1u));
        if (!raw.Mosaic()) {
            full = raw.Center();
            return;
        }
        float sums[kMostSignalChannels];
        float counts[kMostSignalChannels];
        [ForceUnroll]
        for (uint c = 0; c < kMostSignalChannels; ++c) {
            sums[c] = 0;
            counts[c] = 0;
        }
        const uint center = raw.Site(int2(0));
        [ForceUnroll]
        for (int dy = -1; dy <= 1; ++dy) {
            [ForceUnroll]
            for (int dx = -1; dx <= 1; ++dx) {
                const uint site = raw.Site(int2(dx, dy));
                const float value = raw.At(int2(dx, dy)).Get()[0];
                [ForceUnroll]
                for (uint c = 0; c < kMostSignalChannels; ++c) {
                    if (c == site) {
                        sums[c] += value;
                        counts[c] += 1;
                    }
                }
            }
        }
        [ForceUnroll]
        for (uint c = 0; c < kMostSignalChannels; ++c) {
            if (c < result.count) {
                result[c] = c == center ? raw.Center().Get()[0] : sums[c] / max(counts[c], 1.0f);
            }
        }
        full = Digital.Make(result);
    }
}

struct GaussianPsfNode {
    static void Evaluate(PointContext context, Neighborhood<ElectronRate> input,
                         out ElectronRate output) {
        const float sigma = max(context.Parameter(0), 1e-3f);
        const bool mosaic = input.Mosaic();
        const uint site = input.Site(int2(0));
        Channels sums = Channels(input.Center().Get().count);
        float total = 0;
        for (int dy = -kPsfRadius; dy <= kPsfRadius; ++dy) {
            for (int dx = -kPsfRadius; dx <= kPsfRadius; ++dx) {
                if (mosaic && input.Site(int2(dx, dy)) != site) {
                    continue;
                }
                const float weight = exp(-float(dx * dx + dy * dy) / (2 * sigma * sigma));
                const Channels sample = input.At(int2(dx, dy)).Get();
                [ForceUnroll]
                for (uint c = 0; c < kMostSignalChannels; ++c) {
                    sums[c] += weight * sample[c];
                }
                total += weight;
            }
        }
        [ForceUnroll]
        for (uint c = 0; c < kMostSignalChannels; ++c) {
            sums[c] /= total;
        }
        output = ElectronRate.Make(sums);
    }
}

struct DenoiseNode {
    static void Evaluate(PointContext context, Neighborhood<LinearColor> input,
                         out LinearColor output) {
        const float space = max(context.Parameter(0), 1e-3f);
        const float range = max(context.Parameter(1), 1e-6f);
        const Channels center = input.Center().Get();
        Channels sums = Channels(center.count);
        float total = 0;
        for (int dy = -kDenoiseRadius; dy <= kDenoiseRadius; ++dy) {
            for (int dx = -kDenoiseRadius; dx <= kDenoiseRadius; ++dx) {
                const Channels sample = input.At(int2(dx, dy)).Get();
                float distance = 0;
                [ForceUnroll]
                for (uint c = 0; c < kMostSignalChannels; ++c) {
                    const float d = sample[c] - center[c];
                    distance += d * d;
                }
                const float weight = exp(-float(dx * dx + dy * dy) / (2 * space * space) -
                                         distance / (2 * range * range));
                [ForceUnroll]
                for (uint c = 0; c < kMostSignalChannels; ++c) {
                    sums[c] += weight * sample[c];
                }
                total += weight;
            }
        }
        [ForceUnroll]
        for (uint c = 0; c < kMostSignalChannels; ++c) {
            sums[c] /= total;
        }
        output = LinearColor.Make(sums);
    }
}

struct VignetteNode {
    static void Evaluate(PointContext context, ElectronRate input, out ElectronRate output) {
        const float2 size = float2(context.element.capture->width, context.element.capture->height);
        const float2 offset = (float2(context.element.coord) + 0.5f) - 0.5f * size;
        const float radius2 = dot(offset, offset) / max(dot(0.5f * size, 0.5f * size), 1e-6f);
        const float gain = max(1 - context.Parameter(0) * radius2, 0.0f);
        Channels channels = input.Get();
        [ForceUnroll]
        for (uint c = 0; c < kMostSignalChannels; ++c) {
            channels[c] *= gain;
        }
        output = ElectronRate.Make(channels);
    }
}

struct BlackLevelNode {
    static void Evaluate(PointContext context, Digital input, out Digital output) {
        const float black = context.Detector().blackLevel;
        Channels channels = input.Get();
        [ForceUnroll]
        for (uint c = 0; c < kMostSignalChannels; ++c) {
            channels[c] = max(channels[c] - black, 0.0f);
        }
        output = Digital.Make(channels);
    }
}

struct WhiteBalanceNode {
    static void Evaluate(PointContext context, LinearColor input, out LinearColor output) {
        Channels channels = input.Get();
        [ForceUnroll]
        for (uint c = 0; c < 3; ++c) {
            channels[c] *= context.Parameter(c);
        }
        output = LinearColor.Make(channels);
    }
}

struct CcmNode {
    static void Evaluate(PointContext context, LinearColor input, out LinearColor output) {
        const Channels channels = input.Get();
        const float3 color = float3(channels[0], channels[1], channels[2]);
        Channels result = Channels(3);
        [ForceUnroll]
        for (uint row = 0; row < 3; ++row) {
            const float3 weights = float3(context.Parameter(row * 3), context.Parameter(row * 3 + 1),
                                          context.Parameter(row * 3 + 2));
            result[row] = dot(weights, color);
        }
        output = LinearColor.Make(result);
    }
}

float3 AcesFilm(float3 x) {
    return saturate((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

struct ToneNode<let curve : uint> {
    static void Evaluate(PointContext context, LinearColor input, out LinearColor output) {
        Channels channels = input.Get();
        const float3 color = max(float3(channels[0], channels[1], channels[2]), float3(0));
        float3 mapped = color;
        if (curve == 1) {
            mapped = color / (1 + color);
        } else if (curve == 2) {
            mapped = AcesFilm(color);
        }
        [ForceUnroll]
        for (uint c = 0; c < 3; ++c) {
            channels[c] = mapped[c];
        }
        output = LinearColor.Make(channels);
    }
}
