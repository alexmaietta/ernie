// SPDX-License-Identifier: BSD-3-Clause

import ernie.sensor_pipeline;

using namespace ernie;

static const int kPsfRadius = 3;
static const int kDenoiseRadius = 2;
static const int kDiffractionRadius = 8;

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
        if (method == 1 && Bayer(context.Detector().pattern) && result.count >= 3) {
            Malvar(raw, center, result);
        }
        full = Digital.Make(result);
    }

    static void Malvar(Neighborhood<Digital> raw, uint center, inout Channels result) {
        const float c = raw.Center().Get()[0];
        const float cross = raw.At(0, -1).Get()[0] + raw.At(0, 1).Get()[0] +
                            raw.At(-1, 0).Get()[0] + raw.At(1, 0).Get()[0];
        const float diagonal = raw.At(-1, -1).Get()[0] + raw.At(1, -1).Get()[0] +
                               raw.At(-1, 1).Get()[0] + raw.At(1, 1).Get()[0];
        const float rows = raw.At(-2, 0).Get()[0] + raw.At(2, 0).Get()[0];
        const float columns = raw.At(0, -2).Get()[0] + raw.At(0, 2).Get()[0];
        if (center != 1) {
            result[1] = max((4 * c + 2 * cross - rows - columns) / 8, 0.0f);
            result[2 - center] = max((6 * c + 2 * diagonal - 1.5f * (rows + columns)) / 8, 0.0f);
            return;
        }
        const float row = raw.At(-1, 0).Get()[0] + raw.At(1, 0).Get()[0];
        const float column = raw.At(0, -1).Get()[0] + raw.At(0, 1).Get()[0];
        const uint across = raw.Site(int2(1, 0));
        result[across] = max((5 * c + 4 * row - rows - diagonal + 0.5f * columns) / 8, 0.0f);
        result[2 - across] =
            max((5 * c + 4 * column - columns - diagonal + 0.5f * rows) / 8, 0.0f);
    }
}

bool Bayer(uint pattern) {
    switch (DetectorPattern(pattern)) {
    case DetectorPattern::Rggb:
    case DetectorPattern::Bggr:
    case DetectorPattern::Grbg:
    case DetectorPattern::Gbrg:
        return true;
    default:
        return false;
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
    static void Evaluate(PointContext context, LinearColor input, CaptureStat stat,
                         out LinearColor output) {
        Channels channels = input.Get();
        const bool metered = stat.Weight() > 0;
        [ForceUnroll]
        for (uint c = 0; c < 3; ++c) {
            const float world = metered ? stat.values[1] / max(stat.values[c], 1e-12f) : 1;
            channels[c] *= context.Parameter(c) * world;
        }
        output = LinearColor.Make(channels);
    }
}

float CenterRadius2(PointContext context) {
    const float2 size = float2(context.element.capture->width, context.element.capture->height);
    const float2 offset = (float2(context.element.coord) + 0.5f) - 0.5f * size;
    return dot(offset, offset) / max(dot(0.5f * size, 0.5f * size), 1e-6f);
}

struct MeterNode<let mode : uint> {
    static float Weight(PointContext context) {
        const float radius = max(context.Parameter(0), 1e-3f);
        const float r2 = CenterRadius2(context);
        if (mode == 1) {
            return exp(-r2 / (2 * radius * radius));
        }
        if (mode == 2) {
            return r2 <= radius * radius ? 1 : 0;
        }
        return 1;
    }

    static CaptureStat Contribute(PointContext context, LinearColor color) {
        const float weight = Weight(context);
        const Channels channels = color.Get();
        CaptureStat stat = CaptureStat.Zero();
        [ForceUnroll]
        for (uint c = 0; c < kMostSignalChannels; ++c) {
            stat.values[c] = weight * channels[c];
        }
        const float luminance =
            channels.count >= 3
                ? dot(float3(channels[0], channels[1], channels[2]), float3(0.2126f, 0.7152f, 0.0722f))
                : channels[0];
        stat.values[kMostSignalChannels] = weight * luminance;
        stat.values[kStatWords - 1] = weight;
        return stat;
    }

    static CaptureStat Finish(PointContext context, CaptureStat sum) {
        CaptureStat mean = sum;
        const float weight = sum.Weight();
        [ForceUnroll]
        for (uint word = 0; word <= kMostSignalChannels; ++word) {
            mean.values[word] = weight > 0 ? sum.values[word] / weight : 0;
        }
        return mean;
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

float3 AgX(float3 color) {
    const float3x3 inset = float3x3(0.856627153315983f, 0.137318972929847f, 0.11189821299995f,
                                    0.0951212405381588f, 0.761241990602591f, 0.0767994186031903f,
                                    0.0482516061458583f, 0.101439036467562f, 0.811302368396859f);
    const float3x3 outset =
        float3x3(1.1271005818144368f, -0.1413297634984383f, -0.14132976349843826f,
                 -0.11060664309660323f, 1.157823702216272f, -0.11060664309660294f,
                 -0.016493938717834573f, -0.016493938717834257f, 1.2519364065950405f);
    const float lowest = -12.47393f;
    const float highest = 4.026069f;
    float3 x = mul(color, inset);
    x = clamp(log2(max(x, float3(1e-10f))), float3(lowest), float3(highest));
    x = (x - lowest) / (highest - lowest);
    const float3 x2 = x * x;
    const float3 x4 = x2 * x2;
    x = 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4 - 6.868f * x2 * x + 0.4298f * x2 +
        0.1191f * x - 0.00232f;
    x = mul(x, outset);
    return pow(max(x, float3(0)), float3(2.2f));
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
        } else if (curve == 3) {
            mapped = AgX(color);
        }
        [ForceUnroll]
        for (uint c = 0; c < 3; ++c) {
            channels[c] = mapped[c];
        }
        output = LinearColor.Make(channels);
    }
}

struct AutoExposureNode {
    static void Evaluate(PointContext context, LinearColor input, CaptureStat stat,
                         out LinearColor output) {
        const float gain = context.State(0);
        Channels channels = input.Get();
        [ForceUnroll]
        for (uint c = 0; c < kMostSignalChannels; ++c) {
            channels[c] *= gain;
        }
        output = LinearColor.Make(channels);
    }

    static void Update(PointContext context, CaptureStat stat) {
        if (stat.Weight() <= 0) {
            return;
        }
        const float lowest = context.Parameter(2);
        const float highest = max(context.Parameter(3), lowest);
        const float luminance = max(stat.values[kMostSignalChannels], 1e-6f);
        const float ideal = clamp(context.Parameter(0) / luminance, lowest, highest);
        const float gain = max(context.State(0), 1e-12f);
        const float next = gain * pow(ideal / gain, saturate(context.Parameter(1)));
        context.SetState(0, clamp(next, lowest, highest));
    }
}

float BesselJ1(float x) {
    const float ax = abs(x);
    if (ax < 8) {
        const float y = x * x;
        const float numerator =
            x * (72362614232.0f +
                 y * (-7895059235.0f +
                      y * (242396853.1f +
                           y * (-2972611.439f + y * (15704.48260f + y * (-30.16036606f))))));
        const float denominator =
            144725228442.0f +
            y * (2300535178.0f +
                 y * (18583304.74f + y * (99447.43394f + y * (376.9991397f + y))));
        return numerator / denominator;
    }
    const float z = 8 / ax;
    const float y = z * z;
    const float shifted = ax - 2.356194491f;
    const float p = 1 + y * (0.183105e-2f + y * (-0.3516396496e-4f +
                                                  y * (0.2457520174e-5f + y * (-0.240337019e-6f))));
    const float q = 0.04687499995f +
                    y * (-0.2002690873e-3f +
                         y * (0.8449199096e-5f + y * (-0.88228987e-6f + y * 0.105787412e-6f)));
    const float value = sqrt(0.636619772f / ax) * (cos(shifted) * p - z * sin(shifted) * q);
    return x < 0 ? -value : value;
}

float Airy(float radius) {
    if (radius < 1e-6f) {
        return 1;
    }
    const float ratio = 2 * BesselJ1(radius) / radius;
    return ratio * ratio;
}

struct DiffractionPsfNode {
    static void Evaluate(PointContext context, Neighborhood<ElectronRate> input,
                         out ElectronRate output) {
        const float fNumber = max(context.Parameter(0), 0.5f);
        const float wavelength = max(context.Parameter(1), 1.0f) * 1e-3f;
        const float pitch = max(context.Parameter(2), 0.01f);
        const float scale = 3.14159265f * pitch / (wavelength * fNumber);
        const bool mosaic = input.Mosaic();
        const uint site = input.Site(int2(0));
        Channels sums = Channels(input.Center().Get().count);
        float total = 0;
        for (int dy = -kDiffractionRadius; dy <= kDiffractionRadius; ++dy) {
            for (int dx = -kDiffractionRadius; dx <= kDiffractionRadius; ++dx) {
                if (mosaic && input.Site(int2(dx, dy)) != site) {
                    continue;
                }
                const float weight = Airy(scale * sqrt(float(dx * dx + dy * dy)));
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

static const int kMostGlareRadius = 64;

struct GlareNode {
    static void Evaluate(PointContext context, uint pass, Neighborhood<ElectronRate> input,
                         Neighborhood<ElectronRate> previous, out ElectronRate output) {
        const float sigma = max(context.Parameter(0), 1e-3f);
        if (pass == 2) {
            const float strength = saturate(context.Parameter(1));
            output = input.Center().Lerp(previous.Center(), strength);
            return;
        }
        const int step = input.Mosaic() ? 2 : 1;
        const int radius = min(int(ceil(3 * sigma)), kMostGlareRadius);
        const int2 axis = pass == 0 ? int2(1, 0) : int2(0, 1);
        Channels sums = Channels(previous.Center().Get().count);
        float total = 0;
        for (int offset = -radius - radius % step; offset <= radius; offset += step) {
            const float weight = exp(-float(offset * offset) / (2 * sigma * sigma));
            const Channels sample = previous.At(axis * offset).Get();
            [ForceUnroll]
            for (uint c = 0; c < kMostSignalChannels; ++c) {
                sums[c] += weight * sample[c];
            }
            total += weight;
        }
        [ForceUnroll]
        for (uint c = 0; c < kMostSignalChannels; ++c) {
            sums[c] /= total;
        }
        output = ElectronRate.Make(sums);
    }
}

struct BloomingNode {
    static void Evaluate(PointContext context, uint pass, Neighborhood<Electrons> input,
                         Neighborhood<Electrons> previous, out Electrons output) {
        const float well = context.Detector().fullWell;
        const int reach = max(asint(context.ParameterBits(0)), 1);
        const int height = int(input.Size().y);
        const int row = int(input.element.coord.y);
        Channels own = input.Center().Get();
        Channels arriving = Channels(own.count);
        for (int direction = -1; direction <= 1; direction += 2) {
            Channels between = Channels(own.count);
            for (int k = 1; k <= reach; ++k) {
                const int at = row + direction * k;
                if (at < 0 || at >= height) {
                    break;
                }
                const Channels other = input.At(int2(0, direction * k)).Get();
                [ForceUnroll]
                for (uint c = 0; c < kMostSignalChannels; ++c) {
                    arriving[c] += max(0.5f * max(other[c] - well, 0.0f) - between[c], 0.0f);
                    between[c] += max(well - other[c], 0.0f);
                }
            }
        }
        [ForceUnroll]
        for (uint c = 0; c < kMostSignalChannels; ++c) {
            own[c] = min(own[c], well) + min(arriving[c], max(well - own[c], 0.0f));
        }
        output = Electrons.Make(own);
    }
}
