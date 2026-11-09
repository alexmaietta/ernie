// SPDX-License-Identifier: BSD-3-Clause

import ernie.pipeline;
import ernie.render.kernels.random;

using namespace ernie;

static const float kDarkReferenceKelvin = 293.15f;

float ExposureTime(PointContext context, float time) {
    return time > 0 ? time : context.Camera().exposureTime;
}

float DigitalMax(DetectorRecord detector) {
    return exp2(float(clamp(detector.bitDepth, 1u, 32u))) - 1;
}

struct IntegrateNode {
    static void Evaluate(PointContext context, ElectronRate rate, out Electrons electrons) {
        const float time = ExposureTime(context, context.Parameter(0));
        Channels channels = rate.Get();
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            channels[i] *= time;
        }
        electrons = Electrons.Make(channels);
    }
}

struct Emva1288Node<let shot : uint, let fixedPatterns : uint, let clipping : uint> {
    static void Evaluate(PointContext context, Electrons mean, out Electrons electrons) {
        const DetectorRecord detector = context.Detector();
        const float time = ExposureTime(context, context.Parameter(0));
        const float dark = detector.darkCurrent * time *
                           exp2((detector.temperature - kDarkReferenceKelvin) /
                                max(detector.darkCurrentDoubling, 1e-3f));
        NoiseStream temporal = context.Noise(0);
        NoiseStream fixedStream = context.FixedNoise(Hash(detector.seed));
        Channels channels = mean.Get();
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            if (i < channels.count) {
                float expected = channels[i];
                float offset = 0;
                if (fixedPatterns == 1) {
                    expected *= 1 + detector.prnu * fixedStream.Normal();
                    offset = detector.dsnu * fixedStream.Normal();
                }
                expected = max(expected, 0.0f) + dark;
                float collected = expected;
                if (shot == 0) {
                    collected = temporal.Poisson(expected);
                } else if (shot == 1) {
                    collected = max(expected + sqrt(expected) * temporal.Normal(), 0.0f);
                }
                if (clipping == 0) {
                    collected = min(collected, detector.fullWell);
                }
                channels[i] = collected + detector.readNoise * temporal.Normal() + offset;
            }
        }
        electrons = Electrons.Make(channels);
    }
}

struct AdcNode {
    static void Evaluate(PointContext context, Electrons electrons, out Digital digital) {
        const DetectorRecord detector = context.Detector();
        const float gain = detector.systemGain * context.Camera().iso / 100;
        const float most = DigitalMax(detector);
        Channels channels = electrons.Get();
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            channels[i] = clamp(round(channels[i] * gain + detector.blackLevel), 0.0f, most);
        }
        digital = Digital.Make(channels);
    }
}

struct GainNode {
    static void Evaluate(PointContext context, Digital input, out Digital digital) {
        const float gain = context.Parameter(0);
        Channels channels = input.Get();
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            channels[i] *= gain;
        }
        digital = Digital.Make(channels);
    }
}

struct ToColorNode {
    static void Evaluate(PointContext context, Digital digital, out LinearColor color) {
        const DetectorRecord detector = context.Detector();
        const float black = detector.blackLevel;
        const float range = max(DigitalMax(detector) - black, 1.0f);
        const Channels channels = digital.Get();
        Channels rgb = Channels(3);
        [ForceUnroll]
        for (uint i = 0; i < 3; ++i) {
            const uint from = channels.count >= 3 ? i : 0;
            rgb[i] = max(channels[from] - black, 0.0f) / range;
        }
        color = LinearColor.Make(rgb);
    }
}

float EncodeSrgb(float linear) {
    return linear <= 0.0031308f ? 12.92f * linear : 1.055f * pow(linear, 1 / 2.4f) - 0.055f;
}

float EncodeRec709(float linear) {
    return linear < 0.018f ? 4.5f * linear : 1.099f * pow(linear, 0.45f) - 0.099f;
}

struct EncodeNode<let transfer : uint> {
    static void Evaluate(PointContext context, LinearColor color, out DisplayColor display) {
        Channels channels = color.Get();
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            const float linear = saturate(channels[i]);
            channels[i] = transfer == 0 ? EncodeSrgb(linear) : EncodeRec709(linear);
        }
        display = DisplayColor.Make(channels);
    }
}
