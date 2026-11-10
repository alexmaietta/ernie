// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensor_pipeline.context;

using namespace ernie;

float HalfToFloat(uint bits) {
    const uint sign = (bits & 0x8000u) << 16;
    const uint exponent = (bits >> 10) & 0x1fu;
    const uint mantissa = bits & 0x3ffu;
    if (exponent == 0) {
        const float magnitude = float(mantissa) * 5.9604645e-8f;
        return sign != 0 ? -magnitude : magnitude;
    }
    if (exponent == 31) {
        return asfloat(sign | 0x7f800000u | (mantissa << 13));
    }
    return asfloat(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

uint FloatToHalf(float value) {
    const uint bits = asuint(value);
    const uint sign = (bits >> 16) & 0x8000u;
    const int exponent = int((bits >> 23) & 0xffu) - 112;
    uint mantissa = bits & 0x7fffffu;
    if (int((bits >> 23) & 0xffu) == 255) {
        return sign | 0x7c00u | (mantissa != 0 ? 0x200u : 0);
    }
    if (exponent >= 31) {
        return sign | 0x7c00u;
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000u;
        const uint shift = uint(14 - exponent);
        uint half = mantissa >> shift;
        if (((mantissa >> (shift - 1)) & 1) != 0) {
            half += 1;
        }
        return sign | half;
    }
    uint half = sign | (uint(exponent) << 10) | (mantissa >> 13);
    if ((mantissa & 0x1000u) != 0) {
        half += 1;
    }
    return half;
}

uint HalfStride(uint channels) {
    return (channels + 1) & ~1u;
}

uint MosaicSite(uint pattern, uint2 coord) {
    const uint site = (coord.x & 1) | ((coord.y & 1) << 1);
    switch (DetectorPattern(pattern)) {
    case DetectorPattern::Rggb:
        return uint4(0, 1, 1, 2)[site];
    case DetectorPattern::Bggr:
        return uint4(2, 1, 1, 0)[site];
    case DetectorPattern::Grbg:
        return uint4(1, 0, 2, 1)[site];
    case DetectorPattern::Gbrg:
        return uint4(1, 2, 0, 1)[site];
    case DetectorPattern::Rccb:
        return uint4(0, 1, 1, 2)[site];
    case DetectorPattern::Rccc:
        return site == 0 ? 0 : 1;
    default:
        return 0;
    }
}

SensorPipelineValue ValueSlot(SensorPipelineParams p, SensorPipelineCapture* capture, uint slot) {
    return p.values[capture->firstValue + slot];
}

S ReadSignalAt<S : ISignal>(SensorPipelineParams p, SensorPipelineCapture* capture, uint slot, uint index) {
    const SensorPipelineValue value = ValueSlot(p, capture, slot);
    Channels channels = Channels(value.channels);
    if ((value.flags & kSignalRecords) != 0) {
        Ptr<float> record = Ptr<float>(value.address) + uint64_t(index) * kRecordWords;
        channels[0] = float(Ptr<uint>(value.returned)[index]);
        channels[1] = record[3];
        channels[2] = record[4];
        channels[3] = record[0];
        channels[4] = record[1];
        channels[5] = record[2];
        channels[6] = record[5];
        channels[7] = (value.flags & kSignalWorld) != 0 ? 1 : 0;
    } else if ((value.flags & kSignalSampled) != 0) {
        const uint stride = value.flags >> kSignalStrideShift;
        const uint2 coord = uint2(index % capture->width, index / capture->width);
        const uint site = MosaicSite(p.tables->detectors[capture->detector].pattern, coord);
        channels[0] = Ptr<float>(value.address)[uint64_t(index) * stride + site];
    } else if ((value.flags & kSignalHalf) != 0) {
        Ptr<uint> words = Ptr<uint>(value.address);
        const uint64_t first = uint64_t(index) * HalfStride(value.channels) / 2;
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            if (i < value.channels) {
                channels[i] = HalfToFloat(words[first + i / 2] >> (16 * (i % 2)));
            }
        }
    } else {
        Ptr<float> floats = Ptr<float>(value.address);
        const uint64_t first = uint64_t(index) * value.channels;
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            if (i < value.channels) {
                channels[i] = floats[first + i];
            }
        }
    }
    return S.Make(channels);
}

S ReadSignal<S : ISignal>(SensorPipelineParams p, SensorPipelineElement element, uint slot) {
    return ReadSignalAt<S>(p, element.capture, slot, element.index);
}

void WriteSignal<S : ISignal>(SensorPipelineParams p, SensorPipelineElement element, uint slot, S signal) {
    const SensorPipelineValue value = ValueSlot(p, element.capture, slot);
    const Channels channels = signal.Get();
    if ((value.flags & kSignalRecords) != 0) {
        Ptr<float> record = Ptr<float>(value.address) + uint64_t(element.index) * kRecordWords;
        Ptr<uint>(value.returned)[element.index] = channels[0] != 0 ? 1 : 0;
        record[3] = channels[1];
        record[4] = channels[2];
        record[0] = channels[3];
        record[1] = channels[4];
        record[2] = channels[5];
        record[5] = channels[6];
    } else if ((value.flags & kSignalHalf) != 0) {
        Ptr<uint> words = Ptr<uint>(value.address);
        const uint64_t first = uint64_t(element.index) * HalfStride(value.channels) / 2;
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; i += 2) {
            if (i < value.channels) {
                const uint high = i + 1 < value.channels ? FloatToHalf(channels[i + 1]) : 0;
                words[first + i / 2] = FloatToHalf(channels[i]) | (high << 16);
            }
        }
    } else {
        Ptr<float> floats = Ptr<float>(value.address);
        const uint64_t first = uint64_t(element.index) * value.channels;
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            if (i < value.channels) {
                floats[first + i] = channels[i];
            }
        }
    }
}
