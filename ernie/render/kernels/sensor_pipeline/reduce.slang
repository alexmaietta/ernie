// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensor_pipeline.stencil;

using namespace ernie;

struct CaptureStat {
    float values[kStatWords];

    static CaptureStat Zero() {
        CaptureStat stat;
        [ForceUnroll]
        for (uint word = 0; word < kStatWords; ++word) {
            stat.values[word] = 0;
        }
        return stat;
    }

    CaptureStat Add(CaptureStat other) {
        CaptureStat sum;
        [ForceUnroll]
        for (uint word = 0; word < kStatWords; ++word) {
            sum.values[word] = values[word] + other.values[word];
        }
        return sum;
    }

    float Weight() {
        return values[kStatWords - 1];
    }
}

SensorPipelineElement ElementAt(SensorPipelineCapture* capture, uint index) {
    SensorPipelineElement element;
    element.capture = capture;
    element.index = index;
    element.coord = uint2(index % capture->width, index / capture->width);
    return element;
}

CaptureStat ReadStatWords(Ptr<float> words) {
    CaptureStat stat;
    [ForceUnroll]
    for (uint word = 0; word < kStatWords; ++word) {
        stat.values[word] = words[word];
    }
    return stat;
}

void WriteStatWords(Ptr<float> words, CaptureStat stat) {
    [ForceUnroll]
    for (uint word = 0; word < kStatWords; ++word) {
        words[word] = stat.values[word];
    }
}

Ptr<float> StatWords(SensorPipelineParams p, SensorPipelineCapture* capture, uint slot) {
    return Ptr<float>(ValueSlot(p, capture, slot).address);
}

CaptureStat ReadStat(SensorPipelineParams p, SensorPipelineCapture* capture, uint slot) {
    return ReadStatWords(StatWords(p, capture, slot));
}

void WriteStat(SensorPipelineParams p, SensorPipelineCapture* capture, uint slot, CaptureStat stat) {
    WriteStatWords(StatWords(p, capture, slot), stat);
}

CaptureStat ReadStatPartial(SensorPipelineParams p, SensorPipelineCapture* capture, uint slot, uint group) {
    return ReadStatWords(StatWords(p, capture, slot) + (1 + group) * kStatWords);
}

void WriteStatPartial(SensorPipelineParams p, SensorPipelineCapture* capture, uint slot, uint group,
                      CaptureStat stat) {
    WriteStatWords(StatWords(p, capture, slot) + (1 + group) * kStatWords, stat);
}

CaptureStat SumStatPartials(SensorPipelineParams p, SensorPipelineCapture* capture, uint slot) {
    CaptureStat sum = CaptureStat.Zero();
    for (uint group = 0; group < kReduceGroups; ++group) {
        sum = sum.Add(ReadStatPartial(p, capture, slot, group));
    }
    return sum;
}

bool FindCapture(SensorPipelineParams p, out SensorPipelineCapture* capture) {
    const uint index = LinearThreadIndex();
    capture = p.captures;
    if (index >= p.captureCount) {
        return false;
    }
    capture = p.captures + index;
    return true;
}

groupshared float s_reduceStat[kPipelineGroup * kStatWords];

struct ReduceGroup {
    SensorPipelineCapture* capture;
    uint part;
    uint thread;
    uint first;
    uint stride;
    uint count;

    static bool Find(SensorPipelineParams p, out ReduceGroup group) {
        const uint index = LinearGroupIndex();
        group = {};
        if (index / kReduceGroups >= p.captureCount) {
            return false;
        }
        group.capture = p.captures + index / kReduceGroups;
        group.part = index % kReduceGroups;
        group.thread = LocalThreadIndex();
        group.first = group.part * kPipelineGroup + group.thread;
        group.stride = kReduceGroups * kPipelineGroup;
        group.count = group.capture->width * group.capture->height;
        return true;
    }

    void Store(SensorPipelineParams p, uint slot, CaptureStat sum) {
        [ForceUnroll]
        for (uint word = 0; word < kStatWords; ++word) {
            s_reduceStat[thread * kStatWords + word] = sum.values[word];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint half = kPipelineGroup / 2; half > 0; half /= 2) {
            if (thread < half) {
                [ForceUnroll]
                for (uint word = 0; word < kStatWords; ++word) {
                    s_reduceStat[thread * kStatWords + word] +=
                        s_reduceStat[(thread + half) * kStatWords + word];
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (thread == 0) {
            CaptureStat partial;
            [ForceUnroll]
            for (uint word = 0; word < kStatWords; ++word) {
                partial.values[word] = s_reduceStat[word];
            }
            WriteStatPartial(p, capture, slot, part, partial);
        }
    }
}
