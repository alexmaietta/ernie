// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.pipeline.stencil;

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

PipelineElement ElementAt(PipelineCapture* capture, uint index) {
    PipelineElement element;
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

Ptr<float> StatWords(PipelineParams p, PipelineCapture* capture, uint slot) {
    return Ptr<float>(ValueSlot(p, capture, slot).address);
}

CaptureStat ReadStat(PipelineParams p, PipelineCapture* capture, uint slot) {
    return ReadStatWords(StatWords(p, capture, slot));
}

void WriteStat(PipelineParams p, PipelineCapture* capture, uint slot, CaptureStat stat) {
    WriteStatWords(StatWords(p, capture, slot), stat);
}

CaptureStat ReadStatPartial(PipelineParams p, PipelineCapture* capture, uint slot, uint group) {
    return ReadStatWords(StatWords(p, capture, slot) + (1 + group) * kStatWords);
}

void WriteStatPartial(PipelineParams p, PipelineCapture* capture, uint slot, uint group,
                      CaptureStat stat) {
    WriteStatWords(StatWords(p, capture, slot) + (1 + group) * kStatWords, stat);
}
