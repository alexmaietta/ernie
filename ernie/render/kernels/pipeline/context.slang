// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.pipeline.noise;
__exported import ernie.render.kernels.pipeline.signals;

using namespace ernie;

struct PipelineElement {
    PipelineCapture* capture;
    uint index;
    uint2 coord;
}

bool FindElement(PipelineParams p, out PipelineElement element) {
    const uint64_t global = uint64_t(LinearThreadIndex());
    element = {};
    if (global >= p.elementCount || p.captureCount == 0) {
        return false;
    }
    uint low = 0;
    uint high = p.captureCount - 1;
    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        if (p.captures[middle].firstElement <= global) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    element.capture = p.captures + low;
    element.index = uint(global - element.capture->firstElement);
    element.coord = uint2(element.index % element.capture->width, element.index / element.capture->width);
    return true;
}

struct PointContext {
    PipelineParams params;
    PipelineElement element;
    uint node;
    uint parameters;

    __init(PipelineParams params, PipelineElement element, uint node, uint parameters) {
        this.params = params;
        this.element = element;
        this.node = node;
        this.parameters = parameters;
    }

    float Parameter(uint index) {
        return asfloat(element.capture->words[parameters + index]);
    }

    uint ParameterBits(uint index) {
        return element.capture->words[parameters + index];
    }

    DetectorRecord Detector() {
        return params.tables->detectors[element.capture->detector];
    }

    CameraRecord Camera() {
        return params.tables->cameras[element.capture->camera];
    }

    NoiseStream Noise(uint stream) {
        const PipelineCapture capture = *element.capture;
        return NoiseStream(capture.identity, element.index, node, stream, capture.capture);
    }

    NoiseStream FixedNoise(uint stream) {
        return NoiseStream(element.capture->identity, element.index, node, stream, kFixedCapture);
    }
}
