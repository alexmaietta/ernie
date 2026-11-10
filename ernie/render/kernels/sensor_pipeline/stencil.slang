// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensor_pipeline.segment;

using namespace ernie;

bool Mosaics(uint pattern) {
    switch (DetectorPattern(pattern)) {
    case DetectorPattern::Rggb:
    case DetectorPattern::Bggr:
    case DetectorPattern::Grbg:
    case DetectorPattern::Gbrg:
    case DetectorPattern::Rccb:
    case DetectorPattern::Rccc:
        return true;
    default:
        return false;
    }
}

struct Neighborhood<S : ISignal> {
    SensorPipelineParams params;
    SensorPipelineElement element;
    uint slot;

    __init(SensorPipelineParams params, SensorPipelineElement element, uint slot) {
        this.params = params;
        this.element = element;
        this.slot = slot;
    }

    int2 Clamped(int2 offset) {
        const int2 size = int2(element.capture->width, element.capture->height);
        return clamp(int2(element.coord) + offset, int2(0), size - 1);
    }

    S At(int2 offset) {
        const int2 at = Clamped(offset);
        return ReadSignalAt<S>(params, element.capture, slot,
                               uint(at.y) * element.capture->width + uint(at.x));
    }

    uint2 Size() {
        return uint2(element.capture->width, element.capture->height);
    }

    bool Inside(int2 offset) {
        const int2 at = int2(element.coord) + offset;
        return all(at >= 0) && at.x < int(element.capture->width) &&
               at.y < int(element.capture->height);
    }

    S At(int x, int y) {
        return At(int2(x, y));
    }

    S Center() {
        return ReadSignal<S>(params, element, slot);
    }

    bool Mosaic() {
        return (ValueSlot(params, element.capture, slot).flags & kSignalMosaic) != 0;
    }

    uint Site(int2 offset) {
        return MosaicSite(params.tables->detectors[element.capture->detector].pattern,
                          uint2(Clamped(offset)));
    }
}
