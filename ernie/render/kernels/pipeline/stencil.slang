// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.pipeline.segment;

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

struct Neighborhood<S : ISignal> {
    PipelineParams params;
    PipelineElement element;
    uint slot;

    __init(PipelineParams params, PipelineElement element, uint slot) {
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
