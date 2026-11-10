// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/products.h"
#include "ernie/scene/records/sensors.h"
#include "ernie/stage/schema/schema.h"

namespace ernie {

static const uint kViewPinhole = 1u << 0;

struct ViewRecord {
    Float2 scale;
    Float2 bias;
    Float2 clipping;
    float shutter;
    float readout;
    float spread;
    float exposure;
    uint flags;
    uint padding;
};

ERNIE_CHECK_SIZE(ViewRecord, 48);

ERNIE_SHARED Float2 ConformAperture(Float2 aperture, ProductRecord product) {
    const float image = float(product.width) * product.pixelAspect / float(max(product.height, 1u));
    const bool wider = aperture.x > aperture.y * image;
    Float2 wide;
    wide.x = aperture.y * image;
    wide.y = aperture.y;
    Float2 tall;
    tall.x = aperture.x;
    tall.y = aperture.x / image;
    switch (AspectRatioConformPolicy(product.conform)) {
    case AspectRatioConformPolicy::ExpandAperture:
        return wider ? tall : wide;
    case AspectRatioConformPolicy::CropAperture:
        return wider ? wide : tall;
    case AspectRatioConformPolicy::AdjustApertureWidth:
        return wide;
    case AspectRatioConformPolicy::AdjustApertureHeight:
        return tall;
    default:
        return aperture;
    }
}

}
