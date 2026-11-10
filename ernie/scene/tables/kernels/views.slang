// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/tables/kernels/views.h"

using namespace ernie;

[shader("compute")]
[numthreads(kViewGroup, 1, 1)]
void WriteViews(uniform ViewParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.count) {
        return;
    }
    const ViewEntry entry = p.entries[index];
    ViewRecord view = {};
    if (entry.camera != kNoRow) {
        const CameraRecord camera = p.cameras[entry.camera];
        const ProductRecord product = p.products[entry.product];
        const Projection projection = Projection(camera.projection);
        const float2 aperture = projection == Projection::Spherical
                                    ? camera.aperture.Vector()
                                    : ConformAperture(camera.aperture, product).Vector();
        const float focal = camera.focalLength;
        const bool undistorted =
            entry.distortion == kNoRow ||
            DistortionModel(p.distortions[entry.distortion].model) == DistortionModel::None;
        view.scale = Float2(0.5f * aperture / focal);
        view.bias = Float2(camera.offset.Vector() / focal);
        view.clipping = camera.clipping;
        view.shutter = lerp(camera.shutter.x, camera.shutter.y, 0.5f);
        view.readout = camera.readout;
        view.spread = aperture.y / (float(max(product.height, 1u)) * focal);
        view.exposure = camera.exposure;
        view.clippingPlanes = camera.clippingPlanes;
        view.flags = projection == Projection::Perspective && undistorted ? kViewPinhole : 0;
    }
    p.views[index] = view;
}
