// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.distortion.angular;

using namespace ernie;

struct KannalaBrandt : IAngularMap {
    no_diff float4 k;
    no_diff float focalLength;
    no_diff float maxAngle;

    static KannalaBrandt Load(RenderTables* tables, uint row, float focalLength) {
        const DistortionRecord record = tables->distortions[row];
        KannalaBrandt model;
        model.k = record.radial.Vector();
        model.focalLength = focalLength;
        model.maxAngle = record.maxAngle;
        return model;
    }

    [Differentiable]
    float Radius(float angle) {
        const float a2 = angle * angle;
        return focalLength * angle * (1 + a2 * (k.x + a2 * (k.y + a2 * (k.z + a2 * k.w))));
    }

    [Differentiable]
    float Slope(float angle) {
        const float a2 = angle * angle;
        return focalLength *
               (1 + a2 * (3 * k.x + a2 * (5 * k.y + a2 * (7 * k.z + a2 * 9 * k.w))));
    }

    float MaxAngle() {
        return maxAngle;
    }
}
