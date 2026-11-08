// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.distortion.angular;

using namespace ernie;

static const uint kMostFThetaTerms = 8;

struct FTheta : IAngularMap {
    no_diff ArrayRef terms;
    no_diff float focalLength;
    no_diff float maxAngle;

    static FTheta Load(RenderTables* tables, uint row, float focalLength) {
        const DistortionRecord record = tables->distortions[row];
        FTheta model;
        model.terms = record.fTheta;
        model.focalLength = focalLength;
        model.maxAngle = record.maxAngle;
        return model;
    }

    float Term(uint i) {
        return Ptr<float>(terms.address)[i];
    }

    [Differentiable]
    float Radius(float angle) {
        if (terms.count == 0) {
            return focalLength * angle;
        }
        float radius = 0;
        [MaxIters(kMostFThetaTerms)]
        for (uint i = min(terms.count, kMostFThetaTerms); i > 0; --i) {
            radius = radius * angle + no_diff Term(i - 1);
        }
        return radius;
    }

    [Differentiable]
    float Slope(float angle) {
        if (terms.count < 2) {
            return terms.count == 0 ? focalLength : 0;
        }
        float slope = 0;
        [MaxIters(kMostFThetaTerms)]
        for (uint i = min(terms.count, kMostFThetaTerms); i > 1; --i) {
            slope = slope * angle + float(i - 1) * no_diff Term(i - 1);
        }
        return slope;
    }

    float MaxAngle() {
        return maxAngle;
    }
}
