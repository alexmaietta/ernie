// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.render.kernels.random;

static const uint kFixedCapture = 0xffffffffu;
static const float kPoissonInversionLimit = 20;

struct NoiseStream {
    uint state;

    __init(uint identity, uint element, uint node, uint stream, uint capture) {
        state = Hash(element ^ Hash(node ^ Hash(stream ^ Hash(capture ^ Hash(identity)))));
    }

    [mutating]
    float Uniform() {
        state = Hash(state);
        return (float(state >> 8) + 0.5f) * (1.0f / 16777216.0f);
    }

    [mutating]
    float Normal() {
        const float radius = sqrt(-2 * log(Uniform()));
        return radius * cos(2 * kPi * Uniform());
    }

    [mutating]
    float Poisson(float mean) {
        if (mean <= 0) {
            return 0;
        }
        if (mean >= kPoissonInversionLimit) {
            return max(round(mean + sqrt(mean) * Normal()), 0.0f);
        }
        const float u = Uniform();
        float term = exp(-mean);
        float total = term;
        float count = 0;
        [MaxIters(64)]
        while (u > total && count < 64) {
            count += 1;
            term *= mean / count;
            total += term;
        }
        return count;
    }
}
