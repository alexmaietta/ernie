// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

uint Hash(uint x) {
    const uint state = x * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}

struct Rng {
    uint state;

    __init(uint camera, uint pixel, uint sample, uint seed) {
        state = Hash(pixel ^ Hash(sample ^ Hash(camera ^ Hash(seed))));
    }

    [mutating]
    float Next() {
        state = Hash(state);
        return float(state >> 8) * (1.0f / 16777216.0f);
    }

    [mutating]
    float2 Next2() {
        const float x = Next();
        return float2(x, Next());
    }
}

float2 ConcentricDisk(float2 u) {
    const float2 offset = 2 * u - 1;
    if (offset.x == 0 && offset.y == 0) {
        return float2(0);
    }

    const bool wide = abs(offset.x) > abs(offset.y);
    const float radius = wide ? offset.x : offset.y;
    const float theta = wide ? 0.25f * kPi * (offset.y / offset.x)
                             : 0.5f * kPi - 0.25f * kPi * (offset.x / offset.y);
    return radius * float2(cos(theta), sin(theta));
}
