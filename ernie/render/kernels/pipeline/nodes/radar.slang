// SPDX-License-Identifier: BSD-3-Clause

import ernie.pipeline;
import ernie.render.kernels.sensors.frame;

using namespace ernie;

static const int kCfarRadius = 4;

struct RadarNoiseNode {
    static void Evaluate(PointContext context, Returns input, out Returns output) {
        output = input;
        if (!input.returned) {
            return;
        }
        const float distance = max(input.distance + context.Parameter(0) * context.Normal(0), 0.0f);
        if (input.distance > 0) {
            float3 origin = float3(0);
            if (input.world) {
                const SensorFrame frame = LoadSensorFrameAt(uint64_t(context.params.tables),
                                                            context.element.capture->sensor);
                origin = frame.position + frame.origin;
            }
            output.position = origin + (input.position - origin) * (distance / input.distance);
        }
        output.distance = distance;
        output.radialVelocity = input.radialVelocity + context.Parameter(1) * context.Normal(1);
        const float floor = context.Parameter(2);
        if (floor > 0) {
            const float spread = sqrt(floor / 2);
            const float real = sqrt(max(input.intensity, 0.0f)) + spread * context.Normal(2);
            const float imaginary = spread * context.Normal(3);
            output.intensity = real * real + imaginary * imaginary;
        }
    }
}

struct CfarNode {
    static void Evaluate(PointContext context, Neighborhood<Returns> input, out Returns output) {
        output = input.Center();
        if (!output.returned) {
            return;
        }
        const int guard = clamp(asint(context.ParameterBits(0)), 0, kCfarRadius);
        const int reach = clamp(guard + asint(context.ParameterBits(1)), guard, kCfarRadius);
        float sum = 0;
        float cells = 0;
        for (int dy = -reach; dy <= reach; ++dy) {
            for (int dx = -reach; dx <= reach; ++dx) {
                if (max(abs(dx), abs(dy)) <= guard || !input.Inside(int2(dx, dy))) {
                    continue;
                }
                const Returns cell = input.At(dx, dy);
                sum += cell.returned ? cell.intensity : 0.0f;
                cells += 1;
            }
        }
        if (cells == 0) {
            return;
        }
        const float rate = clamp(context.Parameter(2), 1e-12f, 1.0f);
        const float scale = cells * (pow(rate, -1.0f / cells) - 1);
        output.returned = output.intensity > scale * sum / cells;
    }
}
