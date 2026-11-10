// SPDX-License-Identifier: BSD-3-Clause

import ernie.sensor_pipeline;
import ernie.render.kernels.sensors.frame;

using namespace ernie;

float3 BeamOrigin(PointContext context, Returns returns) {
    if (!returns.world) {
        return float3(0);
    }
    const SensorFrame frame = LoadSensorFrameAt(uint64_t(context.params.tables),
                                                context.element.capture->sensor);
    return frame.position + frame.origin;
}

Returns WithRange(PointContext context, Returns returns, float distance) {
    const float range = max(distance, 0.0f);
    if (returns.distance > 0) {
        const float3 origin = BeamOrigin(context, returns);
        returns.position = origin + (returns.position - origin) * (range / returns.distance);
    }
    returns.distance = range;
    return returns;
}

struct RangeNoiseNode {
    static void Evaluate(PointContext context, Returns input, out Returns output) {
        output = input;
        if (!input.returned) {
            return;
        }
        const float sigma = context.Parameter(0) + context.Parameter(1) * input.distance;
        output = WithRange(context, input, input.distance + sigma * context.Normal(0));
    }
}

struct IntensityNoiseNode {
    static void Evaluate(PointContext context, Returns input, out Returns output) {
        output = input;
        if (!input.returned) {
            return;
        }
        const float sigma = context.Parameter(0) + context.Parameter(1) * input.intensity;
        output.intensity = max(input.intensity + sigma * context.Normal(0), 0.0f);
    }
}

struct DropoutNode {
    static void Evaluate(PointContext context, Returns input, out Returns output) {
        output = input;
        if (!input.returned) {
            return;
        }
        if (input.intensity < context.Parameter(0) || context.Uniform(0) < context.Parameter(1)) {
            output.returned = false;
        }
    }
}

struct RangeQuantizeNode {
    static void Evaluate(PointContext context, Returns input, out Returns output) {
        output = input;
        const float step = context.Parameter(0);
        if (!input.returned || step <= 0) {
            return;
        }
        output = WithRange(context, input, round(input.distance / step) * step);
    }
}

struct ToPointCloudNode {
    static void Evaluate(PointContext context, Returns input, out CloudSignal output) {
        output = CloudSignal.Make(input.Get());
    }
}
