// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensor_pipeline.reduce;

using namespace ernie;

interface IPointNode {}

interface IStencilNode {}

[__AttributeUsage(_AttributeTargets.Struct)]
struct NodeAttribute {
    string displayName;
};

[__AttributeUsage(_AttributeTargets.Var)]
struct InputAttribute {
    float minimum;
    float maximum;
};

[__AttributeUsage(_AttributeTargets.Var)]
struct DisplayNameAttribute {
    string name;
};

extension<S : ISignal> S {
    static S Zero(S like) {
        return S.Make(Channels(like.Get().count));
    }

    S Add(float value) {
        Channels channels = Get();
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            channels[i] += value;
        }
        return S.Make(channels);
    }

    S Add(S other) {
        Channels channels = Get();
        const Channels added = other.Get();
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            channels[i] += added[i];
        }
        return S.Make(channels);
    }

    S Scaled(float factor) {
        Channels channels = Get();
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            channels[i] *= factor;
        }
        return S.Make(channels);
    }

    S Lerp(S other, float amount) {
        Channels channels = Get();
        const Channels target = other.Get();
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            channels[i] = lerp(channels[i], target[i], amount);
        }
        return S.Make(channels);
    }
}
