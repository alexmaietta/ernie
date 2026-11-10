// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.device.kernels.device;

#include "ernie/render/kernels/sensor_pipeline/pipeline.h"

using namespace ernie;

struct Channels {
    float values[kMostSignalChannels];
    uint count;

    __init(uint count) {
        [ForceUnroll]
        for (uint i = 0; i < kMostSignalChannels; ++i) {
            values[i] = 0;
        }
        this.count = count;
    }

    __subscript(uint i)->float {
        get {
            return values[i];
        }
        set {
            values[i] = newValue;
        }
    }
}

interface ISignal {
    static This Make(Channels channels);
    Channels Get();
}

struct ElectronRate : ISignal {
    Channels channels;

    static ElectronRate Make(Channels channels) {
        return { channels };
    }

    Channels Get() {
        return channels;
    }
}

struct Electrons : ISignal {
    Channels channels;

    static Electrons Make(Channels channels) {
        return { channels };
    }

    Channels Get() {
        return channels;
    }
}

struct Digital : ISignal {
    Channels channels;

    static Digital Make(Channels channels) {
        return { channels };
    }

    Channels Get() {
        return channels;
    }
}

struct LinearColor : ISignal {
    Channels channels;

    static LinearColor Make(Channels channels) {
        return { channels };
    }

    Channels Get() {
        return channels;
    }
}

struct DisplayColor : ISignal {
    Channels channels;

    static DisplayColor Make(Channels channels) {
        return { channels };
    }

    Channels Get() {
        return channels;
    }
}

struct Radiance : ISignal {
    Channels channels;

    static Radiance Make(Channels channels) {
        return { channels };
    }

    Channels Get() {
        return channels;
    }
}

struct Returns : ISignal {
    Channels channels;

    static Returns Make(Channels channels) {
        return { channels };
    }

    Channels Get() {
        return channels;
    }

    property bool returned {
        get {
            return channels[0] != 0;
        }
        set {
            channels[0] = newValue ? 1 : 0;
        }
    }

    property float distance {
        get {
            return channels[1];
        }
        set {
            channels[1] = newValue;
        }
    }

    property float intensity {
        get {
            return channels[2];
        }
        set {
            channels[2] = newValue;
        }
    }

    property float3 position {
        get {
            return float3(channels[3], channels[4], channels[5]);
        }
        set {
            channels[3] = newValue.x;
            channels[4] = newValue.y;
            channels[5] = newValue.z;
        }
    }

    property float radialVelocity {
        get {
            return channels[6];
        }
        set {
            channels[6] = newValue;
        }
    }

    property bool world {
        get {
            return channels[7] != 0;
        }
    }
}

struct CloudSignal : ISignal {
    Channels channels;

    static CloudSignal Make(Channels channels) {
        return { channels };
    }

    Channels Get() {
        return channels;
    }
}
