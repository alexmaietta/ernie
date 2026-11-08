// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.device.kernels.device;

#include "ernie/render/kernels/pipeline/pipeline.h"

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
