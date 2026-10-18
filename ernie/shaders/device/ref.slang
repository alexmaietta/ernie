// SPDX-License-Identifier: BSD-3-Clause
/// @file ref.slang
/// @brief `Ref<T>`, the one way shaders reach buffers, so a backend without device addresses can
/// run them. Its 8 bytes are a buffer's device address on Vulkan. WebGPU has no pointers, so
/// there (`ERNIE_WEBGPU`) they are an arena index and a byte offset, read through the arenas
/// bound as byte buffers. Data keeps the same packed layout either way.
///
///     struct Parameters { Ref<float> values; uint count; };
///     parameters.values[i] = 2 * parameters.values[i];

#if defined(ERNIE_WEBGPU)

/// The most arenas a WebGPU kernel reaches.
static const uint kArenas = 8;
[[vk::binding(0, 1)]] RWByteAddressBuffer arenas[kArenas];

struct Ref<T> {
    uint arena;
    /// Bytes into the arena; `kNull` for none.
    uint offset;

    static const uint kNull = 0xffffffff;

    bool IsNull() {
        return offset == kNull;
    }

    __subscript(uint index) -> T {
        get {
            return arenas[arena].Load<T>(offset + index * sizeof(T));
        }
        [nonmutating] set {
            arenas[arena].Store<T>(offset + index * sizeof(T), newValue);
        }
    }
}

#else

struct Ref<T> {
    /// The buffer's device address; 0 for none.
    uint64_t handle;

    bool IsNull() {
        return handle == 0;
    }

    __subscript(uint index) -> T {
        get {
            return ((T*)handle)[index];
        }
        [nonmutating] set {
            ((T*)handle)[index] = newValue;
        }
    }
}

#endif
