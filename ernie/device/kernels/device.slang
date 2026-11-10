// SPDX-License-Identifier: BSD-3-Clause

#include "ernie/device/kernels/device.h"

using namespace ernie;

uniform DeviceGlobals Device;

[__AttributeUsage(_AttributeTargets.Function)]
public struct MaxRegistersAttribute {
    int count;
};

public static const float kPi = 3.14159265358979f;
public static const float kInfinity = asfloat(0x7F800000u);

public uint LinearGroupIndex() {
    __target_switch {
    case cuda:
        __intrinsic_asm "ernie_linear_group_index()";
    case spirv:
        const uint3 group = spirv_asm { result:$$uint3 = OpLoad builtin(WorkgroupId:uint3); };
        const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
        return (group.z * groups.y + group.y) * groups.x + group.x;
    }
}

public uint LocalThreadIndex() {
    __target_switch {
    case cuda:
        __intrinsic_asm "((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x)";
    case spirv:
        return spirv_asm { result:$$uint = OpLoad builtin(LocalInvocationIndex:uint); };
    }
}

public uint LinearThreadIndex() {
    const uint3 size = WorkgroupSize();
    return LinearGroupIndex() * (size.x * size.y * size.z) + LocalThreadIndex();
}

public uint3 ThreadIndex() {
    __target_switch {
    case cuda:
        __intrinsic_asm "make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z)";
    case spirv:
        return spirv_asm { result:$$uint3 = OpLoad builtin(GlobalInvocationId:uint3); };
    }
}

public uint LaneCount() {
    return WaveGetLaneCount();
}

#if ERNIE_HIP
uint64_t HipBallot(bool value) {
    __intrinsic_asm "__ballot($0)";
}
uint HipLaneIndex() {
    __intrinsic_asm "__lane_id()";
}
#endif

public uint LaneIndex() {
#if ERNIE_HIP
    return HipLaneIndex();
#else
    return WaveGetLaneIndex();
#endif
}

public uint WaveOrderedThreadIndex() {
    __target_switch {
    case cuda:
        __intrinsic_asm "((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x)";
    case spirv:
        const uint wave = spirv_asm { result:$$uint = OpLoad builtin(SubgroupId:uint); };
        return wave * WaveGetLaneCount() + WaveGetLaneIndex();
    }
}

public uint64_t Ballot(bool value) {
#if ERNIE_HIP
    return HipBallot(value);
#else
    const uint4 lanes = WaveActiveBallot(value);
    const uint64_t upper = LaneCount() > 32 ? uint64_t(lanes.y) << 32 : 0;
    return uint64_t(lanes.x) | upper;
#endif
}

extern static const uint kKernelHash;

public interface IPrintable {
    uint2 Printed();
}
public extension uint : IPrintable {
    uint2 Printed() { return uint2(this, kPrintedUint); }
}
public extension int : IPrintable {
    uint2 Printed() { return uint2(asuint(this), kPrintedInt); }
}
public extension float : IPrintable {
    uint2 Printed() { return uint2(asuint(this), kPrintedFloat); }
}
public extension bool : IPrintable {
    uint2 Printed() { return uint2(this ? 1 : 0, kPrintedBool); }
}

void Write(uint kind, String format, uint values, uint2 a, uint2 b, uint2 c, uint2 d) {
    uint* header = Device.debug;
    uint slot;
    InterlockedAdd(header[kTakenWord], 1u, slot);
    if (slot >= Device.debugCapacity) {
        return;
    }
    uint* record = header + kDebugHeaderWords + slot * kDebugRecordWords;
    const uint3 thread = ThreadIndex();
    record[kKindWord] = kind | (values << 8);
    record[kHashWord] = getStringHash(format);
    record[kThreadWord] = thread.x;
    record[kThreadWord + 1] = thread.y;
    record[kThreadWord + 2] = thread.z;
    record[kValuesWord] = a.x;
    record[kValuesWord + 1] = a.y;
    record[kValuesWord + 2] = b.x;
    record[kValuesWord + 3] = b.y;
    record[kValuesWord + 4] = c.x;
    record[kValuesWord + 5] = c.y;
    record[kValuesWord + 6] = d.x;
    record[kValuesWord + 7] = d.y;
    record[kLinearWord] = LinearThreadIndex();
    record[kKernelWord] = kKernelHash;
    record[kStampWord] = header[kSubmissionWord];
}

public void Assert(bool condition, String message) {
    if (!condition) {
        Write(kAssertRecord, message, 0, uint2(0), uint2(0), uint2(0), uint2(0));
    }
}

public void Print(String format) {
    Write(kPrintRecord, format, 0, uint2(0), uint2(0), uint2(0), uint2(0));
}
public void Print<A : IPrintable>(String format, A a) {
    Write(kPrintRecord, format, 1, a.Printed(), uint2(0), uint2(0), uint2(0));
}
public void Print<A : IPrintable, B : IPrintable>(String format, A a, B b) {
    Write(kPrintRecord, format, 2, a.Printed(), b.Printed(), uint2(0), uint2(0));
}
public void Print<A : IPrintable, B : IPrintable, C : IPrintable>(String format, A a, B b, C c) {
    Write(kPrintRecord, format, 3, a.Printed(), b.Printed(), c.Printed(), uint2(0));
}
public void Print<A : IPrintable, B : IPrintable, C : IPrintable, D : IPrintable>(
    String format, A a, B b, C c, D d) {
    Write(kPrintRecord, format, 4, a.Printed(), b.Printed(), c.Printed(), d.Printed());
}
