// SPDX-License-Identifier: BSD-3-Clause
/// @file transform.slang
/// @brief The transform graph on the device: programs of ops evaluated into local transforms,
/// composed level by level into worlds relative to the render origin, all in double-float, and
/// the worlds read back as floats by the kernels that place things.
///
///     import transform;
///     Affine world = LoadWorld(parameters.worlds, slot);
import affine;
import double_float;

/// A program's op kinds, as `TransformGraph::DeviceOp` encodes them; a test holds these to the
/// C++ constants. A folded constant, then each `XformOpKind` after it.
public static const uint kConstant = 0;
public static const uint kTranslate = 1;
public static const uint kScale = 2;
public static const uint kRotateX = 3;
public static const uint kRotateXYZ = 6;
public static const uint kOrient = 12;

/// `DeviceOp::flags`: bit 0 inverts; bits 8 to 15 count the floats the op reads.
public static const uint kInvert = 1;
public static const uint kFloatsShift = 8;

public static const uint kMaxOps = 8;
/// No slot: a root's parent.
public static const uint kNone = 0xffffffff;

/// A slot's world transform relative to the render origin, as float rows.
public Affine LoadWorld(float4* worlds, uint slot) {
    return Round(LoadDouble(worlds, slot));
}

struct Op { uint kind; uint flags; float* source; float4 hi[3]; float4 lo[3]; };

static const uint3 kAxes[6] = { uint3(0, 1, 2), uint3(0, 2, 1), uint3(1, 0, 2),
                                uint3(1, 2, 0), uint3(2, 0, 1), uint3(2, 1, 0) };

/// A right-handed turn about one axis, as the host's AxisRotation builds it.
Affine Turn(uint axis, float degrees) {
    float c = cos(radians(degrees));
    float s = sin(radians(degrees));
    uint a = (axis + 1) % 3;
    uint b = (axis + 2) % 3;
    Affine m = Identity();
    m.rows[a][a] = c;
    m.rows[a][b] = -s;
    m.rows[b][a] = s;
    m.rows[b][b] = c;
    return m;
}

Affine Inverse(Affine m) {
    float3 a = m.rows[0].xyz;
    float3 b = m.rows[1].xyz;
    float3 c = m.rows[2].xyz;
    float3 r0 = cross(b, c);
    float3 r1 = cross(c, a);
    float3 r2 = cross(a, b);
    float invDet = 1 / dot(a, r0);
    // The inverse's columns are the cofactor rows over the determinant.
    float3x3 inv = transpose(float3x3(r0, r1, r2)) * invDet;
    float3 t = float3(m.rows[0].w, m.rows[1].w, m.rows[2].w);
    float3 back = -mul(inv, t);
    Affine result;
    for (uint r = 0; r < 3; ++r) {
        result.rows[r] = float4(inv[r], back[r]);
    }
    return result;
}

/// A bound op's matrix from the floats it reads.
Affine OpMatrix(Op op) {
    Affine m = Identity();
    float* v = op.source;
    uint floats = (op.flags >> kFloatsShift) & 0xff;
    if (op.kind == kTranslate) {
        m.rows[0].w = v[0];
        m.rows[1].w = v[1];
        m.rows[2].w = v[2];
    } else if (op.kind == kScale) {
        float3 s = floats == 1 ? float3(v[0]) : float3(v[0], v[1], v[2]);
        m.rows[0].x = s.x;
        m.rows[1].y = s.y;
        m.rows[2].z = s.z;
    } else if (op.kind < kRotateXYZ) {
        m = Turn(op.kind - kRotateX, v[0]);
    } else if (op.kind < kOrient) {
        // The order's first axis turns first: R = Rk * Rj * Ri, each by its own axis' angle.
        uint3 axes = kAxes[op.kind - kRotateXYZ];
        m = Mul(Mul(Turn(axes.z, v[axes.z]), Turn(axes.y, v[axes.y])),
                Turn(axes.x, v[axes.x]));
    } else if (op.kind == kOrient) {
        float4 q = normalize(float4(v[0], v[1], v[2], v[3]));
        m.rows[0] = float4(1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w),
                           2 * (q.x * q.z + q.y * q.w), 0);
        m.rows[1] = float4(2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z),
                           2 * (q.y * q.z - q.x * q.w), 0);
        m.rows[2] = float4(2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w),
                           1 - 2 * (q.x * q.x + q.y * q.y), 0);
    } else {
        // Column-major, as the scene's matrices are.
        for (uint r = 0; r < 3; ++r) {
            m.rows[r] = float4(v[r], v[4 + r], v[8 + r], v[12 + r]);
        }
    }
    return (op.flags & kInvert) != 0 ? Inverse(m) : m;
}

/// An op's matrix: a folded constant exactly as the host split it, a bound op from its floats.
DoubleAffine OpDoubleMatrix(Op op) {
    if (op.kind != kConstant) {
        return Promote(OpMatrix(op));
    }
    DoubleAffine m;
    for (uint r = 0; r < 3; ++r) {
        m.rows[r].hi = op.hi[r];
        m.rows[r].lo = op.lo[r];
    }
    return m;
}

struct LocalParameters { uint* slots; Op* programs; uint* lengths; float4* locals; uint count; };

/// One thread a programmed node, its program at its place in the list.
[shader("compute")]
[numthreads(64, 1, 1)]
void Local(uint3 id : SV_DispatchThreadID, uniform LocalParameters parameters) {
    uint at = id.x;
    if (at >= parameters.count) {
        return;
    }
    DoubleAffine local = Promote(Identity());
    for (uint k = 0; k < parameters.lengths[at]; ++k) {
        local = Mul(local, OpDoubleMatrix(parameters.programs[at * kMaxOps + k]));
    }
    StoreDouble(parameters.locals, parameters.slots[at], local);
}

struct ComposeParameters { uint* slots; uint* parents; float4* locals; float4* worlds; uint count; };

/// One level of the graph: each node's world from its parent's and its local.
[shader("compute")]
[numthreads(64, 1, 1)]
void Compose(uint3 id : SV_DispatchThreadID, uniform ComposeParameters parameters) {
    if (id.x >= parameters.count) {
        return;
    }
    uint slot = parameters.slots[id.x];
    uint parent = parameters.parents[slot];
    DoubleAffine local = LoadDouble(parameters.locals, slot);
    StoreDouble(parameters.worlds, slot,
                parent == kNone ? local : Mul(LoadDouble(parameters.worlds, parent), local));
}
