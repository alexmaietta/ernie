// SPDX-License-Identifier: BSD-3-Clause
/// @file double_float.slang
/// @brief Double-float arithmetic after Dekker (1971) and Hida, Li and Bailey's QD library (2001):
/// error-free sums and Veltkamp-split products, about 48 bits of mantissa from pairs of floats.
/// `precise` keeps every step as written (SPIR-V NoContraction).
import affine;
import buffer_ref;

/// Four values, each the unevaluated sum of a high float and a low one.
struct Double4 {
    float4 hi;
    float4 lo;
};

Double4 Pair(float4 hi, float4 lo) {
    Double4 d;
    d.hi = hi;
    d.lo = lo;
    return d;
}

/// The sum and its rounding error, for any a and b.
Double4 TwoSum(float4 a, float4 b) {
    precise float4 s = a + b;
    precise float4 v = s - a;
    precise float4 e = (a - (s - v)) + (b - v);
    return Pair(s, e);
}

/// The sum renormalised, for |a| >= |b|.
Double4 QuickTwoSum(float4 a, float4 b) {
    precise float4 s = a + b;
    precise float4 e = b - (s - a);
    return Pair(s, e);
}

Double4 Add(Double4 a, Double4 b) {
    Double4 s = TwoSum(a.hi, b.hi);
    precise float4 e = s.lo + (a.lo + b.lo);
    return QuickTwoSum(s.hi, e);
}

/// Veltkamp's split of each float into two 12-bit halves, whose products are exact.
void Halves(float4 a, out float4 high, out float4 low) {
    precise float4 c = 4097.0 * a;
    precise float4 big = c - a;
    precise float4 h = c - big;
    precise float4 l = a - h;
    high = h;
    low = l;
}

/// The product and its rounding error, without fused multiply-adds: Vulkan and WebGPU do not
/// promise `fma` rounds once.
Double4 TwoProduct(float4 a, float4 b) {
    float4 ah, al, bh, bl;
    Halves(a, ah, al);
    Halves(b, bh, bl);
    precise float4 p = a * b;
    precise float4 e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    return Pair(p, e);
}

/// `a` times the double-float scalar `hi + lo`.
Double4 Scale(Double4 a, float hi, float lo) {
    Double4 p = TwoProduct(a.hi, float4(hi));
    precise float4 e = p.lo + (a.hi * lo + a.lo * hi);
    return QuickTwoSum(p.hi, e);
}

/// A 3x4 affine transform's rows in double-float.
struct DoubleAffine {
    Double4 rows[3];
};

DoubleAffine Promote(Affine m) {
    DoubleAffine d;
    for (uint r = 0; r < 3; ++r) {
        d.rows[r] = Pair(m.rows[r], float4(0));
    }
    return d;
}

DoubleAffine Mul(DoubleAffine a, DoubleAffine b) {
    DoubleAffine c;
    for (uint r = 0; r < 3; ++r) {
        Double4 row = a.rows[r];
        Double4 sum = Pair(float4(0, 0, 0, row.hi.w), float4(0, 0, 0, row.lo.w));
        sum = Add(sum, Scale(b.rows[0], row.hi.x, row.lo.x));
        sum = Add(sum, Scale(b.rows[1], row.hi.y, row.lo.y));
        c.rows[r] = Add(sum, Scale(b.rows[2], row.hi.z, row.lo.z));
    }
    return c;
}

/// Each transform as six float4: its three high rows, then its three low rows.
DoubleAffine LoadDouble(BufferRef<float4> matrices, uint index) {
    DoubleAffine m;
    for (uint r = 0; r < 3; ++r) {
        m.rows[r] = Pair(matrices[6 * index + r], matrices[6 * index + 3 + r]);
    }
    return m;
}

void StoreDouble(BufferRef<float4> matrices, uint index, DoubleAffine m) {
    for (uint r = 0; r < 3; ++r) {
        matrices[6 * index + r] = m.rows[r].hi;
        matrices[6 * index + 3 + r] = m.rows[r].lo;
    }
}

Affine Round(DoubleAffine m) {
    Affine f;
    for (uint r = 0; r < 3; ++r) {
        f.rows[r] = m.rows[r].hi + m.rows[r].lo;
    }
    return f;
}
