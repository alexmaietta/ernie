// SPDX-License-Identifier: BSD-3-Clause
/// @file affine.slang
/// @brief 3x4 affine transforms as float rows: identity, loads, stores and products.
import ref;

/// A 3x4 affine transform's rows; its fourth row is (0, 0, 0, 1).
struct Affine {
    float4 rows[3];
};

Affine Identity() {
    Affine m;
    m.rows[0] = float4(1, 0, 0, 0);
    m.rows[1] = float4(0, 1, 0, 0);
    m.rows[2] = float4(0, 0, 1, 0);
    return m;
}

/// Each transform as three float4 rows.
Affine Load(Ref<float4> matrices, uint index) {
    Affine m;
    for (uint r = 0; r < 3; ++r) {
        m.rows[r] = matrices[3 * index + r];
    }
    return m;
}

void Store(Ref<float4> matrices, uint index, Affine m) {
    for (uint r = 0; r < 3; ++r) {
        matrices[3 * index + r] = m.rows[r];
    }
}

Affine Mul(Affine a, Affine b) {
    Affine c;
    for (uint r = 0; r < 3; ++r) {
        float4 row = a.rows[r];
        c.rows[r] = row.x * b.rows[0] + row.y * b.rows[1] + row.z * b.rows[2] +
                    float4(0, 0, 0, row.w);
    }
    return c;
}
