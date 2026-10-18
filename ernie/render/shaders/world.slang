// SPDX-License-Identifier: BSD-3-Clause
/// @file world.slang
/// @brief Reading the transform graph's worlds, which kernels import to find where things are.
import affine;
import double_float;

/// A slot's world transform relative to the render origin, as float rows.
public Affine LoadWorld(float4* worlds, uint slot) {
    return Round(LoadDouble(worlds, slot));
}
