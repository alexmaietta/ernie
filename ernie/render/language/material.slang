// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.standard;

[__AttributeUsage(_AttributeTargets.Var)]
struct InputAttribute {
    float minimum;
    float maximum;
};

[__AttributeUsage(_AttributeTargets.Var)]
struct ColorAttribute {};

[__AttributeUsage(_AttributeTargets.Var)]
struct DisplayNameAttribute {
    string name;
};
