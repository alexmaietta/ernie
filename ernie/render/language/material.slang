// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.material.standard;

[__AttributeUsage(_AttributeTargets.Var)]
[__AttributeUsage(_AttributeTargets.Param)]
struct InputAttribute {
    float minimum;
    float maximum;
};

[__AttributeUsage(_AttributeTargets.Var)]
[__AttributeUsage(_AttributeTargets.Param)]
struct ColorAttribute {};

[__AttributeUsage(_AttributeTargets.Var)]
[__AttributeUsage(_AttributeTargets.Param)]
struct DisplayNameAttribute {
    string name;
};

[__AttributeUsage(_AttributeTargets.Function)]
struct PatternAttribute {};
