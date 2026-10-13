// SPDX-License-Identifier: BSD-3-Clause
// The entry point tests compile generated MaterialX code under, after
// stub_bridge.slang and the generated code; in groups of 2x2 quads, as
// MaterialX's code takes derivatives.

[shader("compute")]
[numthreads(8, 8, 1)]
void stubMain(uint3 id : SV_DispatchThreadID) {
    float3 color;
    float3 transparency;
    ernieMaterialX(color, transparency);
    stubOutput[0] = float4(color + stubLight, transparency.x);
}
