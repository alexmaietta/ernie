// SPDX-License-Identifier: BSD-3-Clause
// Shadow map pass: planet depth from the light, one cascade per draw.
import planet;

[shader("vertex")]
float4 shadowVertex(uint vertexIndex : SV_VertexID, uint instanceIndex : SV_InstanceID)
    : SV_Position {
    return mul(shadowViewProj(), float4(planetVertex(vertexIndex, instanceIndex).world, 1.0));
}
