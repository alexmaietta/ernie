// SPDX-License-Identifier: BSD-3-Clause
// Shadow map pass: terrain depth from the light, one cascade per draw.
import terrain;
import virtual_shadow_caster;

[shader("vertex")]
ShadowPosition shadowVertex(uint vertexIndex : SV_VertexID, uint instanceIndex : SV_InstanceID) {
    return shadowPosition(gridVertex(vertexIndex, instanceIndex).world);
}

// A virtual shadow map's level (`Shadows::virtualMap`): the fragment written through its pages.
[shader("fragment")]
void virtualShadowFragment(float4 position : SV_Position) {
    writeVirtualShadow(position);
}
