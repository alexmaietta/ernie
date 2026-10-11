// SPDX-License-Identifier: BSD-3-Clause
// Shadow map pass: planet depth from the light, one cascade per draw.
import planet;

[shader("vertex")]
ShadowPosition shadowVertex(uint vertexIndex : SV_VertexID, uint instanceIndex : SV_InstanceID) {
    return shadowPosition(planetVertex(vertexIndex, instanceIndex).world);
}
