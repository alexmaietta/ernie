// SPDX-License-Identifier: BSD-3-Clause
// Shadow map pass: terrain depth from the light, one cascade per draw.
import terrain;

[shader("vertex")]
ShadowPosition shadowVertex(uint vertexIndex : SV_VertexID, uint instanceIndex : SV_InstanceID) {
    return shadowPosition(gridVertex(vertexIndex, instanceIndex).world);
}
