// Shadow map pass: terrain depth from the light, one cascade per draw.
import terrain;

[shader("vertex")]
float4 shadowVertex(uint vertexIndex : SV_VertexID, uint instanceIndex : SV_InstanceID)
    : SV_Position {
    return mul(shadowViewProj(), float4(gridVertex(vertexIndex, instanceIndex).world, 1.0));
}
