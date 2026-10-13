// SPDX-License-Identifier: BSD-3-Clause
// A stand-in for the bridge functions MaterialX shaders call
// (ernie/session/materialx.h, MaterialXShader), exported to the
// generated module it follows, for tests that compile generated code
// without the renderer: constant geometry, an empty uniform buffer, a
// grey texture, one light overhead and a grey sky.

RWStructuredBuffer<float4> stubOutput;

export float3 ernieMxPositionWorld() { return float3(0.0); }
export float3 ernieMxNormalWorld() { return float3(0.0, 0.0, 1.0); }
export float3 ernieMxTangentWorld() { return float3(1.0, 0.0, 0.0); }
export float3 ernieMxBitangentWorld() { return float3(0.0, 1.0, 0.0); }
export float3 ernieMxPositionObject() { return float3(0.0); }
export float3 ernieMxNormalObject() { return float3(0.0, 0.0, 1.0); }
export float3 ernieMxTangentObject() { return float3(1.0, 0.0, 0.0); }
export float3 ernieMxBitangentObject() { return float3(0.0, 1.0, 0.0); }
export float2 ernieMxTexcoord(int set) { return float2(0.5); }
export float4 ernieMxColor(int set) { return float4(1.0); }
export float ernieMxGeomPropFloat(int id) { return 0.0; }
export float2 ernieMxGeomPropFloat2(int id) { return float2(0.0); }
export float3 ernieMxGeomPropFloat3(int id) { return float3(0.0); }
export float4 ernieMxGeomPropFloat4(int id) { return float4(0.0); }
export int ernieMxGeomPropInt(int id) { return 0; }
export float3 ernieMxViewPosition() { return float3(0.0, 0.0, 5.0); }
export float ernieMxTime() { return 0.0; }
export float ernieMxFrame() { return 1.0; }
export float4x4 ernieMxWorldMatrix() { return float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1); }
export float4x4 ernieMxWorldInverseMatrix() { return ernieMxWorldMatrix(); }
export float4x4 ernieMxViewMatrix() { return ernieMxWorldMatrix(); }
export float4x4 ernieMxViewInverseMatrix() { return ernieMxWorldMatrix(); }
export float4x4 ernieMxProjectionMatrix() { return ernieMxWorldMatrix(); }

export float ernieMxLoadFloat(uint offset) { return 0.5; }
export int ernieMxLoadInt(uint offset) { return 0; }
export float2 ernieMxLoadFloat2(uint offset) { return float2(0.5); }
export float3 ernieMxLoadFloat3(uint offset) { return float3(0.5); }
export float4 ernieMxLoadFloat4(uint offset) { return float4(0.5); }
export float3x3 ernieMxLoadFloat3x3(uint offset) { return float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1); }
export float4x4 ernieMxLoadFloat4x4(uint offset) { return ernieMxWorldMatrix(); }

export float4 ernieMxSample(uint slot, float2 uv) { return float4(0.5); }
export float4 ernieMxSampleLevel(uint slot, float2 uv, float lod) { return float4(0.5); }
export float4 ernieMxSampleGrad(uint slot, float2 uv, float2 dx, float2 dy) { return float4(0.5); }
export int2 ernieMxTextureSize(uint slot, int level) { return int2(1); }
export int ernieMxTextureLevels(uint slot) { return 1; }

static float3 stubLight = float3(0.0);
export int ernieMxLightCount() { return 1; }
export float3 ernieMxLightDirection(int light) { return float3(0.0, 0.0, 1.0); }
export void ernieMxLightResponse(int light, float3 response) { stubLight += response; }
export float3 ernieMxEnvironmentRadiance(float3 direction, float alpha) { return float3(0.2); }
export float3 ernieMxEnvironmentIrradiance(float3 normal) { return float3(0.2); }
