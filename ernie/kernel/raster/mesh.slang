// SPDX-License-Identifier: BSD-3-Clause
// Mesh and PointInstancer passes. Geometry is pulled from storage buffers rather than vertex
// buffers, read through their addresses. Besides the frame (bindings 0 and 1, frame.slang) and
// the draw's material (5, surface.slang, whose textures come from the bindless heap: its albedo
// texture, normal map, roughness, metallic and opacity texture, each with its own sampler, through
// `materialTexture`) the passes bind
//   10 the buffers' addresses (`MeshResources`).
// The colour pass writes the G-buffer for spectral shading (frame.slang).
import frame;
import gbuffer;
import surface;

static const uint kHasNormals = 1;
// The draw's light link group (`LinkGroups`) in the flags' top bits.
static const uint kLinkGroupShift = 24;
static const uint kHasColors = 2;
static const uint kInstanced = 4;
static const uint kHasOrientations = 8;
static const uint kHasScales = 16;
static const uint kHasUvs = 32;
static const uint kHasInstanceColors = 64;
// The frame has thermal emission on: the material's temperature texture applies.
static const uint kThermal = 128;
// The mesh has temperatures per point (`temperatures`).
static const uint kHasTemperatures = 256;
// The mesh has a thermal network's nodes and their temperatures (`nodeIds`).
static const uint kHasNodes = 512;
// Smooth tangents per point, for anisotropic materials, in place of a triangle's own.
static const uint kHasTangents = 1024;
// Instanced: kelvin per instance in place of the draw's own (`instanceTemperatures`).
static const uint kHasInstanceTemperatures = 2048;
// Instanced: whether each instance is drawn (`instanceVisible`); a hidden one's triangles have
// no area.
static const uint kHasInstanceVisibility = 4096;
// Normals per triangle corner (`cornerNormals`, face-varying or uniform) in place of per point.
static const uint kHasCornerNormals = 8192;

struct MeshDraw {
    // World transform of the mesh, or of the instancer for instanced draws.
    float4x4 model;
    // Instanced draws: the prototype's own transform, applied before the instance's.
    float4x4 prototype;
    // rgb: tint on the material's albedo where the mesh has no vertex colours, a: its own
    // temperature in kelvin (0 for its material's, or with thermal emission off).
    float4 tint;
    // First index, first vertex, first instance id, flags.
    uint4 offsets;
    // First point in `points` (which differs from the first vertex for bound points), material,
    // prim id, and the instancer's first entry in the instance arrays.
    uint4 points;
    // `model` at the last frame, for motion blur (Uniforms.motion).
    float4x4 previous;
    // Where the instance arrays and points are in their double-buffered pools: x the instances'
    // current offset and y their last frame's, z the points' current offset and w their last
    // frame's (equal to the current ones when they stood still).
    uint4 motion;
    // x: the mesh's first corner in `uvs` (its levels of detail's corners follow, as in
    // `indices`); y: its slot in `activeCounts` plus 1, 0 for all its triangles drawn; z: with a
    // slot, the points it holds; w its thermal network's first node in `nodeTemperatures`.
    uint4 uv;
    // x: the mesh's first corner in `cornerNormals` (as `uv.x` in `uvs`); yzw unused.
    uint4 corners;
}

// The passes' buffers' addresses (renderer's `MeshResources`, mesh_drawable.cpp).
struct MeshResources {
    float* points;
    float* normals;
    // Triangle indices.
    uint* indices;
    MeshDraw* draws;
    // Vertex colours.
    float* colors;
    float* positions;
    // Quaternions as (x, y, z, w), imaginary part first, as in USD's quath memory layout.
    float* orientations;
    float* scales;
    // Per instanced draw, the instances of its prototype as indices into the instance arrays
    // (those culling left, for culled draws).
    uint* instanceIds;
    // Per draw of an indirect multi-draw, what its push constants would hold (draw, instance
    // offset, index offset, unused).
    uint4* drawRecords;
    // Texture coordinates (s, t) per corner of every mesh's triangles, its levels of detail's too.
    float* uvs;
    // Linear RGB per instance, double-buffered as `scales`, in place of the prototype's colours
    // (`kHasInstanceColors`).
    float* instanceColors;
    // Meshes' points and triangle indices in use, pairs by slot (`MeshDraw.uv.y`).
    int* activeCounts;
    // Meshes' temperatures in kelvin per point (`kHasTemperatures`), a thermal solver's.
    float* temperatures;
    // Their thermal networks (`kHasNodes`): each point's node, and every mesh's nodes'
    // temperatures from its first (`MeshDraw.uv.w`).
    uint* nodeIds;
    float* nodeTemperatures;
    // Smooth tangents per point (`kHasTangents`), as `normals`.
    float* tangents;
    // Kelvin per instance (`kHasInstanceTemperatures`), and whether each instance is drawn
    // (`kHasInstanceVisibility`, 0 hides it), double-buffered as `scales`.
    float* instanceTemperatures;
    float* instanceVisible;
    // Normals per corner of every mesh's triangles (`kHasCornerNormals`), its levels of detail's
    // too.
    float* cornerNormals;
}
[[vk::binding(10, 0)]] ConstantBuffer<MeshResources> meshResources;
static float* points = meshResources.points;
static float* normals = meshResources.normals;
static uint* indices = meshResources.indices;
static MeshDraw* draws = meshResources.draws;
static float* colors = meshResources.colors;
static float* positions = meshResources.positions;
static float* orientations = meshResources.orientations;
static float* scales = meshResources.scales;
static uint* instanceIds = meshResources.instanceIds;
static uint4* drawRecords = meshResources.drawRecords;
static float* uvs = meshResources.uvs;
static float* instanceColors = meshResources.instanceColors;
static int* activeCounts = meshResources.activeCounts;
static float* temperatures = meshResources.temperatures;
static uint* nodeIds = meshResources.nodeIds;
static float* nodeTemperatures = meshResources.nodeTemperatures;
static float* tangents = meshResources.tangents;
static float* instanceTemperatures = meshResources.instanceTemperatures;
static float* instanceVisible = meshResources.instanceVisible;
static float* cornerNormals = meshResources.cornerNormals;
// The material's textures (`materialTexture` slots, `MaterialTextureSlot`): albedo; tangent-space
// normals as `surface.textures.y` says (1 stored as n * 127 + 128 in 8 bits, with the deviation of
// the slopes each texel covers in alpha, `MipFilter::NormalSlopes`, as the detail normal map; from
// DDS and KTX2 files as stored, 2 n / 2 + 1/2 in rgb, 3 the same in rg, BC5, with z rebuilt);
// roughness and metallic, one channel each in r; opacity, one channel in r, for cutouts
// (`meshCutoutFragmentMain`, `meshShadowCutoutFragment`).
static const uint kAlbedoTexture = 0;
static const uint kNormalTexture = 1;
static const uint kRoughnessTexture = 2;
static const uint kMetallicTexture = 3;
static const uint kOpacityTexture = 4;

// What one draw draws: the draw (`draws` entry), its first instance id and its first index past
// the mesh's own.
struct DrawCall {
    uint item;
    uint instanceOffset;
    uint indexOffset;
}

// Draw `drawIndex` of the current (multi-)draw call.
DrawCall drawCall(uint drawIndex) {
    DrawCall call;
    if (draw.records != 0) {
        let record = drawRecords[draw.records - 1 + drawIndex];
        call.item = record.x;
        call.instanceOffset = record.y;
        call.indexOffset = record.z;
    } else {
        call.item = draw.first;
        call.instanceOffset = draw.instanceOffset;
        call.indexOffset = draw.indexOffset;
    }
    return call;
}

float3 load3(float* buffer, uint i) {
    return float3(buffer[3 * i], buffer[3 * i + 1], buffer[3 * i + 2]);
}

float3 transformPoint(float4x4 m, float3 p) {
    return mul(m, float4(p, 1.0)).xyz;
}

// Normal through the upper 3x3 of `m` by its cofactor matrix, proportional to the inverse
// transpose, so non-uniform scales keep normals perpendicular; not normalised.
float3 transformNormal(float4x4 m, float3 n) {
    let c0 = float3(m[0][0], m[1][0], m[2][0]);
    let c1 = float3(m[0][1], m[1][1], m[2][1]);
    let c2 = float3(m[0][2], m[1][2], m[2][2]);
    let product = n.x * cross(c1, c2) + n.y * cross(c2, c0) + n.z * cross(c0, c1);
    return dot(c0, cross(c1, c2)) < 0.0 ? -product : product;
}

float3 rotate(float4 q, float3 v) {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

struct MeshVertex {
    float3 world;
    float3 normal;
    float3 tangent;
    float3 color;
    float kelvin;
    float2 uv;
    uint corner;
    uint instance;
    uint item;
}

// A vertex's world position and normal: its point at `pointBase` in the points pool, placed as
// instance `instance` at `instanceBase` in the instance pools (instanced draws), through `model`.
struct Placed {
    float3 world;
    float3 normal;
    float3 tangent;
}

Placed placeVertex(MeshDraw d, uint corner, uint instance, uint pointBase, uint instanceBase,
                   float4x4 model, uint triangleCorner) {
    let flags = d.offsets.w;
    var p = load3(points, pointBase + d.points.x + corner);
    var n = (flags & kHasCornerNormals) != 0 ? load3(cornerNormals, d.corners.x + triangleCorner)
            : (flags & kHasNormals) != 0     ? load3(normals, d.offsets.y + corner)
                                             : float3(0.0);
    // A tangent lies in the surface, so moves as its points do (without their translation).
    var t = (flags & kHasTangents) != 0 ? load3(tangents, d.offsets.y + corner) : float3(0.0);
    if ((flags & kInstanced) != 0) {
        let i = instanceBase + instance;
        p = transformPoint(d.prototype, p);
        n = transformNormal(d.prototype, n);
        t = mul((float3x3)d.prototype, t);
        if ((flags & kHasScales) != 0) {
            let scale = load3(scales, i);
            p = p * scale;
            n = n / scale;
            t = t * scale;
        }
        if ((flags & kHasOrientations) != 0) {
            let q = float4(orientations[4 * i], orientations[4 * i + 1], orientations[4 * i + 2],
                           orientations[4 * i + 3]);
            let unit = q / max(length(q), 1e-20);
            p = rotate(unit, p);
            n = rotate(unit, n);
            t = rotate(unit, t);
        }
        p = p + load3(positions, i);
    }
    Placed placed;
    placed.world = transformPoint(model, p);
    placed.normal = transformNormal(model, n);
    placed.tangent = mul((float3x3)model, t);
    return placed;
}

// Vertex `vertexIndex` (three per triangle) of instance `instanceIndex` of the current draw, at
// `motion` of the way from the last frame (0) to this one (1): its transform, instance arrays and
// points each interpolated between their last frame's and this one's.
MeshVertex meshVertex(uint vertexIndex, uint instanceIndex, DrawCall call, float motion = 1.0) {
    let d = draws[call.item];
    let flags = d.offsets.w;
    // A triangle past the mesh's indices in use (`activeCounts`, slot d.uv.y - 1, whole
    // triangles of them), or with a corner past its points in use (as a simulator's overflow
    // leaves), takes its first point at all three corners: no area, so nothing drawn.
    var dead = false;
    if (d.uv.y != 0) {
        let slot = d.uv.y - 1;
        let at = call.indexOffset + vertexIndex;
        let first = d.offsets.x + at - vertexIndex % 3;
        let points = min(uint(max(activeCounts[2 * slot], 0)), d.uv.z);
        dead = at / 3 >= uint(max(activeCounts[2 * slot + 1], 0)) / 3 || indices[first] >= points ||
               indices[first + 1] >= points || indices[first + 2] >= points;
    }
    let instance = (flags & kInstanced) != 0
                       ? instanceIds[d.offsets.z + call.instanceOffset + instanceIndex]
                       : 0u;
    // A hidden instance draws as dead triangles.
    if ((flags & kHasInstanceVisibility) != 0 && instanceVisible[d.motion.x + instance] == 0.0) {
        dead = true;
    }
    let corner = dead ? 0u : indices[d.offsets.x + call.indexOffset + vertexIndex];
    let index = d.offsets.y + corner;
    let triangleCorner = call.indexOffset + vertexIndex;
    let now = placeVertex(d, corner, instance, d.motion.z, d.motion.x, d.model, triangleCorner);
    MeshVertex v;
    v.world = now.world;
    v.normal = now.normal;
    v.tangent = now.tangent;
    if (motion != 1.0) {
        let before =
            placeVertex(d, corner, instance, d.motion.w, d.motion.y, d.previous, triangleCorner);
        v.world = lerp(before.world, now.world, motion);
        v.normal = lerp(before.normal, now.normal, motion);
        v.tangent = lerp(before.tangent, now.tangent, motion);
    }
    v.color = (flags & kHasColors) != 0 ? load3(colors, index) : d.tint.rgb;
    if ((flags & kHasInstanceColors) != 0) {
        v.color = load3(instanceColors, d.motion.x + instance);
    }
    // A solver's temperature: the point's own, else its node's.
    v.kelvin = 0.0;
    if ((flags & kHasTemperatures) != 0) {
        v.kelvin = temperatures[index];
    } else if ((flags & kHasNodes) != 0) {
        v.kelvin = nodeTemperatures[d.uv.w + nodeIds[index]];
    } else if ((flags & kHasInstanceTemperatures) != 0) {
        v.kelvin = instanceTemperatures[d.motion.x + instance];
    }
    v.uv = float2(0.0);
    if ((flags & kHasUvs) != 0) {
        let at = d.uv.x + call.indexOffset + vertexIndex;
        v.uv = float2(uvs[2 * at], uvs[2 * at + 1]);
    }
    v.corner = vertexIndex % 3;
    v.item = call.item;
    v.instance = (flags & kInstanced) != 0
                     ? instanceIds[d.offsets.z + call.instanceOffset + instanceIndex] - d.points.w
                     : 0u;
    return v;
}

struct VertexOutput {
    float4 position : SV_Position;
    float3 world : WORLD;
    float3 normal : NORMAL;
    float3 color : COLOR;
    float kelvin : KELVIN;
    float2 uv : TEXCOORD;
    float3 barycentric : BARYCENTRIC;
    float3 tangent : TANGENT;
    nointerpolation uint instance : INSTANCE;
    nointerpolation uint item : ITEM;
}

// The vertex stage's outputs: the fragment stage's inputs (`VertexOutput`, in the same order)
// and the view's layer of the G-buffer, for one layered pass over every view
// (Uniforms.flags.z).
struct LayeredVertexOutput {
    float4 position : SV_Position;
    float3 world : WORLD;
    float3 normal : NORMAL;
    float3 color : COLOR;
    float kelvin : KELVIN;
    float2 uv : TEXCOORD;
    float3 barycentric : BARYCENTRIC;
    float3 tangent : TANGENT;
    nointerpolation uint instance : INSTANCE;
    nointerpolation uint item : ITEM;
    uint layer : SV_RenderTargetArrayIndex;
}

[shader("vertex")]
LayeredVertexOutput meshVertexMain(uint vertexIndex : SV_VertexID, uint instanceIndex : SV_InstanceID,
                            uint drawIndex : SV_DrawIndex) {
    let barycentric = float3[3](float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
    let v = meshVertex(vertexIndex, instanceIndex, drawCall(drawIndex), u.motion.x);
    LayeredVertexOutput out;
    out.layer = u.flags.z;
    out.position = projectWorld(v.world);
    out.world = v.world;
    out.normal = v.normal;
    out.color = v.color;
    out.kelvin = v.kelvin;
    out.uv = v.uv;
    out.barycentric = barycentric[v.corner];
    out.tangent = v.tangent;
    out.instance = v.instance;
    out.item = v.item;
    return out;
}

// The MotionVector var's pass (`RasterDrawable::RecordMotion`): the moving draws again over the
// G-buffer's depth, each pixel's motion from where its point was at the last frame.
[[vk::binding(20, 0)]] ConstantBuffer<MotionUniforms> motionView;

struct MotionVertexOutput {
    float4 position : SV_Position;
    float4 previous : PREVIOUS;  // the point's clip position at the last frame
    uint layer : SV_RenderTargetArrayIndex;
}

struct MotionFragmentInput {
    float4 position : SV_Position;
    float4 previous : PREVIOUS;
}

[shader("vertex")]
MotionVertexOutput meshMotionVertex(uint vertexIndex : SV_VertexID, uint instanceIndex : SV_InstanceID,
                                    uint drawIndex : SV_DrawIndex) {
    let call = drawCall(drawIndex);
    // As meshVertexMain, so the depth matches the G-buffer's.
    let v = meshVertex(vertexIndex, instanceIndex, call, u.motion.x);
    let d = draws[call.item];
    let corner = indices[d.offsets.x + call.indexOffset + vertexIndex];
    let instance = (d.offsets.w & kInstanced) != 0
                       ? instanceIds[d.offsets.z + call.instanceOffset + instanceIndex]
                       : 0u;
    let before = placeVertex(d, corner, instance, d.motion.w, d.motion.y, d.previous,
                             call.indexOffset + vertexIndex);
    MotionVertexOutput out;
    out.layer = u.flags.z;
    out.position = projectWorld(v.world);
    out.previous = mul(motionView.previousViewProj, float4(before.world, 1.0));
    return out;
}

[shader("fragment")]
float4 meshMotionFragment(MotionFragmentInput in) : SV_Target0 {
    return float4(pixelMotion(motionView, in.position.xy, in.previous), 0.0, 0.0);
}

[shader("vertex")]
ShadowPosition meshShadowVertex(uint vertexIndex : SV_VertexID,
                                uint instanceIndex : SV_InstanceID,
                                uint drawIndex : SV_DrawIndex) {
    // Where the caster is at the pass's sub-frame (Uniforms.motion, 1 without motion blur).
    let v = meshVertex(vertexIndex, instanceIndex, drawCall(drawIndex), u.motion.x);
    return shadowPosition(v.world);
}

// Detail textures projected along the three world axes and blended by the normal (triplanar
// mapping); detail slopes perturb the normal within each projection plane (UDN blending).
//
// Reference: B. Golus, "Normal Mapping for a Triplanar Shader", 2017.
// https://bgolus.medium.com/normal-mapping-for-a-triplanar-shader-10bf39dca05a
DetailSample triplanarDetail(float3 world, inout float3 normal) {
    var w = pow(abs(normal), float3(4.0));
    w = w / max(w.x + w.y + w.z, 1e-6);
    let x = sampleDetail(detailCoords(world.yz, kProjectYz));
    let y = sampleDetail(detailCoords(world.xz, kProjectXz));
    let z = sampleDetail(detailCoords(world.xy, kProjectXy));
    DetailSample d;
    d.albedo = w.x * x.albedo + w.y * y.albedo + w.z * z.albedo;
    d.slope = float2(0.0);
    d.slopeVariance = w.x * x.slopeVariance + w.y * y.slopeVariance + w.z * z.slopeVariance;
    let perturbation = w.x * float3(0.0, -x.slope.x, -x.slope.y) +
                       w.y * float3(-y.slope.x, 0.0, -y.slope.y) +
                       w.z * float3(-z.slope.x, -z.slope.y, 0.0);
    normal = normalize(normal + perturbation);
    return d;
}

// Texture coordinates `st` through a UsdTransform2d (the rows of a 2 x 3 matrix), with t running
// up the image as USD's does (UsdUVTexture reads t = 0 at the image's bottom row).
float2 transformUv(float4 rows[2], float2 st) {
    return float2(dot(rows[0].xy, st) + rows[0].z, dot(rows[1].xy, st) + rows[1].z);
}

float2 imageUv(float2 uv) {
    return float2(uv.x, 1.0 - uv.y);
}

struct UvSample {
    float3 albedo;        // multiplier on the albedo modulation
    float slopeVariance;  // of the normal map's slopes the pixel covers, and the roughness's
    float metallic;       // the metallic texture's value, 0 without one
}

// The material's textures through the mesh's texture coordinates `st`: the albedo texture's
// colour (scaled and biased), the normal map perturbing `normal` in the tangent frame the
// coordinates' screen-space derivatives give, so meshes need no stored tangents, the roughness
// as slope variance alpha^2 / 2 (alpha its square), and the metallic value.
//
// References:
// - C. Schüler, "Normal Mapping without Precomputed Tangents", ShaderX5, 2006, and
//   "Followup: Normal Mapping Without Precomputed Tangents", 2013.
//   http://www.thetenthplanet.de/archives/1180
// - M. S. Mikkelsen, "Surface Gradient-Based Bump Mapping Framework", Journal of Computer
//   Graphics Techniques 9(3), 2020.
UvSample uvTextures(float2 st, float3 world, inout float3 normal) {
    // Derivatives first, outside the branches on the material's textures.
    let normalUv = transformUv(surface.normalUv, st);
    let dpdx = ddx(world);
    let dpdy = ddy(world);
    let duvdx = ddx(normalUv);
    let duvdy = ddy(normalUv);
    UvSample s;
    s.albedo = float3(1.0);
    s.slopeVariance = 0.0;
    s.metallic = 0.0;
    if (surface.textures.z > 0.0) {
        let roughness =
            materialTexture(kRoughnessTexture, imageUv(transformUv(surface.roughnessUv, st))).r;
        let alpha = roughness * roughness;
        s.slopeVariance = 0.5 * alpha * alpha;
    }
    if (surface.textures.w > 0.0) {
        s.metallic =
            materialTexture(kMetallicTexture, imageUv(transformUv(surface.metallicUv, st))).r;
    }
    if (surface.textures.x > 0.0) {
        let c = materialTexture(kAlbedoTexture, imageUv(transformUv(surface.albedoUv, st)));
        s.albedo = max(c.rgb * surface.albedoTextureScale.rgb + surface.albedoTextureBias.rgb,
                       float3(0.0));
    }
    if (surface.textures.y > 0.0) {
        // The cotangent frame: tangent along increasing s, bitangent along increasing t, the
        // gradients of s and t up to the screen Jacobian's determinant, whose sign (negative
        // with Vulkan's downward y, and on mirrored faces) the frame keeps.
        let dp2perp = cross(dpdy, normal);
        let dp1perp = cross(normal, dpdx);
        let handedness = dot(dpdx, dp2perp) < 0.0 ? -1.0 : 1.0;
        let tangent = handedness * (dp2perp * duvdx.x + dp1perp * duvdy.x);
        let bitangent = handedness * (dp2perp * duvdx.y + dp1perp * duvdy.y);
        let size = max(dot(tangent, tangent), dot(bitangent, bitangent));
        if (size > 1e-30) {
            let texel = materialTexture(kNormalTexture, imageUv(normalUv));
            var n = (texel.xyz * 255.0 - 128.0) / 127.0;
            if (surface.textures.y > 2.5) {
                let xy = texel.xy * 2.0 - 1.0;
                n = float3(xy, sqrt(saturate(1.0 - dot(xy, xy))));
            } else if (surface.textures.y > 1.5) {
                n = texel.xyz * 2.0 - 1.0;
            } else {
                let sigma = min(texel.w, 0.999) / (1.0 - min(texel.w, 0.999));
                s.slopeVariance += sigma * sigma;
            }
            let scale = rsqrt(size);
            normal = normalize(tangent * scale * n.x + bitangent * scale * n.y + normal * n.z);
        }
    }
    return s;
}

// The surface's tangent at texture coordinates `st`, along which u grows: the cotangent frame's
// (as `uvTextures`' normal map's), up to scale, which anisotropic materials orient their
// highlights by.
float3 uvTangent(float2 st, float3 world, float3 normal) {
    let dpdx = ddx(world);
    let dpdy = ddy(world);
    let dudx = ddx(st.x);
    let dudy = ddy(st.x);
    let dp2perp = cross(dpdy, normal);
    let dp1perp = cross(normal, dpdx);
    let handedness = dot(dpdx, dp2perp) < 0.0 ? -1.0 : 1.0;
    return handedness * (dp2perp * dudx + dp1perp * dudy);
}

// Meshes are two-sided: normals turn towards the viewer. Without authored normals each triangle
// is lit flat with the normal of its plane. The albedo modulation is the mesh's vertex colours or
// tint times the albedo texture and the detail albedo; the normal map then the detail normals
// perturb the normal; the material's albedo map scales its reflectance by world x and y.
[shader("fragment")]
GBuffer meshFragmentMain(VertexOutput in) {
    return meshGBuffer(in);
}

// The surface's opacity at texture coordinates `st`: its texture's, or its constant.
float surfaceOpacity(float2 st) {
    if (surface.opacity.z > 0.0) {
        return materialTexture(kOpacityTexture, imageUv(transformUv(surface.opacityUv, st))).r;
    }
    return surface.opacity.x;
}

// As `meshFragmentMain` for materials with an opacity threshold, cutting out what falls below
// it (alpha testing), in its own pipeline so the discard costs the others no early depth tests.
// The G-buffer is shaded first, so derivatives hold across the quad.
[shader("fragment")]
GBuffer meshCutoutFragmentMain(VertexOutput in) {
    let out = meshGBuffer(in);
    if (surfaceOpacity(in.uv) < surface.opacity.y) {
        discard;
    }
    return out;
}

// Shading normal `normal` of a two-sided surface, turned to the side the viewer sees: flipped
// where it disagrees with the triangle's own face towards the viewer (`flat`, the screen
// derivatives' cross product, either way round), as on a back face. Not where it merely points a
// little away from the viewer, as interpolated normals do on a curved mesh's silhouette: flipping
// those there turned the unlit side's limb towards the light, speckling it with lit pixels.
float3 twoSided(float3 normal, float3 flat, float3 toEye) {
    let facing = dot(flat, toEye) < 0.0 ? -flat : flat;
    return dot(normal, facing) < 0.0 ? -normal : normal;
}

// Minimal shading without colour (kMinimalNone): the ids and the geometric normal, two-sided,
// with no material evaluated or lighting; and its cutout variant.
Minimal meshIds(VertexOutput in) {
    let d = draws[in.item];
    let flat = cross(ddx(in.world), ddy(in.world));
    let normal = twoSided((d.offsets.w & (kHasNormals | kHasCornerNormals)) != 0 &&
                                  dot(in.normal, in.normal) > 0.0
                              ? normalize(in.normal)
                              : normalize(flat),
                          flat, towardsViewer(in.world));
    Minimal out;
    out.color = float4(0.0, 0.0, 0.0, 1.0);
    out.ids = uint4(d.points.z, in.instance, packOctahedral(normal), 0u);
    return out;
}

[shader("fragment")]
Minimal meshIdsFragmentMain(VertexOutput in) {
    return meshIds(in);
}

[shader("fragment")]
Minimal meshIdsCutoutFragmentMain(VertexOutput in) {
    let out = meshIds(in);
    if (surfaceOpacity(in.uv) < surface.opacity.y) {
        discard;
    }
    return out;
}

// Minimal shading (`minimalShade`) of what `meshFragmentMain` writes, and of its cutout variant.
[shader("fragment")]
Minimal meshMinimalFragmentMain(VertexOutput in) {
    return minimalShade(meshGBuffer(in), in.world);
}

[shader("fragment")]
Minimal meshMinimalCutoutFragmentMain(VertexOutput in) {
    let out = minimalShade(meshGBuffer(in), in.world);
    if (surfaceOpacity(in.uv) < surface.opacity.y) {
        discard;
    }
    return out;
}

// The shadow passes' cutout variant: the caster's texture coordinates, to test its opacity.
struct ShadowCutoutOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
}

// Its vertex stage's outputs: the fragment stage's inputs and the shadow map's layer.
struct ShadowCutoutLayered {
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
    uint layer : SV_RenderTargetArrayIndex;
}

[shader("vertex")]
ShadowCutoutLayered meshShadowCutoutVertex(uint vertexIndex : SV_VertexID,
                                           uint instanceIndex : SV_InstanceID,
                                           uint drawIndex : SV_DrawIndex) {
    let v = meshVertex(vertexIndex, instanceIndex, drawCall(drawIndex), u.motion.x);
    let at = shadowPosition(v.world);
    ShadowCutoutLayered out;
    out.position = at.position;
    out.uv = v.uv;
    out.layer = at.layer;
    return out;
}

[shader("fragment")]
void meshShadowCutoutFragment(ShadowCutoutOutput in) {
    if (surfaceOpacity(in.uv) < surface.opacity.y) {
        discard;
    }
}

// `color` coloured by texture loading when its debug view is on (`Uniforms::flags.y`): magenta
// while the material's albedo texture loads; on a virtual one, green where its tiles are as fine
// as wanted, through yellow a mip coarser to red three or more.
float3 textureLoadingTint(float3 color) {
    if (u.flags.y == 0) {
        return color;
    }
    if (surface.detailMean.a > 0.5) {
        return lerp(color, float3(1.0, 0.0, 1.0), 0.7);
    }
    if (surface.textures.x > 1.5) {
        let coarser = saturate(virtualAlbedoCoarseness() / 3.0);
        let heat = coarser < 0.5
                       ? lerp(float3(0.1, 0.9, 0.1), float3(1.0, 0.9, 0.1), coarser * 2.0)
                       : lerp(float3(1.0, 0.9, 0.1), float3(1.0, 0.1, 0.1), coarser * 2.0 - 1.0);
        return lerp(color, heat, 0.6);
    }
    return color;
}

GBuffer meshGBuffer(VertexOutput in) {
    let d = draws[in.item];
    let toEye = towardsViewer(in.world);
    let flat = cross(ddx(in.world), ddy(in.world));
    var normal = (d.offsets.w & (kHasNormals | kHasCornerNormals)) != 0 &&
                         dot(in.normal, in.normal) > 0.0
                     ? normalize(in.normal)
                     : normalize(flat);
    normal = twoSided(normal, flat, toEye);
    var shadingNormal = normal;
    // Derivatives outside the branch on texture coordinates.
    let tangent = uvTangent(in.uv, in.world, normal);
    let textured = uvTextures(in.uv, in.world, shadingNormal);
    let detail = triplanarDetail(in.world, shadingNormal);
    let modulation = textureLoadingTint(
        cascadeTint(in.color * textured.albedo * detail.albedo, in.world));
    let unlit = hiddenLine();
    GBuffer out;
    // Its temperature: its points' or their nodes' (a solver's), or its instance's, else its own,
    // else its material's
    // thermal texture's (`thermalTextureSlot`: a thermal inertia below 0), else 0 for the
    // material's.
    let thermal = (d.offsets.w & kThermal) != 0;
    let solved =
        (d.offsets.w & (kHasTemperatures | kHasNodes | kHasInstanceTemperatures)) != 0;
    var kelvin = thermal && solved && in.kelvin > 0.0 ? in.kelvin : d.tint.a;
    if (kelvin == 0.0 && (d.offsets.w & (kThermal | kHasUvs)) == (kThermal | kHasUvs)) {
        kelvin = thermalTextureSlot(imageUv(in.uv));
    }
    out.normal = packNormal(shadingNormal, kelvin);
    out.albedo = float4(unlit ? 1.0 : surfaceAlbedoScale(in.world),
                        wireframe(modulation, in.barycentric));
    out.ids = uint4(d.points.y, d.points.z, in.instance,
                    packSurfaceFlags((unlit ? kUnlit : 0u) |
                                         linkGroupFlags((d.offsets.w >> kLinkGroupShift) & 31u),
                                     textured.slopeVariance + detail.slopeVariance,
                                     textured.metallic));
    // The mesh's smooth tangents where it has them, else its triangle's own.
    out.tangent = (d.offsets.w & kHasTangents) != 0 && dot(in.tangent, in.tangent) > 0.0
                      ? packTangent(in.tangent)
                  : (d.offsets.w & kHasUvs) != 0 ? packTangent(tangent)
                                                 : float2(0.0);
    return out;
}
