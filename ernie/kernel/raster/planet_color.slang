// SPDX-License-Identifier: BSD-3-Clause
// Planet G-buffer pass: the planet's normals, shadows, albedo and ids for spectral shading.
import planet;
import gbuffer;
import surface;

struct VertexOutput {
    float4 position : SV_Position;
    float3 world : WORLD;
    float height : HEIGHT;
    float2 grid : GRID;
    float3 barycentric : BARYCENTRIC;
    nointerpolation uint face : FACE;
    nointerpolation uint level : LEVEL;
    nointerpolation uint4 page : PAGE;
}

// The vertex stage's outputs: the fragment stage's inputs (`VertexOutput`, in the same order)
// and the view's layer of the G-buffer, for one layered pass over every view
// (Uniforms.flags.z).
struct LayeredVertexOutput {
    float4 position : SV_Position;
    float3 world : WORLD;
    float height : HEIGHT;
    float2 grid : GRID;
    float3 barycentric : BARYCENTRIC;
    nointerpolation uint face : FACE;
    nointerpolation uint level : LEVEL;
    nointerpolation uint4 page : PAGE;
    uint layer : SV_RenderTargetArrayIndex;
}

[shader("vertex")]
LayeredVertexOutput vertexMain(uint vertexIndex : SV_VertexID, uint instanceIndex : SV_InstanceID) {
    let barycentric = float3[6](float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1), float3(1, 0, 0),
                                float3(0, 1, 0), float3(0, 0, 1));
    let v = planetVertex(vertexIndex, instanceIndex);
    LayeredVertexOutput out;
    out.layer = u.flags.z;
    out.position = projectWorld(v.world);
    out.world = v.world;
    out.height = v.height;
    out.grid = v.grid;
    out.barycentric = barycentric[v.corner];
    out.face = v.face;
    out.level = v.level;
    out.page = v.page;
    return out;
}

float3 colormap(float t) {
    let stops = float3[5](float3(0.16, 0.18, 0.30), float3(0.20, 0.42, 0.52),
                          float3(0.56, 0.62, 0.44), float3(0.80, 0.72, 0.52),
                          float3(0.96, 0.95, 0.92));
    let x = clamp(t, 0.0, 1.0) * 4.0;
    let i = uint(min(floor(x), 3.0));
    return lerp(stops[i], stops[i + 1], x - float(i));
}

// Detail textures projected along the three axes and blended by the normal (triplanar), as on
// meshes, since no one plane suits a whole sphere; their slopes perturb `normal`, and the
// variance of the slopes below the pixel goes to `slopeVariance`.
float3 triplanarDetail(float3 world, inout float3 normal, out float slopeVariance) {
    var w = pow(abs(normal), float3(4.0));
    w = w / max(w.x + w.y + w.z, 1e-6);
    let x = sampleDetail(detailCoords(world.yz, kProjectYz));
    let y = sampleDetail(detailCoords(world.xz, kProjectXz));
    let z = sampleDetail(detailCoords(world.xy, kProjectXy));
    let perturbation = w.x * float3(0.0, -x.slope.x, -x.slope.y) +
                       w.y * float3(-y.slope.x, 0.0, -y.slope.y) +
                       w.z * float3(-z.slope.x, -z.slope.y, 0.0);
    normal = normalize(normal + perturbation);
    slopeVariance = w.x * x.slopeVariance + w.y * y.slopeVariance + w.z * z.slopeVariance;
    return w.x * x.albedo + w.y * y.albedo + w.z * z.albedo;
}

static const float kPi = 3.14159265358979;

// The material's albedo texture over the whole body in simple cylindrical projection, at the
// point of face `face` and data grid coordinate `grid`: s the longitude east from -180 degrees, t
// the latitude from the south pole, through the reader's coordinate transform (UsdTransform2d, so
// a grid that starts at 0 degrees east shifts by half a turn), t up the image as USD's.
// Longitude's derivatives across the screen drop the turn it jumps at its seam, so filtering and
// virtual textures' requests see the footprint as it is.
float3 latLongAlbedo(uint face, float2 grid) {
    let d = faceDirection(face, faceCoords(grid));
    let st = float2(atan2(d.y, d.x) / (2.0 * kPi) + 0.5, asin(clamp(d.z, -1.0, 1.0)) / kPi + 0.5);
    var dx = ddx(st);
    var dy = ddy(st);
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    let rows = surface.albedoUv;
    let uv = float2(dot(rows[0].xy, st) + rows[0].z, 1.0 - (dot(rows[1].xy, st) + rows[1].z));
    let uvDx = float2(dot(rows[0].xy, dx), -dot(rows[1].xy, dx));
    let uvDy = float2(dot(rows[0].xy, dy), -dot(rows[1].xy, dy));
    let c = materialTextureGrad(0, uv, uvDx, uvDy);
    return max(c.rgb * surface.albedoTextureScale.rgb + surface.albedoTextureBias.rgb, float3(0.0));
}

// The material's albedo map is not applied: it is a planar raster, which has no place on a
// sphere. Its albedo texture is, in latitude and longitude (`latLongAlbedo`).
[shader("fragment")]
GBuffer fragmentMain(VertexOutput in) {
    float3 up;
    let normal = planetNormal(in.face, in.grid, in.page, up);
    var shadingNormal = toWorld(normal);
    float slopeVariance;
    var modulation = triplanarDetail(in.world, shadingNormal, slopeVariance);
    if (surface.textures.x > 0.0) {
        modulation = modulation * latLongAlbedo(in.face, in.grid);
    }
    let range = max(p.body.w - p.body.z, 1e-6);
    let fraction = (in.height - p.body.z) / range;

    var flags = 0u;
    if (p.dims.y == 1) {
        modulation = colormap(fraction);
        flags = kUnlit;
    } else if (p.dims.y == 2) {
        modulation = modulation * colormap(fraction);
    } else if (p.dims.y == 3) {
        // East and north at the point, north towards +z (at the poles, +y stands in for it).
        let axis = abs(up.z) < 0.9999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
        let east = normalize(cross(axis, up));
        let north = cross(up, east);
        let towards = p.hillshade.x * east + p.hillshade.y * north + p.hillshade.z * up;
        modulation = float3(max(dot(normal, towards), 0.0));
        flags = kUnlit;
    }
    if (hiddenLine()) {
        flags = kUnlit;
    }
    modulation = cascadeTint(modulation, in.world);
    if (p.dims.z == 1) {
        let levelTints = float3[6](float3(1.0, 0.6, 0.6), float3(0.6, 1.0, 0.6),
                                   float3(0.6, 0.7, 1.0), float3(1.0, 1.0, 0.55),
                                   float3(1.0, 0.6, 1.0), float3(0.55, 1.0, 1.0));
        modulation = modulation * levelTints[in.level % 6];
    }
    GBuffer out;
    out.normal = packNormal(shadingNormal, asfloat(p.ids.w));
    let scale = flags == kUnlit ? 1.0 : surface.albedo.a;
    out.albedo = float4(scale, wireframe(modulation, in.barycentric));
    out.ids = uint4(p.ids.x, p.ids.y, 0,
                    packSurfaceFlags(flags | linkGroupFlags(p.ids.z), slopeVariance));
    out.tangent = float2(0.0);  // none
    return out;
}
