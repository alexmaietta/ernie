// Planet G-buffer pass: the planet's normals, shadows, albedo and ids for spectral shading.
import planet;
import surface;

struct VertexOutput {
    float4 position : SV_Position;
    float3 world : WORLD;
    float height : HEIGHT;
    float2 grid : GRID;
    float3 barycentric : BARYCENTRIC;
    nointerpolation uint face : FACE;
    nointerpolation uint level : LEVEL;
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
// meshes, since no one plane suits a whole sphere; their slopes perturb `normal`.
float3 triplanarDetail(float3 world, inout float3 normal) {
    var w = pow(abs(normal), float3(4.0));
    w = w / max(w.x + w.y + w.z, 1e-6);
    let x = sampleDetail(detailCoords(world.yz, kProjectYz));
    let y = sampleDetail(detailCoords(world.xz, kProjectXz));
    let z = sampleDetail(detailCoords(world.xy, kProjectXy));
    let perturbation = w.x * float3(0.0, -x.slope.x, -x.slope.y) +
                       w.y * float3(-y.slope.x, 0.0, -y.slope.y) +
                       w.z * float3(-z.slope.x, -z.slope.y, 0.0);
    normal = normalize(normal + perturbation);
    return w.x * x.albedo + w.y * y.albedo + w.z * z.albedo;
}

// The material's albedo map is not applied: it is a planar raster, which has no place on a
// sphere.
[shader("fragment")]
GBuffer fragmentMain(VertexOutput in) {
    float3 up;
    let normal = planetNormal(in.face, in.grid, up);
    var shadingNormal = normal;
    var modulation = triplanarDetail(in.world, shadingNormal);
    let range = max(p.body.w - p.body.z, 1e-6);
    let fraction = (in.height - p.body.z) / range;

    var flags = 0u;
    if (p.dims.y == 1) {
        modulation = colormap(fraction);
        flags = kUnlit;
    } else if (p.dims.y == 2) {
        modulation = modulation * colormap(fraction);
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
    out.normal = float4(shadingNormal, flags == kUnlit ? 1.0
                                                       : sunVisibility(in.world, normal,
                                                                       in.position.xy));
    let scale = flags == kUnlit ? 1.0 : surface.albedo.a;
    out.albedo = float4(scale, wireframe(modulation, in.barycentric));
    out.ids = uint4(p.ids.x, p.ids.y, 0, flags);
    return out;
}
