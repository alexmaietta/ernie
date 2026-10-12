// SPDX-License-Identifier: BSD-3-Clause
// Materials, shared by every raster drawable: the material uniform at binding 5, whose `heap`
// names its albedo map, tileable detail albedo and normal textures (with hex-tiling) and mesh
// textures in the bindless heap; 6 to 9 are free.
import bindless;
import brdf;
import frame;

public struct Surface {
    // rgb: albedo where no albedo map applies (shading uses the material's spectrum instead),
    // a: luminance of the material's reflectance.
    public float4 albedo;
    // Albedo map: origin x, origin y (top edge), pixel size, 1 when the map is used.
    public float4 albedoMap;
    // Albedo map width and height in cells, detail tiles per world unit, normal green sign
    // (+1 OpenGL, -1 DirectX).
    public float4 albedoMapSize;
    // Detail albedo strength, detail normal strength, has detail albedo, has detail normal.
    public float4 detail;
    // rgb: mean linear colour of the detail albedo texture, a: unused.
    public float4 detailMean;
    // Hex tiling: enabled, rotation strength, luminance contrast, blend exponent.
    public float4 hex;
    // Detail tiles at the frame's origin (`FrameParams::origin`), so tiling stays fixed to the
    // world while positions are relative to it: xyz the fractional tile along each axis; then the
    // hex lattice's phase for the xy and yz projections, and xy the phase for xz (`kHexPeriod`).
    public float4 detailPhase;
    public float4 hexPhaseXyYz;
    public float4 hexPhaseXz;
    // Textures through mesh texture coordinates (mesh.slang): the albedo texture's scale and
    // bias, each texture's coordinate transform (UsdTransform2d) as the rows of a 2 x 3 matrix in
    // xyz, and whether there is an albedo texture (x: 1, or virtual, 2 on a sparse image and 3 in
    // pages), a normal map (y, its encoding: see mesh.slang's `normalTexture`), a roughness
    // texture (z) and a metallic texture (w).
    public float4 albedoTextureScale;
    public float4 albedoTextureBias;
    public float4 albedoUv[2];
    public float4 normalUv[2];
    public float4 roughnessUv[2];
    public float4 metallicUv[2];
    public float4 textures;
    // The opacity texture's coordinate transform, and opacity (x), the threshold below which a
    // surface is cut out (y, 0 for none) and whether an opacity texture replaces x (z).
    public float4 opacityUv[2];
    public float4 opacity;
    // Parameters of the active reflectance model.
    public BrdfParams brdf;
    // Minimal shading's diffuse, glossy and emission mode (`kMinimalGlossy`): the model's
    // roughness and metallic, and 1 when it has a roughness (else no highlights); the emission,
    // linear RGB cd/m^2.
    public float4 gloss;
    public float4 emission;
    // Slots in the bindless heap (`surfaceSlot`): 0 the albedo map, 1 the detail albedo and 2
    // the detail normal textures (`heapTextures`), 3 the detail sampler (`heapSamplers`); 4 to 8
    // the mesh textures' images and 9 to 13 their samplers, by `MaterialTextureSlot`; 14 a virtual
    // albedo texture's min-LOD map, or its pages (`heapTextureArrays`, slot 4 then its tail).
    public uint4 heap[4];
    // A virtual albedo texture (`VirtualTexture`, `virtualTexture`): its first tail mip, mips (0
    // for none), width and height; texels across and down a tile; this frame slot's request
    // flags, one per tile of each mip above the tail.
    public uint4 virtualAlbedo;
    public uint2 virtualTile;
    public uint* virtualRequests;
}

[[vk::binding(5, 0)]] public ConstantBuffer<Surface> surface;

// The material's heap slot `i` (`Surface::heap`).
public uint surfaceSlot(uint i) {
    return surface.heap[i / 4][i % 4];
}

// Albedo map: r albedo times validity, g validity, so filtering fades to the constant albedo at
// gaps.
Texture2D<float4> albedoMap() {
    return heapTextures[surfaceSlot(0)];
}
Texture2D<float4> detailAlbedo() {
    return heapTextures[surfaceSlot(1)];
}
Texture2D<float4> detailNormal() {
    return heapTextures[surfaceSlot(2)];
}
SamplerState detailSampler() {
    return heapSamplers[surfaceSlot(3)];
}

// Tiles across and down mip `mip` of a virtual texture `size` texels across and down.
uint2 virtualTiles(uint2 size, uint mip) {
    let tile = surface.virtualTile;
    return (max(size >> mip, uint2(1)) + tile - 1) / tile;
}

// The first flag (and page table entry) of mip `mip`'s tiles: those of the mips before it.
uint virtualFirstTile(uint2 size, uint mip) {
    uint first = 0;
    for (uint m = 0; m < mip; ++m) {
        let tiles = virtualTiles(size, m);
        first += tiles.x * tiles.y;
    }
    return first;
}

// Mip `mip`'s tile under `wrapped` (texture coordinates in [0, 1)), and where in it, in texels.
uint2 virtualTileAt(uint2 size, uint mip, float2 wrapped, out float2 texel) {
    texel = wrapped * float2(max(size >> mip, uint2(1)));
    return min(uint2(texel) / surface.virtualTile, virtualTiles(size, mip) - 1);
}

// Marks the tile a footprint of level of detail `lod` wants in the request flags
// (`VirtualTexture::BeginFrame` loads it, with its coarser tiles, for later frames).
void virtualRequest(uint2 size, float lod, float2 wrapped) {
    let mip = uint(clamp(floor(lod), 0.0, 31.0));
    if (mip >= surface.virtualAlbedo.x) {
        return;
    }
    float2 texel;
    let tile = virtualTileAt(size, mip, wrapped, texel);
    let flag = surface.virtualRequests + virtualFirstTile(size, mip) +
               tile.y * virtualTiles(size, mip).x + tile.x;
    if (*flag == 0) {
        *flag = 1;
    }
}

// The virtual albedo texture on a sparse image (`textures.x` 2) at `uv`: requests its
// footprint's tile, then samples no finer than the min-LOD map says is resident around it, so
// filtering spans tiles in hardware and never reads an unbound one.
float4 virtualTexture(Texture2D<float4> texture, SamplerState filter, float2 uv) {
    let size = surface.virtualAlbedo.zw;
    let wrapped = uv - floor(uv);
    let lod = texture.CalculateLevelOfDetailUnclamped(filter, uv);
    virtualRequest(size, lod, wrapped);
    float2 texel;
    let tile0 = virtualTileAt(size, 0, wrapped, texel);
    let minLod = heapTextures[surfaceSlot(14)].Load(int3(int2(tile0), 0)).r;
    // Gradients scaled up by 2^(minLod - lod) raise the level of detail to minLod and no further,
    // keeping anisotropic filtering, without the MinLod capability devices may lack.
    let scale = exp2(max(minLod - lod, 0.0));
    return texture.SampleGrad(filter, uv, ddx(uv) * scale, ddy(uv) * scale);
}

// Pages are this many texels square: a tile and a border of `kPageBorder` on each side
// (`PagePool`).
static const float kPageTexels = 128.0;
static const float kPageBorder = 4.0;

// Mip `mip` of the paged virtual albedo texture at `wrapped`, bilinear: from the page of its
// tile there, or of the nearest coarser resident tile (`VirtualTexture`'s page table, after the
// request flags), else from the tail image (heap slot 4) with `filter`.
float4 virtualPageSample(uint2 size, uint mip, float2 wrapped, SamplerState filter) {
    let tail = surface.virtualAlbedo.x;
    let table = surface.virtualRequests + max(virtualFirstTile(size, tail), 1u);
    for (uint m = mip; m < tail; ++m) {
        float2 texel;
        let tile = virtualTileAt(size, m, wrapped, texel);
        let page = table[virtualFirstTile(size, m) + tile.y * virtualTiles(size, m).x + tile.x];
        if (page != 0) {
            let local = texel - float2(tile * surface.virtualTile) + kPageBorder;
            return heapTextureArrays[surfaceSlot(14)].SampleLevel(
                clampSampler(), float3(local / kPageTexels, float(page - 1)), 0.0);
        }
    }
    return heapTextures[surfaceSlot(4)].SampleLevel(filter, wrapped,
                                                    float(max(mip, tail) - tail));
}

// The virtual albedo texture in pages (`textures.x` 3) at `uv`: requests its footprint's tile,
// then blends the two mips around the footprint's level of detail (trilinear, isotropic), each
// from the finest resident page.
float4 virtualPagedTexture(SamplerState filter, float2 uv) {
    let size = surface.virtualAlbedo.zw;
    let wrapped = uv - floor(uv);
    let dx = ddx(uv) * float2(size);
    let dy = ddy(uv) * float2(size);
    let lod = max(0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-12)), 0.0);
    virtualRequest(size, lod, wrapped);
    let mip = min(uint(floor(lod)), surface.virtualAlbedo.y - 1);
    let next = min(mip + 1, surface.virtualAlbedo.y - 1);
    return lerp(virtualPageSample(size, mip, wrapped, filter),
                virtualPageSample(size, next, wrapped, filter), frac(lod));
}

// Mesh texture `slot` (0 albedo, 1 normal, 2 roughness, 3 metallic, 4 opacity) at `uv`, with its
// own sampler; in fragment shaders.
public float4 materialTexture(uint slot, float2 uv) {
    let texture = heapTextures[surfaceSlot(4 + slot)];
    let filter = heapSamplers[surfaceSlot(9 + slot)];
    if (slot == 0 && surface.textures.x > 2.5) {
        return virtualPagedTexture(filter, uv);
    }
    if (slot == 0 && surface.textures.x > 1.5) {
        return virtualTexture(texture, filter, uv);
    }
    return texture.Sample(filter, uv);
}

// Luminance albedo at a world position, which scales the material's reflectance spectrum
// normalised to luminance 1 (white where the material is black): the reflectance's luminance
// without an albedo map, else the map's albedo, falling back to the reflectance's at gaps. So the
// map sets the albedo and the material's spectrum its shape.
public float surfaceAlbedoScale(float3 world) {
    if (surface.albedoMap.w == 0.0) {
        return surface.albedo.a;
    }
    let extent = surface.albedoMapSize.xy * surface.albedoMap.z;
    let uv = float2(world.x - surface.albedoMap.x, surface.albedoMap.y - world.y) / extent;
    let m = albedoMap().SampleLevel(clampSampler(), uv, 0);
    return m.r + (1.0 - m.g) * surface.albedo.a;
}

// Detail texture coordinates: world x east and y north scaled to tiles. Textures are sampled with
// v = -y so image rows run southwards and the image's up direction points north.
public struct DetailCoords {
    public float2 tile;     // in the detail texture's period: the world's tile, plus whole tiles
    public float2 hexTile;  // in the hex lattice's period (`kHexPeriod`)
    public float2 dx;
    public float2 dy;
}

// The planar projections of the world detail textures use: horizontal (terrain, and meshes'
// faces looking up or down), and the two vertical ones meshes blend in.
public static const uint kProjectXy = 0;
public static const uint kProjectYz = 1;
public static const uint kProjectXz = 2;

// Hex tiling's random rotations repeat every this many lattice cells along each lattice axis
// (about 1200 tiles), so the lattice's phase at the frame's origin stays small and exact in float.
static const int kHexPeriod = 4096;

// Detail coordinates of planar projection `projection` of a point, `planar` its coordinates
// relative to the frame's origin, in world units, with x east and y north for the horizontal one.
public DetailCoords detailCoords(float2 planar, uint projection) {
    let local = planar * surface.albedoMapSize.z;
    let phase = surface.detailPhase;
    let tilePhase = projection == kProjectXy   ? phase.xy
                    : projection == kProjectYz ? phase.yz
                                               : phase.xz;
    let hexPhase = projection == kProjectXy   ? surface.hexPhaseXyYz.xy
                   : projection == kProjectYz ? surface.hexPhaseXyYz.zw
                                              : surface.hexPhaseXz.xy;
    DetailCoords c;
    c.tile = local + tilePhase;
    c.hexTile = local + hexPhase;
    c.dx = ddx(local);
    c.dy = ddy(local);
    return c;
}

public struct DetailSample {
    // Multiplier on the base albedo.
    public float3 albedo;
    // Slope (dz/dx east, dz/dy north) added to the terrain's.
    public float2 slope;
    // Variance per axis of the detail slopes the pixel covers about `slope`, which filtering
    // averaged away: roughness below the pixel that shading adds to the model's
    // (`BrdfInput::slopeVariance`).
    public float slopeVariance;
}

float2 toTexture(float2 tile) {
    return float2(tile.x, -tile.y);
}

float2 rotate(float2 v, float2 cs) {
    return float2(cs.x * v.x - cs.y * v.y, cs.y * v.x + cs.x * v.y);
}

public struct DetailTap {
    public float3 albedo;
    public float2 slope;
    public float slopeVariance;
}

// One tap of both detail textures at `tile` (tile space, y north) with derivatives, its tangent
// slope rotated back by `inverse` (cos, sin) into tile space. The normal map's mips hold the mean
// slope's normal and, in alpha, the deviation of the slopes it covers (`MipFilter::NormalSlopes`:
// sigma / (1 + sigma)), which is the same along any axis, so needs no rotating.
DetailTap detailTap(float2 tile, float2 dx, float2 dy, float2 inverse) {
    DetailTap t;
    t.albedo = float3(1.0);
    t.slope = float2(0.0);
    t.slopeVariance = 0.0;
    if (surface.detail.z > 0.0) {
        t.albedo = detailAlbedo()
                       .SampleGrad(detailSampler(), toTexture(tile), toTexture(dx), toTexture(dy))
                       .rgb;
    }
    if (surface.detail.w > 0.0) {
        let texel =
            detailNormal().SampleGrad(detailSampler(), toTexture(tile), toTexture(dx), toTexture(dy));
        // Stored as n * 127 + 128 in 8 bits (`RasterMaterial::SetDetailNormal`).
        var n = (texel.xyz * 255.0 - 128.0) / 127.0;
        n.y = n.y * surface.albedoMapSize.w;
        t.slope = rotate(-n.xy / max(n.z, 0.05), inverse);
        let sigma = min(texel.w, 0.999) / (1.0 - min(texel.w, 0.999));
        t.slopeVariance = sigma * sigma;
    }
    return t;
}

// pcg2d (Jarzynski and Olano 2020).
uint2 pcg2d(uint2 seed) {
    var v = seed * 1664525u + 1013904223u;
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}

// Hex-tiling (Mikkelsen 2022): the plane is covered by a triangle grid whose vertices are hexagon
// centres; each vertex samples the texture with its own random rotation and offset, and the three
// samples are blended with barycentric weights sharpened by an exponent and biased towards brighter
// samples, which hides the seams without visible blur. Slopes from the normal map are rotated back
// with each sample's rotation so lighting stays consistent across tiles.
public DetailTap hexDetail(float2 tile, float2 dx, float2 dy) {
    // Skew into the simplex triangle grid with unit edges of the hexagon lattice.
    let st = tile * 2.0 * sqrt(3.0);
    let skewed = float2(st.x - 0.57735027 * st.y, 1.15470054 * st.y);
    let base = int2(floor(skewed));
    let f = frac(skewed);
    let z = 1.0 - f.x - f.y;
    let upper = z < 0.0 ? 1.0 : 0.0;
    let flip = 2.0 * upper - 1.0;
    let weights = float3(-z * flip, upper - f.y * flip, upper - f.x * flip);
    let iu = int(upper);
    let vertices = int2[3](base + int2(iu, iu), base + int2(iu, 1 - iu), base + int2(1 - iu, iu));

    DetailTap taps[3];
    var luminance = float3(1.0);
    for (uint k = 0; k < 3; ++k) {
        let vertex = vertices[k];
        let centre = float2(float(vertex.x) + 0.5 * float(vertex.y), float(vertex.y) / 1.15470054) /
                     (2.0 * sqrt(3.0));
        // Random per cell of a lattice repeating every kHexPeriod cells, and sampled around the
        // cell's centre, so the pattern depends only on where the cell is in the world.
        let h = pcg2d(asuint(((vertex % kHexPeriod) + kHexPeriod) % kHexPeriod));
        let angle = (float(h.x) / 4294967295.0 * 2.0 - 1.0) * PI * surface.hex.y;
        let cs = float2(cos(angle), sin(angle));
        let offset = float2(pcg2d(h)) / 4294967295.0;
        let rotated = rotate(tile - centre, cs) + offset;
        taps[k] = detailTap(rotated, rotate(dx, cs), rotate(dy, cs), float2(cs.x, -cs.y));
        luminance[k] = dot(taps[k].albedo, float3(0.299, 0.587, 0.114));
    }
    // Weights relative to the largest one keep the dominant term at 1, so high exponents cannot
    // underflow the normalisation.
    let contrast = lerp(float3(1.0), luminance, surface.hex.z);
    let largest = max(weights.x, max(weights.y, weights.z));
    var w = contrast * pow(weights / largest, float3(surface.hex.w));
    w = w / max(w.x + w.y + w.z, 1e-6);

    DetailTap result;
    result.albedo = w.x * taps[0].albedo + w.y * taps[1].albedo + w.z * taps[2].albedo;
    result.slope = w.x * taps[0].slope + w.y * taps[1].slope + w.z * taps[2].slope;
    // The blend's variance: each tap's own, plus its slope's spread about the blend's.
    result.slopeVariance = 0.0;
    for (uint k = 0; k < 3; ++k) {
        let offset = taps[k].slope - result.slope;
        result.slopeVariance += w[k] * (taps[k].slopeVariance + 0.5 * dot(offset, offset));
    }
    return result;
}

public DetailSample sampleDetail(DetailCoords c) {
    DetailTap tap;
    if (surface.hex.x > 0.0) {
        tap = hexDetail(c.hexTile, c.dx, c.dy);
    } else {
        tap = detailTap(c.tile, c.dx, c.dy, float2(1.0, 0.0));
    }
    DetailSample d;
    let normalised = tap.albedo / max(surface.detailMean.rgb, float3(1e-4));
    d.albedo = lerp(float3(1.0), normalised, surface.detail.x * surface.detail.z);
    d.slope = tap.slope * surface.detail.y;
    d.slopeVariance = tap.slopeVariance * surface.detail.y * surface.detail.y;
    return d;
}

// Packs the slope variance of `DetailSample` with a pixel's flags into the G-buffer's ids.w: the
// flags in the low 8 bits; a textured model parameter in [0, 1] (`MaterialSettings::
// metallicParam`), 0 without one, in the next 8 as 8-bit fixed point; and in the high 16 the
// slopes' deviation sigma as sigma / (1 + sigma) in 16-bit fixed point (shading.slang unpacks
// them).
// The G-buffer flags' bits for light link group `group` (`LinkGroups`), bits 1 to 5: shading
// lights a surface only with the lights whose masks hold its group.
public uint linkGroupFlags(uint group) {
    return (group & 31u) << 1;
}

public uint packSurfaceFlags(uint flags, float slopeVariance, float parameter = 0.0) {
    let sigma = sqrt(max(slopeVariance, 0.0));
    let encoded = uint(round(sigma / (1.0 + sigma) * 65535.0));
    let value = uint(round(saturate(parameter) * 255.0));
    return (flags & 0xFFu) | (value << 8) | (min(encoded, 65534u) << 16);
}

// Minimal shading (`Uniforms::minimal`), RTX Minimal's modes for throughput: the w mode's low
// bits are no colour (black, for the other outputs alone), one constant colour, the surface's
// diffuse colour, or that with glossy highlights and emission.
public static const uint kMinimalNone = 0;
public static const uint kMinimalConstant = 1;
public static const uint kMinimalTextured = 2;
public static const uint kMinimalGlossy = 3;

// What minimal passes write (`MinimalTargets`): sRGB colour, and the prim id, instance id, unit
// normal (octahedral, two 16-bit snorm) and surface flags.
public struct Minimal {
    public float4 color : SV_Target0;
    public uint4 ids : SV_Target1;
}

// Unit vector `n` as two 16-bit snorm octahedral coordinates.
//
// Reference: Z. H. Cigolle, S. Donow, D. Evangelakos, M. Mara, M. McGuire and Q. Meyer, "A Survey
// of Efficient Representations for Independent Unit Vectors", JCGT 3(2), 2014.
// http://jcgt.org/published/0003/02/01/
public uint packOctahedral(float3 n) {
    var p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-12);
    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * select(p >= 0.0, float2(1.0), float2(-1.0));
    }
    let q = int2(round(clamp(p, -1.0, 1.0) * 32767.0));
    return (uint(q.x) & 0xFFFFu) | (uint(q.y) << 16);
}

float srgbEncode(float c) {
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// Minimal shading of what a G-buffer pass wrote at `world`: Lambert under the distant lights
// (`u.minimalLights`), the brightest shadowed (`minimalShadow`) and the others not, plus the
// scene's ambient light (`u.motion.yzw`), each as a share of what the brightest gives a white
// surface facing it, of the constant colour or of the material's linear colour scaled to the
// albedo (an albedo map's) and modulated (textures, vertex colours, tints); unlit surfaces show
// that colour as it is. With glossy highlights and emission (`kMinimalGlossy`), a model with a
// roughness (`Surface.gloss`) adds each light's GGX highlight, the metal's tinted, its diffuse
// light falling with its metallic, and every surface its emission (`Surface.emission`) on the
// lights' scale (`u.minimal.x`). Exposed, clamped and encoded as sRGB.
//
// References:
// - B. Walter, S. R. Marschner, H. Li and K. E. Torrance, "Microfacet Models for Refraction
//   through Rough Surfaces", EGSR 2007 (GGX).
// - B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 course notes (Schlick's Fresnel
//   and the Smith term's k = alpha / 2).
public Minimal minimalShade(GBuffer g, float3 world) {
    Minimal out;
    out.ids = uint4(g.ids.y, g.ids.z, packOctahedral(normalize(g.normal.xyz)), g.ids.w);
    let mode = uint(u.minimal.w) & 3u;
    float3 color = float3(0.0);
    if (mode != kMinimalNone) {
        var base = u.minimal.rgb;
        if (mode == kMinimalTextured || mode == kMinimalGlossy) {
            base = surface.albedo.rgb / max(surface.albedo.a, 1e-6) * g.albedo.x * g.albedo.yzw;
        }
        let glossy = mode == kMinimalGlossy && surface.gloss.z > 0.5;
        if ((g.ids.w & kUnlit) != 0) {
            color = base;
        } else {
            let n = normalize(g.normal.xyz);
            let v = normalize(u.eye.xyz - world);
            let alpha = max(surface.gloss.x * surface.gloss.x, 1e-3);
            let metallic = glossy ? saturate(surface.gloss.y) : 0.0;
            let f0 = lerp(float3(0.04), base, metallic);
            var lit = 0.0;
            var highlight = float3(0.0);
            for (uint i = 0; i < 4; ++i) {
                let light = u.minimalLights[i];
                let nl = max(dot(n, light.xyz), 0.0);
                if (light.w <= 0.0 || nl <= 0.0) {
                    continue;
                }
                // The main light (the first) casts the shadows.
                let shadow = i == 0 ? minimalShadow(world, n) : 1.0;
                lit += light.w * nl * shadow;
                if (glossy) {
                    let h = normalize(light.xyz + v);
                    let nh = max(dot(n, h), 0.0);
                    let nv = max(dot(n, v), 1e-4);
                    let a2 = alpha * alpha;
                    let d = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
                    let k = 0.5 * alpha;
                    let visibility = 1.0 / ((nl * (1.0 - k) + k) * (nv * (1.0 - k) + k));
                    let fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
                    // The lights' shares are of a white Lambert surface's pi E / pi, so a BRDF f
                    // gives pi f nl of them.
                    highlight += light.w * shadow * PI * 0.25 * d * visibility * fresnel * nl;
                }
            }
            color = base * (1.0 - metallic) * (lit + u.motion.yzw) + highlight;
        }
        if (mode == kMinimalGlossy) {
            color += surface.emission.rgb * u.minimal.x;
        }
        color = saturate(color * u.eye.w);
    }
    out.color = float4(srgbEncode(color.r), srgbEncode(color.g), srgbEncode(color.b), 1.0);
    return out;
}
