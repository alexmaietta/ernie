// Spectral shading pass: turns the G-buffer into spectral radiance, one image layer per band of
// the camera's spectral range. The host composes it with a reflectance model by prepending
//     import <model module>;
//     typealias ActiveBrdf = <model type>;
// and dispatches one pipeline per model, each shading the pixels whose material uses its model.
//
// Radiance is L = E / pi * (a * albedo + (1 - a) * F * V) + sum over lights of E_k / pi * F_k * V_k
// + albedo * L_dome * D, with E the sun's spectral irradiance at normal incidence, F the model's
// radiance factor (pi times the BRDF times cos(i)), V the shadow visibility and a the ambient
// fraction. A distant light's E_k is its irradiance; a sphere light's is its intensity over the
// squared distance (capped at its radius) times its cone's falloff. A rect or disk light's E_k is
// its radiance times the length of its vector form factor (the polygon clipped to the receiver's
// horizon, Lambert's formula), with F_k along that vector. The dome adds its radiance times its
// map's irradiance over pi, D, from spherical harmonics, diffusely. Unlit pixels get
// L = E / pi * albedo.
//
// One dispatch shades every view of a set of render targets, view `z` of the dispatch in their
// layer `z` (a group of `bands` layers of the spectral image) with the uniforms of view
// `firstView + z`.
//
// `skyMain` fills the pixels no geometry covers: the dome's radiance times its map in the pixel's
// direction, plus the stars of a star field, points at infinity spread over the four pixels nearest
// each (a tent filter, which sums to one, so a star's irradiance is kept at any resolution): each
// adds its spectral irradiance times its weight over the pixel's solid angle. Each pixel gathers
// the stars in the grid cells around its direction, in a fixed order, so images repeat exactly.
//
// Bindings: 0 every view's ShadingUniforms, 1 G-buffer normal, 2 G-buffer albedo, 3 G-buffer ids,
// 4 depth (one layer per view each), 5 materials, 6 RGB-to-spectrum coefficient table, 7 its
// sampler, 8 spectral radiance, 9 lights, 10 the lights' shadow maps, 11 their comparison sampler,
// 12 their view-projections, 13 the dome's map, 14 the stars, 15 their grid cells.
//
// References:
// - W. Jakob and J. Hanika, "A Low-Dimensional Function Space for Efficient Spectral Upsampling",
//   Computer Graphics Forum 38(2), 2019 (sigmoid spectra of RGB colours).
// - H. W. Jensen et al., "A Physically-Based Night Sky Model", SIGGRAPH 2001, 399-408 (stars as
//   black bodies of their magnitude's irradiance). https://doi.org/10.1145/383259.383306
// - M. Planck, "Ueber das Gesetz der Energieverteilung im Normalspectrum", Annalen der Physik
//   309(3):553-563, 1901 (black-body spectra).
import brdf;

static const uint kUnlit = 1;
static const uint kBackground = 0xFFFFFFFF;

struct ShadingUniforms {
    float4x4 invViewProj;
    // xyz: camera position.
    float4 eye;
    // xyz: unit direction towards the sun, w: ambient fraction.
    float4 sun;
    // First band's lower wavelength (nm), band width (nm), band count, unused.
    float4 range;
    // Image width and height, coefficient table resolution, light count.
    uint4 size;
    // Sun spectral irradiance at normal incidence per band (W m^-2 nm^-1), four bands per float4.
    float4 irradiance[16];
    // Lights' shadow filter radius (texels), normal offset (texels), map resolution, unused.
    float4 lightShadow;
    // 16 Poisson disk samples in the unit disk, two per float4.
    float4 poisson[8];
    // Dome: irradiance over pi of its RGB map as order-2 spherical harmonics (rgb), and its
    // spectral radiance per band (W m^-2 sr^-1 nm^-1), four per float4; zero without a dome.
    float4 domeSh[9];
    float4 dome[16];
    // x: the lights this view sees, a bit per light (a batch's views see their world's and the
    // shared ones); yzw unused.
    uint4 view;
    // Rows of the rotation from world directions to the dome's frame, and to the star catalogue's.
    float4 toDome[3];
    float4 toCatalogue[3];
    // x: what shows behind the scene (kSky* bits); y, z: the dome map's width and height; w: its
    // mip levels.
    uint4 sky;
    // x: the stars' grid cells around in right ascension, y: in declination; zw unused.
    float4 starGrid;
    // To the views' stride in the buffer, 1280 bytes.
    float4 padding[13];
}

// What `skyMain` draws behind the scene.
static const uint kSkyDome = 1;   // the dome's radiance
static const uint kSkyMap = 2;    // times its map
static const uint kSkyStars = 4;  // and the stars

static const uint kMaxLights = 16;

struct LightGpu {
    // xyz: position (sphere, rect, disk) or unit direction towards the light (distant), w: type,
    // 0 distant, 1 sphere, 2 rect, 3 disk.
    float4 position;
    // xyz: unit cone axis, w: sphere radius.
    float4 axis;
    // Cosine of the cone's angle (-2 without a cone), cosine where its falloff starts, first
    // shadow view, shadow view count (0 without shadows, 1, or 6 for a cube).
    float4 cone;
    // x: world size of a shadow texel, at unit distance for perspective views.
    float4 shadow;
    // Rect and disk: half width along the light's +x and half height along its +y, as vectors.
    float4 extentX;
    float4 extentY;
    // Spectral irradiance (distant, W m^-2 nm^-1) or intensity (sphere, W sr^-1 nm^-1) per band.
    float4 spectrum[16];
}

struct MaterialGpu {
    // Model index, unused.
    uint4 info;
    BrdfParams params;
    // Reflectance per band normalised to luminance 1 (G-buffer albedo.x scales it), four per
    // float4.
    float4 reflectance[16];
}

struct ShadingConstants {
    // x: index of the model this dispatch shades, y: the first view's uniforms, zw unused.
    uint4 model;
}

[[vk::binding(0, 0)]] StructuredBuffer<ShadingUniforms> shadingViews;
[[vk::push_constant]] ConstantBuffer<ShadingConstants> pass;
[[vk::binding(1, 0)]] Texture2DArray<float4> gNormal;
[[vk::binding(2, 0)]] Texture2DArray<float4> gAlbedo;
[[vk::binding(3, 0)]] Texture2DArray<uint4> gIds;
[[vk::binding(4, 0)]] Texture2DArray<float> gDepth;
[[vk::binding(5, 0)]] StructuredBuffer<MaterialGpu> materials;
// Jakob-Hanika sigmoid coefficients (c0, c1, c2) over linear RGB in [0, 1]^3: red along x, green
// along y and blue across layers, at `size.z` samples per axis.
[[vk::binding(6, 0)]] Texture2DArray<float4> sigmoidTable;
[[vk::binding(7, 0)]] SamplerState tableSampler;
[[vk::image_format("r32f")]] [[vk::binding(8, 0)]] RWTexture2DArray<float> spectral;
[[vk::binding(9, 0)]] StructuredBuffer<LightGpu> lights;
[[vk::binding(10, 0)]] Texture2DArray<float> lightShadowMaps;
[[vk::binding(11, 0)]] SamplerComparisonState lightShadowSampler;
[[vk::binding(12, 0)]] StructuredBuffer<float4x4> lightViews;
// The dome's latitude-longitude RGB map with its mips (rows from +z down, columns from +x
// through +y), its stars (right ascension and declination in radians, temperature in kelvin,
// spectral irradiance at 555 nm) sorted by grid cell, and each cell's first star and count.
[[vk::binding(13, 0)]] Texture2D<float4> domeMap;
[[vk::binding(14, 0)]] StructuredBuffer<float4> stars;
[[vk::binding(15, 0)]] StructuredBuffer<uint2> starCells;

// The uniforms of the view this thread shades (`shadeMain` sets it).
static uint shadedView;
#define sh shadingViews[shadedView]

float bandValue(float4 values[16], uint band) {
    return values[band / 4][band % 4];
}

// The modulation colour as a spectrum value at `wavelength`: its largest component times the
// sigmoid spectrum of the colour divided by it (Jakob and Hanika 2019), which reproduces the colour
// for any brightness, including modulations above 1.
float modulationAt(float3 coefficients, float scale, float wavelength) {
    let t = (wavelength - 360.0) / (830.0 - 360.0);
    let x = (coefficients.x * t + coefficients.y) * t + coefficients.z;
    return scale * (0.5 + x / (2.0 * sqrt(1.0 + x * x)));
}

// Sigmoid spectrum coefficients (Jakob and Hanika 2019) of a linear RGB colour whose largest
// component is 1, interpolated from the coefficient table.
float3 sigmoidCoefficients(float3 colour) {
    let resolution = float(sh.size.z);
    let uv = (colour.xy * (resolution - 1.0) + 0.5) / resolution;
    let layer = colour.z * (resolution - 1.0);
    let lower = floor(layer);
    let upper = min(lower + 1.0, resolution - 1.0);
    let a = sigmoidTable.SampleLevel(tableSampler, float3(uv, lower), 0).xyz;
    let b = sigmoidTable.SampleLevel(tableSampler, float3(uv, upper), 0).xyz;
    return lerp(a, b, layer - lower);
}

// The dome map's irradiance over pi at unit normal `n` (Ramamoorthi and Hanrahan 2001).
float3 domeIrradiance(float3 n) {
    let c = sh.domeSh;
    let e = c[0].rgb * 0.282095 + c[1].rgb * 0.488603 * n.y + c[2].rgb * 0.488603 * n.z +
            c[3].rgb * 0.488603 * n.x + c[4].rgb * 1.092548 * n.x * n.y +
            c[5].rgb * 1.092548 * n.y * n.z + c[6].rgb * 0.315392 * (3.0 * n.z * n.z - 1.0) +
            c[7].rgb * 1.092548 * n.x * n.z + c[8].rgb * 0.546274 * (n.x * n.x - n.y * n.y);
    return max(e, float3(0.0));
}

static const uint kDiskSides = 16;
static const uint kMaxPolygon = kDiskSides + 1;

// The vector form factor of a rect or disk light at `world` with unit normal `n`: half the sum
// over the emitter's edges, clipped to the receiver's horizon, of each edge's angle times the unit
// normal of the plane through it and `world` (Lambert's formula; Arvo 1994). Irradiance is the
// light's radiance times dot(n, V). Zero when the emitter faces away or lies below the horizon.
float3 areaFormFactor(LightGpu light, float3 world, float3 n) {
    let centre = light.position.xyz;
    if (dot(world - centre, light.axis.xyz) <= 0.0) {
        return float3(0.0);
    }
    float3 polygon[kMaxPolygon];
    uint count = 0;
    if (uint(light.position.w) == 2) {
        polygon[0] = centre - light.extentX.xyz - light.extentY.xyz;
        polygon[1] = centre + light.extentX.xyz - light.extentY.xyz;
        polygon[2] = centre + light.extentX.xyz + light.extentY.xyz;
        polygon[3] = centre - light.extentX.xyz + light.extentY.xyz;
        count = 4;
    } else {
        // A regular polygon of the disk's area.
        let step = 2.0 * PI / float(kDiskSides);
        let scale = sqrt(step / sin(step));
        for (uint i = 0; i < kDiskSides; ++i) {
            let a = step * float(i);
            polygon[i] = centre + scale * (cos(a) * light.extentX.xyz + sin(a) * light.extentY.xyz);
        }
        count = kDiskSides;
    }
    // Sutherland-Hodgman against the receiver's tangent plane.
    float3 clipped[kMaxPolygon];
    uint kept = 0;
    for (uint i = 0; i < count; ++i) {
        let a = polygon[i];
        let b = polygon[(i + 1) % count];
        let da = dot(n, a - world);
        let db = dot(n, b - world);
        if (da > 0.0) {
            clipped[kept] = a;
            kept += 1;
        }
        if ((da > 0.0) != (db > 0.0) && kept < kMaxPolygon) {
            clipped[kept] = a + (b - a) * (da / (da - db));
            kept += 1;
        }
    }
    if (kept < 3) {
        return float3(0.0);
    }
    var vector = float3(0.0);
    for (uint i = 0; i < kept; ++i) {
        let u = normalize(clipped[i] - world);
        let w = normalize(clipped[(i + 1) % kept] - world);
        let edge = cross(u, w);
        let sine = length(edge);
        if (sine > 1e-7) {
            vector += acos(clamp(dot(u, w), -1.0, 1.0)) * edge / sine;
        }
    }
    vector *= 0.5;
    return dot(vector, n) < 0.0 ? -vector : vector;
}

// Jimenez 2014, "Next Generation Post Processing in Call of Duty: Advanced Warfare".
float interleavedGradientNoise(float2 pixel) {
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

// The shadow view of a light's maps that sees `world`: its only one, or for a cube the face of
// the dominant axis from the light (faces +x, -x, +y, -y, +z, -z).
uint lightView(LightGpu light, float3 world) {
    let first = uint(light.cone.z);
    if (light.cone.w < 6.0) {
        return first;
    }
    let d = world - light.position.xyz;
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        return first + (d.x >= 0.0 ? 0 : 1);
    }
    if (a.y >= a.z) {
        return first + (d.y >= 0.0 ? 2 : 3);
    }
    return first + (d.z >= 0.0 ? 4 : 5);
}

// Fraction of a light reaching `world` past its shadow map: Poisson-disk PCF with normal offset,
// as the sun's shadows (frame.slang). Points outside the map are lit.
float lightVisibility(LightGpu light, float3 world, float3 n, float nDotL, float2 pixel) {
    if (light.cone.w <= 0.0) {
        return 1.0;
    }
    let view = lightView(light, world);
    let distance = light.position.w == 0.0 ? 1.0 : length(world - light.position.xyz);
    let texelWorld = light.shadow.x * distance;
    let offset = n * sh.lightShadow.y * texelWorld * max(1.0 - clamp(nDotL, 0.0, 1.0), 0.25);
    let clip = mul(lightViews[view], float4(world + offset, 1.0));
    if (clip.w <= 0.0) {
        return 1.0;
    }
    // Receivers beyond the far plane compare at it, as the sun's (frame.slang).
    let ndc = float3(clip.xy / clip.w, min(clip.z / clip.w, 0.99999994));
    let uv = float2(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y);
    if (any(uv < float2(0.0)) || any(uv > float2(1.0))) {
        return 1.0;
    }
    let angle = 6.28318530718 * interleavedGradientNoise(pixel);
    let c = cos(angle);
    let sn = sin(angle);
    let radius = sh.lightShadow.x / sh.lightShadow.z;
    var lit = 0.0;
    for (uint i = 0; i < 16; ++i) {
        let pair = sh.poisson[i / 2];
        let p = i % 2 == 0 ? pair.xy : pair.zw;
        let tap = uv + float2(c * p.x - sn * p.y, sn * p.x + c * p.y) * radius;
        lit += lightShadowMaps.SampleCmpLevelZero(lightShadowSampler, float3(tap, float(view)),
                                                  ndc.z);
    }
    return lit / 16.0;
}

// Radiance factors of the active model for three bands at once: the bands' albedos go in the RGB
// slots of BrdfInput, which models treat independently.
float3 radianceFactors(float3 n, float3 l, float3 v, float3 albedo, BrdfParams params) {
    let input = brdfInputOf(n, l, v, albedo);
    if (input.mu0 <= 0.0) {
        return float3(0.0);
    }
    ActiveBrdf model;
    return max(model.radianceFactor(input, params), float3(0.0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shadeMain(uint3 id : SV_DispatchThreadID) {
    shadedView = pass.model.y + id.z;
    if (id.x >= sh.size.x || id.y >= sh.size.y) {
        return;
    }
    let pixel = int4(id.xyz, 0);
    let ids = gIds.Load(pixel);
    if (ids.x == kBackground) {
        return;
    }
    let material = materials[ids.x];
    if (material.info.x != pass.model.x) {
        return;
    }
    let ndc = float3((float(id.x) + 0.5) / float(sh.size.x) * 2.0 - 1.0,
                     1.0 - (float(id.y) + 0.5) / float(sh.size.y) * 2.0, gDepth.Load(pixel));
    let clip = mul(sh.invViewProj, float4(ndc, 1.0));
    let world = clip.xyz / clip.w;
    let normalVisibility = gNormal.Load(pixel);
    let n = normalize(normalVisibility.xyz);
    let visibility = normalVisibility.w;
    let albedo = gAlbedo.Load(pixel);
    let modulation = max(albedo.yzw, float3(0.0));
    let scale = max(modulation.x, max(modulation.y, modulation.z));
    let coefficients = scale > 0.0 ? sigmoidCoefficients(modulation / scale) : float3(0.0);
    let v = normalize(sh.eye.xyz - world);
    let l = sh.sun.xyz;
    let ambient = sh.sun.w;
    let bands = uint(sh.range.z);
    let unlit = (ids.w & kUnlit) != 0;
    let domeRgb = unlit ? float3(0.0) : domeIrradiance(n);
    let domeScale = max(domeRgb.x, max(domeRgb.y, domeRgb.z));
    let domeCoefficients = domeScale > 0.0 ? sigmoidCoefficients(domeRgb / domeScale) : float3(0.0);

    // Per light: direction towards it and the scale on its spectrum (inverse square, cone and
    // shadow); 0 for lights that cannot reach the point.
    let lightCount = unlit ? 0 : min(sh.size.w, kMaxLights);
    float3 lightDirections[kMaxLights];
    float lightScales[kMaxLights];
    for (uint k = 0; k < lightCount; ++k) {
        if (((sh.view.x >> k) & 1u) == 0) {
            lightDirections[k] = float3(0.0);
            lightScales[k] = 0.0;
            continue;
        }
        let light = lights[k];
        var toLight = light.position.xyz;
        var scale = 1.0;
        let type = uint(light.position.w);
        if (type >= 2) {
            let form = areaFormFactor(light, world, n);
            scale = length(form);
            toLight = scale > 0.0 ? form / scale : n;
        } else if (type == 1) {
            let offset = light.position.xyz - world;
            let distance2 = max(dot(offset, offset), 1e-12);
            toLight = offset * rsqrt(distance2);
            scale = 1.0 / max(distance2, light.axis.w * light.axis.w);
            if (light.cone.x > -1.5) {
                scale *= smoothstep(light.cone.x, max(light.cone.y, light.cone.x + 1e-6),
                                    dot(-toLight, light.axis.xyz));
            }
        }
        let nDotL = dot(n, toLight);
        if (nDotL <= 0.0 || scale <= 0.0) {
            scale = 0.0;
        } else {
            scale *= lightVisibility(light, world, n, nDotL, float2(id.xy));
        }
        lightDirections[k] = toLight;
        lightScales[k] = scale;
    }
    for (uint first = 0; first < bands; first += 3) {
        float3 bandAlbedo = float3(0.0);
        float3 bandIrradiance = float3(0.0);
        for (uint k = 0; k < 3; ++k) {
            let band = min(first + k, bands - 1);
            let wavelength = sh.range.x + (float(band) + 0.5) * sh.range.y;
            bandAlbedo[k] = bandValue(material.reflectance, band) * albedo.x *
                            modulationAt(coefficients, scale, wavelength);
            bandIrradiance[k] = bandValue(sh.irradiance, band);
        }
        float3 response;
        if (unlit) {
            response = bandAlbedo;
        } else {
            let factors = radianceFactors(n, l, v, bandAlbedo, material.params);
            response = ambient * bandAlbedo + (1.0 - ambient) * factors * visibility;
        }
        var radiance = bandIrradiance / PI * response;
        if (domeScale > 0.0) {
            for (uint j = 0; j < 3; ++j) {
                let band = min(first + j, bands - 1);
                let wavelength = sh.range.x + (float(band) + 0.5) * sh.range.y;
                radiance[j] += bandAlbedo[j] * bandValue(sh.dome, band) *
                               modulationAt(domeCoefficients, domeScale, wavelength);
            }
        }
        for (uint k = 0; k < lightCount; ++k) {
            if (lightScales[k] <= 0.0) {
                continue;
            }
            float3 lightSpectrum;
            for (uint j = 0; j < 3; ++j) {
                lightSpectrum[j] = bandValue(lights[k].spectrum, min(first + j, bands - 1));
            }
            let factors = radianceFactors(n, lightDirections[k], v, bandAlbedo, material.params);
            radiance += lightScales[k] * lightSpectrum / PI * factors;
        }
        for (uint k = 0; k < 3 && first + k < bands; ++k) {
            spectral[int3(id.xy, id.z * bands + first + k)] = radiance[k];
        }
    }
}

static const uint kMaxBands = 64;
// Planck's second radiation constant, hc/k, in nm K.
static const float kPlanckC2 = 1.4387769e7;

// A black body's spectral radiance at `wavelength` (nm) over its value at 555 nm.
float planckRatio(float wavelength, float kelvin) {
    let r = 555.0 / wavelength;
    return r * r * r * r * r * (exp(kPlanckC2 / (555.0 * kelvin)) - 1.0) /
           (exp(kPlanckC2 / (wavelength * kelvin)) - 1.0);
}

// The direction through normalised device coordinates `ndc`, from the near to the far plane: its
// length varies so that it is linear in `ndc`, a point of the image plane.
float3 rayThrough(float2 ndc) {
    let near = mul(sh.invViewProj, float4(ndc, 0.0, 1.0));
    let far = mul(sh.invViewProj, float4(ndc, 1.0, 1.0));
    return far.xyz / far.w - near.xyz / near.w;
}

// Texel (`x`, `y`) of mip `level` of the dome map, wrapping around in longitude.
float3 domeTexel(int x, int y, uint level, int2 size) {
    let wrapped = int2((x % size.x + size.x) % size.x, clamp(y, 0, size.y - 1));
    return domeMap.Load(int3(wrapped, int(level))).rgb;
}

// The dome map bilinearly at (`u`, `v`) in mip `level`.
float3 domeBilinear(float2 uv, uint level) {
    let size = int2(max(sh.sky.y >> level, 1u), max(sh.sky.z >> level, 1u));
    let p = uv * float2(size) - 0.5;
    let base = int2(floor(p));
    let f = p - float2(base);
    let a = domeTexel(base.x, base.y, level, size);
    let b = domeTexel(base.x + 1, base.y, level, size);
    let c = domeTexel(base.x, base.y + 1, level, size);
    let d = domeTexel(base.x + 1, base.y + 1, level, size);
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// The dome map in world direction `d`, from the mips whose texels match `pixelAngle` (radians):
// trilinear, with the manual filtering that float32 maps need on every device.
float3 domeMapAt(float3 d, float pixelAngle) {
    let local = float3(dot(sh.toDome[0].xyz, d), dot(sh.toDome[1].xyz, d),
                       dot(sh.toDome[2].xyz, d));
    var u = atan2(local.y, local.x) / (2.0 * PI);
    u = u < 0.0 ? u + 1.0 : u;
    let v = acos(clamp(local.z, -1.0, 1.0)) / PI;
    let texelAngle = PI / float(sh.sky.z);
    let lod = clamp(log2(max(pixelAngle / texelAngle, 1e-6)), 0.0, float(sh.sky.w - 1));
    let lower = uint(floor(lod));
    let upper = min(lower + 1, sh.sky.w - 1);
    return lerp(domeBilinear(float2(u, v), lower), domeBilinear(float2(u, v), upper),
                lod - float(lower));
}

// The star grid's row of declination `declination` (radians), of `rows` from the south pole.
int starRow(float declination, int rows) {
    return clamp(int(floor((declination + 0.5 * PI) / PI * float(rows))), 0, rows - 1);
}

// Adds the stars around world direction `d` to `radiance` (per band), as their tent-filtered
// irradiance over the pixel's solid angle. `a`, `b` and `c` are `rayThrough`'s steps in x and y
// and its value at the image centre; `pixel` the pixel's centre in pixels.
void addStars(inout float radiance[kMaxBands], float3 d, float3 a, float3 b, float3 c,
              float2 pixel, float pixelAngle, float solidAngle) {
    // Points within a pixel and a half: farther ones give this pixel no weight.
    let reach = 1.5 * pixelAngle;
    if (reach > 0.2) {
        return;  // pixels so wide a point covers a crowd of cells: no stars
    }
    let cat = float3(dot(sh.toCatalogue[0].xyz, d), dot(sh.toCatalogue[1].xyz, d),
                     dot(sh.toCatalogue[2].xyz, d));
    let declination = asin(clamp(cat.z, -1.0, 1.0));
    var ascension = atan2(cat.y, cat.x);
    ascension = ascension < 0.0 ? ascension + 2.0 * PI : ascension;
    let columns = int(sh.starGrid.x);
    let rows = int(sh.starGrid.y);
    let firstRow = starRow(declination - reach, rows);
    let lastRow = starRow(declination + reach, rows);
    let widest = min(abs(declination) + reach, 0.5 * PI);
    int firstColumn = 0;
    int lastColumn = columns - 1;
    if (widest < 0.5 * PI - 1e-3) {
        let span = reach / cos(widest);
        if (span < PI) {
            firstColumn = int(floor((ascension - span) / (2.0 * PI) * float(columns)));
            lastColumn = int(floor((ascension + span) / (2.0 * PI) * float(columns)));
        }
    }
    // Directions to image coordinates: d = alpha a + beta b + gamma c gives the point
    // (alpha, beta) / gamma in normalised device coordinates.
    let bc = cross(b, c);
    let ca = cross(c, a);
    let ab = cross(a, b);
    let det = dot(a, bc);
    let size = float2(sh.size.xy);
    let bands = uint(sh.range.z);
    for (int r = firstRow; r <= lastRow; ++r) {
        for (int k = firstColumn; k <= lastColumn; ++k) {
            let column = (k % columns + columns) % columns;
            let cell = starCells[uint(r * columns + column)];
            for (uint i = cell.x; i < cell.x + cell.y; ++i) {
                let star = stars[i];
                let cosDec = cos(star.y);
                let local = float3(cosDec * cos(star.x), cosDec * sin(star.x), sin(star.y));
                let w = sh.toCatalogue[0].xyz * local.x + sh.toCatalogue[1].xyz * local.y +
                        sh.toCatalogue[2].xyz * local.z;
                let gamma = dot(ab, w) / det;
                if (gamma <= 0.0) {
                    continue;
                }
                let ndc = float2(dot(bc, w), dot(ca, w)) / det / gamma;
                let at = float2((ndc.x + 1.0) * 0.5 * size.x, (1.0 - ndc.y) * 0.5 * size.y);
                let offset = abs(at - pixel);
                let weight = max(1.0 - offset.x, 0.0) * max(1.0 - offset.y, 0.0);
                if (weight <= 0.0) {
                    continue;
                }
                let scale = star.w * weight / solidAngle;
                for (uint band = 0; band < bands; ++band) {
                    let wavelength = sh.range.x + (float(band) + 0.5) * sh.range.y;
                    radiance[band] += scale * planckRatio(wavelength, star.z);
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void skyMain(uint3 id : SV_DispatchThreadID) {
    shadedView = pass.model.y + id.z;
    if (id.x >= sh.size.x || id.y >= sh.size.y) {
        return;
    }
    if (gIds.Load(int4(id.xyz, 0)).x != kBackground) {
        return;
    }
    let size = float2(sh.size.xy);
    let pixel = float2(id.xy) + 0.5;
    let ndc = float2(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0);
    let c = rayThrough(float2(0.0));
    let a = rayThrough(float2(1.0, 0.0)) - c;
    let b = rayThrough(float2(0.0, 1.0)) - c;
    let ray = c + ndc.x * a + ndc.y * b;
    let d = normalize(ray);
    // The image plane, `distance` from the eye along `forward`, and the pixel's solid angle there:
    // its area over the distance squared, times the cube of the cosine off the axis.
    var forward = normalize(cross(a, b));
    forward = dot(forward, c) < 0.0 ? -forward : forward;
    let distance = dot(c, forward);
    let pixelArea = 4.0 * length(a) * length(b) / (size.x * size.y);
    let cosine = dot(d, forward);
    let solidAngle = pixelArea / (distance * distance) * cosine * cosine * cosine;
    let pixelAngle = 2.0 * length(b) / size.y / distance;

    let bands = uint(sh.range.z);
    float radiance[kMaxBands];
    for (uint band = 0; band < kMaxBands; ++band) {
        radiance[band] = 0.0;
    }
    if ((sh.sky.x & kSkyDome) != 0) {
        let rgb = (sh.sky.x & kSkyMap) != 0 ? max(domeMapAt(d, pixelAngle), float3(0.0))
                                             : float3(1.0);
        let scale = max(rgb.x, max(rgb.y, rgb.z));
        let coefficients = scale > 0.0 ? sigmoidCoefficients(rgb / scale) : float3(0.0);
        for (uint band = 0; band < bands; ++band) {
            let wavelength = sh.range.x + (float(band) + 0.5) * sh.range.y;
            radiance[band] =
                bandValue(sh.dome, band) * modulationAt(coefficients, scale, wavelength);
        }
    }
    if ((sh.sky.x & kSkyStars) != 0) {
        addStars(radiance, d, a, b, c, pixel, pixelAngle, solidAngle);
    }
    for (uint band = 0; band < bands; ++band) {
        spectral[int3(id.xy, id.z * bands + band)] = radiance[band];
    }
}
