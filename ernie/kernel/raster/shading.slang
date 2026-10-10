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
// 12 their view-projections, 13 the dome's map, 14 the stars, 15 their grid cells, 16 the map
// that lights the scene (13's, without its sun when that is lit on its own).
//
// References:
// - W. Jakob and J. Hanika, "A Low-Dimensional Function Space for Efficient Spectral Upsampling",
//   Computer Graphics Forum 38(2), 2019 (sigmoid spectra of RGB colours).
// - H. W. Jensen et al., "A Physically-Based Night Sky Model", SIGGRAPH 2001, 399-408 (stars as
//   black bodies of their magnitude's irradiance). https://doi.org/10.1145/383259.383306
// - M. Planck, "Ueber das Gesetz der Energieverteilung im Normalspectrum", Annalen der Physik
//   309(3):553-563, 1901 (black-body spectra).
import brdf;
// The camera's projection: a frame camera's matrices, or a pushbroom's strip
// (projection_pinhole.slang, projection_pushbroom.slang).
import projection;

static const uint kUnlit = 1;
static const uint kBackground = 0xFFFFFFFF;

struct ShadingUniforms {
    float4x4 invViewProj;
    // xyz: camera position.
    float4 eye;
    // xyz: unit direction towards the sun, w: ambient fraction.
    float4 sun;
    // First band's lower wavelength (nm), band width (nm), band count, 1 when the projection is
    // orthographic (every ray along the view direction).
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
    // A pushbroom's intrinsics (`PushbroomStrip::Intrinsics`), its strip in `invViewProj`'s place.
    float4 strip;
    // Dome lighting (`tokens::kDomeLightMode`): x 1 when shadowed (`domeOcclusion`), y 1 when it
    // lights diffusely (0 in the limited mode), z the occlusion's world radius, w its largest
    // radius in pixels.
    float4 domeLighting;
    // To the views' stride in the buffer, 1280 bytes.
    float4 padding[11];
}

// What `skyMain` draws behind the scene.
static const uint kSkyDome = 1;   // the dome's radiance
static const uint kSkyMap = 2;    // times its map
static const uint kSkyStars = 4;  // and the stars
// Not drawn but lighting: a dome lights the scene, visible or not (its map, or a white
// placeholder without one, bound either way).
static const uint kSkyLighting = 8;

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
    // x: model index; y: the parameter a per-pixel value in the G-buffer replaces, plus 1 (0 for
    // none; `packSurfaceFlags`); z: 1 when the surface emits; w unused.
    uint4 info;
    BrdfParams params;
    // Reflectance per band normalised to luminance 1 (G-buffer albedo.x scales it), four per
    // float4.
    float4 reflectance[16];
    // Emitted spectral radiance per band (W m^-2 sr^-1 nm^-1), four per float4.
    float4 emission[16];
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
// The map that lights the scene, of the same size: the dome map, or the dome map without its sun
// when the sun lights the scene as a shadowed distant light of its own (`ExtractDomeSun`).
[[vk::binding(16, 0)]] Texture2D<float4> domeLighting;

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

// Cosine to the viewer below which a shading normal is bent towards it.
static const float kMinFacing = 0.05;

// The shading normal `n` bent towards the viewer `v` just enough that it faces them. Normal maps
// tilt shading normals away from the viewer, which no visible surface does; such pixels shaded
// black, as speckle wherever the light is near the view direction, most of all at the opposition
// surge, where everything the viewer sees is lit.
//
// Reference: V. Schuessler, E. Heitz, J. Hanika and C. Dachsbacher, "Microfacet-based Normal
// Mapping for Robust Monte Carlo Path Tracing", ACM Transactions on Graphics 36(6), 2017, section 2
// (the black and energy-losing pixels of back-facing shading normals).
float3 facingNormal(float3 n, float3 v) {
    let facing = dot(n, v);
    return facing >= kMinFacing ? n : normalize(n + (kMinFacing - facing) * v);
}

// Radiance factors of the active model for three bands at once: the bands' albedos go in the RGB
// slots of BrdfInput, which models treat independently.
float3 radianceFactors(float3 n, float3 l, float3 v, float3 albedo, float slopeVariance,
                       BrdfParams params) {
    var input = brdfInputOf(n, l, v, albedo);
    input.slopeVariance = slopeVariance;
    if (input.mu0 <= 0.0) {
        return float3(0.0);
    }
    ActiveBrdf model;
    return max(model.radianceFactor(input, params), float3(0.0));
}

// The active model's response to light from all around (`IBrdf::environment`) for the viewer
// along `v` and the bands' albedos `albedo`.
EnvironmentResponse environmentOf(float3 n, float3 v, float3 albedo, float slopeVariance,
                                  BrdfParams params) {
    var input = brdfInputOf(n, v, v, albedo);
    input.slopeVariance = slopeVariance;
    ActiveBrdf model;
    return model.environment(input, params);
}

// The dome map's RGB around direction `r`, prefiltered for a lobe of perceptual roughness
// `roughness` by reading the mips whose texels span the lobe (about 2 alpha radians across; the
// box-filtered mips stand in for a GGX convolution), as sigmoid coefficients (xyz) and scale (w).
float4 domeLobe(float3 r, float roughness) {
    let alpha = roughness * roughness;
    let rgb = max(domeMapAt(r, max(2.0 * alpha, 1e-4), true), float3(0.0));
    let scale = max(rgb.x, max(rgb.y, rgb.z));
    return float4(scale > 0.0 ? sigmoidCoefficients(rgb / scale) : float3(0.0), scale);
}

// The world point the G-buffer holds at `pixel` of view layer `layer`, and whether there is one
// (not the background, whose depth is the cleared 1).
bool gbufferPoint(int2 pixel, uint layer, out float3 world) {
    let size = int2(sh.size.xy);
    world = float3(0.0);
    if (any(pixel < 0) || any(pixel >= size)) {
        return false;
    }
    let depth = gDepth.Load(int4(pixel, int(layer), 0));
    if (depth >= 1.0) {
        return false;
    }
    let ndc = float2((float(pixel.x) + 0.5) / float(size.x) * 2.0 - 1.0,
                     1.0 - (float(pixel.y) + 0.5) / float(size.y) * 2.0);
    world = cameraPoint(sh.invViewProj, sh.strip, ndc, depth);
    return true;
}

static const uint kOcclusionDirections = 4;
static const uint kOcclusionSteps = 4;

// The fraction of the dome the point `world` with unit normal `n`, seen along `v` (towards the
// viewer), sees past geometry within the world radius `sh.domeLighting.z`: ground-truth ambient
// occlusion's cosine-weighted visibility from the horizons of the depth buffer, in
// `kOcclusionDirections` screen directions turned per pixel by interleaved gradient noise,
// `kOcclusionSteps` samples to each side, out to the radius in pixels (at most
// `sh.domeLighting.w`).
//
// Reference: J. Jimenez, X. Wu, A. Pesce and A. Jarabo, "Practical Realtime Strategies for
// Accurate Indirect Occlusion", SIGGRAPH 2016 course: Physically Based Shading.
float4 domeOcclusion(int2 pixel, uint layer, float3 world, float depth, float3 n, float3 v) {
    let size = float2(sh.size.xy);
    let radius = sh.domeLighting.z;
    let ndc = float2((float(pixel.x) + 0.5) / size.x * 2.0 - 1.0,
                     1.0 - (float(pixel.y) + 0.5) / size.y * 2.0);
    // The radius in pixels, from the world size of a pixel at this depth.
    let pixelWorld =
        length(cameraPoint(sh.invViewProj, sh.strip, ndc + float2(2.0 / size.x, 0.0), depth) -
               world);
    let pixels = min(radius / max(pixelWorld, 1e-12), sh.domeLighting.w);
    if (pixels < 1.0) {
        return float4(n, 1.0);
    }
    let noise = interleavedGradientNoise(float2(pixel));
    var visibility = 0.0;
    var bent = float3(0.0);
    for (uint k = 0; k < kOcclusionDirections; ++k) {
        let angle = (float(k) + noise) * PI / float(kOcclusionDirections);
        let direction = float2(cos(angle), sin(angle));
        // The slice through the view direction along this screen direction (pixel y runs down,
        // ndc y up), and the normal's projection into it.
        let along = cameraPoint(sh.invViewProj, sh.strip,
                                ndc + float2(direction.x, -direction.y) * 2.0 / size, depth) -
                    world;
        let tangent = normalize(along - dot(along, v) * v);
        let axis = normalize(cross(tangent, v));
        let projected = n - axis * dot(n, axis);
        let projectedLength = length(projected);
        if (projectedLength < 1e-4) {
            visibility += 1.0;
            bent += n;
            continue;
        }
        let cosN = clamp(dot(projected / projectedLength, v), -1.0, 1.0);
        let gamma = (dot(projected, tangent) < 0.0 ? -1.0 : 1.0) * acos(cosN);
        // The highest horizon to each side, as the cosine of its angle from the view direction.
        float horizons[2] = {-1.0, -1.0};
        for (uint side = 0; side < 2; ++side) {
            let sign = side == 0 ? 1.0 : -1.0;
            for (uint step = 1; step <= kOcclusionSteps; ++step) {
                let distance = (float(step) - 0.5 + 0.5 * noise) / float(kOcclusionSteps);
                let at = pixel + int2(round(sign * direction * distance * pixels));
                float3 sample;
                if (!gbufferPoint(at, layer, sample)) {
                    continue;
                }
                let offset = sample - world;
                let length2 = dot(offset, offset);
                if (length2 > radius * radius || length2 < 1e-12) {
                    continue;
                }
                horizons[side] = max(horizons[side], dot(offset * rsqrt(length2), v));
            }
        }
        // The horizon angles (positive towards the tangent), clamped to the hemisphere about the
        // normal, and the cosine-weighted arc between them.
        let h1 = gamma + min(acos(horizons[0]) - gamma, 0.5 * PI);
        let h0 = gamma + max(-acos(horizons[1]) - gamma, -0.5 * PI);
        let arc = 0.25 * (-cos(2.0 * h0 - gamma) + cosN + 2.0 * h0 * sin(gamma)) +
                  0.25 * (-cos(2.0 * h1 - gamma) + cosN + 2.0 * h1 * sin(gamma));
        visibility += projectedLength * arc;
        // The slice's bent normal: the cosine-weighted mean of the directions between the
        // horizons, sin(h) along the tangent plus cos(h) along the view (Jimenez et al. 2016).
        let t0 = (6.0 * sin(h0 - gamma) - sin(3.0 * h0 - gamma) + 6.0 * sin(h1 - gamma) -
                  sin(3.0 * h1 - gamma) + 16.0 * sin(gamma) -
                  3.0 * (sin(h0 + gamma) + sin(h1 + gamma))) /
                 12.0;
        let t1 = (-cos(3.0 * h0 - gamma) - cos(3.0 * h1 - gamma) + 8.0 * cos(gamma) -
                  3.0 * (cos(h0 + gamma) + cos(h1 + gamma))) /
                 12.0;
        bent += projectedLength * (t0 * tangent + t1 * v);
    }
    let length = sqrt(dot(bent, bent));
    return float4(length > 1e-6 ? bent / length : n,
                  saturate(visibility / float(kOcclusionDirections)));
}

// The view vector at world point `world` of a pixel at `ndc`: back along its ray, towards the eye,
// along the view when orthographic, towards the row's pose for a pushbroom.
float3 viewVector(float2 ndc, float3 world) {
    return kPushbroomCamera ? -cameraRay(sh.invViewProj, sh.strip, ndc)
           : sh.range.w != 0.0 ? -normalize(rayThrough(ndc))
                               : normalize(sh.eye.xyz - world);
}

// Dome occlusion per pixel: bent normal (xyz) and visibility (w), as the occlusion pass computes
// it (`occlusionMain`), then smoothed by `occlusionFilterMain`, which shading reads.
[[vk::image_format("rgba16f")]] [[vk::binding(17, 0)]] RWTexture2DArray<float4> occlusionRaw;
[[vk::image_format("rgba16f")]] [[vk::binding(18, 0)]] RWTexture2DArray<float4> occlusionFiltered;

// The dome occlusion of each covered pixel, for the full dome lighting mode.
[shader("compute")]
[numthreads(8, 8, 1)]
void occlusionMain(uint3 id : SV_DispatchThreadID) {
    shadedView = pass.model.y + id.z;
    if (id.x >= sh.size.x || id.y >= sh.size.y) {
        return;
    }
    let pixel = int4(id.xyz, 0);
    if (gIds.Load(pixel).x == kBackground) {
        occlusionRaw[id] = float4(0.0, 0.0, 1.0, 1.0);
        return;
    }
    let ndc = float3((float(id.x) + 0.5) / float(sh.size.x) * 2.0 - 1.0,
                     1.0 - (float(id.y) + 0.5) / float(sh.size.y) * 2.0, gDepth.Load(pixel));
    let world = cameraPoint(sh.invViewProj, sh.strip, ndc.xy, ndc.z);
    let v = viewVector(ndc.xy, world);
    let n = facingNormal(normalize(gNormal.Load(pixel).xyz), v);
    occlusionRaw[id] = domeOcclusion(int2(id.xy), id.z, world, ndc.z, n, v);
}

// `occlusionMain`'s results averaged over 5 x 5 pixels of the same surface, weighted by how far
// each lies off the pixel's plane (relative to its distance) and how its normal turns, which
// smooths the per-pixel rotation's noise without blurring across edges. Spatial only, so a batch
// of views needs no history.
[shader("compute")]
[numthreads(8, 8, 1)]
void occlusionFilterMain(uint3 id : SV_DispatchThreadID) {
    shadedView = pass.model.y + id.z;
    if (id.x >= sh.size.x || id.y >= sh.size.y) {
        return;
    }
    float3 world;
    if (!gbufferPoint(int2(id.xy), id.z, world)) {
        occlusionFiltered[id] = float4(0.0, 0.0, 1.0, 1.0);
        return;
    }
    let n = normalize(gNormal.Load(int4(id.xyz, 0)).xyz);
    let scale = max(0.01 * length(world - sh.eye.xyz), 1e-4);
    var sum = float4(0.0);
    var weights = 0.0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            let at = int2(id.xy) + int2(dx, dy);
            float3 other;
            if (!gbufferPoint(at, id.z, other)) {
                continue;
            }
            let otherNormal = normalize(gNormal.Load(int4(at, id.z, 0)).xyz);
            let offPlane = abs(dot(n, other - world)) / scale;
            let weight = exp(-offPlane * offPlane) * pow(saturate(dot(n, otherNormal)), 8.0);
            let value = occlusionRaw[uint3(at, id.z)];
            sum += weight * float4(value.xyz * value.w, value.w);
            weights += weight;
        }
    }
    let average = weights > 0.0 ? sum / weights : occlusionRaw[id];
    let length = sqrt(dot(average.xyz, average.xyz));
    occlusionFiltered[id] = float4(length > 1e-6 ? average.xyz / length : n, average.w);
}

// How much of a specular lobe of perceptual roughness `roughness` around reflection `r` the sky
// still feeds, from occlusion `occlusion` (bent normal, visibility): the overlap of the lobe's cone
// with the visible cone around the bent normal, whose width the visibility sets (Unreal's distance
// field AO specular occlusion), fading to none as the visible cone closes.
//
// Reference: Unreal Engine, SkyLightingShared.ush, `GetDistanceFieldAOSpecularOcclusion` and
// `ApproximateConeConeIntersection`.
float coneOcclusion(float4 occlusion, float3 r, float roughness) {
    let reflectionCone = max(roughness, 0.1) * PI;
    let visibleCone = occlusion.w * PI;
    let between = acos(clamp(dot(occlusion.xyz, r), -1.0, 1.0));
    let difference = abs(reflectionCone - visibleCone);
    let overlap =
        smoothstep(0.0, 1.0, 1.0 - saturate((between - difference) /
                                            max(reflectionCone + visibleCone - difference, 1e-4)));
    return overlap * saturate((visibleCone - 0.1) / 0.2);
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
    let world = cameraPoint(sh.invViewProj, sh.strip, ndc.xy, ndc.z);
    let normalVisibility = gNormal.Load(pixel);
    // Back along the pixel's ray: towards the eye, along the view when orthographic, towards the
    // row's pose for a pushbroom.
    let v = viewVector(ndc.xy, world);
    let n = facingNormal(normalize(normalVisibility.xyz), v);
    let visibility = normalVisibility.w;
    let albedo = gAlbedo.Load(pixel);
    let modulation = max(albedo.yzw, float3(0.0));
    let scale = max(modulation.x, max(modulation.y, modulation.z));
    let coefficients = scale > 0.0 ? sigmoidCoefficients(modulation / scale) : float3(0.0);
    let l = sh.sun.xyz;
    let ambient = sh.sun.w;
    let bands = uint(sh.range.z);
    let unlit = (ids.w & kUnlit) != 0;
    // Variance of the slopes below the pixel, which widens the model's roughness: their deviation
    // sigma as sigma / (1 + sigma) in the high 16 bits (surface.slang's `packSurfaceFlags`).
    let encodedSigma = float(ids.w >> 16) / 65535.0;
    let sigma = encodedSigma / (1.0 - encodedSigma);
    let slopeVariance = sigma * sigma;
    // A textured parameter (such as metallic) replaces the material's value per pixel.
    var params = material.params;
    if (material.info.y != 0) {
        let index = material.info.y - 1;
        params.values[index / 4][index % 4] = float((ids.w >> 8) & 0xFFu) / 255.0;
    }
    // The dome lighting mode: its shadowing (the full mode, from the occlusion passes) and its
    // diffuse light (none in the limited one). The visibility darkens the ambient term too, and
    // the diffuse dome is looked up towards the bent normal, as far as the sky is hidden
    // (Unreal's sky lighting with distance field AO).
    let domeLit = !unlit && (sh.sky.x & kSkyLighting) != 0;
    let shadowed = domeLit && sh.domeLighting.x > 0.0;
    let occlusionState = shadowed ? occlusionFiltered[id] : float4(n, 1.0);
    let occlusion = occlusionState.w;
    let lookup = normalize(lerp(occlusionState.xyz, n, occlusion) + 1e-6 * n);
    let domeRgb =
        unlit ? float3(0.0) : domeIrradiance(lookup) * (occlusion * sh.domeLighting.y);
    let domeScale = max(domeRgb.x, max(domeRgb.y, domeRgb.z));
    let domeCoefficients = domeScale > 0.0 ? sigmoidCoefficients(domeRgb / domeScale) : float3(0.0);
    // The dome's specular lobes, around the mirror direction, where the model has any: their
    // roughness does not depend on the albedo, so one probe finds them for every band.
    float4 lobes[2] = {float4(0.0), float4(0.0)};
    if (domeLit) {
        let probe = environmentOf(n, v, float3(1.0), slopeVariance, params);
        let mirror = 2.0 * dot(n, v) * n - v;
        if (any(probe.specular0 > 0.0)) {
            lobes[0] = domeLobe(mirror, probe.roughness0);
            lobes[0].w *= shadowed ? coneOcclusion(occlusionState, mirror, probe.roughness0) : 1.0;
        }
        if (any(probe.specular1 > 0.0)) {
            lobes[1] = domeLobe(mirror, probe.roughness1);
            lobes[1].w *= shadowed ? coneOcclusion(occlusionState, mirror, probe.roughness1) : 1.0;
        }
    }

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
        // How the model reflects light from all around: the ambient fraction of the sun's (from
        // every direction alike, so every lobe sees it whole) and the dome's.
        let environment = unlit ? diffuseEnvironment(bandAlbedo)
                                : environmentOf(n, v, bandAlbedo, slopeVariance, params);
        float3 response;
        if (unlit) {
            response = bandAlbedo;
        } else {
            let factors = radianceFactors(n, l, v, bandAlbedo, slopeVariance, params);
            let uniform = environment.diffuse + environment.specular0 + environment.specular1;
            response = ambient * occlusion * uniform + (1.0 - ambient) * factors * visibility;
        }
        var radiance = bandIrradiance / PI * response;
        for (uint j = 0; j < 3; ++j) {
            let band = min(first + j, bands - 1);
            let wavelength = sh.range.x + (float(band) + 0.5) * sh.range.y;
            let dome = bandValue(sh.dome, band);
            if (material.info.z != 0 && !unlit) {
                radiance[j] += bandValue(material.emission, band);
            }
            if (domeScale > 0.0) {
                radiance[j] += environment.diffuse[j] * dome *
                               modulationAt(domeCoefficients, domeScale, wavelength);
            }
            if (lobes[0].w > 0.0) {
                radiance[j] += environment.specular0[j] * dome *
                               modulationAt(lobes[0].xyz, lobes[0].w, wavelength);
            }
            if (lobes[1].w > 0.0) {
                radiance[j] += environment.specular1[j] * dome *
                               modulationAt(lobes[1].xyz, lobes[1].w, wavelength);
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
            let factors = radianceFactors(n, lightDirections[k], v, bandAlbedo, slopeVariance,
                                          params);
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

// Texel (`x`, `y`) of mip `level` of the dome map (the lighting map with `lighting`), wrapping
// around in longitude.
float3 domeTexel(int x, int y, uint level, int2 size, bool lighting) {
    let wrapped = int2((x % size.x + size.x) % size.x, clamp(y, 0, size.y - 1));
    let at = int3(wrapped, int(level));
    return lighting ? domeLighting.Load(at).rgb : domeMap.Load(at).rgb;
}

// The dome map bilinearly at (`u`, `v`) in mip `level`.
float3 domeBilinear(float2 uv, uint level, bool lighting) {
    let size = int2(max(sh.sky.y >> level, 1u), max(sh.sky.z >> level, 1u));
    let p = uv * float2(size) - 0.5;
    let base = int2(floor(p));
    let f = p - float2(base);
    let a = domeTexel(base.x, base.y, level, size, lighting);
    let b = domeTexel(base.x + 1, base.y, level, size, lighting);
    let c = domeTexel(base.x, base.y + 1, level, size, lighting);
    let d = domeTexel(base.x + 1, base.y + 1, level, size, lighting);
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// The dome map in world direction `d`, from the mips whose texels match `pixelAngle` (radians):
// trilinear, with the manual filtering that float32 maps need on every device. With `lighting`,
// the map that lights the scene (binding 16), else the one the sky shows.
float3 domeMapAt(float3 d, float pixelAngle, bool lighting = false) {
    let local = float3(dot(sh.toDome[0].xyz, d), dot(sh.toDome[1].xyz, d),
                       dot(sh.toDome[2].xyz, d));
    var u = atan2(local.y, local.x) / (2.0 * PI);
    u = u < 0.0 ? u + 1.0 : u;
    let v = acos(clamp(local.z, -1.0, 1.0)) / PI;
    let texelAngle = PI / float(sh.sky.z);
    let lod = clamp(log2(max(pixelAngle / texelAngle, 1e-6)), 0.0, float(sh.sky.w - 1));
    let lower = uint(floor(lod));
    let upper = min(lower + 1, sh.sky.w - 1);
    return lerp(domeBilinear(float2(u, v), lower, lighting),
                domeBilinear(float2(u, v), upper, lighting), lod - float(lower));
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
    let bands = uint(sh.range.z);
    float radiance[kMaxBands];
    for (uint band = 0; band < kMaxBands; ++band) {
        radiance[band] = 0.0;
    }
    if (kPushbroomCamera || sh.range.w != 0.0) {
        // Orthographic: every pixel sees the dome along the view direction, with no solid angle
        // for stars to fill. A pushbroom's pixels look along their rows' rays; its stars are not
        // drawn.
        let d = kPushbroomCamera ? cameraRay(sh.invViewProj, sh.strip, ndc) : normalize(c);
        if ((sh.sky.x & kSkyDome) != 0) {
            let rgb = (sh.sky.x & kSkyMap) != 0 ? max(domeMapAt(d, 0.0), float3(0.0))
                                                 : float3(1.0);
            let scale = max(rgb.x, max(rgb.y, rgb.z));
            let coefficients = scale > 0.0 ? sigmoidCoefficients(rgb / scale) : float3(0.0);
            for (uint band = 0; band < bands; ++band) {
                let wavelength = sh.range.x + (float(band) + 0.5) * sh.range.y;
                radiance[band] =
                    bandValue(sh.dome, band) * modulationAt(coefficients, scale, wavelength);
            }
        }
        for (uint band = 0; band < bands; ++band) {
            spectral[int3(id.xy, id.z * bands + band)] = radiance[band];
        }
        return;
    }
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
