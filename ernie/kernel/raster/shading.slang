// SPDX-License-Identifier: BSD-3-Clause
// Spectral shading pass: turns the G-buffer into spectral radiance, one image layer per band of
// the camera's spectral range. The host composes it with a reflectance model by prepending
//     import <model module>;
//     typealias ActiveBrdf = <model type>;
// and dispatches one pipeline per model, each shading the pixels whose material uses its model.
//
// Radiance is L = E / pi * F * V + sum over lights of E_k / pi * F_k * V_k + albedo * L_dome * D
// + albedo * L_a * O, with E the sun's spectral irradiance at normal incidence, F the model's
// radiance factor (pi times the BRDF times cos(i)), V the shadow visibility, L_a the scene's
// ambient radiance (uniform white light from all around, `ernie:ambient`) and O the ambient
// occlusion. A distant light's E_k is its irradiance; a sphere light's is its intensity over the
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
// A pushbroom's rows each see the stars from their own pose, through the detector line's pixels
// (`starOffset`).
//
// Bindings: 0 the pass's resources (`ShadingResources`: its buffers' addresses and its textures'
// and samplers' indices in the bindless heap, bindless.slang's set 1), 1 G-buffer normal, 2
// G-buffer albedo, 3 G-buffer ids, 4 depth (one layer per view each), 8 spectral radiance, 17 and
// 18 the dome occlusion, raw and filtered, 29 the light patterns' data (light_pattern's).
//
// References:
// - W. Jakob and J. Hanika, "A Low-Dimensional Function Space for Efficient Spectral Upsampling",
//   Computer Graphics Forum 38(2), 2019 (sigmoid spectra of RGB colours).
// - H. W. Jensen et al., "A Physically-Based Night Sky Model", SIGGRAPH 2001, 399-408 (stars as
//   black bodies of their magnitude's irradiance). https://doi.org/10.1145/383259.383306
// - M. Planck, "Ueber das Gesetz der Energieverteilung im Normalspectrum", Annalen der Physik
//   309(3):553-563, 1901 (black-body spectra).
import brdf;
// The device's textures and samplers by index (bindless.slang).
import bindless;
// The planet's atmosphere: the sun's transmittance through it (atmosphere.slang).
import atmosphere;
import light_pattern;
import light_patterns;
// The camera's projection: a frame camera's matrices, or a pushbroom's strip
// (projection_pinhole.slang, projection_pushbroom.slang).
import projection;

static const uint kUnlit = 1;
static const uint kBackground = 0xFFFFFFFF;

struct ShadingUniforms {
    float4x4 invViewProj;
    // xyz: camera position.
    float4 eye;
    // xyz unused, w: the scene's ambient spectral radiance as white of
    // its luminance (W m^-2 sr^-1 nm^-1), which `ambient`'s colour shapes.
    float4 sun;
    // First band's lower wavelength (nm), band width (nm), band count, 1 when the projection is
    // orthographic (every ray along the view direction).
    float4 range;
    // Image width and height, coefficient table resolution, light count.
    uint4 size;
    // The spectral irradiance unlit surfaces show their colour under, as a white surface facing it
    // (W m^-2 nm^-1): the brightest distant light's, or relative units without one; four bands per
    // float4.
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
    // shared ones); y: its world's slot among a distant light's per-world maps (0 the shared
    // one); z: where its lights start in `lights` (its sub-frame's copy when they move across
    // the shutter); w: the float bits of the camera's aperture diameter, m (`laserSpeckle`).
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
    // The atmosphere (atmosphere.slang), in km: xyz the planet's centre relative to the frame's
    // origin and w its radius; then x the atmosphere's top radius, y 1 with an atmosphere, z
    // unused, w the sky's greatest degree of linear polarization (`skyPolarization`).
    float4 atmosphere[2];
    // xyz: the scene's ambient light's colour, linear RGB over its luminance (`sun.w` its
    // radiance); w unused.
    float4 ambient;
    // Celestial bodies, in km: xyz each centre relative to the frame's origin, w its radius (0 for
    // none); and rgb its normal albedo. They fill the views' stride in the buffer, 1280 bytes.
    float4 bodies[4];
    float4 bodyAlbedo[4];
}

// What `skyMain` draws behind the scene.
static const uint kSkyDome = 1;   // the dome's radiance
static const uint kSkyMap = 2;    // times its map
static const uint kSkyStars = 4;  // and the stars
// Not drawn but lighting: a dome lights the scene, visible or not (its map, or a white
// placeholder without one, bound either way).
static const uint kSkyLighting = 8;
static const uint kSkyAtmosphere = 16;  // the atmosphere, over the dome and stars
static const uint kSkyBodies = 32;      // celestial bodies, over the dome and stars

static const uint kMaxLights = 16;

struct LightGpu {
    // xyz: position (sphere, rect, disk) or unit direction towards the light (distant), w: type,
    // 0 distant, 1 sphere, 2 rect, 3 disk, 4 cylinder, 5 laser.
    float4 position;
    // xyz: unit cone axis, w: sphere or cylinder radius.
    float4 axis;
    // Cosine of the cone's angle (-2 without a cone), cosine where its falloff starts, first
    // shadow view, shadow view count (0 without shadows, 1, or 6 for a cube).
    float4 cone;
    // x: world size of a shadow texel, at unit distance for perspective views; y: 1 for a dome's
    // direction (`Light::domeDiffuse`), which lights with the model's diffuse response alone; z:
    // focus exponent (sphere, rect, disk; 0 unfocused); w: its IES profile's slot in
    // `lightProfiles` plus 1, 0 without.
    float4 shadow;
    // Rect and disk: half width along the light's +x and half height along its +y, as vectors;
    // a cylinder's half length along +x.
    float4 extentX;
    float4 extentY;
    // Spectral irradiance (distant, W m^-2 nm^-1), intensity (sphere, W sr^-1 nm^-1) or
    // radiance (rect, disk) per band: of a focused light the part its focus scales, |cos|^focus
    // off its axis.
    float4 spectrum[16];
    // A focused light's part its focus leaves alone: its emission times its focus tint.
    float4 tinted[16];
    // A laser's beam (`LightParams::Beam`): its 1/e^2 waist radii along x and y, its Rayleigh
    // ranges along each; x: the waist's distance along the beam.
    float4 beam;
    float4 beamExtra;
    // beamExtra yzw: whether its pattern is the beam itself (0, a spot) or spreads from the
    // aperture (1), the efficiency into it and its zero order. pattern: its pattern's index
    // (`lightPattern`), the laser's wavelength (nm) and the time (s); patternExtent xy: its
    // tangent-plane half extents (0 unbounded); patternLayout: its data and image
    // (LightPatternParams); its parameters.
    float4 pattern;
    float4 patternExtent;
    uint4 patternLayout;
    float4 patternParams[4];
    // x: texels along each side of its shadow maps, in that corner of their layers of the map
    // array (`ShadingUniforms::lightShadow.z` texels across); y: 1 for a batch's distant light,
    // whose maps are the shared one then one per world (`ShadingUniforms::view.y`); z: 1 for a
    // laser showing speckle (`laserSpeckle`), w: its linewidth, nm.
    float4 map;
    // Light linking (`LinkGroups`): x the link groups of the surfaces it lights, y those whose
    // geometry casts its shadows, a bit each (surface.slang's `linkGroupFlags`); z 1 when it casts
    // shadows at all (volume.slang's volume shadows); w unused.
    uint4 links;
    // xyz: its electric field's direction, across the way its light goes; w: the fraction of its
    // light linearly polarized along it.
    float4 polarization;
}

struct MaterialGpu {
    // x: model index; y: the parameter a per-pixel value in the G-buffer replaces, plus 1 (0 for
    // none; `packSurfaceFlags`); z: 1 when the surface emits; w: the float bits of the share of
    // incident polarization its diffuse reflection keeps.
    uint4 info;
    BrdfParams params;
    // Reflectance per band normalised to luminance 1 (G-buffer albedo.x scales it), four per
    // float4.
    float4 reflectance[16];
    // Emitted spectral radiance per band (W m^-2 sr^-1 nm^-1), four per float4.
    float4 emission[16];
}

struct ShadingConstants {
    // x: index of the model this dispatch shades, y: the first view's uniforms, z: the layers
    // from a raster radiance layer to its Stokes Q, and on to U, with a polarizer (0 without
    // them); w: the sky's part (`kSkyPart*`) a sky dispatch draws.
    uint4 model;
    // The terrain's horizon map (`terrainOcclusion`): its north-west corner's x and y relative
    // to the frame's origin, its texel size and the terrain's vertical exaggeration; and its
    // width, height, the terrain's prim id and 1 when there is one.
    float4 horizonGrid;
    uint4 horizonInfo;
    // The terrain's heights (`terrainHeights`): their cell size, the frame origin's height
    // (subtracted from the terrain's), the no-data value and 1 when there is one; their width
    // and height, and 1 when the horizon map holds the ground's horizon over what stands on the
    // terrain (layers 2 kHorizonLevels on).
    float4 heightsGrid;
    uint4 heightsSize;
    // The heights above the ground the horizon map's levels see from, in height units.
    float4 horizonLevels;
}

// The pass's buffers, read through their addresses, and its textures and samplers, by their
// indices in the bindless heap (the renderer's `ShadingResources`).
struct ShadingResources {
    // Every view's ShadingUniforms.
    ShadingUniforms* views;
    MaterialGpu* materials;
    LightGpu* lights;
    // The lights' shadow views' view-projections.
    float4x4* lightViews;
    // Lights' IES profiles (`LightParams::profile`): per slot kProfileRows rows from the light's
    // axis (0) to opposite it (180 degrees), of kProfileColumns columns around it from its +x
    // towards +y.
    float* lightProfiles;
    // Per light slot, its laser's pattern's integral over its extent (1 for a normalized pattern,
    // else the renderer's `patternNormMain`), which its emission is divided by so the laser emits
    // its power.
    float* patternNorms;
    // Per shaded view and light slot, kMaxLit (sequence frame, weight) pairs: the frames a laser
    // shows over the view's span of the sensor clock and how much, its pulses' energy then over
    // their average included (the renderer's `WriteLightTiming`).
    float2* lightTiming;
    // The stars (right ascension and declination in radians, temperature in kelvin, spectral
    // irradiance at 555 nm) sorted by grid cell, and each cell's first star and count.
    float4* stars;
    uint2* starCells;
    // The atmosphere's medium, its constituents' densities and their phase functions.
    float4* atmosphereMedium;
    float* atmosphereDensities;
    float* atmospherePhases;
    // Heap indices. Jakob-Hanika sigmoid coefficients (c0, c1, c2) over linear RGB in [0, 1]^3:
    // red along x, green along y and blue across layers, at `size.z` samples per axis (an array),
    // and its linear sampler.
    uint sigmoidTable;
    uint tableSampler;
    // The lights' shadow maps (an array) and their comparison sampler.
    uint lightShadowMaps;
    uint lightShadowSampler;
    // The dome's latitude-longitude RGB map with its mips (rows from +z down, columns from +x
    // through +y), and the map that lights the scene, of the same size: the dome map, or the dome
    // map without its sun when the sun lights the scene as a shadowed distant light of its own
    // (`ExtractDomeSun`); both 2D.
    uint domeMap;
    uint domeLighting;
    // The terrain's horizon map (an array) and its heights (2D).
    uint horizonMap;
    uint terrainHeights;
    // The atmosphere's transmittance, multiple scattering, sky irradiance and refraction tables
    // (arrays).
    uint atmosphereTransmittance;
    uint atmosphereMultiple;
    uint atmosphereSkyIrradiance;
    uint atmosphereRefraction;
    // The volumes' shadow maps (volume.slang's `volumeShadowMain`, an array; kNoVolumeShadows
    // without volumes) and their texels across.
    uint volumeShadows;
    uint volumeShadowSize;
}

[[vk::binding(0, 0)]] ConstantBuffer<ShadingResources> resources;
[[vk::push_constant]] ConstantBuffer<ShadingConstants> pass;
[[vk::binding(1, 0)]] Texture2DArray<float4> gNormal;
[[vk::binding(2, 0)]] Texture2DArray<float4> gAlbedo;
[[vk::binding(3, 0)]] Texture2DArray<uint4> gIds;
[[vk::binding(4, 0)]] Texture2DArray<float> gDepth;
[[vk::image_format("r32f")]] [[vk::binding(8, 0)]] RWTexture2DArray<float> spectral;

// The resources under the names the pass reads them by: buffers as arrays, textures as the
// heap's (float4; one-channel ones read as `.x`).
#define shadingViews (resources.views)
#define materials (resources.materials)
#define lights (resources.lights)
#define lightViews (resources.lightViews)
#define lightProfiles (resources.lightProfiles)
#define patternNorms (resources.patternNorms)
#define lightTiming (resources.lightTiming)
#define stars (resources.stars)
#define starCells (resources.starCells)
#define atmosphereMedium (resources.atmosphereMedium)
#define atmosphereDensities (resources.atmosphereDensities)
#define atmospherePhases (resources.atmospherePhases)
#define sigmoidTable (heapTextureArrays[resources.sigmoidTable])
#define tableSampler (heapSamplers[resources.tableSampler])
#define lightShadowMaps (heapTextureArrays[resources.lightShadowMaps])
#define lightShadowSampler (heapComparisonSamplers[resources.lightShadowSampler])
#define domeMap (heapTextures[resources.domeMap])
#define domeLightingMap (heapTextures[resources.domeLighting])
#define horizonMap (heapTextureArrays[resources.horizonMap])
#define terrainHeights (heapTextures[resources.terrainHeights])
#define atmosphereTransmittance (heapTextureArrays[resources.atmosphereTransmittance])
#define atmosphereMultiple (heapTextureArrays[resources.atmosphereMultiple])
#define atmosphereSkyIrradiance (heapTextureArrays[resources.atmosphereSkyIrradiance])
#define atmosphereRefraction (heapTextureArrays[resources.atmosphereRefraction])
static const uint kMaxLit = 4;
static const uint kProfileRows = 129;
static const uint kProfileColumns = 64;

// A light's IES profile towards unit `direction` from it, bilinear: candela.
float lightProfile(LightGpu light, float3 direction) {
    let base = (uint(light.shadow.w) - 1) * kProfileRows * kProfileColumns;
    let x = normalize(light.extentX.xyz);
    let y = cross(x, light.axis.xyz);
    let theta = acos(clamp(dot(direction, light.axis.xyz), -1.0, 1.0));
    var phi = atan2(dot(direction, y), dot(direction, x));
    phi = phi < 0.0 ? phi + 2.0 * PI : phi;
    let r = theta / PI * float(kProfileRows - 1);
    let c = phi / (2.0 * PI) * float(kProfileColumns);
    let r0 = min(uint(r), kProfileRows - 2);
    let c0 = uint(c) % kProfileColumns;
    let c1 = (c0 + 1) % kProfileColumns;
    let tr = saturate(r - float(r0));
    let tc = frac(c);
    let row0 = base + r0 * kProfileColumns;
    let row1 = row0 + kProfileColumns;
    return lerp(lerp(lightProfiles[row0 + c0], lightProfiles[row0 + c1], tc),
                lerp(lightProfiles[row1 + c0], lightProfiles[row1 + c1], tc), tr);
}
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
static const uint kCylinderArc = 8;  // points on each end's arc of a cylinder's outline
static const uint kMaxPolygon = kDiskSides + 1;

// The vector form factor of a rect or disk light at `world` with unit normal `n`: half the sum
// over the emitter's edges, clipped to the receiver's horizon, of each edge's angle times the unit
// normal of the plane through it and `world` (Lambert's formula; Arvo 1994). Irradiance is the
// light's radiance times dot(n, V). Zero when the emitter faces away or lies below the horizon.
float3 areaFormFactor(LightGpu light, float3 world, float3 n) {
    let centre = light.position.xyz;
    let type = uint(light.position.w);
    float3 polygon[kMaxPolygon];
    uint count = 0;
    if (type == 4) {
        // A cylinder's side, by its outline from `world` (a uniformly bright surface lights a
        // point by its outline alone): at each end the arc of the end circle facing `world`,
        // between the tangent points, the ends joined along the side.
        let along = normalize(light.extentX.xyz);
        let out = world - centre;
        let facing = out - dot(out, along) * along;
        let distance = length(facing);
        let radius = light.axis.w;
        if (distance <= radius) {
            return float3(0.0);  // inside it
        }
        let toward = facing / distance;
        let across = cross(along, toward);
        let tangent = acos(radius / distance);
        for (uint i = 0; i < kCylinderArc; ++i) {
            let a = tangent * (2.0 * float(i) / float(kCylinderArc - 1) - 1.0);
            let rim = radius * (cos(a) * toward + sin(a) * across);
            polygon[i] = centre + light.extentX.xyz + rim;
            polygon[2 * kCylinderArc - 1 - i] = centre - light.extentX.xyz + rim;
        }
        count = 2 * kCylinderArc;
    } else if (dot(world - centre, light.axis.xyz) <= 0.0) {
        return float3(0.0);
    } else if (type == 2) {
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

// The shadow view of a light's maps that sees `world`: its only one; for a distant light's
// cascades over the view (`RasterRenderer::FitViewCascades`), the finest whose map holds it, else
// the coarsest; or for a cube the face of the dominant axis from the light (faces +x, -x, +y,
// -y, +z, -z).
uint lightView(LightGpu light, float3 world) {
    let first = uint(light.cone.z);
    let count = uint(light.cone.w);
    if (light.map.y != 0.0) {
        // A batch's distant light: the map of the view's world.
        return first + min(sh.view.y, count - 1);
    }
    if (light.position.w == 0.0 && count > 1) {
        for (uint c = 0; c + 1 < count; ++c) {
            let clip = mul(lightViews[first + c], float4(world, 1.0)).xyz;
            if (all(abs(clip.xy) <= float2(1.0)) && clip.z >= 0.0 && clip.z <= 1.0) {
                return first + c;
            }
        }
        return first + count - 1;
    }
    if (count < 6) {
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

// Largest PCF radius in texels for a light's penumbra: wider than the sun's (frame.slang), as a
// lamp near its casters throws penumbrae of several degrees (a 1024-texel cube face's texel is
// 1.5 mrad).
static const float kMaxLightFilterTexels = 64.0;

// Poisson sample `i` of 16 in the unit disk.
float2 lightPoisson(uint i) {
    let pair = sh.poisson[i / 2];
    return i % 2 == 0 ? pair.xy : pair.zw;
}

// Where a point falls in a light's shadow maps: the view that sees it, its place in that view's
// map and its depth there; not `inside` off the map or behind the light.
struct LightTap {
    uint view;
    float4 clip;
    float2 uv;
    float depth;
    bool inside;
}

// The world width of a texel of orthographic view `view` of `light`: its map's width over its
// texels, from the view's first row (2 over the map's width in world units).
float lightTexelWorld(LightGpu light, uint view) {
    return 2.0 / (light.map.x * max(length(lightViews[view][0].xyz), 1e-12));
}

// Poisson sample `i` of a disk `radius` wide across a light's ray, turned by `rotation` (cosine,
// sine), on unit axes `across` and `across2`.
float3 lightDisk(uint i, float2 rotation, float3 across, float3 across2, float radius) {
    let p = lightPoisson(i);
    let q = float2(rotation.x * p.x - rotation.y * p.y, rotation.y * p.x + rotation.x * p.y);
    return (q.x * across + q.y * across2) * radius;
}

LightTap lightTap(LightGpu light, float3 point) {
    LightTap tap;
    tap.view = lightView(light, point);
    tap.clip = mul(lightViews[tap.view], float4(point, 1.0));
    let ndc = tap.clip.xyz / max(tap.clip.w, 1e-20);
    // Receivers beyond the far plane compare at it, as the sun's (frame.slang).
    tap.depth = min(ndc.z, 0.99999994);
    let uv = float2(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y);
    tap.inside = tap.clip.w > 0.0 && all(uv >= float2(0.0)) && all(uv <= float2(1.0));
    // In the light's corner of the layer, short of its last texel's far edge, so filtering does
    // not reach past it.
    let corner = light.map.x / sh.lightShadow.z;
    tap.uv = min(uv * corner, float2(corner - 0.5 / sh.lightShadow.z));
    return tap;
}

// Fraction of a light reaching `world` past its shadow map: Poisson-disk PCF with normal offset,
// as the sun's shadows (frame.slang). Points outside the map are lit.
//
// The taps lie on a disk across the light's ray through the receiver, each looked up in the view
// that sees it, so a cube's filter crosses the seams between its faces.
//
// Lights with a size get percentage-closer soft shadows, as the sun's: a blocker search averages
// the distance of occluders within the light's footprint, and the filter widens to the penumbra
// they cast. A local light of radius R (`extentX.w`), seen from the receiver at distance d_r past
// blockers at d_b, casts a penumbra of R (d_r - d_b) / (d_b d_r) radians as the map sees it; a
// distant light's (`extentY.w`, the tangent of its angular radius) is its gap to the blockers
// times that tangent. Every tap compares against the receiver's plane, which the disk across the
// ray meets at a slant, so tilted surfaces do not shadow themselves.
//
// Reference: R. Fernando, "Percentage-Closer Soft Shadows", SIGGRAPH 2005 Sketches.
// https://doi.org/10.1145/1187112.1187153
float lightVisibility(LightGpu light, float3 world, float3 n, float nDotL, float2 pixel) {
    if (light.cone.w <= 0.0) {
        return 1.0;
    }
    // Orthographic views (the sun-like distant light's, a laser's along its beam) have texels of
    // one size; perspective ones grow with distance.
    let distant = light.position.w == 0.0;
    let orthographic = distant || (light.position.w == 5.0 && light.beamExtra.y < 0.5);
    let distance = orthographic ? 1.0 : length(world - light.position.xyz);
    // A distant light's texels are its cascade's: the world width of its map over its texels.
    let texelWorld = distant ? lightTexelWorld(light, lightView(light, world))
                             : light.shadow.x * distance;
    let offset = n * sh.lightShadow.y * texelWorld * max(1.0 - clamp(nDotL, 0.0, 1.0), 0.25);
    let receiver = world + offset;
    let centre = lightTap(light, receiver);
    if (!centre.inside) {
        return 1.0;
    }
    // Towards the light along its ray through the receiver; a laser's orthographic view looks
    // down its beam.
    let towards = distant ? light.position.xyz
                          : (orthographic ? -light.axis.xyz
                                          : normalize(light.position.xyz - receiver));
    let across = normalize(cross(towards, abs(towards.z) < 0.9 ? float3(0.0, 0.0, 1.0)
                                                               : float3(1.0, 0.0, 0.0)));
    let across2 = cross(towards, across);
    let angle = 6.28318530718 * interleavedGradientNoise(pixel);
    let rotation = float2(cos(angle), sin(angle));
    let size = sh.lightShadow.z;
    let softness = sh.lightShadow.x;
    let cosine = clamp(nDotL, 0.05, 1.0);
    // How far the receiver's plane falls towards the light per world unit across it.
    let tilt = min(sqrt(1.0 - cosine * cosine) / cosine, 10.0);
    var filterTexels = softness;

    let lightRadius = orthographic ? 0.0 : light.extentX.w;
    let tanRadius = distant ? light.extentY.w : 0.0;
    if (lightRadius > 0.0 || tanRadius > 0.0) {
        // An orthographic view's depth per world unit towards the light.
        let ahead = mul(lightViews[centre.view], float4(receiver + towards, 1.0));
        let depthPerWorld = max(abs(ahead.z / ahead.w - centre.depth), 1e-12);
        // Search where blockers halfway to the light (a local light) or between the near plane
        // and the receiver (a distant one) would shadow it.
        let searchTexels =
            clamp(orthographic ? centre.depth / depthPerWorld * tanRadius / texelWorld
                               : lightRadius / max(distance * light.shadow.x, 1e-12),
                  softness, kMaxLightFilterTexels);
        var blockers = 0.0;
        var blockerSum = 0.0;
        for (uint i = 0; i < 16; ++i) {
            let point = receiver + lightDisk(i, rotation, across, across2,
                                             searchTexels * texelWorld);
            let tap = lightTap(light, point);
            if (!tap.inside) {
                continue;
            }
            let texel = min(int2(tap.uv * size), int2(int(size) - 1));
            let stored = lightShadowMaps.Load(int4(texel, int(tap.view), 0)).x;
            let drop =
                length(lightPoisson(i)) * searchTexels * texelWorld * tilt;
            if (orthographic) {
                // In the tap's own view, which may be another cascade.
                let tapDepthPerWorld =
                    max(abs(dot(lightViews[tap.view][2].xyz, towards)), 1e-12);
                let gap = (tap.depth - stored) / tapDepthPerWorld;
                if (gap > drop) {
                    blockers += 1.0;
                    blockerSum += gap;
                }
            } else {
                // Along the ray from the light (where clip w is 0) through the tap, clip z and w
                // are affine in the fraction s of the way: the stored depth gives its s.
                let atLight = mul(lightViews[tap.view], float4(light.position.xyz, 1.0)).z;
                let s = atLight / (stored * tap.clip.w - tap.clip.z + atLight);
                let tapDistance = length(point - light.position.xyz);
                let blocker = clamp(s, 1e-3, 1.0) * tapDistance;
                if (tapDistance - blocker > drop) {
                    blockers += 1.0;
                    blockerSum += blocker;
                }
            }
        }
        if (blockers == 0.0) {
            return 1.0;
        }
        var penumbraTexels = 0.0;
        if (orthographic) {
            penumbraTexels = blockerSum / blockers * tanRadius / texelWorld;
        } else {
            let blocker = blockerSum / blockers;
            let radians = lightRadius * (distance - blocker) / (blocker * distance);
            penumbraTexels = radians / light.shadow.x;
        }
        filterTexels = clamp(penumbraTexels, softness, kMaxLightFilterTexels);
    }
    var lit = 0.0;
    for (uint i = 0; i < 16; ++i) {
        let point =
            receiver + lightDisk(i, rotation, across, across2, filterTexels * texelWorld);
        let tap = lightTap(light, point);
        if (!tap.inside) {
            lit += 1.0;
            continue;
        }
        // Compared at the receiver's plane, lowered towards the light by the tap's distance
        // across it.
        let drop = length(lightPoisson(i)) * filterTexels * texelWorld * tilt;
        let toLight = orthographic ? towards : normalize(light.position.xyz - point);
        let lowered = mul(lightViews[tap.view], float4(point + toLight * drop, 1.0));
        let depth = min(lowered.z / max(lowered.w, 1e-20), 0.99999994);
        lit += lightShadowMaps.SampleCmpLevelZero(lightShadowSampler,
                                                  float3(tap.uv, float(tap.view)), depth);
    }
    return lit / 16.0;
}

static const uint kNoVolumeShadows = 0xFFFFFFFFu;

// The transmittance of a light's volume shadow map (volume.slang's `volumeShadowMain`) at texel
// `texel` of shadow view `view`, at the view's depth `depth`: 1 before the volumes begin, their
// slices' transmittance, linear between them, within, and the last beyond.
float volumeShadowAt(int2 texel, uint view, float depth) {
    let maps = heapTextureArrays[resources.volumeShadows];
    let span = maps.Load(int4(texel, int(3 * view), 0));
    if (span.x >= span.y || depth <= span.x) {
        return 1.0;
    }
    let a = maps.Load(int4(texel, int(3 * view + 1), 0));
    let b = maps.Load(int4(texel, int(3 * view + 2), 0));
    let slices = float[9](1.0, a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
    let x = saturate((depth - span.x) / (span.y - span.x)) * 8.0;
    let i = min(uint(x), 7u);
    return lerp(slices[i], slices[i + 1], x - float(i));
}

// How much of a light the volumes let through to `world`: its volume shadow map in the shadow
// view shading reads its geometry's shadows from (`lightView`), bilinear between texels. 1 for a
// light without shadow maps, or without volumes.
float volumeTransmittance(LightGpu light, float3 world) {
    if (resources.volumeShadows == kNoVolumeShadows || light.cone.w <= 0.0) {
        return 1.0;
    }
    let view = lightView(light, world);
    let clip = mul(lightViews[view], float4(world, 1.0));
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    let uv = float2(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y);
    if (any(uv < float2(0.0)) || any(uv > float2(1.0))) {
        return 1.0;
    }
    let size = int(resources.volumeShadowSize);
    let p = uv * float(size) - 0.5;
    let base = int2(floor(p));
    let f = p - floor(p);
    float t[4];
    for (int c = 0; c < 4; ++c) {
        let texel = clamp(base + int2(c & 1, c >> 1), int2(0), int2(size - 1));
        t[c] = volumeShadowAt(texel, view, ndc.z);
    }
    return lerp(lerp(t[0], t[1], f.x), lerp(t[2], t[3], f.x), f.y);
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

// The active model's specular part of `radianceFactors` (`IBrdf::specularReflection`).
SpecularReflection specularOf(float3 n, float3 l, float3 v, float3 albedo, float slopeVariance,
                              BrdfParams params) {
    var input = brdfInputOf(n, l, v, albedo);
    input.slopeVariance = slopeVariance;
    ActiveBrdf model;
    var r = model.specularReflection(input, params);
    r.factor = input.mu0 > 0.0 ? max(r.factor, float3(0.0)) : float3(0.0);
    return r;
}

// Fresnel's reflectances at cos(i) `cosI` onto a medium of relative index of refraction `ior`, for
// light polarized across (s) and in (p) the plane of incidence, and the sign of r_s r_p, which
// turns negative past Brewster's angle.
//
// Reference: M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge, 1999, sec. 1.5.
float3 fresnelSP(float cosI, float ior) {
    let sinT2 = (1.0 - cosI * cosI) / (ior * ior);
    if (sinT2 >= 1.0) {
        return float3(1.0, 1.0, 1.0);
    }
    let cosT = sqrt(1.0 - sinT2);
    let rs = (cosI - ior * cosT) / (cosI + ior * cosT);
    let rp = (ior * cosI - cosT) / (ior * cosI + cosT);
    return float3(rs * rs, rp * rp, rs * rp < 0.0 ? -1.0 : 1.0);
}

// Specular reflection's linear Stokes Q and U over I in the pixel's frame (`frameX`, `frameY`),
// for light arriving from unit direction `l`, `polarized` of it along its field `field`, mirrored
// towards `v` by facets of index `ior` facing their half vector: the incident Stokes in the s
// and p frame of the plane holding `l` and `v`, through Fresnel's Mueller matrix (R_s + R_p and
// R_s - R_p halved, and sqrt(R_s R_p) on U), then its polarized part along its field's
// direction, seen in the pixel's frame. Unpolarized light comes off polarized across that plane.
//
// Reference: E. Collett, "Field Guide to Polarization", SPIE FG05, 2005 (the Mueller matrix of
// reflection).
float2 specularStokes(float3 l, float3 v, float polarized, float3 field, float ior,
                      float3 frameX, float3 frameY) {
    let across = cross(l, v);
    if (dot(across, across) < 1e-12) {
        return float2(0.0);
    }
    let s = normalize(across);
    let h = normalize(l + v);
    let f = fresnelSP(saturate(dot(l, h)), ior);
    // The incident light's Stokes over I in its s and p frame, p across s and its travel, -l.
    let pIn = cross(s, -l);
    let fs = dot(field, s);
    let fp = dot(field, pIn);
    let fields = fs * fs + fp * fp;
    let qIn = fields > 1e-8 ? polarized * (fs * fs - fp * fp) / fields : 0.0;
    let uIn = fields > 1e-8 ? polarized * 2.0 * fs * fp / fields : 0.0;
    let mean = 0.5 * (f.x + f.y);
    let difference = 0.5 * (f.x - f.y);
    let i = mean + difference * qIn;
    let q = difference + mean * qIn;
    let u = f.z * sqrt(f.x * f.y) * uIn;
    let degree = sqrt(q * q + u * u) / max(i, 1e-12);
    if (degree < 1e-6) {
        return float2(0.0);
    }
    // The reflected field's direction, chi from s towards p across its travel, v.
    let chi = 0.5 * atan2(u, q);
    let reflected = cos(chi) * s + sin(chi) * cross(s, v);
    let x = dot(reflected, frameX);
    let y = dot(reflected, frameY);
    let seen = max(x * x + y * y, 1e-12);
    return degree * float2(x * x - y * y, 2.0 * x * y) / seen;
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
    // Reverse-Z: the background keeps the cleared 0.
    let depth = gDepth.Load(int4(pixel, int(layer), 0));
    if (depth <= 0.0) {
        return false;
    }
    let ndc = float2((float(pixel.x) + 0.5) / float(size.x) * 2.0 - 1.0,
                     1.0 - (float(pixel.y) + 0.5) / float(size.y) * 2.0);
    world = cameraPoint(sh.invViewProj, sh.strip, ndc, depth);
    return true;
}

// Pixel `pixel`'s footprint on its surface at `world`: the steps in world units to its
// neighbours' surface points across (xyz of the first) and down (of the second), to the nearer
// of the two either side, so at an edge the neighbour on the pixel's own surface counts rather
// than one on a surface far behind it (whose step, metres long, would spread a laser's beam over
// the pixel and light the edge); zero where no neighbour has a surface. They grow with distance
// and with how obliquely the surface is seen.
//
// References:
// - H. Igehy, "Tracing Ray Differentials", SIGGRAPH 1999.
float3x2 pixelFootprint(int2 pixel, uint layer, float3 world) {
    float3 steps[2] = {float3(0.0), float3(0.0)};
    for (uint axis = 0; axis < 2; ++axis) {
        let step = axis == 0 ? int2(1, 0) : int2(0, 1);
        float3 after;
        float3 before;
        let hasAfter = gbufferPoint(pixel + step, layer, after);
        let hasBefore = gbufferPoint(pixel - step, layer, before);
        if (hasAfter && hasBefore) {
            steps[axis] = length(after - world) <= length(before - world) ? after - world
                                                                          : world - before;
        } else if (hasAfter) {
            steps[axis] = after - world;
        } else if (hasBefore) {
            steps[axis] = world - before;
        }
    }
    return float3x2(steps[0].x, steps[1].x, steps[0].y, steps[1].y, steps[0].z, steps[1].z);
}

// A laser's irradiance at `world` per watt of its power, across the light's ray, m^-2: its pattern
// (`LightParams::Beam`, an `ILightPattern` in light slot `slot`), averaged over the pixel's
// `footprint` (`pixelFootprint`) as it spans the beam's x and y, so a feature smaller than a pixel
// spreads over it and the pixels it falls on carry its whole power, however obliquely; 0 behind
// the laser.
//
// A pattern gives its relative power per unit area of the tangent plane, t = (x / z, y / z) at the
// point's distance z along the axis; over its integral there (`patternNorms`), per square metre of
// the plane across the axis that is its emission over z^2. It is given the beam's own 1/e^2
// radius there, w(z) / z with w = w0 sqrt(1 + (z / z_R)^2) along each axis, so the spot (the beam
// itself) is 2 / (pi w_x w_y) exp(-2 (x^2 / w_x^2 + y^2 / w_y^2)) near its waist and far from it
// alike. `efficiency` of the power goes into the pattern, the zero order into a spot. A sequence
// shows its frame `frame`.
//
// Across the ray, that irradiance is over the cosine between the ray and the axis: a pattern's
// rays leave the aperture, while the beam's own follow its wavefronts, of radius
// R = z (1 + (z_R / z)^2) from its waist, so parallel to the axis near the waist.
//
// References:
// - A. E. Siegman, Lasers, University Science Books, 1986, ch. 17.
float laserIrradiance(LightGpu light, uint slot, float3 world, float3x2 footprint, uint frame) {
    let offset = world - light.position.xyz;
    let along = dot(offset, light.axis.xyz);
    if (along <= 0.0) {
        return 0.0;
    }
    let x = normalize(light.extentX.xyz);
    let y = cross(x, light.axis.xyz);
    let z = along - light.beamExtra.x;
    let zr = max(light.beam.zw, float2(1e-12));
    let w = light.beam.xy * sqrt(1.0 + (z / zr) * (z / zr));
    // The pixel's extent across the beam along each of its axes.
    let spanX = abs(mul(x, footprint));
    let spanY = abs(mul(y, footprint));
    let width = float2(spanX.x + spanX.y, spanY.x + spanY.y);
    let across = float2(dot(offset, x), dot(offset, y));
    LightPatternInput input;
    input.t = across / along;
    input.footprint = width / along;
    input.beam = w / along;
    input.wavelength = light.pattern.y;
    input.time = light.pattern.z;
    input.frame = frame;
    input.frames = max(uint(light.pattern.w), 1);
    // The beam's rays' slope r / R and a pattern's from the aperture.
    let slope = across * z / (z * z + zr * zr);
    let beamOblique = sqrt(1.0 + dot(slope, slope));
    let patternOblique = length(offset) / along;
    // What the element leaves undiffracted, along the axis.
    var irradiance = light.beamExtra.w * beamOblique * beamlet(float2(0.0), input);
    let efficiency = light.beamExtra.z;
    let norm = patternNorms[slot];
    if (efficiency > 0.0 && norm > 0.0) {
        // Nothing beyond a bounded pattern's extent, the pixel's footprint covering it in part at
        // its edge.
        let extent = light.patternExtent.xy;
        let bounded = extent > 0.0;
        let cover = select(bounded,
                           saturate((extent - abs(input.t)) / max(input.footprint, 1e-9) + 0.5),
                           float2(1.0));
        if (cover.x * cover.y > 0.0) {
            input.t = select(bounded, clamp(input.t, -extent, extent), input.t);
            LightPatternParams p;
            p.values = light.patternParams;
            p.layout = light.patternLayout;
            let emission = max(lightPattern(uint(light.pattern.x), input, p), 0.0);
            let oblique = light.beamExtra.y > 0.5 ? patternOblique : beamOblique;
            irradiance += oblique * efficiency * cover.x * cover.y * emission / norm;
        }
    }
    return irradiance / (along * along);
}

// Speckle's field: layers of smoothed value noise on rotated lattices, each node a complex
// Gaussian amplitude, summed so the field is close to circular Gaussian and its intensity
// exponential (fully developed speckle) with mean 1. One lattice cell is `kSpeckleCell` times
// lambda z / D: the intensity's correlation area is 1.08 cells^2, measured over the field, and
// Goodman's (lambda z)^2 / A for a pupil of area A = pi D^2 / 4.
static const uint kSpeckleLayers = 4;
static const float kSpeckleCell = 1.086;
static const float kSpeckleCorrelation = 1.08;  // cells^2
// The mean of a node's squared smoothstep weights over a cell, per axis cubed.
static const float kSpeckleWeightVariance = 0.409913;
static const float3x3 kSpeckleRotations[kSpeckleLayers] = {
    float3x3(-0.001378, -0.222370, -0.974961, 0.997727, -0.065997, 0.013642, -0.067379, -0.972726,
             0.221956),
    float3x3(-0.661598, -0.482648, 0.573880, 0.112401, -0.820503, -0.560483, 0.741386, -0.306310,
             0.597094),
    float3x3(-0.989136, 0.096774, 0.110657, -0.122309, -0.959364, -0.254287, 0.081552, -0.265059,
             0.960777),
    float3x3(-0.297184, -0.055924, -0.953181, -0.844140, -0.451140, 0.289656, -0.446217, 0.890699,
             0.086864),
};

// pcg3d (Jarzynski and Olano 2020).
uint3 pcg3d(uint3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// The speckle field's intensity at `p` (lattice cells) for seed `seed`, mean 1.
float speckleIntensity(float3 p, uint seed) {
    var field = float2(0.0);
    for (uint layer = 0; layer < kSpeckleLayers; ++layer) {
        let q = mul(kSpeckleRotations[layer], p);
        let cell = floor(q);
        let t = q - cell;
        let s = t * t * (3.0 - 2.0 * t);
        let base = asuint(int3(cell));
        for (uint corner = 0; corner < 8; ++corner) {
            let o = uint3(corner & 1u, (corner >> 1) & 1u, corner >> 2);
            let w = (o.x != 0 ? s.x : 1.0 - s.x) * (o.y != 0 ? s.y : 1.0 - s.y) *
                    (o.z != 0 ? s.z : 1.0 - s.z);
            let h = pcg3d(base + o + uint3(seed * 0x9E3779B9u, layer * 0x85EBCA6Bu, seed));
            // A complex Gaussian amplitude (Box-Muller).
            let u1 = (float(h.x >> 8) + 0.5) / 16777216.0;
            let u2 = float(h.y >> 8) / 16777216.0;
            let r = sqrt(-2.0 * log(u1));
            field += w * r * float2(cos(2.0 * PI * u2), sin(2.0 * PI * u2));
        }
    }
    return dot(field, field) / (2.0 * float(kSpeckleLayers) * kSpeckleWeightVariance);
}

// The factor laser `light` (slot `slot`) is scaled by at `world` for speckle, 1 without it
// (`LightGpu::map.z`): subjective speckle, the field the camera's aperture (D, `sh.view.w`)
// resolves, fixed to the scene. Its grain on a surface at distance z is lambda z / D across; its
// contrast is 1 / sqrt(M) for M independent patterns summed in the pixel: the pixel's area over
// the grain's correlation area plus 1, the two polarizations a depolarizing surface (`retention`
// of the laser's polarization kept) gives, 2 / (1 + P^2), and the laser's linewidth across the
// paths' spread over the pixel, sqrt(1 + (2 pi sigma_l dlambda / lambda^2)^2). Optically rough
// surfaces only, the grain isotropic in the scene (foreshortened on surfaces seen obliquely).
//
// References:
// - J. W. Goodman, "Speckle Phenomena in Optics", Roberts and Company, 2007, ch. 3-5 (statistics,
//   correlation area, contrast reduction by integration, polarization and wavelength diversity).
// - M. Jarzynski and M. Olano, "Hash Functions for GPU Rendering", JCGT 9(3), 2020.
float laserSpeckle(LightGpu light, uint slot, float3 world, float3 v, float3 toLight,
                   float3x2 footprint, float retention) {
    let aperture = asfloat(sh.view.w);
    if (light.map.z < 0.5 || !(aperture > 0.0)) {
        return 1.0;
    }
    let lambda = light.pattern.y * 1e-9;
    let cell = kSpeckleCell * lambda * length(world - sh.eye.xyz) / aperture;
    let stepX = float3(footprint[0].x, footprint[1].x, footprint[2].x);
    let stepY = float3(footprint[0].y, footprint[1].y, footprint[2].y);
    let spatial = 1.0 + length(cross(stepX, stepY)) / (kSpeckleCorrelation * cell * cell);
    let polarized = light.polarization.w * retention;
    let polarization = 2.0 / (1.0 + polarized * polarized);
    // The rms spread of the light's path, in and out, over the pixel.
    let bisector = toLight + v;
    let spread = length(float2(dot(stepX, bisector), dot(stepY, bisector))) / sqrt(12.0);
    let diversity = 2.0 * PI * spread * light.map.w * 1e-9 / (lambda * lambda);
    let looks = spatial * polarization * sqrt(1.0 + diversity * diversity);
    if (looks > 1e4) {
        return 1.0;
    }
    let intensity = speckleIntensity(world / cell, slot + 1u);
    return max(0.0, 1.0 + (intensity - 1.0) * rsqrt(looks));
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
        let slice = occlusionSlice(n, tangent, v, float2(horizons[0], horizons[1]));
        bent += slice.xyz;
        visibility += slice.w;
    }
    let length = sqrt(dot(bent, bent));
    return float4(length > 1e-6 ? bent / length : n,
                  saturate(visibility / float(kOcclusionDirections)));
}

// One slice of ground-truth ambient occlusion about axis `v`: the plane through `v` and unit
// `tangent` (perpendicular to it), with the highest horizons towards +tangent (x) and -tangent
// (y) as the cosines of their angles from `v`. Returns the slice's bent normal contribution (xyz)
// and its cosine-weighted visibility (w), both weighted by the normal's projection into it.
float4 occlusionSlice(float3 n, float3 tangent, float3 v, float2 horizons) {
    let axis = normalize(cross(tangent, v));
    let projected = n - axis * dot(n, axis);
    let projectedLength = length(projected);
    if (projectedLength < 1e-4) {
        return float4(n, 1.0);
    }
    let cosN = clamp(dot(projected / projectedLength, v), -1.0, 1.0);
    let gamma = (dot(projected, tangent) < 0.0 ? -1.0 : 1.0) * acos(cosN);
    // The horizon angles (positive towards the tangent), clamped to the hemisphere about the
    // normal, and the cosine-weighted arc between them.
    let h1 = gamma + min(acos(clamp(horizons.x, -1.0, 1.0)) - gamma, 0.5 * PI);
    let h0 = gamma + max(-acos(clamp(horizons.y, -1.0, 1.0)) - gamma, -0.5 * PI);
    let arc = 0.25 * (-cos(2.0 * h0 - gamma) + cosN + 2.0 * h0 * sin(gamma)) +
              0.25 * (-cos(2.0 * h1 - gamma) + cosN + 2.0 * h1 * sin(gamma));
    // The slice's bent normal: the cosine-weighted mean of the directions between the horizons,
    // sin(h) along the tangent plus cos(h) along `v` (Jimenez et al. 2016).
    let t0 = (6.0 * sin(h0 - gamma) - sin(3.0 * h0 - gamma) + 6.0 * sin(h1 - gamma) -
              sin(3.0 * h1 - gamma) + 16.0 * sin(gamma) -
              3.0 * (sin(h0 + gamma) + sin(h1 + gamma))) /
             12.0;
    let t1 = (-cos(3.0 * h0 - gamma) - cos(3.0 * h1 - gamma) + 8.0 * cos(gamma) -
              3.0 * (cos(h0 + gamma) + cos(h1 + gamma))) /
             12.0;
    return float4(projectedLength * (t0 * tangent + t1 * v), projectedLength * arc);
}

// `horizonMap`, the terrain's horizon map (terrain_horizon.slang): per texel and level (a height
// above the ground, `ShadingConstants::horizonLevels`), the tangents of the horizon's elevation
// towards azimuths 0 to 135 degrees (layer 2 level) and 180 to 315 (layer 2 level + 1), before
// vertical exaggeration. `terrainHeights`, the terrain's heights before exaggeration, rows from its
// north edge. `atmosphereTransmittance`, the atmosphere's transmittance to its top
// (atmosphere.slang), a layer per four bands (then the planet's shadow, its layers after).

// The apparent direction from `eye` (relative to the frame's origin, metres) of something at
// infinity in true direction `w`, lifted towards the zenith by the atmosphere's refraction in the
// middle band; `w` without an atmosphere or where its ray misses it. The bending is for the
// apparent direction, which one fixed-point step finds from the true one.
float3 apparentDirection(float3 eye, float3 w) {
    if (sh.atmosphere[1].y == 0.0) {
        return w;
    }
    let bottom = sh.atmosphere[0].w;
    let top = sh.atmosphere[1].x;
    let middle = uint(sh.range.z) / 2;
    let p = aboveGround(eye * 1e-3 - sh.atmosphere[0].xyz, sh.atmosphere[0].w);
    // Started from the bending of a ray along the eye's horizon, so a body just below the
    // geometric horizon, whose own ray would meet the ground, starts lifted clear of it; then
    // apparent = true + bending(apparent), which contracts by about a quarter each step near the
    // horizon (where the bending changes fastest), to within a tenth of an arcminute in four.
    let r0 = length(p);
    var apparent = w;
    if (r0 <= top) {
        let horizontal = refractionDeviation(atmosphereRefraction, tableSampler, r0,
                                             horizonCosine(r0, bottom), middle / 4, bottom, top);
        apparent = turnTowardsZenith(w, p / r0, horizontal[middle % 4]);
    }
    for (uint i = 0; i < 4; ++i) {
        float start;
        float end;
        bool ground;
        if (!clipToAtmosphere(p, apparent, bottom, top, start, end, ground)) {
            return apparent;
        }
        let first = p + start * apparent;
        let r = length(first);
        let vertical = first / r;
        let bending = refractionDeviation(atmosphereRefraction, tableSampler, r,
                                          dot(vertical, apparent), middle / 4, bottom, top);
        apparent = turnTowardsZenith(w, vertical, bending[middle % 4]);
    }
    return apparent;
}

// The true direction of what is seen from `eye` along apparent direction `d` beyond the atmosphere,
// in the middle band: `d` turned down by its refraction.
float3 trueDirection(float3 eye, float3 d) {
    if (sh.atmosphere[1].y == 0.0) {
        return d;
    }
    let bottom = sh.atmosphere[0].w;
    let top = sh.atmosphere[1].x;
    let middle = uint(sh.range.z) / 2;
    let p = aboveGround(eye * 1e-3 - sh.atmosphere[0].xyz, sh.atmosphere[0].w);
    float start;
    float end;
    bool ground;
    if (!clipToAtmosphere(p, d, bottom, top, start, end, ground)) {
        return d;
    }
    let first = p + start * d;
    let r = length(first);
    let vertical = first / r;
    let bending = refractionDeviation(atmosphereRefraction, tableSampler, r, dot(vertical, d),
                                      middle / 4, bottom, top);
    return turnTowardsZenith(d, vertical, -bending[middle % 4]);
}

// The wavelengths (nm) of band group `group`'s bands.
float4 groupWavelengths(uint group) {
    float4 wavelengths;
    for (uint k = 0; k < 4; ++k) {
        wavelengths[k] = sh.range.x + (float(group * 4 + k) + 0.5) * sh.range.y;
    }
    return wavelengths;
}

// The celestial bodies as an `Eclipse` of band group `group`, in km from `centre` (km from the
// frame's origin), leaving out body `skip`.
Eclipse bodiesEclipse(uint group, float3 centre, uint skip = 4) {
    Eclipse eclipse;
    for (uint i = 0; i < 4; ++i) {
        eclipse.bodies[i] =
            i == skip ? float4(0.0) : float4(sh.bodies[i].xyz - centre, sh.bodies[i].w);
    }
    eclipse.wavelengths = groupWavelengths(group);
    return eclipse;
}

// Whether any celestial body is set.
bool anyBodies() {
    return any(float4(sh.bodies[0].w, sh.bodies[1].w, sh.bodies[2].w, sh.bodies[3].w) > 0.0);
}

// Whether `lights[k]` is a distant light this view sees, not a dome's direction.
bool isSkyLight(uint k) {
    let light = lights[k];
    return ((sh.view.x >> k) & 1u) != 0 && light.position.w == 0.0 && light.shadow.y < 0.5;
}

// The distant lights the atmosphere carries (`SkyLights`), with their irradiance in band group
// `group`: the distant lights this view sees, such as the sun and earthshine, as many as fit, in
// their order. Each scatters in the air, lights the ground, the bodies and the planet's night
// side, and shows its disc.
SkyLights skyLights(uint group) {
    SkyLights sky;
    sky.count = 0;
    let count = min(sh.size.w, kMaxLights);
    for (uint k = 0; k < count && sky.count < kSkyLights; ++k) {
        if (!isSkyLight(k)) {
            continue;
        }
        let tangent = lights[k].extentY.w;
        sky.towards[sky.count] =
            float4(lights[k].position.xyz, tangent * rsqrt(1.0 + tangent * tangent));
        sky.irradiance[sky.count] = lights[k].spectrum[group] + lights[k].tinted[group];
        ++sky.count;
    }
    return sky;
}

// The nearest celestial body along `d` from `eye` (km from the frame's origin) whose disc covers
// some of a pixel of angular radius `pixelRadius`: its index (4 for none), the point on it (or
// nearest it) that the ray sees, and the fraction of the pixel it covers, from the angle between
// the ray and its limb.
uint skyBody(float3 eye, float3 d, float pixelRadius, out float3 point, out float coverage) {
    uint nearest = 4;
    var nearestDistance = 1e30;
    point = float3(0.0);
    coverage = 0.0;
    for (uint i = 0; i < 4; ++i) {
        let body = sh.bodies[i];
        if (body.w <= 0.0) {
            continue;
        }
        let toBody = body.xyz - eye;
        let distance = length(toBody);
        let along = dot(toBody, d);
        if (along <= 0.0 || distance <= body.w || distance >= nearestDistance) {
            continue;
        }
        let bodyRadius = asin(body.w / distance);
        let offset = 2.0 * asin(min(0.5 * length(toBody / distance - d), 1.0));
        let covers = saturate(0.5 + (bodyRadius - offset) / max(2.0 * pixelRadius, 1e-9));
        if (covers <= 0.0) {
            continue;
        }
        nearest = i;
        nearestDistance = distance;
        coverage = covers;
        // Where the ray meets the sphere, or the limb's point nearest the ray.
        let miss = toBody - along * d;
        let closest = length(miss);
        point = closest < body.w ? eye + (along - sqrt(body.w * body.w - closest * closest)) * d
                                 : body.xyz - miss / max(closest, 1e-9) * body.w;
    }
    return nearest;
}

// The radiance of celestial body `index` at `point` on it (km from the frame's origin) seen along
// `d` in band group `group`, lit by each of the sky's distant lights (`skyLights`):
// Lommel-Seeliger reflectance scaled to its normal albedo, a radiance factor of
// A 2 mu0 / (mu0 + mu), which lights a full moon's disc evenly as the Moon's is; dimmed where the
// other bodies eclipse the light there, and the planet, whose atmosphere refracts reddened light
// into its umbra (`planetShadow`, whose table takes the sun's size for every light).
//
// References:
// - B. Hapke, "Theory of Reflectance and Emittance Spectroscopy", 2nd ed., Cambridge University
//   Press, 2012, section 8.5 (the Lommel-Seeliger law, and the radiance factor).
//   https://doi.org/10.1017/CBO9781139025683
// - A. S. McEwen, "Photometric functions for photoclinometry and other applications", Icarus
//   92(2):298-311, 1991 (the Lommel-Seeliger law for the Moon's maria and highlands).
//   https://doi.org/10.1016/0019-1035(91)90053-V
float4 bodyRadiance(uint index, float3 point, float3 d, uint group) {
    let body = sh.bodies[index];
    let normal = normalize(point - body.xyz);
    let mu = max(dot(normal, -d), 0.0);
    let sky = skyLights(group);
    let eclipse = bodiesEclipse(group, float3(0.0), index);
    var lit = float4(0.0);
    for (uint i = 0; i < sky.count; ++i) {
        let towards = sky.towards[i].xyz;
        let sine = sky.towards[i].w;
        let mu0 = dot(normal, towards);
        if (mu0 <= 0.0) {
            continue;
        }
        var visible = eclipseVisibility(eclipse, point, towards, sine);
        if (sh.atmosphere[1].y != 0.0) {
            // The planet's shadow follows the transmittance in its table.
            let groups = (uint(sh.range.z) + 3) / 4;
            visible *= planetShadow(atmosphereTransmittance, tableSampler,
                                    sh.atmosphere[0].xyz - point, towards, sine,
                                    groupWavelengths(group), groups + group, sh.atmosphere[1].x);
        }
        lit += sky.irradiance[i] * 2.0 * mu0 / (mu0 + mu) * visible;
    }
    if (all(lit == 0.0)) {
        return float4(0.0);
    }
    let rgb = sh.bodyAlbedo[index].xyz;
    let scale = max(rgb.x, max(rgb.y, rgb.z));
    let coefficients = scale > 0.0 ? sigmoidCoefficients(rgb / scale) : float3(0.0);
    let wavelengths = groupWavelengths(group);
    float4 albedo;
    for (uint k = 0; k < 4; ++k) {
        albedo[k] = modulationAt(coefficients, scale, wavelengths[k]);
    }
    return albedo / PI * lit;
}

// Celestial bodies over what lies behind them along `d` from `eye` (relative to the frame's
// origin, metres) where there is no atmosphere to draw them through (`atmosphereSky` does where
// there is), in pixels of angular radius `pixelRadius`.
void bodiesSky(inout float radiance[kMaxBands], float3 eye, float3 d, float pixelRadius) {
    if (!anyBodies() || ((sh.sky.x & kSkyAtmosphere) != 0 && sh.atmosphere[1].y != 0.0)) {
        return;
    }
    float3 point;
    float coverage;
    let index = skyBody(eye * 1e-3, d, pixelRadius, point, coverage);
    if (index == 4) {
        return;
    }
    let bands = uint(sh.range.z);
    for (uint group = 0; group * 4 < bands; ++group) {
        let light = bodyRadiance(index, point, d, group);
        for (uint k = 0; k < 4 && group * 4 + k < bands; ++k) {
            let band = group * 4 + k;
            radiance[band] = lerp(radiance[band], light[k], coverage);
        }
    }
}

// The light of the sky and the ground around a surface at `world` (relative to the frame's
// origin, metres) facing `normal` (its bent normal where geometry occludes part of the sky), seen
// over the fraction `visibility` of its surroundings: per band, the radiance of a uniform
// surround giving the same irradiance, from each of the sky's distant lights (`skyLights`); 0
// without an atmosphere. The sky lights it as isotropically over the part of the sky it faces,
// (1 + cos tilt) / 2, of the sky's irradiance on level ground (the sky irradiance table); the
// ground, Lambertian with the atmosphere's ground albedo and lit by the sun and the sky, over the
// rest. Both dim as the celestial bodies eclipse the sun here, as if they eclipsed it for all the
// air around: in a total eclipse's umbra this leaves out the twilight scattered in from beyond it.
//
// References:
// - B. Y. H. Liu and R. C. Jordan, "The long-term average performance of flat-plate solar-energy
//   collectors", Solar Energy 7(2):53-74, 1963 (irradiance on tilted surfaces from the sky and
//   the ground). https://doi.org/10.1016/0038-092X(63)90006-9
// - E. Bruneton, "Precomputed Atmospheric Scattering: a New Implementation", 2017,
//   `GetSunAndSkyIrradiance`. https://ebruneton.github.io/precomputed_atmospheric_scattering/
void skyLight(float3 world, float3 normal, float visibility, out float light[kMaxBands]) {
    let bands = uint(sh.range.z);
    for (uint band = 0; band < kMaxBands; ++band) {
        light[band] = 0.0;
    }
    if (sh.atmosphere[1].y == 0.0) {
        return;
    }
    let bottom = sh.atmosphere[0].w;
    let top = sh.atmosphere[1].x;
    let p = world * 1e-3 - sh.atmosphere[0].xyz;
    let r = max(length(p), bottom);
    let up = normalize(p);
    let facing = 0.5 * (1.0 + dot(normal, up));
    for (uint group = 0; group * 4 < bands; ++group) {
        let sources = skyLights(group);
        let eclipse = bodiesEclipse(group, float3(0.0));
        let albedo = groundAlbedo(atmosphereMedium, group);
        var surround = float4(0.0);
        for (uint i = 0; i < sources.count; ++i) {
            let towards = sources.towards[i].xyz;
            let sine = sources.towards[i].w;
            let muSun = dot(up, towards);
            let sky = skyIrradiance(atmosphereSkyIrradiance, tableSampler, r, muSun, group,
                                    bottom, top);
            // The ground below, lit by the light through the air and by the sky.
            let toSun = sunTransmittance(atmosphereTransmittance, tableSampler, up * bottom,
                                         towards, sine, group, bottom, top);
            let ground = albedo * (toSun * max(muSun, 0.0) + sky);
            surround += sources.irradiance[i] * (sky * facing + ground * (1.0 - facing)) *
                        eclipseVisibility(eclipse, world * 1e-3, towards, sine);
        }
        surround *= visibility / PI;
        for (uint k = 0; k < 4 && group * 4 + k < bands; ++k) {
            light[group * 4 + k] = surround[k];
        }
    }
}

// Steps of the sky's march through the atmosphere per pixel, and of the aerial perspective's
// between the eye and a surface.
static const uint kSkySteps = 48;
static const uint kAerialSteps = 32;

// Aerial perspective: the atmosphere between the eye and the surface point `world` (relative to
// the frame's origin, metres) seen through the pixel at `ndc`. Per band, the transmittance across
// it and the light of the sky's distant lights (`skyLights`) it scatters towards the eye; 1 and 0
// without an atmosphere. The eye is the camera's centre, the near plane's point for an
// orthographic camera, or the row's pose for a pushbroom.
void aerialPerspective(float3 world, float2 ndc, out float transmittance[kMaxBands],
                       out float scattered[kMaxBands]) {
    let bands = uint(sh.range.z);
    for (uint band = 0; band < kMaxBands; ++band) {
        transmittance[band] = 1.0;
        scattered[band] = 0.0;
    }
    if (sh.atmosphere[1].y == 0.0) {
        return;
    }
    float3 eye;
    if (kPushbroomCamera) {
        float3 ray;
        cameraRayThrough(sh.invViewProj, sh.strip, ndc, eye, ray);
    } else if (sh.range.w != 0.0) {
        eye = cameraPoint(sh.invViewProj, sh.strip, ndc, 1.0);
    } else {
        eye = sh.eye.xyz;
    }
    let toSurface = (world - eye) * 1e-3;
    let span = length(toSurface);
    if (span <= 0.0) {
        return;
    }
    let p = aboveGround(eye * 1e-3 - sh.atmosphere[0].xyz, sh.atmosphere[0].w);
    let d = toSurface / span;
    float start;
    float end;
    if (!clipSegment(p, d, span, sh.atmosphere[1].x, start, end)) {
        return;
    }
    for (uint group = 0; group * 4 < bands; ++group) {
        float4 light;
        float4 through;
        scatterAlongLights(atmosphereMedium, atmosphereDensities, atmospherePhases,
                           atmosphereTransmittance, atmosphereMultiple, tableSampler, p, d,
                           max(start, 0.0), end, skyLights(group),
                           bodiesEclipse(group, sh.atmosphere[0].xyz), group, kAerialSteps, light,
                           through);
        for (uint k = 0; k < 4 && group * 4 + k < bands; ++k) {
            transmittance[group * 4 + k] = through[k];
            scattered[group * 4 + k] = light[k];
        }
    }
}

// The atmosphere between the eye at `origin` (relative to the frame's origin, metres) and the
// dome and stars along `d`: their light dims by its transmittance and the light of the sky's
// distant lights (`skyLights`) it scatters adds; their discs, of `pixelSolidAngle` or larger,
// show dimmed through it; a ray that meets the planet where no geometry is drawn sees the air in
// front of it over black (the atmosphere is the air alone: its planet's surface is the scene's own
// geometry, a planet Terrain or a sphere, shown whether the air is or not). Celestial bodies
// show in front of the lights' discs, the dome and the stars (`bodyRadiance`), and eclipse the
// lights for the air and the ground (`eclipseVisibility`).
void atmosphereSky(inout float radiance[kMaxBands], float3 origin, float3 d,
                   float pixelSolidAngle) {
    for (uint band = 0; band < kMaxBands; ++band) {
        skyScattered[band] = 0.0;
    }
    if ((sh.sky.x & kSkyAtmosphere) == 0 || sh.atmosphere[1].y == 0.0) {
        return;
    }
    let bands = uint(sh.range.z);
    let bottom = sh.atmosphere[0].w;
    let top = sh.atmosphere[1].x;
    let pixelRadius = sqrt(pixelSolidAngle / PI);
    let p = aboveGround(origin * 1e-3 - sh.atmosphere[0].xyz, bottom);
    float start;
    float end;
    bool ground;
    let inside = clipToAtmosphere(p, d, bottom, top, start, end, ground);
    for (uint group = 0; group * 4 < bands; ++group) {
        var scattered = float4(0.0);
        var transmittance = float4(1.0);
        // Refraction: rays that the air bends back up clear the ground just below the
        // horizon, and what lies beyond the atmosphere shows from further down, where the bent
        // ray came from (judged where the ray is in the atmosphere first).
        var bending = float4(0.0);
        var escapes = bool4(!ground);
        var vertical = float3(0.0, 0.0, 1.0);
        let eclipse = bodiesEclipse(group, sh.atmosphere[0].xyz);
        let sources = skyLights(group);
        if (inside) {
            scatterAlongLights(atmosphereMedium, atmosphereDensities, atmospherePhases,
                               atmosphereTransmittance, atmosphereMultiple, tableSampler, p, d,
                               start, end, sources, eclipse, group, kSkySteps, scattered,
                               transmittance);
            let first = p + start * d;
            let r = length(first);
            vertical = first / r;
            let mu = dot(vertical, d);
            bending = refractionDeviation(atmosphereRefraction, tableSampler, r, mu, group,
                                          bottom, top);
            escapes = escapesAtmosphere(atmosphereMedium, atmosphereDensities, r, mu, group);
            // A ray towards the ground that the air bends clear of it passes through the
            // air low along the horizon, not just to the ground: its light and transmittance are
            // those of the ray grazing the horizon, out to the top.
            if (ground && any(escapes)) {
                let level = d - mu * vertical;
                let grazing = horizonCosine(r, bottom) + 1e-4;
                let along = normalize(level) * sqrt(max(1.0 - grazing * grazing, 0.0)) +
                            grazing * vertical;
                float grazingStart;
                float grazingEnd;
                bool grazingGround;
                if (clipToAtmosphere(p, along, bottom, top, grazingStart, grazingEnd,
                                     grazingGround) &&
                    !grazingGround) {
                    float4 light;
                    float4 through;
                    scatterAlongLights(atmosphereMedium, atmosphereDensities, atmospherePhases,
                                       atmosphereTransmittance, atmosphereMultiple, tableSampler,
                                       p, along, grazingStart, grazingEnd, sources, eclipse, group,
                                       kSkySteps, light, through);
                    scattered = select(escapes, light, scattered);
                    transmittance = select(escapes, through, transmittance);
                }
            }
        }
        // A light's disc smaller than the pixel shows dimmed by the bodies as a whole, from the
        // eye; a larger one shows where no body covers it.
        bool resolved[kSkyLights];
        float4 eyeVisible[kSkyLights];
        for (uint i = 0; i < sources.count; ++i) {
            let sine = sources.towards[i].w;
            resolved[i] = pixelSolidAngle <= PI * sine * sine;
            eyeVisible[i] = resolved[i] || !anyBodies()
                                ? float4(1.0)
                                : eclipseVisibility(eclipse, p, sources.towards[i].xyz, sine);
        }
        for (uint k = 0; k < 4; ++k) {
            let band = group * 4 + k;
            if (band >= bands) {
                break;
            }
            let wavelength = sh.range.x + (float(band) + 0.5) * sh.range.y;
            let beyond = turnTowardsZenith(d, vertical, -bending[k]);
            var behind = 0.0;
            if (escapes[k]) {
                // A celestial body in front of the stars and the dome, its limb's pixels
                // partly covering them; and in front of a light's disc, where the pixel's centre
                // lies on it: in a total eclipse the Moon is barely larger than the sun, and a
                // partly covered pixel would let the sun's light through all around its limb.
                behind = radiance[band];
                var covered = false;
                float3 point;
                float coverage;
                let index = skyBody(origin * 1e-3, beyond, pixelRadius, point, coverage);
                if (index < 4) {
                    let light = bodyRadiance(index, point, beyond, group);
                    behind = lerp(behind, light[k], coverage);
                    covered = coverage >= 0.5;
                }
                for (uint i = 0; i < sources.count; ++i) {
                    let shows = resolved[i] && covered ? 0.0 : eyeVisible[i][k];
                    behind += sources.irradiance[i][k] * shows *
                              sunDisc(beyond, sources.towards[i].xyz, sources.towards[i].w,
                                      wavelength, pixelSolidAngle);
                }
            }
            radiance[band] = behind * transmittance[k] + scattered[k];
            skyScattered[band] = scattered[k];
        }
    }
}

// The light the air scattered into the last `atmosphereSky` per band, which `skyPolarization`
// polarizes; the rest of the sky's light is unpolarized.
static float skyScattered[kMaxBands];

// Light the sky's lights scatter towards the eye along view direction `d`, as each band of
// `group`'s Stokes Q and U over I in the frame (`frameX`, `frameY`) across it: each light's
// single-scattering Rayleigh pattern, sin^2 g / (1 + cos^2 g) at scattering angle g, its field
// across the plane of the light and the line of sight, scaled by the sky's greatest degree
// (`sh.atmosphere[1].w`, which multiple scattering and haze keep below 1) and weighted by the
// light's share of the lights' irradiance.
//
// References:
// - S. Chandrasekhar, "Radiative Transfer", Oxford, 1950 (Rayleigh scattering's polarization).
// - K. L. Coulson, "Polarization and Intensity of Light in the Atmosphere", A. Deepak, 1988
//   (the clear sky's greatest degree, about 0.75 to 0.85).
void skyPolarization(uint group, float3 d, float3 frameX, float3 frameY, out float4 q,
                     out float4 u) {
    q = float4(0.0);
    u = float4(0.0);
    let greatest = sh.atmosphere[1].w;
    if (greatest <= 0.0 || sh.atmosphere[1].y == 0.0) {
        return;
    }
    let sources = skyLights(group);
    var total = float4(0.0);
    for (uint i = 0; i < sources.count; ++i) {
        total += sources.irradiance[i];
    }
    for (uint i = 0; i < sources.count; ++i) {
        let towards = sources.towards[i].xyz;
        let across = cross(d, towards);
        if (dot(across, across) < 1e-12) {
            continue;
        }
        let cosG = dot(d, towards);
        let degree = greatest * (1.0 - cosG * cosG) / (1.0 + cosG * cosG);
        let field = normalize(across);
        let x = dot(field, frameX);
        let y = dot(field, frameY);
        let seen = max(x * x + y * y, 1e-12);
        let share = sources.irradiance[i] / max(total, float4(1e-30));
        q += share * degree * (x * x - y * y) / seen;
        u += share * degree * 2.0 * x * y / seen;
    }
}

// The fraction of distant light `k`'s light that reaches `world` (relative to the frame's
// origin, metres) in band `band`: its transmittance through the atmosphere along `apparent`, its
// apparent direction there (`apparentDirection`), so the ground stays lit while a refracted sun
// still shows above its horizon (1 without an atmosphere); and past the celestial bodies
// (`eclipseVisibility`). Its irradiance is what arrives above the atmosphere.
float distantLightThrough(uint k, float3 world, float3 apparent, uint band) {
    let tangent = lights[k].extentY.w;
    let sine = tangent * rsqrt(1.0 + tangent * tangent);
    var through = 1.0;
    if (sh.atmosphere[1].y != 0.0) {
        through = sunTransmittance(atmosphereTransmittance, tableSampler,
                                   world * 1e-3 - sh.atmosphere[0].xyz, apparent, sine, band / 4,
                                   sh.atmosphere[0].w, sh.atmosphere[1].x)[band % 4];
    }
    if (anyBodies()) {
        through *= eclipseVisibility(bodiesEclipse(band / 4, float3(0.0)), world * 1e-3,
                                     lights[k].position.xyz, sine)[band % 4];
    }
    return through;
}

static const uint kHorizonLevels = 4;

// The horizon tangents of level `level` around texel `texel` (clamped to the map), as eight
// values, bilinear at fraction `f` towards the next texels.
void horizonAround(int2 texel, float2 f, uint level, out float4 first, out float4 second) {
    let last = int2(pass.horizonInfo.xy) - 1;
    float4 a[2], b[2], c[2], d[2];
    for (uint side = 0; side < 2; ++side) {
        let layer = 2 * level + side;
        a[side] = horizonMap.Load(int4(clamp(texel, int2(0), last), layer, 0));
        b[side] = horizonMap.Load(int4(clamp(texel + int2(1, 0), int2(0), last), layer, 0));
        c[side] = horizonMap.Load(int4(clamp(texel + int2(0, 1), int2(0), last), layer, 0));
        d[side] = horizonMap.Load(int4(clamp(texel + int2(1, 1), int2(0), last), layer, 0));
    }
    first = lerp(lerp(a[0], b[0], f.x), lerp(c[0], d[0], f.x), f.y);
    second = lerp(lerp(a[1], b[1], f.x), lerp(c[1], d[1], f.x), f.y);
}

// How far world point `world` (relative to the frame's origin) stands above the terrain under
// it, in height units (before exaggeration): bilinear in the heights. False off the terrain or
// over no-data.
bool aboveTerrain(float3 world, out float above) {
    above = 0.0;
    let grid = pass.horizonGrid;
    let size = int2(pass.heightsSize.xy);
    let cell = float2(world.x - grid.x, grid.y - world.y) / pass.heightsGrid.x - 0.5;
    if (any(cell < -0.5) || any(cell > float2(size) - 0.5)) {
        return false;
    }
    let base = int2(floor(cell));
    let f = saturate(cell - float2(base));
    float h[4];
    for (uint k = 0; k < 4; ++k) {
        let at = clamp(base + int2(k & 1, k >> 1), int2(0), size - 1);
        h[k] = terrainHeights.Load(int3(at, 0)).x;
        if ((pass.heightsGrid.w > 0.5 && h[k] == pass.heightsGrid.z) || isnan(h[k])) {
            return false;
        }
    }
    let ground = lerp(lerp(h[0], h[1], f.x), lerp(h[2], h[3], f.x), f.y);
    above = (world.z + pass.heightsGrid.y) / grid.w - ground;
    return true;
}

// The dome's visibility and bent normal at world point `world` (relative to the frame's origin)
// with unit normal `n`, `above` height units over the terrain (0 on it), from the terrain's
// horizon map: the horizons around the point (bilinear in its texels, linear between the levels
// bracketing `above`, extrapolated past the highest), in four vertical slices whose horizons to
// either side are the map's opposite azimuths, integrated as `occlusionSlice` does about the
// vertical. They see the whole terrain, on screen or not, at the map's resolution.
float4 terrainOcclusion(float3 world, float3 n, float above, bool onTerrain) {
    let grid = pass.horizonGrid;
    let cell = float2(world.x - grid.x, grid.y - world.y) / grid.z - 0.5;
    let base = int2(floor(cell));
    let f = cell - float2(base);
    uint lower = 0;
    while (lower + 2 < kHorizonLevels && above > pass.horizonLevels[lower + 1]) {
        lower += 1;
    }
    float4 first, second;
    // The terrain's own pixels see what stands on it too, when the map holds that horizon
    // (terrain_horizon.slang's `horizonCastersMain`, the layers after the levels').
    horizonAround(base, f, onTerrain && pass.heightsSize.z != 0 ? kHorizonLevels : lower, first,
                  second);
    if (!onTerrain && above > 0.0) {
        float4 upperFirst, upperSecond;
        horizonAround(base, f, lower + 1, upperFirst, upperSecond);
        let span = pass.horizonLevels[lower + 1] - pass.horizonLevels[lower];
        let t = span > 0.0 ? (above - pass.horizonLevels[lower]) / span : 0.0;
        first = max(lerp(first, upperFirst, t), float4(-100.0));
        second = max(lerp(second, upperSecond, t), float4(-100.0));
    }
    first *= grid.w;
    second *= grid.w;
    // Elevation tangents as cosines of the angle from the vertical: sin(atan(t)).
    let up = float3(0.0, 0.0, 1.0);
    let towards = first * rsqrt(1.0 + first * first);
    let opposite = second * rsqrt(1.0 + second * second);
    var visibility = 0.0;
    var bent = float3(0.0);
    for (uint k = 0; k < 4; ++k) {
        let azimuth = float(k) * 0.25 * PI;
        let slice = occlusionSlice(n, float3(cos(azimuth), sin(azimuth), 0.0), up,
                                   float2(towards[k], opposite[k]));
        bent += slice.xyz;
        visibility += slice.w;
    }
    let length = sqrt(dot(bent, bent));
    return float4(length > 1e-6 ? bent / length : n, saturate(visibility / 4.0));
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
    var occlusion = domeOcclusion(int2(id.xy), id.z, world, ndc.z, n, v);
    // Over terrain, and on what stands over it, the horizon map's occlusion too, which sees the
    // whole terrain: the less visible of the two (as Unreal combines distance field and
    // screen-space occlusion).
    if (pass.horizonInfo.w != 0) {
        float above = 0.0;
        let onTerrain = gIds.Load(pixel).y == pass.horizonInfo.z;
        if (onTerrain || aboveTerrain(world, above)) {
            let terrain =
                terrainOcclusion(world, n, onTerrain ? 0.0 : max(above, 0.0), onTerrain);
            if (terrain.w < occlusion.w) {
                occlusion = terrain;
            }
        }
    }
    occlusionRaw[id] = occlusion;
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
    // Back along the pixel's ray: towards the eye, along the view when orthographic, towards the
    // row's pose for a pushbroom.
    let v = viewVector(ndc.xy, world);
    let n = facingNormal(normalize(gNormal.Load(pixel).xyz), v);
    // With a polarizer, the frame its linear Stokes Q and U are measured in (`pass.model.z`
    // layers on to Q, as many more to U): the image's +x and +y across the light reaching the eye.
    let stokes = pass.model.z;
    float3 frameX = float3(0.0);
    float3 frameY = float3(0.0);
    if (stokes != 0) {
        pixelFrame(ndc.xy, ndc.z, v, frameX, frameY);
    }
    // The air's polarization of the light it scatters between the surface and the eye, per
    // group of four bands as `skyPolarization` finds it.
    var airGroup = 0xFFFFFFFFu;
    var airQ = float4(0.0);
    var airU = float4(0.0);
    let albedo = gAlbedo.Load(pixel);
    let modulation = max(albedo.yzw, float3(0.0));
    let scale = max(modulation.x, max(modulation.y, modulation.z));
    let coefficients = scale > 0.0 ? sigmoidCoefficients(modulation / scale) : float3(0.0);
    // The scene's ambient light: white radiance of its luminance, shaped by its colour's
    // spectrum (normalised to its luminance on the host).
    let ambient = sh.sun.w;
    let ambientScale = max(sh.ambient.x, max(sh.ambient.y, sh.ambient.z));
    let ambientCoefficients =
        ambientScale > 0.0 ? sigmoidCoefficients(sh.ambient.xyz / ambientScale) : float3(0.0);
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
    // With a polarizer, the dome's mirrored light's Stokes Q and U over I: unpolarized light
    // polarized by the facets' Fresnel reflection about the mirror direction.
    var mirrorStokes = float2(0.0);
    if (stokes != 0 && (lobes[0].w > 0.0 || lobes[1].w > 0.0)) {
        let mirror = 2.0 * dot(n, v) * n - v;
        let facets = specularOf(n, mirror, v, float3(albedo.x), slopeVariance, params);
        mirrorStokes = specularStokes(mirror, v, 0.0, float3(0.0), facets.ior, frameX, frameY);
    }

    // Per light: direction towards it and the scale on its spectrum (inverse square, cone and
    // shadow); 0 for lights that cannot reach the point.
    let lightCount = unlit ? 0 : min(sh.size.w, kMaxLights);
    float3 lightDirections[kMaxLights];
    float lightScales[kMaxLights];
    float lightFocus[kMaxLights];  // |cos|^focus off a focused light's axis, else 1
    // Per light, with a polarizer: its reflected light's linear Stokes Q and U over I in the
    // pixel's frame, its polarization as far as the surface's diffuse reflection keeps it, along
    // its electric field seen across the light reaching the eye.
    float2 lightStokes[kMaxLights];
    let retention = stokes != 0 ? saturate(asfloat(material.info.w)) : 0.0;
    float3x2 footprint = float3x2(0.0);  // the pixel's, once a laser wants it
    bool footprinted = false;
    // The surface's light link group: lights whose masks lack it leave it unlit.
    let linkGroup = (ids.w >> 1) & 31u;
    for (uint k = 0; k < lightCount; ++k) {
        lightFocus[k] = 1.0;
        lightStokes[k] = float2(0.0);
        if (((sh.view.x >> k) & 1u) == 0 ||
            ((lights[sh.view.z + k].links.x >> linkGroup) & 1u) == 0) {
            lightDirections[k] = float3(0.0);
            lightScales[k] = 0.0;
            continue;
        }
        // The view's sub-frame's copy, where lights move across the shutter.
        let light = lights[sh.view.z + k];
        var shadowPoint = world;  // where its shadow map is read
        var toLight = light.position.xyz;
        var scale = 1.0;
        let type = uint(light.position.w);
        if (type == 5) {
            // A laser: its beam's irradiance, arriving along the beam.
            if (!footprinted) {
                footprint = pixelFootprint(int2(id.xy), id.z, world);
                footprinted = true;
            }
            // Over the frames of its sequence the view's span of time shows, each weighted by
            // its share of the span and its pulses then.
            scale = 0.0;
            for (uint s = 0; s < kMaxLit; ++s) {
                let lit = lightTiming[(shadedView * kMaxLights + k) * kMaxLit + s];
                if (lit.y > 0.0) {
                    scale += lit.y * laserIrradiance(light, k, world, footprint, uint(lit.x));
                }
            }
            let reach = max(length(float3(footprint[0].x, footprint[1].x, footprint[2].x)),
                            length(float3(footprint[0].y, footprint[1].y, footprint[2].y)));
            if (light.beamExtra.y > 0.5) {
                // A pattern spreads from the aperture, as a spot light's light does; its
                // perspective map is read a footprint back towards it.
                toLight = normalize(light.position.xyz - world);
                shadowPoint = world + toLight * reach;
            } else {
                toLight = -light.axis.xyz;
                // A spot's shadow map is read where the beam meets the pixel's surface, when
                // that lies within the pixel's footprint (the light of a beam narrower than the
                // pixel comes from there, which its map, as narrow, sees), moved a footprint back
                // up the beam: the map's texels can be far finer than the pixel, whose surface (at
                // the camera's level of detail) is only as true as its footprint.
                let crossing = dot(light.axis.xyz, n);
                if (abs(crossing) > 1e-4) {
                    let hit = light.position.xyz +
                              light.axis.xyz * (dot(world - light.position.xyz, n) / crossing);
                    if (length(hit - world) <= reach) {
                        shadowPoint = hit;
                    }
                }
                shadowPoint -= light.axis.xyz * reach;
            }
            if (scale > 0.0) {
                scale *= laserSpeckle(light, k, world, v, toLight, footprint,
                                      saturate(asfloat(material.info.w)));
            }
        } else if (type >= 2) {
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
        if (type >= 1 && (light.shadow.z > 0.0 || light.shadow.w > 0.0)) {
            // Shaped by the direction from the light's centre (UsdLux's focus and IES profile):
            // for an area light, as if it were small beside its distance.
            let fromLight = world - light.position.xyz;
            let distance = length(fromLight);
            let direction = distance > 0.0 ? fromLight / distance : light.axis.xyz;
            if (light.shadow.z > 0.0) {
                lightFocus[k] = pow(abs(dot(direction, light.axis.xyz)), light.shadow.z);
            }
            if (light.shadow.w > 0.0) {
                scale *= lightProfile(light, direction);
            }
        }
        let nDotL = dot(n, toLight);
        if (nDotL <= 0.0 || scale <= 0.0) {
            scale = 0.0;
        } else {
            scale *= lightVisibility(light, shadowPoint, n, nDotL, float2(id.xy)) *
                     volumeTransmittance(light, shadowPoint);
        }
        lightDirections[k] = toLight;
        lightScales[k] = scale;
        let polarized = light.polarization.w * retention;
        if (polarized > 0.0) {
            let field = light.polarization.xyz;
            let x = dot(field, frameX);
            let y = dot(field, frameY);
            let seen = x * x + y * y;
            if (seen > 1e-8) {
                lightStokes[k] = polarized * float2(x * x - y * y, 2.0 * x * y) / seen;
            }
        }
    }
    float aerialTransmittance[kMaxBands];
    float aerialScattered[kMaxBands];
    aerialPerspective(world, ndc.xy, aerialTransmittance, aerialScattered);
    float surround[kMaxBands];
    skyLight(world, lookup, unlit ? 0.0 : occlusion, surround);
    // Distant lights as refraction shows them here, such as the sun still above the horizon a
    // little after it has set, each dimmed through the air and past the bodies.
    let throughAir = sh.atmosphere[1].y != 0.0 || anyBodies();
    float3 apparentLights[kMaxLights];
    for (uint k = 0; k < lightCount; ++k) {
        apparentLights[k] = throughAir && lightScales[k] > 0.0 && isSkyLight(k)
                                ? apparentDirection(world, lights[k].position.xyz)
                                : float3(0.0);
    }
    for (uint first = 0; first < bands; first += 3) {
        float3 bandAlbedo = float3(0.0);
        float3 unlitIrradiance = float3(0.0);
        for (uint k = 0; k < 3; ++k) {
            let band = min(first + k, bands - 1);
            let wavelength = sh.range.x + (float(band) + 0.5) * sh.range.y;
            bandAlbedo[k] = bandValue(material.reflectance, band) * albedo.x *
                            modulationAt(coefficients, scale, wavelength);
            unlitIrradiance[k] = bandValue(sh.irradiance, band);
        }
        // How the model reflects light from all around: the scene's ambient light (from every
        // direction alike, so every lobe sees it whole) and the dome's.
        let environment = unlit ? diffuseEnvironment(bandAlbedo)
                                : environmentOf(n, v, bandAlbedo, slopeVariance, params);
        // Unlit surfaces show their colour as a white one facing the brightest distant light.
        var radiance = unlit ? unlitIrradiance / PI * bandAlbedo : float3(0.0);
        // Its linear Stokes Q and U, with a polarizer: polarized lights' share of it.
        var q = float3(0.0);
        var u = float3(0.0);
        if (!unlit && ambient > 0.0) {
            let uniform = environment.diffuse + environment.specular0 + environment.specular1;
            for (uint j = 0; j < 3; ++j) {
                let wavelength =
                    sh.range.x + (float(min(first + j, bands - 1)) + 0.5) * sh.range.y;
                radiance[j] += ambient * occlusion * uniform[j] *
                               modulationAt(ambientCoefficients, ambientScale, wavelength);
            }
        }
        for (uint j = 0; j < 3; ++j) {
            let band = min(first + j, bands - 1);
            let wavelength = sh.range.x + (float(band) + 0.5) * sh.range.y;
            let dome = bandValue(sh.dome, band);
            if (material.info.z != 0 && !unlit) {
                radiance[j] += bandValue(material.emission, band);
            }
            // The atmosphere's sky and ground around it, diffusely.
            radiance[j] += environment.diffuse[j] * surround[band];
            if (domeScale > 0.0) {
                radiance[j] += environment.diffuse[j] * dome *
                               modulationAt(domeCoefficients, domeScale, wavelength);
            }
            // The dome's light mirrored in the specular lobes, polarized by Fresnel's laws.
            var mirrored = 0.0;
            if (lobes[0].w > 0.0) {
                mirrored += environment.specular0[j] * dome *
                            modulationAt(lobes[0].xyz, lobes[0].w, wavelength);
            }
            if (lobes[1].w > 0.0) {
                mirrored += environment.specular1[j] * dome *
                            modulationAt(lobes[1].xyz, lobes[1].w, wavelength);
            }
            radiance[j] += mirrored;
            q[j] += mirrored * mirrorStokes.x;
            u[j] += mirrored * mirrorStokes.y;
        }
        for (uint k = 0; k < lightCount; ++k) {
            if (lightScales[k] <= 0.0) {
                continue;
            }
            float3 lightSpectrum;
            for (uint j = 0; j < 3; ++j) {
                let band = min(first + j, bands - 1);
                lightSpectrum[j] = bandValue(lights[k].spectrum, band) * lightFocus[k] +
                                   bandValue(lights[k].tinted, band);
                if (any(apparentLights[k] != 0.0)) {
                    lightSpectrum[j] *= distantLightThrough(k, world, apparentLights[k], band);
                }
            }
            // A dome's direction stands for its diffuse light alone: the model's diffuse
            // response to surrounding light times cos(i), as a Lambertian radiance factor (the
            // dome keeps its specular light).
            let factors =
                lights[k].shadow.y > 0.5
                    ? environment.diffuse * max(dot(n, lightDirections[k]), 0.0)
                    : radianceFactors(n, lightDirections[k], v, bandAlbedo, slopeVariance, params);
            let reflected = lightScales[k] * lightSpectrum / PI * factors;
            radiance += reflected;
            if (stokes != 0) {
                // Its specular part, polarized by Fresnel's laws; the rest diffuse, keeping the
                // light's polarization as far as the material does.
                var specular = float3(0.0);
                var specularPart = float2(0.0);
                if (lights[k].shadow.y <= 0.5) {
                    let facets =
                        specularOf(n, lightDirections[k], v, bandAlbedo, slopeVariance, params);
                    specular = min(lightScales[k] * lightSpectrum / PI * facets.factor, reflected);
                    if (any(specular > 0.0)) {
                        let polarization = lights[sh.view.z + k].polarization;
                        specularPart = specularStokes(lightDirections[k], v, polarization.w,
                                                      polarization.xyz, facets.ior, frameX,
                                                      frameY);
                    }
                }
                let diffuse = reflected - specular;
                q += diffuse * lightStokes[k].x + specular * specularPart.x;
                u += diffuse * lightStokes[k].y + specular * specularPart.y;
            }
        }
        // Through the air to the eye: dimmed, with the sun's light it scatters on the way.
        for (uint k = 0; k < 3 && first + k < bands; ++k) {
            let band = first + k;
            let layer = id.z * bands + band;
            spectral[int3(id.xy, layer)] =
                radiance[k] * aerialTransmittance[band] + aerialScattered[band];
            if (stokes != 0) {
                if (band / 4 != airGroup) {
                    airGroup = band / 4;
                    skyPolarization(airGroup, -v, frameX, frameY, airQ, airU);
                }
                spectral[int3(id.xy, stokes + layer)] =
                    q[k] * aerialTransmittance[band] + aerialScattered[band] * airQ[band % 4];
                spectral[int3(id.xy, 2 * stokes + layer)] =
                    u[k] * aerialTransmittance[band] + aerialScattered[band] * airU[band % 4];
            }
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
    // Reverse-Z: from the near plane (depth 1) to twice its distance (1/2), which an infinite far
    // plane, with no point at depth 0, still has.
    let near = mul(sh.invViewProj, float4(ndc, 1.0, 1.0));
    let beyond = mul(sh.invViewProj, float4(ndc, 0.5, 1.0));
    return beyond.xyz / beyond.w - near.xyz / near.w;
}

// Texel (`x`, `y`) of mip `level` of the dome map (the lighting map with `lighting`), wrapping
// around in longitude.
float3 domeTexel(int x, int y, uint level, int2 size, bool lighting) {
    let wrapped = int2((x % size.x + size.x) % size.x, clamp(y, 0, size.y - 1));
    let at = int3(wrapped, int(level));
    return lighting ? domeLightingMap.Load(at).rgb : domeMap.Load(at).rgb;
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

// How a pixel's stars land on it. A frame camera's: `a`, `b` and `c` are `rayThrough`'s steps in
// x and y and its value at the image centre, `pixel` the pixel's centre in pixels. A pushbroom's
// row: its pose's camera axes `x`, `y` and `z` in the world, the pixel's normalised image point
// `point` (x across the line, the line's y) and the focal length in pixels `focal`.
struct StarFrame {
    float3 a;
    float3 b;
    float3 c;
    float2 pixel;
    float3 x;
    float3 y;
    float3 z;
    float2 point;
    float focal;
}

// The offset in pixels of direction `w` from the pixel, or a negative one when it is behind.
// A pushbroom's row sees stars from its own pose: how far, in its detector's pixels, a star is
// from the pixel across and along the line, so a turning camera sweeps them across its rows as
// a frame camera's image holds them, and one held still draws each down its column.
float2 starOffset(StarFrame frame, float3 w) {
    if (kPushbroomCamera) {
        let local = float3(dot(frame.x, w), dot(frame.y, w), dot(frame.z, w));
        if (local.z >= 0.0) {
            return float2(-1.0);
        }
        return abs(local.xy / -local.z - frame.point) * frame.focal;
    }
    // Directions to image coordinates: w = alpha a + beta b + gamma c gives the point
    // (alpha, beta) / gamma in normalised device coordinates.
    let bc = cross(frame.b, frame.c);
    let ca = cross(frame.c, frame.a);
    let ab = cross(frame.a, frame.b);
    let det = dot(frame.a, bc);
    let gamma = dot(ab, w) / det;
    if (gamma <= 0.0) {
        return float2(-1.0);
    }
    let ndc = float2(dot(bc, w), dot(ca, w)) / det / gamma;
    let size = float2(sh.size.xy);
    let at = float2((ndc.x + 1.0) * 0.5 * size.x, (1.0 - ndc.y) * 0.5 * size.y);
    return abs(at - frame.pixel);
}

// Adds the stars around world direction `d` (seen from `eye`, relative to the frame's origin) to
// `radiance` (per band), as their tent-filtered irradiance over the pixel's solid angle, landing as
// `frame` puts them: where the atmosphere's refraction lifts them, searching around where the
// pixel looks beyond it.
void addStars(inout float radiance[kMaxBands], float3 eye, float3 d, StarFrame frame,
              float pixelAngle, float solidAngle) {
    // Points within a pixel and a half: farther ones give this pixel no weight.
    let reach = 1.5 * pixelAngle;
    if (reach > 0.2) {
        return;  // pixels so wide a point covers a crowd of cells: no stars
    }
    let beyond = trueDirection(eye, d);
    let cat = float3(dot(sh.toCatalogue[0].xyz, beyond), dot(sh.toCatalogue[1].xyz, beyond),
                     dot(sh.toCatalogue[2].xyz, beyond));
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
                let offset = starOffset(frame, apparentDirection(eye, w));
                if (offset.x < 0.0) {
                    continue;
                }
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

// The sky's parts, one dispatch each in this order (`pass.model.w`), so each has its own GPU zone:
// the dome (which also clears the pixel), the stars, the celestial bodies, then the atmosphere,
// which dims what the others left and adds the light it scatters. Each part after the first
// adds to the radiance the one before stored.
static const uint kSkyPartDome = 0;
static const uint kSkyPartStars = 1;
static const uint kSkyPartBodies = 2;
static const uint kSkyPartAtmosphere = 3;

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
    let part = pass.model.w;
    let size = float2(sh.size.xy);
    let pixel = float2(id.xy) + 0.5;
    let ndc = float2(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0);
    let c = rayThrough(float2(0.0));
    let a = rayThrough(float2(1.0, 0.0)) - c;
    let b = rayThrough(float2(0.0, 1.0)) - c;
    let bands = uint(sh.range.z);
    float radiance[kMaxBands];
    for (uint band = 0; band < kMaxBands; ++band) {
        radiance[band] =
            part == kSkyPartDome || band >= bands ? 0.0 : spectral[int3(id.xy, id.z * bands + band)];
        // Only the atmosphere's part scatters light into the pixel (`writeSky`).
        skyScattered[band] = 0.0;
    }
    if (kPushbroomCamera) {
        // Its pixels look along their rows' rays, each row from its own pose: the row's camera
        // axes from its rays, which run (x, line, -1) in them, x across the line.
        let line = sh.strip.z;
        let x = (ndc.x - sh.strip.y) / sh.strip.x;
        float3 origin;
        float3 ray;
        cameraRayThrough(sh.invViewProj, sh.strip, ndc, origin, ray);
        float3 next;
        cameraRayThrough(sh.invViewProj, sh.strip, ndc + float2(1.0, 0.0), origin, next);
        StarFrame frame;
        frame.x = normalize((next - ray) * sh.strip.x);
        // v = line y - z and x cross v = y + line z.
        let v = ray - x * frame.x;
        let w = cross(frame.x, v);
        frame.y = normalize(line * v + w);
        frame.z = normalize(line * w - v);
        frame.point = float2(x, line);
        frame.focal = 0.5 * sh.strip.x * size.x;
        let d = normalize(ray);
        if (part == kSkyPartDome && (sh.sky.x & kSkyDome) != 0) {
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
        if (part == kSkyPartStars && (sh.sky.x & kSkyStars) != 0) {
            // A detector pixel's solid angle: square, a focal length's pixel on a side, seen
            // off the axis.
            let pixelAngle = 1.0 / frame.focal;
            let cosine = 1.0 / length(float3(x, line, 1.0));
            addStars(radiance, origin, d, frame, pixelAngle,
                     pixelAngle * pixelAngle * cosine * cosine * cosine);
        }
        let detector = 1.0 / frame.focal;
        let offAxis = 1.0 / length(float3(x, line, 1.0));
        if (part == kSkyPartBodies) {
            bodiesSky(radiance, origin, d, 0.5 * detector);
        }
        if (part == kSkyPartAtmosphere) {
            atmosphereSky(radiance, origin, d,
                          detector * detector * offAxis * offAxis * offAxis);
        }
        writeSky(id, radiance, ndc, d);
        return;
    }
    if (sh.range.w != 0.0) {
        // Orthographic: every pixel sees the dome along the view direction, with no solid angle
        // for stars to fill.
        let d = normalize(c);
        if (part == kSkyPartDome && (sh.sky.x & kSkyDome) != 0) {
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
        // Each pixel's ray starts on the near plane.
        if (part == kSkyPartAtmosphere) {
            atmosphereSky(radiance, cameraPoint(sh.invViewProj, sh.strip, ndc, 1.0), d, 0.0);
        }
        writeSky(id, radiance, ndc, d);
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

    if (part == kSkyPartDome && (sh.sky.x & kSkyDome) != 0) {
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
    if (part == kSkyPartStars && (sh.sky.x & kSkyStars) != 0) {
        StarFrame frame;
        frame.a = a;
        frame.b = b;
        frame.c = c;
        frame.pixel = pixel;
        addStars(radiance, sh.eye.xyz, d, frame, pixelAngle, solidAngle);
    }
    if (part == kSkyPartBodies) {
        bodiesSky(radiance, sh.eye.xyz, d, 0.5 * pixelAngle);
    }
    if (part == kSkyPartAtmosphere) {
        atmosphereSky(radiance, sh.eye.xyz, d, solidAngle);
    }
    writeSky(id, radiance, ndc, d);
}

// Writes the sky's radiance per band to pixel `id`, at `ndc` looking along `d`, and with a
// polarizer its Stokes Q and U: the light the air scattered into it (`skyScattered`) polarized
// as `skyPolarization` gives, the rest unpolarized.
void writeSky(uint3 id, float radiance[kMaxBands], float2 ndc, float3 d) {
    let bands = uint(sh.range.z);
    let stokes = pass.model.z;
    float3 frameX = float3(0.0);
    float3 frameY = float3(0.0);
    if (stokes != 0) {
        pixelFrame(ndc, 0.5, d, frameX, frameY);
    }
    for (uint group = 0; group * 4 < bands; ++group) {
        float4 q = float4(0.0);
        float4 u = float4(0.0);
        if (stokes != 0) {
            skyPolarization(group, d, frameX, frameY, q, u);
        }
        for (uint k = 0; k < 4 && group * 4 + k < bands; ++k) {
            let band = group * 4 + k;
            let layer = id.z * bands + band;
            spectral[int3(id.xy, layer)] = radiance[band];
            if (stokes != 0) {
                spectral[int3(id.xy, stokes + layer)] = skyScattered[band] * q[k];
                spectral[int3(id.xy, 2 * stokes + layer)] = skyScattered[band] * u[k];
            }
        }
    }
}

// The frame a pixel's linear Stokes Q and U are measured in: the image's +x and +y across the
// direction `d` it sees along, from the points a pixel over at depth `depth` from its own at
// `ndc`.
void pixelFrame(float2 ndc, float depth, float3 d, out float3 frameX, out float3 frameY) {
    let step = 2.0 / float2(sh.size.xy);
    let here = cameraPoint(sh.invViewProj, sh.strip, ndc, depth);
    let across = cameraPoint(sh.invViewProj, sh.strip, ndc + float2(step.x, 0.0), depth) - here;
    let up = cameraPoint(sh.invViewProj, sh.strip, ndc + float2(0.0, step.y), depth) - here;
    frameX = normalize(across - dot(across, d) * d + 1e-12 * float3(1.0, 0.0, 0.0));
    let upAcross = up - dot(up, d) * d - dot(up, frameX) * frameX;
    frameY = normalize(upAcross + 1e-12 * cross(d, frameX));
}
