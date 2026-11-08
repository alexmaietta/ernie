// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.query.material;
__exported import ernie.render.kernels.surface.point;
__exported import ernie.texture.kernels.tiling;
import ernie.render.kernels.surface.primvars;
import ernie.texture.kernels.texture;
import ernie.render.kernels.basis;

using namespace ernie;

ObjectSpace TransformSpace(HitQuery query, uint slot, WorldTransform w) {
    ObjectSpace space;
    const SlotHit hit = query.layout->hits[slot];
    const uint source = query.tables->prims.world[hit.instancer != kNoPrim ? hit.instancer : hit.prim];
    space.origin = RenderTranslation(query.tables, w, query.world) +
                   ShownOffset(query.layout, query.world, source);
    space.axes = float3x3(w.x.Vector(), w.y.Vector(), w.z.Vector());
    return space;
}

float2 CornerSt(PrimvarRef primvar, SurfaceElement element, uint corner) {
    SurfaceElement at = element;
    at.weights = float3(corner == 0 ? 1 : 0, corner == 1 ? 1 : 0, corner == 2 ? 1 : 0);
    float4 value;
    return ReadPrimvar(primvar, at, value) ? value.xy : float2(0);
}

bool UvTangents(HitQuery query, SurfacePoint point, PrimvarRef primvar, uint name,
                WorldTransform w, out float3 tangent, out float3 bitangent, out float density) {
    tangent = float3(0);
    bitangent = float3(0);
    density = 1;
    const SurfaceElement element = point.element;
    if (!element.mesh || !primvar.Found()) {
        return false;
    }
    const SurfaceRecord surface = query.tables->surfaces[point.slot];
    const Ptr<Float3> positions = Ptr<Float3>(surface.points);
    const float3 a = positions[element.points.x].Vector();
    const float3 e1 = w.Apply(positions[element.points.y].Vector() - a);
    const float3 e2 = w.Apply(positions[element.points.z].Vector() - a);
    float2 sa = float2(0);
    float2 sb = float2(0);
    float2 sc = float2(0);
    [loop]
    for (uint corner = 0; corner < 3; ++corner) {
        const float2 st = CornerSt(primvar, element, corner);
        sa = corner == 0 ? st : sa;
        sb = corner == 1 ? st : sb;
        sc = corner == 2 ? st : sc;
    }
    const float2 d1 = sb - sa;
    const float2 d2 = sc - sa;
    const float det = d1.x * d2.y - d2.x * d1.y;
    const float area = length(cross(e1, e2));
    density = area > 0 ? sqrt(abs(det) / area) : 1;
    const TangentRecord record = query.tables->tangents[surface.meshRow];
    const ArrayRef tangents = record.tangents;
    if (tangents.count > 0 && record.primvar == name) {
        const Ptr<float4> corners = Ptr<float4>(tangents.address);
        const uint3 c = element.corners;
        const float3 weights = element.weights;
        const float4 first = corners[c.x];
        const float3 local = first.xyz * weights.x + corners[c.y].xyz * weights.y +
                             corners[c.z].xyz * weights.z;
        tangent = normalize(w.Apply(local));
        bitangent = (first.w < 0 ? -1.0f : 1.0f) * cross(point.shadingNormal, tangent);
        return true;
    }
    if (abs(det) < 1e-12f) {
        return false;
    }
    tangent = (e1 * d2.y - e2 * d1.y) / det;
    bitangent = (e2 * d1.x - e1 * d2.x) / det;
    return true;
}

Placement ProjectedPlacement(SurfacePoint point, float2 st, float weight, float3 tangent,
                             float3 bitangent, float footprint) {
    Placement placement;
    placement.st = st;
    placement.weight = weight;
    placement.tangent = tangent;
    placement.bitangent = bitangent;
    placement.footprint = footprint;
    placement.angle = 0;
    placement = Framed(placement, point.shadingNormal);
    const float density = footprint / max(point.footprint, 1e-20f);
    placement.axis =
        float2(dot(point.stretch, placement.tangent), dot(point.stretch, placement.bitangent)) *
        density;
    return placement;
}

void ForEachPlacement<V : IPlacementVisitor>(HitQuery query, SurfacePoint point, MappingRecord m,
                                       PlacementSettings settings, inout V visitor) {
    const TextureProjection projection = TextureProjection(m.projection);
    const bool uv = projection == TextureProjection::Uv;
    Placement surface = {};
    const bool object = !uv && TextureSpace(m.projectionSpace) == TextureSpace::Object;
    ObjectSpace space = {};
    float3 p = point.position;
    float3 n = point.shadingNormal;
    float tile = 1;
    float3 blend = float3(0);
    WorldTransform w = {};
    if (uv ? point.element.mesh : object) {
        w = query.tables->slotWorlds[point.slot];
    }
    if (uv) {
        const PrimvarRef primvar = FindPrimvar(query.tables, point.slot, m.uvPrimvar);
        float4 value;
        const float2 st = ReadPrimvar(primvar, point.element, value) ? value.xy : float2(0);
        float3 tangent;
        float3 bitangent;
        float density;
        if (!UvTangents(query, point, primvar, m.uvPrimvar, w, tangent, bitangent,
                        density)) {
            const Basis frame = Basis(point.shadingNormal);
            tangent = frame.tangent;
            bitangent = frame.bitangent;
        }
        surface =
            ProjectedPlacement(point, st, 1, tangent, bitangent, point.footprint * density);
    } else {
        if (object) {
            space = TransformSpace(query, point.slot, w);
            p = space.ToObject(point.position - space.origin);
            n = normalize(mul(space.axes, point.shadingNormal));
        } else {
            p = point.position - RenderShift(query.tables, query.world, query.world);
        }
        tile = max(m.tileSize, 1e-6f);
        const float3 a = abs(n);
        if (projection == TextureProjection::Planar) {
            blend = float3(0, 0, 1);
        } else if (projection == TextureProjection::Box || settings.single) {
            const uint dominant = a.x >= a.y && a.x >= a.z ? 0 : (a.y >= a.z ? 1 : 2);
            blend[dominant] = 1;
        } else {
            const float k = 1 / max(m.triplanarBlend, 1.0f / 32);
            const float3 sharp = pow(max(a, float3(1e-6f)), float3(k));
            blend = sharp / (sharp.x + sharp.y + sharp.z);
        }
    }
    const uint probes = ProbeCount(settings);
    const uint count = (uv ? 1 : 3) * probes;
    [loop]
    for (uint i = 0; i < count; ++i) {
        Placement placement = surface;
        if (!uv) {
            const uint axis = i / probes;
            const float weight = blend[axis];
            if (weight <= 0) {
                continue;
            }
            const bool flipped = n[axis] < 0 && projection != TextureProjection::Planar;
            const float side = flipped ? -1.0f : 1.0f;
            float3 u = axis == 0 ? float3(0, 1, 0) : float3(1, 0, 0);
            const float3 v = axis == 2 ? float3(0, 1, 0) : float3(0, 0, 1);
            u *= axis == 1 ? -side : side;
            const float2 st = float2(dot(p, u), dot(p, v)) / tile;
            placement = ProjectedPlacement(point, st, weight, object ? mul(u, space.axes) : u,
                                           object ? mul(v, space.axes) : v,
                                           point.footprint / tile);
        }
        visitor.Visit(Probe(placement, settings, i % probes));
    }
}

float Channel(float4 value, TextureChannel channel) {
    switch (channel) {
    case TextureChannel::G:
        return value.y;
    case TextureChannel::B:
        return value.z;
    case TextureChannel::A:
        return value.w;
    default:
        return value.x;
    }
}

float3 MixedColor(Float3 value, float4 texel, uint64_t handle, uint channel, float influence) {
    if (handle == 0) {
        return value.Vector();
    }
    const TextureChannel read = TextureChannel(channel);
    const float3 texture = read == TextureChannel::Rgb ? texel.xyz : float3(Channel(texel, read));
    return lerp(value.Vector(), texture, influence);
}

float MixedFloat(float value, float4 texel, uint64_t handle, uint channel, float influence) {
    if (handle == 0) {
        return value;
    }
    return lerp(value, Channel(texel, TextureChannel(channel)), influence);
}

static const uint kTextureInputs = 8;

struct SurfaceTexels<let Inputs : uint> : IPlacementVisitor {
    Ptr<uint64_t> bindings;
    float4 texels[Inputs];
    uint64_t normalHandle;
    float3 normalValue;
    float normalInfluence;
    float normalStrength;
    bool directX;
    float3 normal;
    float3 normalSum;
    float bias;
    uint maxAnisotropy;

    __init(MaterialQuery query, float3 normal) {
        [ForceUnroll]
        for (uint input = 0; input < Inputs; ++input) {
            texels[input] = float4(0);
        }
        bindings = Ptr<uint64_t>(0);
        normalHandle = 0;
        normalValue = float3(0, 0, 1);
        normalInfluence = 1;
        normalStrength = 1;
        directX = false;
        this.normal = normal;
        normalSum = float3(0);
        bias = query.lodBias;
        maxAnisotropy = query.maxAnisotropy;
    }

    uint64_t Handle(uint input) {
        return uint64_t(bindings) == 0 ? 0 : bindings[input];
    }

    [mutating]
    [noinline]
    void Visit(Placement placement) {
        uint pending = normalHandle != 0 ? 1u << Inputs : 0;
        [ForceUnroll]
        for (uint input = 0; input < Inputs; ++input) {
            pending |= Handle(input) != 0 ? 1u << input : 0;
        }
        float4 normalTexel = float4(0);
        [loop]
        while (pending != 0) {
            const uint first = firstbitlow(pending);
            uint64_t handle = first == Inputs ? normalHandle : 0;
            [ForceUnroll]
            for (uint input = 0; input < Inputs; ++input) {
                handle = input == first ? Handle(input) : handle;
            }
            const float4 sample =
                TextureFromHandle(handle).SampleFootprint(placement.st, placement.axis,
                                                          placement.footprint, bias, maxAnisotropy);
            if (first == Inputs) {
                normalTexel = sample;
                pending &= ~(1u << Inputs);
            }
            [ForceUnroll]
            for (uint input = 0; input < Inputs; ++input) {
                if ((pending & (1u << input)) != 0 && Handle(input) == handle) {
                    texels[input] += placement.weight * sample;
                    pending &= ~(1u << input);
                }
            }
        }
        const float3 n =
            normalHandle == 0
                ? float3(normalValue.xy * normalStrength, normalValue.z)
                : TangentNormal(normalTexel, normalValue, normalInfluence, normalStrength, directX,
                                placement.angle);
        normalSum += placement.weight *
                     (placement.tangent * n.x + placement.bitangent * n.y + normal * n.z);
    }

    float3 MappedNormal() {
        return length(normalSum) > 1e-8f ? normalize(normalSum) : normal;
    }
}

float3 CheckerColor(MaterialQuery query, SurfacePoint point, float3 color) {
    RenderTables* tables = query.hit.tables;
    if (!HasApi(tables, point.prim, kApiCheckerboardAPI)) {
        return color;
    }
    const CheckerRecord c = tables->checkers[ApiRow(tables, point.prim, kApiCheckerboardAPI)];
    if (c.enabled == 0) {
        return color;
    }
    const uint axis = c.axis;
    const ObjectSpace space =
        TransformSpace(query.hit, point.slot, tables->slotWorlds[point.slot]);
    const float3 local = space.ToObject(point.position - space.origin);
    const float2 plane = float2(local[(axis + 1) % 3], local[(axis + 2) % 3]);
    const float3 reach = space.ToObject(point.stretch);
    const float2 along = float2(reach[(axis + 1) % 3], reach[(axis + 2) % 3]);
    const float tile = 2 * max(c.size, 1e-6f);
    const uint64_t handle = tables->textureBindings[kCheckerTextureBinding];
    float t;
    if (handle != 0) {
        t = TextureFromHandle(handle)
                .SampleFootprint(plane / tile, along / tile, point.footprint / tile, query.lodBias,
                                 query.maxAnisotropy)
                .x;
    } else {
        const int2 cell = int2(floor(2 * plane / tile));
        t = ((cell.x + cell.y) & 1) != 0 ? 1.0f : 0.0f;
    }
    return lerp(c.colorA.Vector(), c.colorB.Vector(), t);
}
