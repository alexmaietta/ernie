// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.device.kernels.device;
__exported import ernie.scene.records.render_tables;
__exported import ernie.render.kernels.shapes.intersector;
__exported import ernie.render.kernels.trace.tracer;

using namespace ernie;

struct PrimHit {
    Hit hit;
    uint prim;
    uint instancer;
    uint instance;
    uint slot;
    ShapeRecord shape;

    bool IsHit() {
        return hit.IsHit();
    }
}

struct ObjectSpace {
    float3 origin;
    float3x3 axes;

    float3 ToObject(float3 world) {
        const float3 x = axes[0];
        const float3 y = axes[1];
        const float3 z = axes[2];
        return float3(dot(world, cross(y, z)), dot(world, cross(z, x)), dot(world, cross(x, y))) /
               dot(x, cross(y, z));
    }

    Ray ToObject(Ray ray) {
        Ray local = ray;
        local.origin = ToObject(ray.origin - origin);
        local.direction = ToObject(ray.direction);
        return local;
    }

    float3 ToWorld(float3 local) {
        return origin + axes[0] * local.x + axes[1] * local.y + axes[2] * local.z;
    }

    float3 ErrorToWorld(float3 local, float3 error) {
        const float3 placed = abs(axes[0] * local.x) + abs(axes[1] * local.y) +
                              abs(axes[2] * local.z) + abs(origin);
        const float3 carried =
            abs(axes[0]) * error.x + abs(axes[1]) * error.y + abs(axes[2]) * error.z;
        return Gamma(3) * max(placed.x, max(placed.y, placed.z)) + (1 + Gamma(3)) * carried;
    }

    float3 NormalToWorld(float3 normal) {
        const float3 x = axes[0];
        const float3 y = axes[1];
        const float3 z = axes[2];
        const float3 n = cross(y, z) * normal.x + cross(z, x) * normal.y + cross(x, y) * normal.z;
        return normalize(dot(x, cross(y, z)) < 0 ? -n : n);
    }
}

struct LinkFilter : IHitFilter {
    static const uint kRayFlags = 0;

    RenderTables* tables;
    Ptr<SlotHit> hits;
    uint first;
    uint row;

    static bool AcceptsAll() {
        return false;
    }

    bool Opaque() {
        return row == kNoRow;
    }

    bool Accept(Ray ray, Hit candidate) {
        return row == kNoRow || InCollection(tables, row, hits[first + candidate.instance].prim);
    }
}

struct HitQuery {
    RenderTables* tables;
    InstanceLayout* layout;
    uint world;
    uint tlas;

    __init(RenderTables* tables, uint world) {
        this.tables = tables;
        layout = tables->instances;
        this.world = world;
        tlas = TlasIndex(world, layout->worldCount);
    }

    ShapeIntersector Intersector() {
        ShapeIntersector intersector;
        intersector.shapes = layout->shapes;
        intersector.points = tables->points;
        intersector.first = layout->firstInstances[tlas];
        return intersector;
    }

    PrimHit Intersect(Ray ray) {
        return Intersect(ray, kGeometryMask);
    }

    PrimHit Intersect(Ray ray, uint mask) {
        PrimHit primHit;
        primHit.hit = Tracer.Closest(layout->handles[tlas], ray, mask, AcceptAll(), Intersector());
        primHit.prim = kNoPrim;
        primHit.instancer = kNoPrim;
        primHit.instance = 0;
        primHit.slot = 0;
        primHit.shape.kind = uint(ShapeKind::Mesh);
        primHit.shape.param = 0;
        primHit.shape.row = 0;
        primHit.shape.padding = 0;
        if (primHit.IsHit()) {
            const uint slot = layout->firstInstances[tlas] + primHit.hit.instance;
            primHit.slot = slot;
            const SlotHit drawn = layout->hits[slot];
            primHit.prim = drawn.prim;
            primHit.instancer = drawn.instancer;
            primHit.instance = drawn.instance;
            primHit.shape = layout->shapes[slot];
        }
        return primHit;
    }

    bool Occluded(Ray ray) {
        return Tracer.Any(layout->handles[tlas], ray, kGeometryMask, AcceptAll(), Intersector());
    }

    LinkFilter Links(uint row) {
        LinkFilter filter;
        filter.tables = tables;
        filter.hits = layout->hits;
        filter.first = layout->firstInstances[tlas];
        filter.row = row;
        return filter;
    }

    float OccluderDistance(Ray ray, uint links, bool any) {
        const Hit hit = Tracer.Trace(layout->handles[tlas], ray, kGeometryMask, Links(links),
                                     Intersector(), any);
        return hit.IsHit() ? hit.t : kInfinity;
    }

    ObjectSpace LoadObjectSpace(PrimHit primHit) {
        const InstanceRecord record = Ptr<InstanceRecord>(layout->records)[primHit.slot];
        const float t[12] = record.transform;
        ObjectSpace space;
        space.origin = float3(t[3], t[7], t[11]);
        space.axes = float3x3(float3(t[0], t[4], t[8]), float3(t[1], t[5], t[9]),
                              float3(t[2], t[6], t[10]));
        return space;
    }
}
