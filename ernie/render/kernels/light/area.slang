// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
__exported import ernie.render.kernels.light.ilight;
import ernie.texture.kernels.texture;
import ernie.render.kernels.random;
import ernie.render.kernels.basis;

using namespace ernie;

static const float kTinySolidAngle = 1e-5f;

struct AreaLight<Spectrum : ISpectrum> : ILight<Spectrum> {
    no_diff LightRecord record;
    no_diff uint64_t ies;
    no_diff uint64_t texture;

    static AreaLight<Spectrum> Load(LightContext context, LightRecord record) {
        AreaLight<Spectrum> light;
        light.record = record;
        light.ies = record.ies != 0 ? context.tables->textureBindings[record.ies] : 0;
        light.texture = record.textures != 0 && LightType(record.type) == LightType::Rect
                            ? context.tables->textureBindings[record.textures]
                            : 0;
        return light;
    }

    float Profile(float3 emitted) {
        if (ies == 0) {
            return 1;
        }
        const float2 st = IesSt(record, emitted);
        if (st.x < 0) {
            return 0;
        }
        return max(TextureFromHandle(ies).SampleLinear(st, kBaseLevel).r, 0.0f) * record.iesScale;
    }

    float3 Center() {
        return record.position.Vector();
    }

    float3 Normal() {
        return -record.axisZ.Vector();
    }

    float3 Texel(float3 point) {
        if (texture == 0) {
            return float3(1);
        }
        const float3 local = point - Center();
        const float2 st = float2(0.5f + dot(local, record.axisX.Vector()) / record.width,
                                 0.5f + dot(local, record.axisY.Vector()) / record.height);
        return max(TextureFromHandle(texture).SampleLinear(st, kBaseLevel).rgb, float3(0));
    }

    Spectrum Emitted(float3 emitted, float3 point) {
        const float3 weight = ShapingWeight(record, dot(emitted, ForwardAxis(record)));
        return Spectrum.FromRgb(record.radiance.Vector() * weight * Profile(emitted) *
                                Texel(point));
    }

    LightSample<Spectrum> Empty() {
        LightSample<Spectrum> sample;
        sample.direction = float3(0, 0, 1);
        sample.distance = 0;
        sample.radiance = Spectrum.Constant(0);
        sample.pdf = 0;
        sample.delta = false;
        sample.diffuse = record.diffuse;
        sample.specular = record.specular;
        return sample;
    }

    LightSample<Spectrum> Toward(float3 position, float3 point, float3 normal, float area,
                                 bool sided, out float3 emitted, out float3 at) {
        emitted = float3(0, 0, 1);
        at = point;
        LightSample<Spectrum> sample = Empty();
        const float3 offset = point - position;
        const float distance2 = dot(offset, offset);
        if (distance2 <= 0 || area <= 0) {
            return sample;
        }
        const float distance = sqrt(distance2);
        const float3 direction = offset / distance;
        const float cosine = dot(-direction, normal);
        if (cosine <= 0 && sided) {
            return sample;
        }
        sample.direction = direction;
        sample.distance = distance;
        sample.radiance = Spectrum.Constant(1);
        emitted = -direction;
        sample.pdf = distance2 / (area * abs(cosine));
        return sample;
    }

    LightSample<Spectrum> SampleSphere(float3 position, float2 u, out float3 emitted,
                                       out float3 at) {
        emitted = float3(0, 0, 1);
        at = Center();
        const float3 offset = Center() - position;
        const float distance2 = dot(offset, offset);
        const float radius = record.radius;
        LightSample<Spectrum> sample = Empty();
        if ((record.flags & kLightTreatAsPoint) != 0 || radius <= 0) {
            const float distance = sqrt(distance2);
            sample.direction = offset / distance;
            sample.distance = distance;
            sample.radiance = Spectrum.Constant(kPi * radius * radius / max(distance2, 1e-12f));
            emitted = -sample.direction;
            sample.pdf = 1;
            sample.delta = true;
            return sample;
        }
        if (distance2 <= radius * radius) {
            return sample;
        }
        const float distance = sqrt(distance2);
        const float sine2 = radius * radius / distance2;
        const float cosMax = sqrt(max(1 - sine2, 0.0f));
        const float spread = sine2 / (1 + cosMax);
        const float cosine = 1 - u.x * spread;
        const float sine = sqrt(max(1 - cosine * cosine, 0.0f));
        const float phi = 2 * kPi * u.y;
        const Basis frame = Basis(offset / distance);
        const float3 direction = frame.ToWorld(float3(sine * cos(phi), sine * sin(phi), cosine));
        const float along = distance * cosine;
        const float hit = along - sqrt(max(radius * radius - distance2 * sine * sine, 0.0f));
        const float3 point = position + direction * hit;
        sample.direction = direction;
        sample.distance = hit;
        sample.radiance = Spectrum.Constant(1);
        emitted = normalize(point - Center());
        at = point;
        sample.pdf = 1 / (2 * kPi * spread);
        return sample;
    }

    LightSample<Spectrum> SampleRect(float3 position, float2 u, out float3 emitted,
                                     out float3 at) {
        emitted = float3(0, 0, 1);
        at = Center();
        const float3 ex = record.axisX.Vector();
        const float3 ey = record.axisY.Vector();
        const float width = record.width;
        const float height = record.height;
        const float3 corner = Center() - 0.5f * width * ex - 0.5f * height * ey;
        float3 ez = cross(ex, ey);
        const float3 d = corner - position;
        const float x0 = dot(d, ex);
        const float y0 = dot(d, ey);
        float z0 = dot(d, ez);
        if (dot(position - Center(), Normal()) <= 0) {
            return Empty();
        }
        if (z0 > 0) {
            z0 = -z0;
            ez = -ez;
        }
        const float x1 = x0 + width;
        const float y1 = y0 + height;
        const float3 v00 = float3(x0, y0, z0);
        const float3 v01 = float3(x0, y1, z0);
        const float3 v10 = float3(x1, y0, z0);
        const float3 v11 = float3(x1, y1, z0);
        const float3 n0 = normalize(cross(v00, v10));
        const float3 n1 = normalize(cross(v10, v11));
        const float3 n2 = normalize(cross(v11, v01));
        const float3 n3 = normalize(cross(v01, v00));
        const float g0 = acos(clamp(-dot(n0, n1), -1.0f, 1.0f));
        const float g1 = acos(clamp(-dot(n1, n2), -1.0f, 1.0f));
        const float g2 = acos(clamp(-dot(n2, n3), -1.0f, 1.0f));
        const float g3 = acos(clamp(-dot(n3, n0), -1.0f, 1.0f));
        const float k = 2 * kPi - g2 - g3;
        const float solid = g0 + g1 - k;
        if (!(solid > kTinySolidAngle)) {
            const float3 point = corner + u.x * width * ex + u.y * height * ey;
            return Toward(position, point, Normal(), width * height, true, emitted, at);
        }
        const float b0 = n0.z;
        const float b1 = n2.z;
        const float au = u.x * solid + k;
        const float fu = (cos(au) * b0 - b1) / sin(au);
        const float cu = clamp((fu > 0 ? 1.0f : -1.0f) / sqrt(fu * fu + b0 * b0), -1.0f, 1.0f);
        const float xu = clamp(-(cu * z0) / max(sqrt(1 - cu * cu), 1e-12f), x0, x1);
        const float dist = sqrt(xu * xu + z0 * z0);
        const float h0 = y0 / sqrt(dist * dist + y0 * y0);
        const float h1 = y1 / sqrt(dist * dist + y1 * y1);
        const float hv = h0 + u.y * (h1 - h0);
        const float yv = hv * hv < 1 - 1e-6f ? hv * dist / sqrt(1 - hv * hv) : y1;
        const float3 point = position + xu * ex + yv * ey + z0 * ez;
        LightSample<Spectrum> sample = Empty();
        const float3 offset = point - position;
        sample.distance = length(offset);
        sample.direction = offset / sample.distance;
        sample.radiance = Spectrum.Constant(1);
        emitted = -sample.direction;
        at = point;
        sample.pdf = 1 / solid;
        return sample;
    }

    LightSample<Spectrum> SampleDisk(float3 position, float2 u, out float3 emitted,
                                     out float3 at) {
        const float2 offset = ConcentricDisk(u) * record.radius;
        const float3 point =
            Center() + offset.x * record.axisX.Vector() + offset.y * record.axisY.Vector();
        return Toward(position, point, Normal(), kPi * record.radius * record.radius, true,
                      emitted, at);
    }

    LightSample<Spectrum> SampleCylinder(float3 position, float2 u, out float3 emitted,
                                         out float3 at) {
        const float3 axis = record.axisX.Vector();
        const float3 side = record.axisY.Vector();
        const float3 up = record.axisZ.Vector();
        const float3 local = position - Center();
        const float along = (u.x - 0.5f) * record.length;
        if ((record.flags & kLightTreatAsLine) != 0 || record.radius <= 0) {
            const float3 point = Center() + along * axis;
            LightSample<Spectrum> sample = Empty();
            const float3 offset = point - position;
            const float distance2 = dot(offset, offset);
            const float distance = sqrt(distance2);
            sample.direction = offset / distance;
            sample.distance = distance;
            const float across = length(cross(sample.direction, axis));
            sample.radiance = Spectrum.Constant(2 * record.radius * record.length * across /
                                                max(distance2, 1e-12f));
            emitted = -sample.direction;
            at = point;
            sample.pdf = 1;
            sample.delta = true;
            return sample;
        }
        const float facing = atan2(dot(local, up), dot(local, side));
        const float phi = facing + (u.y - 0.5f) * kPi;
        const float3 normal = cos(phi) * side + sin(phi) * up;
        const float3 point = Center() + along * axis + record.radius * normal;
        return Toward(position, point, normal, kPi * record.radius * record.length, true, emitted,
                      at);
    }

    float SphereSpread(float3 position) {
        const float3 offset = Center() - position;
        const float distance2 = dot(offset, offset);
        const float radius2 = record.radius * record.radius;
        if (distance2 <= radius2) {
            return 0;
        }
        const float sine2 = radius2 / distance2;
        return sine2 / (1 + sqrt(max(1 - sine2, 0.0f)));
    }

    float RectSolidAngle(float3 position) {
        const float3 ex = record.axisX.Vector();
        const float3 ey = record.axisY.Vector();
        const float3 corner =
            Center() - 0.5f * record.width * ex - 0.5f * record.height * ey;
        const float3 d = corner - position;
        const float x0 = dot(d, ex);
        const float y0 = dot(d, ey);
        const float z0 = -abs(dot(d, cross(ex, ey)));
        const float x1 = x0 + record.width;
        const float y1 = y0 + record.height;
        const float3 n0 = normalize(cross(float3(x0, y0, z0), float3(x1, y0, z0)));
        const float3 n1 = normalize(cross(float3(x1, y0, z0), float3(x1, y1, z0)));
        const float3 n2 = normalize(cross(float3(x1, y1, z0), float3(x0, y1, z0)));
        const float3 n3 = normalize(cross(float3(x0, y1, z0), float3(x0, y0, z0)));
        const float g0 = acos(clamp(-dot(n0, n1), -1.0f, 1.0f));
        const float g1 = acos(clamp(-dot(n1, n2), -1.0f, 1.0f));
        const float g2 = acos(clamp(-dot(n2, n3), -1.0f, 1.0f));
        const float g3 = acos(clamp(-dot(n3, n0), -1.0f, 1.0f));
        return g0 + g1 + g2 + g3 - 2 * kPi;
    }

    float3 SurfaceNormal(float3 point) {
        if (LightType(record.type) == LightType::Cylinder) {
            const float3 axis = record.axisX.Vector();
            const float3 radial = point - Center();
            return normalize(radial - dot(radial, axis) * axis);
        }
        if (LightType(record.type) == LightType::Sphere) {
            return normalize(point - Center());
        }
        return Normal();
    }

    Spectrum EmittedToward(float3 position, float3 point) {
        const float3 direction = normalize(position - point);
        if (dot(direction, SurfaceNormal(point)) <= 0) {
            return Spectrum.Constant(0);
        }
        return Emitted(direction, point);
    }

    float PointPdf(float3 position, float3 point) {
        const float3 offset = point - position;
        const float distance2 = dot(offset, offset);
        const float cosine = dot(-offset * rsqrt(max(distance2, 1e-30f)), SurfaceNormal(point));
        if (cosine <= 0) {
            return 0;
        }
        switch (LightType(record.type)) {
        case LightType::Sphere: {
            const float spread = SphereSpread(position);
            return spread > 0 ? 1 / (2 * kPi * spread) : 0;
        }
        case LightType::Rect: {
            const float solid = RectSolidAngle(position);
            return solid > kTinySolidAngle ? 1 / solid
                                           : distance2 / (record.width * record.height * cosine);
        }
        case LightType::Disk:
            return distance2 / (kPi * record.radius * record.radius * cosine);
        case LightType::Cylinder:
            return distance2 / (kPi * record.radius * record.length * cosine);
        default:
            return 0;
        }
    }

    LightSample<Spectrum> SampleShape(float3 position, float2 u) {
        float3 emitted;
        float3 at;
        LightSample<Spectrum> sample;
        switch (LightType(record.type)) {
        case LightType::Sphere:
            sample = SampleSphere(position, u, emitted, at);
            break;
        case LightType::Rect:
            sample = SampleRect(position, u, emitted, at);
            break;
        case LightType::Disk:
            sample = SampleDisk(position, u, emitted, at);
            break;
        case LightType::Cylinder:
            sample = SampleCylinder(position, u, emitted, at);
            break;
        default:
            return Empty();
        }
        if (sample.pdf > 0) {
            sample.radiance = sample.radiance * Emitted(emitted, at);
        }
        return sample;
    }

    [Differentiable]
    LightSample<Spectrum> Sample(float3 position, no_diff float2 u) {
        return no_diff SampleShape(detach(position), u);
    }

    [Differentiable]
    Spectrum Radiance(float3 position, float3 incoming) {
        return no_diff Emitted(-detach(incoming), Center());
    }

    float Pdf(float3 position, float3 incoming) {
        return 0;
    }
}
