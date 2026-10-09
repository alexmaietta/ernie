#include "ernie/scene/wide_view.h"

#include <algorithm>
#include <cmath>

namespace ernie {

glm::dvec3 LensRay(const LensModel& lens, glm::dvec2 distorted) {
    if (lens.projection == LensProjection::Fisheye) {
        const double radius = glm::length(distorted);
        if (radius < 1e-12) {
            return {0.0, 0.0, -1.0};
        }
        const double theta = lens.FieldAngle(distorted);
        const glm::dvec2 across = distorted / radius * std::sin(theta);
        return {across.x, across.y, -std::cos(theta)};
    }
    const glm::dvec2 pinhole = lens.HasDistortion() ? lens.Undistort(distorted) : distorted;
    return glm::normalize(glm::dvec3(pinhole, -1.0));
}

glm::dmat3 CubeFaceAxes(CubeFace face) {
    // Columns x, y, z: each face looks down its -z, as the camera does on the front.
    switch (face) {
        case CubeFace::Front:
            return glm::dmat3(1.0);
        case CubeFace::Right:  // looks along +x
            return glm::dmat3(glm::dvec3(0, 0, 1), glm::dvec3(0, 1, 0), glm::dvec3(-1, 0, 0));
        case CubeFace::Left:  // looks along -x
            return glm::dmat3(glm::dvec3(0, 0, -1), glm::dvec3(0, 1, 0), glm::dvec3(1, 0, 0));
        case CubeFace::Up:  // looks along +y
            return glm::dmat3(glm::dvec3(1, 0, 0), glm::dvec3(0, 0, 1), glm::dvec3(0, -1, 0));
        case CubeFace::Down:  // looks along -y
            return glm::dmat3(glm::dvec3(1, 0, 0), glm::dvec3(0, 0, -1), glm::dvec3(0, 1, 0));
    }
    return glm::dmat3(1.0);
}

CubeFace CubeFaceOf(const glm::dvec3& ray) {
    const double forward[kCubeFaces] = {-ray.z, ray.x, -ray.x, ray.y, -ray.y};
    int best = 0;
    for (int face = 1; face < kCubeFaces; ++face) {
        if (forward[face] > forward[best]) {
            best = face;
        }
    }
    return static_cast<CubeFace>(best);
}

glm::dvec2 CubeFacePlane(CubeFace face, const glm::dvec3& ray) {
    const glm::dvec3 local = glm::transpose(CubeFaceAxes(face)) * ray;
    return glm::dvec2(local) / -local.z;
}

glm::dvec3 CubeFaceRay(CubeFace face, glm::dvec2 plane) {
    return glm::normalize(CubeFaceAxes(face) * glm::dvec3(plane, -1.0));
}

double CubeFaceExtent(double facePixels, double marginPixels) {
    // The face's focal length is half its pixels over the extent: solve for a margin of
    // `marginPixels` beyond tangent 1, f = facePixels / (2 extent), extent = 1 + margin / f.
    return facePixels / (facePixels - 2.0 * marginPixels);
}

double MaxFieldAngle(const Camera& camera) {
    // The image's border, where the field is widest.
    const double f = camera.FocalLengthPixels();
    const glm::dvec2 centre = camera.PrincipalPoint();
    const double w = camera.width;
    const double h = camera.height;
    double largest = 0.0;
    constexpr int kSteps = 64;
    for (int i = 0; i <= kSteps; ++i) {
        const double t = double(i) / kSteps;
        for (const glm::dvec2 pixel : {glm::dvec2(t * w, 0.0), glm::dvec2(t * w, h),
                                       glm::dvec2(0.0, t * h), glm::dvec2(w, t * h)}) {
            const glm::dvec2 distorted((pixel.x - centre.x) / f, (centre.y - pixel.y) / f);
            const glm::dvec3 ray = LensRay(camera.lens, distorted);
            largest = std::max(largest, std::acos(std::clamp(-ray.z, -1.0, 1.0)));
        }
    }
    return largest;
}

glm::dvec3 LensRayTable::RayAt(glm::dvec2 pixel) const {
    const glm::dvec2 at = glm::clamp(pixel / double(step), glm::dvec2(0.0),
                                     glm::dvec2(columns - 1, rows - 1));
    const std::uint32_t x0 = std::min(static_cast<std::uint32_t>(at.x), columns - 2);
    const std::uint32_t y0 = std::min(static_cast<std::uint32_t>(at.y), rows - 2);
    const glm::dvec2 t = at - glm::dvec2(x0, y0);
    const auto ray = [&](std::uint32_t x, std::uint32_t y) {
        return glm::dvec3(rays[std::size_t(y) * columns + x]);
    };
    const glm::dvec3 top = glm::mix(ray(x0, y0), ray(x0 + 1, y0), t.x);
    const glm::dvec3 bottom = glm::mix(ray(x0, y0 + 1), ray(x0 + 1, y0 + 1), t.x);
    return glm::normalize(glm::mix(top, bottom, t.y));
}

bool NeedsCubeFaces(const Camera& camera) {
    return camera.lens.projection == LensProjection::Fisheye && !camera.orthographic &&
           !camera.pushbroom && MaxFieldAngle(camera) > kMaxSingleViewAngle;
}

}  // namespace ernie
