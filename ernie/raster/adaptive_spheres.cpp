// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/adaptive_spheres.h"

#include "ernie/util/profile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace ernie {

AdaptiveSphere AdaptiveSphere::Of(const glm::dmat4& transform,
                                  double radius) {
    AdaptiveSphere sphere;
    sphere.centre = glm::dvec3(transform[3]);
    for (int axis = 0; axis < 3; ++axis) {
        const glm::dvec3 column(transform[axis]);
        const double length = glm::length(column);
        sphere.radii[axis] = radius * length;
        sphere.rotation[axis] =
            length > 0.0 ? column / length : glm::dvec3(0.0);
    }
    // A mirroring scale turns the frame left-handed: flip an axis
    // back, the ellipsoid the same.
    if (glm::determinant(sphere.rotation) < 0.0) {
        sphere.rotation[0] = -sphere.rotation[0];
    }
    return sphere;
}

AdaptiveSpheres::AdaptiveSpheres(RasterRenderer& renderer)
    : renderer_(renderer) {}

AdaptiveSpheres::~AdaptiveSpheres() = default;

void AdaptiveSpheres::SetSpheres(
    std::span<const AdaptiveSphere> spheres, const LodSettings& lod) {
    for (const AdaptiveSphere& sphere : spheres) {
        for (int i = 0; i < 3; ++i) {
            if (!(sphere.radii[i] > 0.0) ||
                !std::isfinite(sphere.radii[i])) {
                throw std::invalid_argument(
                    "An adaptive sphere's radii must be positive and "
                    "finite");
            }
        }
    }
    if (!spheres.empty() && shared_ == nullptr) {
        shared_ = std::make_unique<PlanetDrawable>(renderer_);
    }
    // Flat faces of the fewest quads a planet takes.
    constexpr std::size_t kSide = PlanetDrawable::kLodLeafCells + 1;
    static const std::vector<float> kZeros(kSide * kSide, 0.0f);
    RasterView face;
    face.data = kZeros.data();
    face.width = kSide;
    face.height = kSide;
    const std::array<RasterView, CubeSphereLod::kFaces> faces{
        face, face, face, face, face, face};
    while (bodies_.size() < spheres.size()) {
        bodies_.push_back(std::make_unique<PlanetDrawable>(
            renderer_, shared_.get()));
        radii_.emplace_back(0.0);
    }
    for (std::size_t i = 0; i < spheres.size(); ++i) {
        const AdaptiveSphere& sphere = spheres[i];
        PlanetDrawable& body = *bodies_[i];
        if (!body.HasHeights() || radii_[i] != sphere.radii) {
            body.SetHeights(sphere.radii, faces);
            radii_[i] = sphere.radii;
        }
        TerrainParams params;
        params.lod = lod;
        params.material = sphere.material;
        params.primId = sphere.primId;
        params.linkGroup = sphere.linkGroup;
        params.temperature = sphere.temperature;
        params.planetCentre = sphere.centre;
        params.planetRotation = sphere.rotation;
        params.world = sphere.world;
        params.finestCell =
            lod.enabled ? kFinestCell * std::max({sphere.radii.x,
                                                  sphere.radii.y,
                                                  sphere.radii.z})
                        : 0.0;
        body.SetParams(params);
        body.SetOrigin(origin_);
    }
    count_ = spheres.size();
}

std::uint64_t AdaptiveSpheres::TriangleCount() const {
    std::uint64_t triangles = 0;
    for (std::size_t i = 0; i < count_; ++i) {
        triangles += bodies_[i]->TriangleCount();
    }
    return triangles;
}

std::size_t AdaptiveSpheres::PatchCount() const {
    std::size_t patches = 0;
    for (std::size_t i = 0; i < count_; ++i) {
        patches += bodies_[i]->PatchCount();
    }
    return patches;
}

std::uint64_t AdaptiveSpheres::GpuBytes() const {
    std::uint64_t bytes = 0;
    for (const auto& body : bodies_) {
        bytes += body->GpuBytes();
    }
    return bytes;
}

namespace {

void Grow(std::optional<Bounds3>& bounds,
          const std::optional<Bounds3>& more) {
    if (!more) {
        return;
    }
    bounds = bounds ? Bounds3{glm::min(bounds->first, more->first),
                              glm::max(bounds->second, more->second)}
                    : *more;
}

}  // namespace

std::optional<Bounds3> AdaptiveSpheres::BodyBounds() const {
    std::optional<Bounds3> bounds;
    for (std::size_t i = 0; i < count_; ++i) {
        Grow(bounds, bodies_[i]->BodyBounds());
    }
    return bounds;
}

std::optional<Bounds3> AdaptiveSpheres::Bounds() const {
    std::optional<Bounds3> bounds;
    for (std::size_t i = 0; i < count_; ++i) {
        Grow(bounds, bodies_[i]->Bounds());
    }
    return bounds;
}

std::map<std::int32_t, Bounds3>
AdaptiveSpheres::WorldsBounds() const {
    std::map<std::int32_t, Bounds3> bounds;
    for (std::size_t i = 0; i < count_; ++i) {
        for (const auto& [world, box] : bodies_[i]->WorldsBounds()) {
            const auto [it, added] = bounds.try_emplace(world, box);
            if (!added) {
                it->second =
                    Bounds3{glm::min(it->second.first, box.first),
                            glm::max(it->second.second, box.second)};
            }
        }
    }
    return bounds;
}

void AdaptiveSpheres::SetOrigin(const glm::dvec3& origin) {
    origin_ = origin;
    for (const auto& body : bodies_) {
        body->SetOrigin(origin);
    }
}

void AdaptiveSpheres::Prepare(
    const FrameParams& params,
    std::span<const ShadowCascade> cascades) {
    ERNIE_ZONE("Adaptive spheres prepare");
    for (std::size_t i = 0; i < count_; ++i) {
        bodies_[i]->Prepare(params, cascades);
    }
}

void AdaptiveSpheres::RecordShadow(
    VkCommandBuffer commands, const RasterFrame& frame,
    std::size_t cascade, const glm::mat4& lightViewProjection,
    const ShadowCasters& casters) const {
    for (std::size_t i = 0; i < count_; ++i) {
        bodies_[i]->RecordShadow(commands, frame, cascade,
                                 lightViewProjection, casters);
    }
}

void AdaptiveSpheres::RecordColor(VkCommandBuffer commands,
                                  const RasterFrame& frame,
                                  std::size_t view,
                                  DrawPhase phase) const {
    for (std::size_t i = 0; i < count_; ++i) {
        bodies_[i]->RecordColor(commands, frame, view, phase);
    }
}

}  // namespace ernie
