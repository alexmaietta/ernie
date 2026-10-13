// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/raster/drawable.h"
#include "ernie/raster/planet_drawable.h"
#include "ernie/raster/renderer.h"
#include "ernie/raster/terrain_drawable.h"

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ernie {

/// A Sphere prim drawn adaptively (tessellation `adaptive`): its
/// world pose split into centre, axes and the semi-axes its scale
/// gives the radius.
struct AdaptiveSphere {
    glm::dvec3 centre{0.0};
    /// Its axes in the world's, columns x, y and z: a rotation.
    glm::dmat3 rotation{1.0};
    /// Semi-axes along its x, y and z: all its radius, unless scaled
    /// unevenly into an ellipsoid.
    glm::dvec3 radii{1.0};
    std::uint32_t material = RasterRenderer::kDefaultMaterial;
    std::uint32_t primId = 1;
    std::uint32_t linkGroup = 0;
    /// Its own temperature in kelvin; 0 for its material's.
    float temperature = 0.0f;
    /// Its world in a batch (`tokens::kWorld`): only that world's
    /// views draw it and refine it; -1 for every world's.
    std::int32_t world = -1;

    /// `radius` posed by `transform`, which must not shear: the
    /// centre its translation, the axes its columns' directions, the
    /// semi-axes `radius` times their lengths.
    static AdaptiveSphere Of(const glm::dmat4& transform,
                             double radius);
};

/// Spheres and ellipsoids drawn exactly at any scale, from a pebble
/// to a star: each a planet's cube-sphere LOD (`PlanetDrawable`,
/// `CubeSphereLod`) without heights, patches finer towards each eye,
/// so its silhouette and depth stay true however near the eye comes,
/// and anchored in double so bodies far from the origin draw in
/// place. They shade as planets: normals the ellipsoid's, their
/// materials' albedo textures in latitude and longitude
/// (equirectangular, as a uv Sphere's coordinates), detail textures
/// triplanar. In a batch, a sphere of one world refines only towards
/// that world's views.
class AdaptiveSpheres : public RasterDrawable {
  public:
    /// The finest LOD quad, relative to the largest radius: about a
    /// metre and a half on the Earth.
    static constexpr double kFinestCell = 0x1p-22;

    /// Spheres drawn by `renderer`, which must outlive them.
    explicit AdaptiveSpheres(RasterRenderer& renderer);
    ~AdaptiveSpheres() override;
    AdaptiveSpheres(const AdaptiveSpheres&) = delete;
    AdaptiveSpheres& operator=(const AdaptiveSpheres&) = delete;

    /// Draws `spheres` from the next frame, at `lod`'s detail (LOD
    /// off: a fixed cube sphere of `PlanetDrawable::kLodLeafCells`
    /// quads a face edge). A sphere keeps its LOD while its semi-axes
    /// hold.
    /// @throws std::invalid_argument for semi-axes that are not
    /// positive and finite.
    void SetSpheres(std::span<const AdaptiveSphere> spheres,
                    const LodSettings& lod);

    std::size_t Count() const {
        return count_;
    }

    /// Triangles drawn in every view by the last frame.
    std::uint64_t TriangleCount() const;

    /// LOD patches drawn in view 0 by the last frame.
    std::size_t PatchCount() const;

    /// GPU memory held, in bytes.
    std::uint64_t GpuBytes() const;

    /// The spheres' whole bounding box relative to the origin, for
    /// framing them; none without spheres.
    std::optional<Bounds3> BodyBounds() const;

    const char* ProfileName() const override {
        return "Adaptive spheres";
    }

    void SetOrigin(const glm::dvec3& origin) override;

    /// What of the spheres can cast the shadows a view sees
    /// (`PlanetDrawable::Bounds`).
    std::optional<Bounds3> Bounds() const override;

    /// Each world's spheres' `Bounds`.
    std::map<std::int32_t, Bounds3> WorldsBounds() const override;

    void Prepare(const FrameParams& params,
                 std::span<const ShadowCascade> cascades) override;
    void RecordShadow(VkCommandBuffer commands,
                      const RasterFrame& frame, std::size_t cascade,
                      const glm::mat4& lightViewProjection,
                      const ShadowCasters& casters) const override;
    void RecordColor(VkCommandBuffer commands,
                     const RasterFrame& frame, std::size_t view,
                     DrawPhase phase = DrawPhase::All) const override;

  private:
    RasterRenderer& renderer_;
    /// Holds the pipelines the bodies share; draws nothing.
    std::unique_ptr<PlanetDrawable> shared_;
    /// One per sphere, the first `count_` drawn; the semi-axes each
    /// was given heights for.
    std::vector<std::unique_ptr<PlanetDrawable>> bodies_;
    std::vector<glm::dvec3> radii_;
    std::size_t count_ = 0;
    glm::dvec3 origin_{0.0};
};

}  // namespace ernie
