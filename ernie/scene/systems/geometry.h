// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <compare>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ernie/algorithms/radix_sort.h"
#include "ernie/algorithms/scan.h"
#include "ernie/bvh/builder.h"
#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/catalog/catalog.h"
#include "ernie/scene/systems/system.h"

namespace ernie {

struct GeometryStats {
    uint32_t meshes = 0;
    uint32_t shapes = 0;
    uint32_t topologies = 0;
    uint32_t built = 0;
    uint32_t refit = 0;
    uint32_t triangulated = 0;
};

struct MeshShape {
    Ref<bvh::Blas> blas;
    uint32_t points = 0;
    uint32_t triangles = 0;
    uint32_t users = 0;
};

class Geometry : public System {
  public:
    Geometry(Catalog& catalog, bvh::Builder& builder) : m_catalog(catalog), m_builder(builder) {}

    std::string_view Name() const override {
        return "geometry";
    }

    std::span<const ColumnRef> Inputs() const override;
    std::span<const ColumnRef> Outputs() const override;
    void Prepare() override;
    device::Operation Run(const ChangeSet& changes, Stream stream) override;

    uint32_t Runs() const {
        return m_runs;
    }

    const GeometryStats& Stats() const {
        return m_stats;
    }

    std::optional<MeshShape> Find(uint32_t prim) const;

  private:
    struct TopologyKey {
        uint64_t indices = 0;
        uint32_t indexCount = 0;
        uint64_t counts = 0;
        uint32_t faceCount = 0;

        auto operator<=>(const TopologyKey&) const = default;
    };

    struct ShapeKey {
        uint64_t points = 0;
        uint32_t pointCount = 0;
        TopologyKey topology;

        auto operator<=>(const ShapeKey&) const = default;
    };

    struct Topology {
        ArrayRef triangles;
        ArrayRef corners;
        ArrayRef faces;
        ArrayRef starts;
        ArrayRef adjacency;
        uint32_t triangleCount = 0;
        uint32_t covered = 0;
        uint32_t users = 0;
        bool ready = false;
    };

    struct Shape {
        Ref<bvh::Blas> blas;
        ArrayRef normals;
        ArrayRef bounds;
        uint32_t pointCount = 0;
        uint32_t users = 0;
        uint32_t updated = 0;
        bool build = false;
        bool refit = false;
    };

    struct Mesh {
        uint32_t prim = kNoPrim;
        uint32_t generation = 0;
        uint32_t pointsRow = kNoRow;
        uint32_t meshRow = kNoRow;
        std::optional<ShapeKey> key;
        Shape* shape = nullptr;
    };

    void Place(uint32_t prim);
    void Settle();
    void Prune();
    void Triangulate();
    void Release(Mesh& mesh);
    ShapeKey Key(const Mesh& mesh) const;
    void Link(const Mesh& mesh);
    device::Operation Update();

    Catalog& m_catalog;
    bvh::Builder& m_builder;
    device::Program& Kernels();

    Ref<device::Program> m_program;
    std::unique_ptr<algorithms::Scan> m_scan;
    std::unique_ptr<algorithms::RadixSort> m_sort;
    uint32_t m_scanCapacity = 0;
    uint32_t m_sortCapacity = 0;
    std::vector<std::pair<uint32_t, ShapeKey>> m_placing;
    std::unordered_map<uint32_t, Mesh> m_meshes;
    std::unordered_map<uint32_t, uint32_t> m_byPointsRow;
    std::unordered_map<uint32_t, uint32_t> m_byMeshRow;
    std::map<TopologyKey, Topology> m_topologies;
    std::map<ShapeKey, Shape> m_shapes;
    GeometryStats m_stats;
    uint32_t m_runs = 0;
    bool m_stale = false;
};

}
