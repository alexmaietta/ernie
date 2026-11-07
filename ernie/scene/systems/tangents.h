// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <compare>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/commands.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/catalog/catalog.h"
#include "ernie/scene/kernels/schema/schema.h"
#include "ernie/scene/kernels/tables/meshes.h"
#include "ernie/scene/systems/geometry.h"
#include "ernie/scene/systems/record_table.h"
#include "ernie/scene/systems/system.h"

namespace ernie {

struct TangentMesh {
    std::vector<Float3> points;
    std::vector<int32_t> indices;
    std::vector<int32_t> counts;
    std::vector<Float3> normals;
    std::vector<Float2> st;
};

std::vector<Float4> MikkTangents(const TangentMesh& mesh);

std::vector<Float3> SmoothCornerNormals(std::span<const Float3> points,
                                        std::span<const int32_t> indices,
                                        std::span<const int32_t> counts);

template <class T>
std::optional<std::vector<T>>
CornerValues(std::span<const T> values, std::span<const int32_t> valueIndices,
             Interpolation interpolation, std::span<const int32_t> indices,
             std::span<const int32_t> counts) {
    std::vector<T> corners(indices.size());
    uint32_t corner = 0;
    for (uint32_t face = 0; face < counts.size(); ++face) {
        for (int32_t k = 0; k < counts[face] && corner < indices.size(); ++k, ++corner) {
            uint32_t at = corner;
            switch (interpolation) {
            case Interpolation::Constant:
                at = 0;
                break;
            case Interpolation::Uniform:
                at = face;
                break;
            case Interpolation::FaceVarying:
                at = corner;
                break;
            default:
                at = uint32_t(indices[corner]);
                break;
            }
            if (!valueIndices.empty()) {
                if (at >= valueIndices.size()) {
                    return std::nullopt;
                }
                at = uint32_t(valueIndices[at]);
            }
            if (at >= values.size()) {
                return std::nullopt;
            }
            corners[corner] = values[at];
        }
    }
    return corners;
}

struct TangentStats {
    uint32_t meshes = 0;
    uint32_t sets = 0;
    uint32_t computed = 0;
    uint32_t deformed = 0;
};

class Tangents : public System {
  public:
    Tangents(Catalog& catalog, const Geometry& geometry)
        : m_catalog(catalog), m_geometry(geometry) {}

    std::string_view Name() const override {
        return "tangents";
    }

    std::span<const ColumnRef> Inputs() const override;
    std::span<const ColumnRef> Outputs() const override;

    std::span<const TableId> TableInputs() const override {
        static constexpr TableId kTables[] = {TableId::Meshes, TableId::MeshData};
        return kTables;
    }

    std::span<const TableId> TableOutputs() const override {
        static constexpr TableId kTables[] = {TableId::Tangents};
        return kTables;
    }

    void Describe(RenderTables& tables) const override;
    void Hold(std::vector<Ref<const RefCounted>>& held) const override;

    const TangentRecord& Record(uint32_t meshRow) const {
        return m_records.At(meshRow);
    }

    void Prepare() override;
    device::Operation Run(const ChangeSet& changes, Stream stream) override;

    uint32_t Runs() const {
        return m_runs;
    }

    const TangentStats& Stats() const {
        return m_stats;
    }

  private:
    struct Key {
        uint64_t points = 0;
        uint64_t indices = 0;
        uint64_t counts = 0;
        uint64_t primvars = 0;
        uint64_t normals = 0;
        uint32_t interpolation = 0;
        uint32_t smooth = 0;

        auto operator<=>(const Key&) const = default;
    };

    struct Deform {
        Ref<device::Buffer> st;
        Ref<device::Buffer> normals;
        Ref<device::Buffer> triangles;
        Ref<device::Buffer> starts;
        Ref<device::Buffer> corners;
        Ref<device::Buffer> groups;
        Ref<device::Buffer> results;
        uint32_t groupCount = 0;
        uint32_t cornerCount = 0;
    };

    struct Set {
        ArrayRef tangents;
        uint32_t words = 0;
        uint32_t users = 0;
        std::optional<Deform> deform;
    };

    struct Mesh {
        uint32_t generation = 0;
        uint32_t pointsRow = kNoRow;
        uint32_t meshRow = kNoRow;
        uint32_t gprimRow = kNoRow;
        uint32_t interpolation = 0;
        uint32_t smooth = 0;
        std::optional<Key> key;
        Set* set = nullptr;
    };

    struct Cells {
        std::vector<uint32_t> interpolation;
        std::vector<uint32_t> scheme;
    };

    Key KeyFor(const Mesh& mesh) const;
    void Compute(const Mesh& mesh, const Key& key, Set& set);
    void Recompute(const Mesh& mesh, const Key& key, Set& set, device::CommandEncoder& encoder);
    void Forget(uint32_t prim);
    void Place(uint32_t prim, const Cells& cells);
    void CacheCells(Mesh& mesh, const Cells& cells) const;
    void Release(Mesh& mesh);
    void Prune();
    void Link(uint32_t prim, const Mesh& mesh);
    device::Operation Process(const ChangeSet& changes);

    Catalog& m_catalog;
    const Geometry& m_geometry;
    RecordTable<TangentRecord> m_records{"tangent records"};
    std::vector<uint32_t> m_relinked;
    std::unordered_map<uint32_t, Mesh> m_meshes;
    std::unordered_map<uint32_t, uint32_t> m_byPointsRow;
    std::unordered_map<uint32_t, uint32_t> m_byMeshRow;
    std::unordered_map<uint32_t, uint32_t> m_byGprimRow;
    std::map<Key, Set> m_sets;
    bool m_stale = false;
    device::Program& Kernels();

    Ref<device::Program> m_program;
    uint64_t m_seen = 0;
    uint32_t m_runs = 0;
    Cells m_cells;
    TangentStats m_stats;
};

}
