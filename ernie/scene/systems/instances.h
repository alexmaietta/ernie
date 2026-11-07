// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <vector>

#include "ernie/bvh/builder.h"
#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/catalog/catalog.h"
#include "ernie/scene/systems/regions.h"
#include "ernie/scene/systems/system.h"

namespace ernie {

struct InstanceStats {
    uint32_t tlases = 0;
    uint32_t instances = 0;
    uint32_t written = 0;
    uint32_t built = 0;
    uint32_t refit = 0;
    bool relaid = false;
};

class Instances : public System {
  public:
    Instances(Catalog& catalog, bvh::Builder& builder, uint32_t maxRefits = 16);

    std::string_view Name() const override {
        return "instances";
    }

    std::span<const ColumnRef> Inputs() const override;

    std::span<const TableId> TableInputs() const override {
        static constexpr TableId kTables[] = {TableId::Transforms, TableId::Imageables,
                                              TableId::Meshes, TableId::MeshPoints};
        return kTables;
    }

    void Prepare() override;
    void Run(const ChangeSet& changes, TableLog& log) override;

    const InstanceStats& Stats() const {
        return m_stats;
    }

    const Ref<bvh::Tlas>& Tlas() const {
        return m_tlas;
    }

    const Ref<device::Buffer>& Layout() const {
        return m_layout;
    }

    const Ref<device::Buffer>& Origins() const {
        return m_renderOrigins;
    }

    void Describe(RenderTables& tables) const override;
    void Hold(std::vector<Ref<const RefCounted>>& held) const override;

    const Ref<device::Buffer>& Records() const {
        return m_records;
    }

    const Ref<device::Buffer>& Hits() const {
        return m_hits;
    }

    const std::set<uint32_t>& Members() const {
        return m_members;
    }

    std::span<const std::pair<uint32_t, uint32_t>> LightCopies(uint32_t tlas) const {
        return tlas < m_lightCopies.size() ? std::span(m_lightCopies[tlas])
                                           : std::span<const std::pair<uint32_t, uint32_t>>();
    }

    uint32_t Version() const {
        return m_version;
    }

    std::span<const uint32_t> FirstInstances() const {
        return m_regions.Starts();
    }

    std::span<const uint32_t> InstanceCounts() const {
        return m_regions.Counts();
    }

    std::span<const uint32_t> InstancePrims() const {
        return m_regions.Items();
    }

    uint32_t Relayouts() const {
        return m_relayouts;
    }

  private:
    struct Instancer {
        uint32_t prim = kNoPrim;
        uint32_t world = kGlobalWorld;
        uint32_t count = 0;
        bool native = false;
        std::vector<uint32_t> roots;

        bool operator==(const Instancer&) const = default;
    };

    std::vector<Instancer> GatherInstancers() const;
    bool Member(uint32_t prim) const;
    void Relayout();
    void BuildInstancing();
    void WriteLayoutHeader();
    bool Restack();
    std::optional<std::set<uint32_t>> Apply(const ChangeSet& changes);
    std::vector<uint32_t> RegionItems(uint32_t tlas) const;
    bool ShowsGlobals(uint32_t tlas) const;
    void Compose();

    Catalog& m_catalog;
    bvh::Builder& m_builder;
    uint32_t m_maxRefits;
    Ref<device::Buffer> m_unitBoxes;
    Ref<bvh::Blas> m_unitBox;
    device::Program& Kernels();

    Ref<device::Program> m_program;
    Ref<bvh::Tlas> m_tlas;
    Ref<device::Buffer> m_records;
    Ref<device::Buffer> m_shapes;
    Ref<device::Buffer> m_layout;
    Ref<device::Buffer> m_hits;
    Ref<device::Buffer> m_instancing;
    std::vector<Instancer> m_instancers;
    std::set<uint32_t> m_instancerPrims;
    std::set<uint32_t> m_roots;
    std::set<uint32_t> m_members;
    std::vector<std::vector<uint32_t>> m_blockItems;
    std::map<uint32_t, std::vector<uint32_t>> m_drawables;
    std::map<uint32_t, std::vector<uint32_t>> m_nested;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> m_lightCopies;
    uint32_t m_version = 0;
    std::map<uint32_t, std::set<uint32_t>> m_watch;
    uint64_t m_firstOffset = 0;
    uint64_t m_originsOffset = 0;
    uint64_t m_blocksOffset = 0;
    uint64_t m_entriesOffset = 0;
    uint64_t m_piecesOffset = 0;
    uint64_t m_rangesOffset = 0;
    uint64_t m_hiddenOffset = 0;
    uint64_t m_offsetBasesOffset = 0;
    uint64_t m_offsetsOffset = 0;
    Ref<device::Buffer> m_renderOrigins;
    Ref<device::Buffer> m_surfaces;
    Ref<device::Buffer> m_viewers;
    std::vector<uint32_t> m_products;
    Ref<device::Buffer> m_productBuffer;
    std::map<uint32_t, ComposedWorldDesc> m_composed;
    std::map<uint32_t, std::set<uint32_t>> m_shownIn;
    uint64_t m_compositions = ~uint64_t(0);
    std::vector<std::vector<uint32_t>> m_own;
    std::vector<uint32_t> m_globals;
    std::set<uint32_t> m_origins;
    Regions m_regions;
    std::vector<uint32_t> m_refits;
    uint64_t m_countsOffset = 0;
    uint64_t m_primsOffset = 0;
    uint64_t m_seen = 0;
    uint32_t m_worlds = 0;
    uint32_t m_relayouts = 0;
    InstanceStats m_stats;
};

}
