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
#include "ernie/scene/kernels/instances.h"
#include "ernie/scene/systems/regions.h"
#include "ernie/scene/systems/system.h"
#include "ernie/scene/systems/tables.h"

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

    std::span<const TableId> TableOutputs() const override {
        static constexpr TableId kTables[] = {TableId::Instances, TableId::Slots};
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
        return m_renderOrigins.Buffer();
    }

    void Describe(RenderTables& tables) const override;
    void Hold(std::vector<Ref<const RefCounted>>& held) const override;

    const Ref<device::Buffer>& Records() const {
        return m_records.Buffer();
    }

    const Ref<device::Buffer>& Hits() const {
        return m_hits.Buffer();
    }

    const std::set<uint32_t>& Members() const {
        return m_members;
    }

    std::span<const std::pair<uint32_t, uint32_t>> LightCopies(uint32_t tlas) const {
        return tlas < m_lightCopies.size() ? std::span(m_lightCopies[tlas])
                                           : std::span<const std::pair<uint32_t, uint32_t>>();
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

    std::set<uint32_t> Tlases(uint32_t world) const;

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
    void WriteRegions(const std::set<uint32_t>& tlases);
    device::BufferRange LayoutWords(uint64_t address, uint32_t first, uint32_t count) const;
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
    DeviceTable<InstanceRecord> m_records{"instances", device::BufferUsage::AccelBuildInput};
    DeviceTable<ShapeRecord> m_shapes{"instance shapes"};
    Ref<device::Buffer> m_layout;
    InstanceLayout m_header{};
    DeviceTable<SlotHit> m_hits{"instance hits"};
    DeviceTable<WorldTransform> m_slotWorlds{"slot worlds"};
    Ref<device::Buffer> m_instancing;
    std::vector<Instancer> m_instancers;
    std::set<uint32_t> m_instancerPrims;
    std::set<uint32_t> m_roots;
    std::set<uint32_t> m_members;
    std::vector<std::vector<uint32_t>> m_blockItems;
    std::map<uint32_t, std::vector<uint32_t>> m_drawables;
    std::map<uint32_t, std::vector<uint32_t>> m_nested;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> m_lightCopies;
    std::map<uint32_t, std::set<uint32_t>> m_watch;
    DeviceTable<RenderOrigin> m_renderOrigins{"render origins"};
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
    uint64_t m_seen = 0;
    uint32_t m_worlds = 0;
    uint32_t m_relayouts = 0;
    InstanceStats m_stats;
};

}
