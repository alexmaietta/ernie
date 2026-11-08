// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <set>
#include <span>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/kernels/lights.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/instances.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"

namespace ernie {

class LightTable : public Table {
  public:
    LightTable(Structure& structure, const InstanceTable& instances)
        : m_structure(structure), m_instances(instances) {}

    std::string_view Name() const override {
        return "lights";
    }

    std::span<const ColumnRef> Inputs() const override;

    std::span<const TableId> TableInputs() const override {
        static constexpr TableId kTables[] = {TableId::Transforms, TableId::Imageables,
                                              TableId::Textures, TableId::Instances};
        return kTables;
    }

    void Prepare() override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;

    const Ref<device::Buffer>& Layout() const {
        return m_layout;
    }

    const Ref<device::Buffer>& Records() const {
        return m_records;
    }

    std::span<const uint32_t> Counts() const {
        return m_counts;
    }

    uint32_t Written() const {
        return m_written;
    }

    const Ref<device::Buffer>& Spectra();

    void Describe(RenderTables& tables) const override;
    void Hold(std::vector<Ref<const RefCounted>>& held) const override;

  private:
    bool Restructured(const ChangeSet& changes) const;
    void Relayout();
    void WriteCollectionsAndShapes();
    device::Program& Kernels();

    Structure& m_structure;
    Ref<device::Program> m_program;
    Ref<device::Buffer> m_layout;
    Ref<device::Buffer> m_records;
    Ref<device::Buffer> m_cdf;
    Ref<device::Buffer> m_maps;
    Ref<device::Buffer> m_spectra;
    DeviceTable<CollectionRecord> m_collections{"collection records"};
    DeviceTable<LightShapeRecord> m_shapes{"light shape records"};
    std::set<uint32_t> m_lights;
    std::set<uint32_t> m_worldPrims;
    std::map<uint32_t, std::set<uint32_t>> m_shownIn;
    uint64_t m_compositions = ~uint64_t(0);
    std::vector<uint32_t> m_counts;
    uint32_t m_worlds = 0;
    uint32_t m_domes = 0;
    uint64_t m_seen = 0;
    const InstanceTable& m_instances;
    uint32_t m_written = 0;
};

}
