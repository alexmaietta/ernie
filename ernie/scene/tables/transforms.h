// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/records/transforms.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/regions.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"

namespace ernie {

class TransformTable : public RenderTable {
  public:
    explicit TransformTable(Structure& structure) : m_structure(structure) {}

    std::string_view Name() const override {
        return "hierarchy";
    }

    std::span<const ColumnRef> Inputs() const override;
    std::span<const TableId> TableOutputs() const override;
    void Describe(RenderTables& tables) const override;
    void Hold(std::vector<Ref<const RefCounted>>& held) const override;
    void Prepare() override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;

    uint32_t Segments() const {
        return m_segments.Size();
    }

    uint32_t Dispatched() const {
        return m_dispatched;
    }

    uint32_t Rebuilds() const {
        return m_rebuilds;
    }

    const Ref<device::Buffer>& Transforms() const {
        return m_transforms.Buffer();
    }

    const Ref<device::Buffer>& Imageables() const {
        return m_imageables.Buffer();
    }

  private:
    void Rebuild();
    void WriteMotions(bool restructured);
    bool Apply(const ChangeSet& changes);
    uint32_t ParentRow(uint32_t prim) const;

    Structure& m_structure;
    device::Program& Kernels();

    Ref<device::Program> m_program;
    RecordTable<uint32_t> m_parents{"hierarchy parents"};
    RecordTable<uint32_t> m_xforms{"hierarchy transforms"};
    std::unordered_map<uint32_t, uint32_t> m_rows;
    Regions m_segments;
    Ref<device::Buffer> m_segmentRows;
    Ref<device::Buffer> m_segmentStart;
    Ref<device::Buffer> m_segmentCount;
    DeviceTable<WorldTransform> m_transforms{"transforms"};
    DeviceTable<ResolvedImageable> m_imageables{"imageables"};
    DeviceTable<MotionRecord> m_motions{"motion records"};
    uint64_t m_motionSeen = 0;
    bool m_built = false;
    uint64_t m_seen = 0;
    uint32_t m_dispatched = 0;
    uint32_t m_rebuilds = 0;
};

}
