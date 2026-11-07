// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/catalog/catalog.h"
#include "ernie/scene/kernels/tables/transforms.h"
#include "ernie/scene/systems/regions.h"
#include "ernie/scene/systems/system.h"

namespace ernie {

class Hierarchy : public System {
  public:
    explicit Hierarchy(Catalog& catalog) : m_catalog(catalog) {}

    std::string_view Name() const override {
        return "hierarchy";
    }

    std::span<const ColumnRef> Inputs() const override;
    std::span<const TableId> TableOutputs() const override;
    void Describe(RenderTables& tables) const override;
    void Hold(std::vector<Ref<const RefCounted>>& held) const override;
    void Prepare() override;
    void Run(const ChangeSet& changes, TableLog& log) override;

    uint32_t Segments() const {
        return m_segments.Size();
    }

    uint32_t Dispatched() const {
        return m_dispatched;
    }

    uint32_t Rebuilds() const {
        return m_rebuilds;
    }

    device::Download Gather(const Ref<device::Buffer>& source, uint32_t words,
                            std::span<const uint32_t> rows);
    device::Download ReadTransformsAsync(std::span<const uint32_t> prims);
    std::vector<WorldTransform> Transforms(std::span<const uint32_t> prims);
    std::vector<ResolvedImageable> Imageables(std::span<const uint32_t> prims);

  private:
    void Rebuild();
    bool Apply(const ChangeSet& changes);
    uint32_t ParentRow(uint32_t prim) const;

    Catalog& m_catalog;
    device::Program& Kernels();

    Ref<device::Program> m_program;
    std::vector<uint32_t> m_parentRows;
    std::vector<uint32_t> m_xformRows;
    std::unordered_map<uint32_t, uint32_t> m_rows;
    Regions m_segments;
    Ref<device::Buffer> m_parent;
    Ref<device::Buffer> m_xform;
    Ref<device::Buffer> m_segmentRows;
    Ref<device::Buffer> m_segmentStart;
    Ref<device::Buffer> m_segmentCount;
    Ref<device::Buffer> m_transforms;
    Ref<device::Buffer> m_imageables;
    uint32_t m_transformCapacity = 0;
    bool m_built = false;
    uint64_t m_seen = 0;
    uint32_t m_dispatched = 0;
    uint32_t m_rebuilds = 0;
};

}
