// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/catalog/catalog.h"
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
    std::span<const ColumnRef> Outputs() const override;
    void Prepare() override;
    device::Operation Run(const ChangeSet& changes, Stream stream) override;

    uint32_t Segments() const {
        return m_segments.Size();
    }

    uint32_t Runs() const {
        return m_runs;
    }

    uint32_t Dispatched() const {
        return m_dispatched;
    }

    uint32_t Rebuilds() const {
        return m_rebuilds;
    }

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
    bool m_built = false;
    uint64_t m_seen = 0;
    uint32_t m_dispatched = 0;
    uint32_t m_rebuilds = 0;
    uint32_t m_runs = 0;
};

}
