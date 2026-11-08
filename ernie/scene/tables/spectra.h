// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/ref.h"
#include "ernie/scene/records/spectra.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/table.h"

namespace ernie {

class SpectralTable : public Table {
  public:
    explicit SpectralTable(Structure& structure) : m_structure(structure) {}

    std::string_view Name() const override {
        return "spectra";
    }

    std::span<const ColumnRef> Inputs() const override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_rgb, m_bands, m_ranges, m_samples});
    }

  private:
    Structure& m_structure;
    Ref<device::Buffer> m_rgb;
    Ref<device::Buffer> m_bands;
    Ref<device::Buffer> m_ranges;
    Ref<device::Buffer> m_samples;
    uint32_t m_count = 0;
    uint32_t m_detectors = 0;
};

}
