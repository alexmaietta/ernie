// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/builders/table_builder.h"
#include "ernie/scene/builders/table_storage.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/materials.h"

namespace ernie {

class Materials : public System {
  public:
    explicit Materials(Catalog& catalog) : m_catalog(catalog) {}

    std::string_view Name() const override {
        return "materials";
    }

    std::span<const ColumnRef> Inputs() const override;
    void Prepare() override;
    void Run(const ChangeSet& changes, TableLog& log) override;
    void Describe(RenderTables& tables) const override;

    void Hold(std::vector<Ref<const RefCounted>>& held) const override {
        HoldBuffers(held, {m_previews.Buffer(), m_nonVisuals.Buffer(), m_checkers.Buffer()});
    }

  private:
    device::Program& Kernels();

    Catalog& m_catalog;
    Ref<device::Program> m_program;
    DeviceTable<PreviewRecord> m_previews{"preview surface records"};
    DeviceTable<NonVisualRecord> m_nonVisuals{"nonvisual material records"};
    DeviceTable<CheckerRecord> m_checkers{"checkerboard records"};
};

}
