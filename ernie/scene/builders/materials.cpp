// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/builders/materials.h"

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/kernels/materials.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/tables/render_tables.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

std::vector<ColumnRef> MaterialInputs() {
    std::vector<ColumnRef> inputs = {kPlaneAxis};
    for (const SchemaId schema :
         {SchemaId::UsdPreviewSurface, SchemaId::TextureMappingAPI, SchemaId::SurfaceDetailAPI,
          SchemaId::NonVisualMaterialAPI, SchemaId::CheckerboardAPI}) {
        const Span<const ColumnInfo> columns = Columns(schema);
        for (uint32_t index = 0; index < columns.size(); ++index) {
            inputs.push_back({schema, &columns[index], index});
        }
    }
    return inputs;
}

}

std::span<const ColumnRef> MaterialBuilder::Inputs() const {
    static const std::vector<ColumnRef> inputs = MaterialInputs();
    return inputs;
}

void MaterialBuilder::Describe(RenderTables& tables) const {
    tables.previews.address = m_previews.Address();
    tables.nonVisuals.address = m_nonVisuals.Address();
    tables.checkers.address = m_checkers.Address();
}

void MaterialBuilder::Run(const ChangeSet&, TableWrites&) {
    ERNIE_ZONE("Materials/Run");
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    m_previews.Reserve(device, store.Schema(SchemaId::UsdPreviewSurface).Capacity());
    m_nonVisuals.Reserve(device, store.Schema(SchemaId::NonVisualMaterialAPI).Capacity());
    m_checkers.Reserve(device, store.Schema(SchemaId::CheckerboardAPI).Capacity());
    const uint32_t prims = m_structure.GetPrims().Size();
    if (prims == 0) {
        return;
    }
    MaterialParams params{};
    params.scene.address = store.DeviceData();
    params.previews.address = m_previews.Address();
    params.nonVisuals.address = m_nonVisuals.Address();
    params.checkers.address = m_checkers.Address();
    device::Queue& queue = device.GetQueue();
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Materials/Write");
    device::CommandEncoder encoder = queue.CreateEncoder();
    {
        const device::ZoneScope scope(encoder, zone);
        encoder.Dispatch(Kernels().Entry("WriteMaterials"), params,
                         {.x = device::GroupCount(prims, kMaterialGroup)});
    }
    queue.Submit(encoder.Finish());
}

void MaterialBuilder::Prepare() {
    Kernels();
}

device::Program& MaterialBuilder::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.kernels.materials"},
             .library = &kSceneKernels,
             .entries = {"WriteMaterials"},
             .label = "materials"});
    }
    return *m_program;
}

}
