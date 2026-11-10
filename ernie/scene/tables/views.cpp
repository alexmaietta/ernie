// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/views.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/tables/kernels/views.h"
#include "ernie/stage/publish/change_set.h"
#include "ernie/util/profiler.h"

namespace ernie {

void ViewTable::Describe(RenderTables& tables) const {
    tables.views.address = m_views ? m_views->Address() : 0;
    tables.viewStarts.address = m_starts ? m_starts->Address() : 0;
}

void ViewTable::Prepare() {
    Kernels();
}

void ViewTable::Run(const ChangeSet&, TableVersions&) {
    ERNIE_ZONE("Views/Run");
    const Prims& prims = m_structure.GetPrims();
    ColumnStore& store = m_structure.GetColumnStore();
    device::Device& device = store.GetDevice();
    std::vector<uint32_t> starts(std::max(store.Schema(SchemaId::RenderProduct).Capacity(), 1u), 0);
    std::vector<ViewEntry> entries;
    for (const auto& product : m_products.Products()) {
        starts[prims.Row(product->prim, SchemaId::RenderProduct)] = uint32_t(entries.size());
        for (const uint32_t sensor : product->sensors) {
            ViewEntry entry{.product = prims.Row(product->prim, SchemaId::RenderProduct),
                            .camera = kNoRow,
                            .distortion = kNoRow};
            if (IsA(prims.Record(sensor).schema, SchemaId::Camera)) {
                entry.camera = prims.Row(sensor, SchemaId::Camera);
                if (prims.HasApi(sensor, SchemaId::LensDistortionAPI)) {
                    entry.distortion = prims.Row(sensor, SchemaId::LensDistortionAPI);
                }
            }
            entries.push_back(entry);
        }
    }
    m_count = uint32_t(entries.size());
    m_starts = device.CreateBuffer({.label = "view starts"}, std::span<const uint32_t>(starts));
    m_views = device.CreateBuffer(
        {.bytes = std::max<uint64_t>(entries.size(), 1) * sizeof(ViewRecord), .label = "views"});
    if (entries.empty()) {
        return;
    }
    const Ref<device::Buffer> list =
        device.CreateBuffer({.label = "view entries"}, std::span<const ViewEntry>(entries));
    ViewParams params{};
    params.cameras = m_tables.cameras;
    params.distortions = m_tables.distortions;
    params.products = m_tables.products;
    params.entries.address = list->Address();
    params.views.address = m_views->Address();
    params.count = m_count;
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.Dispatch(Kernels().Entry("WriteViews"), params,
                     {.x = device::GroupCount(m_count, kViewGroup)});
    queue.Submit(encoder.Finish());
}

device::Program& ViewTable::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.tables.kernels.views"},
             .library = &kSceneKernels,
             .entries = {"WriteViews"},
             .label = "views"});
    }
    return *m_program;
}

}
