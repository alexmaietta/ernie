// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/surfaces.h"

#include <algorithm>
#include <set>

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/records/render_tables.h"
#include "ernie/scene/schema/columns.h"
#include "ernie/scene/tables/kernels/surfaces.h"
#include "ernie/util/profiler.h"

namespace ernie {

namespace {

constexpr ColumnRef kInputs[] = {kGprimPrimvars,
                                 kGprimDisplayColor,
                                 kGprimDisplayOpacity,
                                 kPointBasedNormals,
                                 kPointBasedNormalsInterpolation,
                                 kMeshSubdivisionScheme,
                                 kPointInstancerPrimvars,
                                 kPointInstancerIds};

}

std::span<const ColumnRef> SurfaceTable::Inputs() const {
    return kInputs;
}

void SurfaceTable::Describe(RenderTables& tables) const {
    tables.surfaces.address = m_surfaces.Address();
}

void SurfaceTable::Run(const ChangeSet& changes, TableVersions&) {
    ERNIE_ZONE("Surfaces/Run");
    m_written = 0;
    ColumnStore& store = m_structure.GetColumnStore();
    const std::span<const uint32_t> starts = m_instances.FirstInstances();
    const std::span<const uint32_t> counts = m_instances.InstanceCounts();
    if (changes.tables->instances.address == 0 || counts.empty()) {
        return;
    }
    device::Device& device = store.GetDevice();
    const bool grown = m_surfaces.Reserve(device, starts.back());
    std::set<uint32_t> dirty;
    for (const uint32_t world :
         changes.resync || grown ? std::vector<uint32_t>{kGlobalWorld} : changes.worlds) {
        dirty.merge(m_instances.Tlases(world));
    }
    std::vector<uint32_t> tlases(dirty.begin(), dirty.end());
    uint32_t widest = 0;
    for (const uint32_t tlas : tlases) {
        widest = std::max(widest, starts[tlas + 1] - starts[tlas]);
        m_written += counts[tlas];
    }
    if (widest == 0) {
        return;
    }
    const Ref<device::Buffer> list = device.CreateBuffer({.label = "surface TLASes"}, tlases);
    SurfaceParams params{};
    params.scene.address = store.DeviceData();
    params.layout = changes.tables->instances;
    params.meshes = changes.tables->meshes;
    params.surfaces.address = m_surfaces.Address();
    params.tlases.address = list->Address();
    device::Queue& queue = device.GetQueue();
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Surfaces/Write");
    device::CommandEncoder encoder = queue.CreateEncoder();
    {
        const device::ZoneScope scope(encoder, zone);
        encoder.Dispatch(
            Kernels().Entry("WriteSurfaces"), params,
            {.x = device::GroupCount(widest, kSurfaceGroup), .y = uint32_t(tlases.size())});
    }
    queue.Submit(encoder.Finish());
}

void SurfaceTable::Prepare() {
    Kernels();
}

device::Program& SurfaceTable::Kernels() {
    if (!m_program) {
        m_program = m_structure.GetColumnStore().GetDevice().CreateProgram(
            {.modules = {"ernie.scene.tables.kernels.surfaces"},
             .library = &kSceneKernels,
             .entries = {"WriteSurfaces"},
             .label = "surfaces"});
    }
    return *m_program;
}

}
