// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/views/readback.h"

#include "ernie/device/commands.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/kernels/readback.h"

namespace ernie {

std::vector<uint32_t> TableRows(const Prims& prims, std::span<const uint32_t> indices,
                                SchemaId table) {
    std::vector<uint32_t> rows;
    rows.reserve(indices.size());
    for (const uint32_t prim : indices) {
        rows.push_back(prims.Alive(prim) ? prims.FindRow(prim, table) : kNoRow);
    }
    return rows;
}

device::Download Readback::ReadAsync(const Ref<device::Buffer>& source, uint32_t stride,
                                     std::span<const uint32_t> rows) {
    if (rows.empty() || !source) {
        return {};
    }
    if (!m_program) {
        m_program = m_device.CreateProgram({.modules = {"ernie.scene.kernels.readback"},
                                            .library = &kSceneKernels,
                                            .entries = {"GatherRows"},
                                            .label = "readback"});
    }
    const uint64_t bytes = uint64_t(rows.size()) * stride;
    const Ref<device::Buffer> out =
        m_device.CreateBuffer({.bytes = bytes, .label = "gathered rows"});
    const Ref<device::Buffer> list = m_device.CreateBuffer({.label = "gather rows"}, rows);
    GatherParams params{};
    params.source.address = source->Address();
    params.rows.address = list->Address();
    params.out.address = out->Address();
    params.words = stride / uint32_t(sizeof(uint32_t));
    params.count = uint32_t(rows.size());
    device::Queue& queue = m_device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.Dispatch(m_program->Entry("GatherRows"), params,
                     {.x = device::GroupCount(params.count, kReadbackGroup)});
    queue.Submit(encoder.Finish());
    return queue.ReadAsync(out->Slice(0, bytes));
}

}
