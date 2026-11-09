// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/pack.h"

#include <algorithm>

#include "ernie/device/commands.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/kernels.h"
#include "ernie/scene/tables/kernels/pack.h"
#include "ernie/util/profiler.h"

namespace ernie {

void PackPlan::Clear() {
    m_copies.clear();
    m_literals.clear();
    m_columns.clear();
    m_count = 0;
}

void PackPlan::Copy(ColumnRef column, uint32_t row, uint32_t destination) {
    m_copies.push_back({column, row, destination});
    if (!Reads(column.column)) {
        m_columns.push_back(column);
    }
}

uint32_t PackPlan::Literal(uint32_t destination, uint32_t value) {
    m_literals.push_back({destination, value});
    return uint32_t(m_literals.size() - 1);
}

void PackPlan::SetLiteral(uint32_t literal, uint32_t value) {
    m_literals[literal].value = value;
    m_literals[literal].late = true;
}

bool PackPlan::Reads(const ColumnInfo* column) const {
    return std::ranges::find(m_columns, column, &ColumnRef::column) != m_columns.end();
}

bool PackPlan::Stale(const ColumnStore& store, uint64_t since) const {
    return std::ranges::any_of(m_columns,
                               [&](ColumnRef column) { return store.LastWrite(column) > since; });
}

uint32_t PackPlan::Twice(uint32_t words) const {
    std::vector<uint8_t> written(words, 0);
    uint32_t twice = kNoPrim;
    const auto write = [&](uint32_t word) {
        if (word < written.size() && written[word]++ != 0 && twice == kNoPrim) {
            twice = word;
        }
    };
    for (const Cell& copy : m_copies) {
        const uint32_t cells = ElementBytes(copy.column.column->type) / sizeof(uint32_t);
        for (uint32_t word = 0; word < cells; ++word) {
            write(copy.destination + word);
        }
    }
    for (const Word& literal : m_literals) {
        write(literal.destination);
    }
    return twice;
}

void PackPlan::Build(ColumnStore& store, bool zeroed) {
    std::vector<PackEntry> entries;
    entries.reserve(m_copies.size() + m_literals.size());
    for (const Cell& copy : m_copies) {
        const ColumnTable& table = store.Schema(copy.column.schema);
        entries.push_back({.pages = table.Pages(copy.column.index),
                           .row = copy.row,
                           .words = table.CellBytes(copy.column.index) / uint32_t(sizeof(uint32_t)),
                           .destination = copy.destination});
    }
    for (const Word& literal : m_literals) {
        if (!zeroed || literal.late || literal.value != 0) {
            entries.push_back(
                {.words = 1, .destination = literal.destination, .value = literal.value});
        }
    }
    m_count = uint32_t(entries.size());
    if (entries.empty()) {
        return;
    }
    device::Device& device = store.GetDevice();
    const uint64_t bytes = entries.size() * sizeof(PackEntry);
    if (!m_entries || m_entries->Bytes() < bytes) {
        m_entries = device.CreateBuffer({.bytes = bytes, .label = m_label + " pack entries"});
    }
    device.GetQueue().Write(m_entries->Slice(0, bytes), entries.data());
}

void PackPlan::Pack(ColumnStore& store, uint64_t words) {
    if (m_count == 0) {
        return;
    }
    device::Device& device = store.GetDevice();
    if (!m_program) {
        m_program = device.CreateProgram({.modules = {"ernie.scene.tables.kernels.pack"},
                                          .library = &kSceneKernels,
                                          .entries = {"PackWords"},
                                          .label = m_label + " pack"});
    }
    PackParams params{};
    params.entries.address = m_entries->Address();
    params.words.address = words;
    params.count = m_count;
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Tables/Pack");
    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    {
        const device::ZoneScope scope(encoder, zone);
        encoder.Dispatch(m_program->Entry("PackWords"), params,
                         {.x = device::GroupCount(params.count, kPackGroup)});
    }
    queue.Submit(encoder.Finish());
}

}
