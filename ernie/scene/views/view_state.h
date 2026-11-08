// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/ref.h"
#include "ernie/scene/records/primvars.h"
#include "ernie/scene/schema/schema.h"
#include "ernie/scene/structure/structure.h"

namespace ernie {

struct ViewBlock {
    uint32_t first = 0;
    uint32_t firstRow = 0;
    uint32_t count = 0;
};

struct ViewRows {
    uint64_t epoch = ~uint64_t(0);
    std::vector<uint32_t> rows;
    std::vector<ViewBlock> blocks;
    std::vector<uint32_t> worlds;
    uint32_t live = 0;
    Ref<device::Buffer> device;

    bool Dense() const {
        return blocks.size() == 1 && live == rows.size();
    }
};

struct RowSource {
    uint64_t rows = 0;
    uint32_t firstRow = 0;
};

struct PrimvarLayout {
    std::vector<ArrayRef> tables;
    PrimvarEntry entry{};
};

class ViewState {
  public:
    ViewState(Structure& structure, SchemaId schema, std::vector<uint32_t> prims,
              uint32_t instance = kNoInstance);

    static constexpr uint32_t kNoInstance = 0xFFFFFFFF;

    uint32_t Instance() const {
        return m_instance;
    }

    Structure& GetStructure() const {
        return m_structure;
    }

    ColumnStore& GetColumnStore() const {
        return m_structure.GetColumnStore();
    }

    SchemaId Schema() const {
        return m_schema;
    }

    uint32_t Size() const {
        return uint32_t(m_prims.size());
    }

    std::span<const uint32_t> Prims() const {
        return m_prims;
    }

    bool Live(uint32_t index) const;
    const ViewRows& Rows(SchemaId table);
    const Ref<device::Buffer>& DeviceRows(SchemaId table);
    RowSource Source(SchemaId table);

    PrimvarLayout& Layout(ColumnRef column, std::string_view name) {
        return m_primvarLayouts[{column.schema, column.index, std::string(name)}];
    }

  private:
    Structure& m_structure;
    SchemaId m_schema;
    uint32_t m_instance;
    std::vector<uint32_t> m_prims;
    std::vector<uint32_t> m_generations;
    std::map<SchemaId, ViewRows> m_rows;
    std::map<std::tuple<SchemaId, uint32_t, std::string>, PrimvarLayout> m_primvarLayouts;
};

}
