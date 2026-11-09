// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/stage/columns/column_store.h"

namespace ernie {

class PackPlan {
  public:
    explicit PackPlan(std::string label) : m_label(std::move(label)) {}

    void Clear();
    void Copy(ColumnRef column, uint32_t row, uint32_t destination);
    uint32_t Literal(uint32_t destination, uint32_t value);
    void SetLiteral(uint32_t literal, uint32_t value);

    bool Reads(const ColumnInfo* column) const;
    bool Stale(const ColumnStore& store, uint64_t since) const;
    uint32_t Twice(uint32_t words) const;

    uint32_t CopyCount() const {
        return uint32_t(m_copies.size());
    }

    void Build(ColumnStore& store, bool zeroed);
    void Pack(ColumnStore& store, uint64_t words);

  private:
    struct Cell {
        ColumnRef column;
        uint32_t row = 0;
        uint32_t destination = 0;
    };

    struct Word {
        uint32_t destination = 0;
        uint32_t value = 0;
        bool late = false;
    };

    std::string m_label;
    std::vector<Cell> m_copies;
    std::vector<Word> m_literals;
    std::vector<ColumnRef> m_columns;
    Ref<device::Buffer> m_entries;
    uint32_t m_count = 0;
    Ref<device::Program> m_program;
};

}
