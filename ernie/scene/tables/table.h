// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/ref.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/schema/schema.h"

namespace ernie {

struct ChangeSet;
struct RenderTables;

class Table {
  public:
    virtual ~Table() = default;

    virtual std::string_view Name() const = 0;
    virtual std::span<const ColumnRef> Inputs() const = 0;

    virtual std::span<const TableId> TableInputs() const {
        return {};
    }

    virtual std::span<const TableId> TableOutputs() const {
        return {};
    }

    virtual bool Structural() const {
        return true;
    }

    virtual bool Pending() const {
        return false;
    }

    virtual void Prepare() {}

    virtual void Finish() {}

    virtual void Describe(RenderTables&) const {}

    virtual void Hold(std::vector<Ref<const RefCounted>>&) const {}

    virtual void Run(const ChangeSet& changes, TableVersions& versions) = 0;

    uint32_t Runs() const {
        return m_runs;
    }

  private:
    friend class Publisher;

    uint32_t m_runs = 0;
};

inline void HoldBuffers(std::vector<Ref<const RefCounted>>& held,
                        std::initializer_list<Ref<device::Buffer>> buffers) {
    for (const Ref<device::Buffer>& buffer : buffers) {
        if (buffer) {
            held.emplace_back(buffer);
        }
    }
}

}
