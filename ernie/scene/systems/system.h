// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "ernie/device/operation.h"
#include "ernie/device/ref.h"
#include "ernie/device/stream.h"
#include "ernie/scene/kernels/schema/schema.h"
#include "ernie/scene/timeline/table_log.h"

namespace ernie {

struct ChangeSet;
struct RenderTables;

class System {
  public:
    virtual ~System() = default;

    virtual std::string_view Name() const = 0;
    virtual std::span<const ColumnRef> Inputs() const = 0;
    virtual std::span<const ColumnRef> Outputs() const = 0;

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

    virtual device::Operation Run(const ChangeSet& changes, Stream stream) = 0;
};

}
