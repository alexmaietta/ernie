// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/stage/views/mapping.h"

#include <utility>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/stage/columns/column_store.h"
#include "ernie/stage/views/attribute.h"
#include "ernie/stage/views/view_state.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"

namespace ernie {

Mapping::Mapping(Mapping&& other) noexcept {
    *this = std::move(other);
}

Mapping& Mapping::operator=(Mapping&& other) noexcept {
    if (this != &other) {
        Release();
        m_view = std::move(other.m_view);
        m_column = other.m_column;
        m_access = other.m_access;
        m_values = std::move(other.m_values);
        m_rows = std::move(other.m_rows);
        m_offsets = std::move(other.m_offsets);
        m_ready = other.m_ready;
        m_aliased = other.m_aliased;
        m_open = std::exchange(other.m_open, false);
    }
    return *this;
}

Mapping::~Mapping() {
    if (m_open && m_access != MapAccess::Read) {
        Log(LogLevel::Warning,
            "A mapping of {}'s {} was dropped without Close; its writes aren't marked as changes",
            Info(m_column.schema).name, m_column.column->field);
    }
    Release();
}

void Mapping::Release() noexcept {
    if (m_open) {
        m_open = false;
        try {
            m_view->GetColumnStore().Unpin(m_column.schema);
        } catch (...) {
        }
    }
}

device::Operation Mapping::Close(Stream producer, const DLTensor* mask) {
    if (!m_open) {
        throw Error("The mapping is already closed");
    }
    Release();
    UntypedAttribute attribute(m_view, m_column);
    if (m_access == MapAccess::Read || m_values.Empty()) {
        attribute.GetColumnStore().GetDevice().GetQueue().Follow(producer);
        return {};
    }
    if (m_rows.Empty()) {
        return attribute.Stamp(producer, mask);
    }
    return attribute.Scatter(m_values.Address(), producer, attribute.Upload(mask, producer));
}

}
