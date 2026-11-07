// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/device/ref.h"

namespace ernie {

template <class T> class RecordTable {
  public:
    explicit RecordTable(std::string label) : m_label(std::move(label)) {}

    void Assign(uint32_t rows, const T& value) {
        m_rows.assign(rows, value);
        m_dirty.clear();
        m_grown = true;
    }

    void Reserve(uint32_t rows) {
        if (rows > m_rows.size()) {
            m_rows.resize(std::max<size_t>(rows, m_rows.size() * 2), T{});
            m_grown = true;
        }
    }

    bool Set(uint32_t row, const T& value) {
        Reserve(row + 1);
        if (std::memcmp(&m_rows[row], &value, sizeof(T)) == 0) {
            return false;
        }
        m_rows[row] = value;
        m_dirty.push_back(row);
        return true;
    }

    const T& Row(uint32_t row) const {
        static const T kEmpty{};
        return row < m_rows.size() ? m_rows[row] : kEmpty;
    }

    uint32_t Rows() const {
        return uint32_t(m_rows.size());
    }

    bool Upload(device::Device& device) {
        Reserve(1);
        device::Queue& queue = device.GetQueue();
        const bool replaced = m_grown || !m_buffer;
        if (replaced) {
            m_buffer = device.CreateBuffer({.bytes = m_rows.size() * sizeof(T), .label = m_label});
            queue.Write(*m_buffer, m_rows.data());
        } else if (m_dirty.size() * 4 > m_rows.size()) {
            queue.Write(*m_buffer, m_rows.data());
        } else {
            std::ranges::sort(m_dirty);
            m_dirty.erase(std::ranges::unique(m_dirty).begin(), m_dirty.end());
            for (const uint32_t row : m_dirty) {
                queue.Write(m_buffer->Slice(uint64_t(row) * sizeof(T), sizeof(T)), &m_rows[row]);
            }
        }
        m_dirty.clear();
        m_grown = false;
        return replaced;
    }

    const Ref<device::Buffer>& Buffer() const {
        return m_buffer;
    }

  private:
    std::string m_label;
    std::vector<T> m_rows;
    std::vector<uint32_t> m_dirty;
    Ref<device::Buffer> m_buffer;
    bool m_grown = false;
};

template <class T> class DeviceTable {
  public:
    explicit DeviceTable(std::string label, device::BufferUsage usage = device::BufferUsage::None)
        : m_label(std::move(label)), m_usage(usage) {}

    void Allocate(device::Device& device, uint32_t rows) {
        m_rows = std::max(rows, 1u);
        m_buffer = device.CreateBuffer(
            {.bytes = uint64_t(m_rows) * sizeof(T), .usage = m_usage, .label = m_label});
    }

    bool Reserve(device::Device& device, uint32_t rows) {
        if (m_buffer && rows <= m_rows) {
            return false;
        }
        Allocate(device, rows);
        return true;
    }

    uint32_t Rows() const {
        return m_buffer ? m_rows : 0;
    }

    uint64_t Address() const {
        return m_buffer ? m_buffer->Address() : 0;
    }

    const Ref<device::Buffer>& Buffer() const {
        return m_buffer;
    }

  private:
    std::string m_label;
    device::BufferUsage m_usage;
    Ref<device::Buffer> m_buffer;
    uint32_t m_rows = 0;
};

}
