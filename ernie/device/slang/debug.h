// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/ref.h"

namespace ernie::device {

class Device;

class DebugRecords {
  public:
    static constexpr uint32_t kCapacity = 1024;

    static uint64_t Bytes();

    explicit DebugRecords(Ref<Buffer> buffer);

    uint64_t Address() const {
        return m_buffer->Address();
    }

    BufferRange Submission() const;

    void AddKernel(const std::string& kernel, std::span<const std::string> strings);

    void Read(uint64_t finished, bool idle);

    std::optional<std::string> TakeFailures(uint64_t submission);

  private:
    std::string Format(const uint32_t* record) const;

    void Report(const uint32_t* record, uint64_t finished);

    Ref<Buffer> m_buffer;
    uint32_t* m_header = nullptr;
    uint32_t m_read = 0;
    mutable std::mutex m_mutex;
    std::map<uint32_t, std::string> m_strings;
    std::map<uint32_t, std::string> m_kernels;
    std::multimap<uint64_t, std::string> m_failures;
};

uint32_t SlangStringHash(std::string_view text);

}
