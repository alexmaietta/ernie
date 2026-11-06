// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/slang/debug.h"

#include <bit>
#include <cstring>
#include <format>

#include "ernie/device/kernels/device.h"
#include "ernie/util/log.h"
#include "ernie/util/text.h"

namespace ernie::device {
namespace {

constexpr size_t kShownFailures = 8;

bool Done(uint32_t stamp, uint64_t finished) {
    return stamp != 0 && int32_t(uint32_t(finished) - stamp) >= 0;
}

std::string Value(uint32_t bits, uint32_t type) {
    switch (type) {
    case kPrintedInt:
        return std::to_string(int32_t(bits));
    case kPrintedFloat:
        return std::format("{}", std::bit_cast<float>(bits));
    case kPrintedBool:
        return bits != 0 ? "true" : "false";
    default:
        return std::to_string(bits);
    }
}

}

uint64_t DebugRecords::Bytes() {
    return uint64_t(kDebugHeaderWords + kCapacity * kDebugRecordWords) * 4;
}

DebugRecords::DebugRecords(Ref<Buffer> buffer) : m_buffer(std::move(buffer)) {
    m_header = static_cast<uint32_t*>(m_buffer->Mapped());
    std::memset(m_header, 0, Bytes());
}

BufferRange DebugRecords::Submission() const {
    return m_buffer->Slice(uint64_t(kSubmissionWord) * 4, 4);
}

void DebugRecords::AddKernel(const std::string& kernel, std::span<const std::string> strings) {
    const std::lock_guard lock(m_mutex);
    m_kernels.try_emplace(SlangStringHash(kernel), kernel);
    for (const std::string& text : strings) {
        m_strings.try_emplace(SlangStringHash(text), text);
    }
}

std::string DebugRecords::Format(const uint32_t* record) const {
    const auto found = m_strings.find(record[kHashWord]);
    std::string_view format = "{}";
    if (found != m_strings.end()) {
        format = found->second;
    }

    const uint32_t values = (record[kKindWord] >> 8) & 0xff;
    std::string text;
    uint32_t next = 0;
    for (size_t at = format.find("{}"); at != std::string_view::npos; at = format.find("{}")) {
        text.append(format.substr(0, at));
        const uint32_t* value = record + kValuesWord + size_t(next) * 2;
        text += next < values ? Value(value[0], value[1]) : "{}";
        ++next;
        format.remove_prefix(at + 2);
    }
    text.append(format);
    return text;
}

void DebugRecords::Report(const uint32_t* record, uint64_t finished) {
    const auto found = m_kernels.find(record[kKernelWord]);
    const std::string kernel = found != m_kernels.end() ? found->second : "a kernel";
    const std::string message = Format(record);
    const uint32_t* grid = record + kThreadWord;
    const std::string thread =
        std::format("thread {} ({}, {}, {})", record[kLinearWord], grid[0], grid[1], grid[2]);

    if ((record[kKindWord] & 0xff) == kAssertRecord) {
        const uint64_t submission = (finished & ~uint64_t(0xffffffff)) | record[kStampWord];
        m_failures.emplace(
            submission > finished ? submission - (uint64_t(1) << 32) : submission,
            std::format("Assert failed in kernel {} at {}: {}", kernel, thread, message));
    } else if ((record[kKindWord] & 0xff) == kPrintRecord) {
        Log(LogLevel::Info, "kernel {}, {}: {}", kernel, thread, message);
    }
}

void DebugRecords::Read(uint64_t finished, bool idle) {
    const std::lock_guard lock(m_mutex);
    const uint32_t taken = m_header[kTakenWord];
    const uint32_t kept = std::min(taken, kCapacity);
    for (; m_read < kept; ++m_read) {
        uint32_t* record = m_header + kDebugHeaderWords + size_t(m_read) * kDebugRecordWords;
        if (!Done(record[kStampWord], finished)) {
            break;
        }
        Report(record, finished);
        record[kStampWord] = 0;
    }

    if (!idle || m_read != kept) {
        return;
    }
    if (taken > kCapacity) {
        Log(LogLevel::Warning,
            "kernels wrote {} more asserts and prints than the {} the "
            "debug buffer holds; those were dropped",
            taken - kCapacity, kCapacity);
    }
    m_header[kTakenWord] = 0;
    m_read = 0;
}

std::optional<std::string> DebugRecords::TakeFailures(uint64_t submission) {
    const std::lock_guard lock(m_mutex);
    const auto end = m_failures.upper_bound(submission);
    if (end == m_failures.begin()) {
        return std::nullopt;
    }

    std::vector<std::string> messages;
    size_t count = 0;
    for (auto it = m_failures.begin(); it != end; ++it, ++count) {
        if (count < kShownFailures) {
            messages.push_back(it->second);
        }
    }
    if (count > kShownFailures) {
        messages.push_back(std::format("and {} more failed asserts", count - kShownFailures));
    }
    m_failures.erase(m_failures.begin(), end);
    return Join(messages, "\n");
}

uint32_t SlangStringHash(std::string_view text) {
    uint32_t hash = 0;
    for (const char c : text) {
        hash = uint32_t(uint8_t(c)) + (hash << 6) + (hash << 16) - hash;
    }
    return hash;
}

}
