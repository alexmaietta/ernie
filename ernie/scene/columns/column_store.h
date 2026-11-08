// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>

#include <unordered_map>
#include <vector>

#include <dlpack/dlpack.h>

#include "ernie/device/buffer.h"
#include "ernie/device/commands.h"
#include "ernie/device/device.h"
#include "ernie/device/program.h"
#include "ernie/device/queue.h"
#include "ernie/device/ref.h"
#include "ernie/scene/columns/array_pool.h"
#include "ernie/scene/columns/column_table.h"
#include "ernie/scene/columns/device_heap.h"
#include "ernie/scene/columns/kernels/attributes.h"
#include "ernie/scene/columns/kernels/changes.h"
#include "ernie/scene/schema/schema.h"
#include "ernie/scene/tables/kernels/authored.h"

namespace ernie {

using Ordinal = uint64_t;

enum PrimColumn : uint32_t {
    kPrimSchema,
    kPrimParent,
    kPrimWorld,
    kPrimFlags,
    kPrimSlots,
    kPrimApis
};

inline constexpr uint32_t kNoStamp = ~0u;

struct Attachment {
    ColumnRef column;
    std::vector<uint32_t> rows;
    std::vector<uint32_t> sorted;
    bool inPlace = false;
    std::shared_ptr<const DLTensor> tensor;
};

class ColumnStore {
  public:
    explicit ColumnStore(device::Device& device);
    ~ColumnStore();
    ColumnStore(const ColumnStore&) = delete;
    ColumnStore& operator=(const ColumnStore&) = delete;

    device::Device& GetDevice() const {
        return m_device;
    }

    DeviceHeap& GetDeviceHeap() {
        return m_heap;
    }

    ColumnTable& Schema(SchemaId schema);
    const ColumnTable& Schema(SchemaId schema) const;

    void Pin(SchemaId schema);
    void Unpin(SchemaId schema);
    bool Pinned(SchemaId schema) const;

    ColumnTable& Prims() {
        return m_prims;
    }

    ColumnTable& Slots() {
        return m_slots;
    }

    ColumnTable& Worlds() {
        return m_worlds;
    }

    ArrayPool& Arrays(ColumnType type);

    void Attach(Attachment attachment);
    const Attachment* FindAttachment(ColumnRef column, std::span<const uint32_t> rows) const;
    bool Attached(ColumnRef column, std::span<const uint32_t> rows) const;
    Attachment Detach(ColumnRef column, std::span<const uint32_t> rows);

    uint32_t AttachmentCount() const {
        return uint32_t(m_attachments.size());
    }

    uint32_t PrimsColumn(SchemaId schema) const;
    uint32_t StampColumn(ColumnRef column) const;
    bool Tracked(ColumnRef column) const;

    void Touch(ColumnRef column);
    uint64_t LastWrite(ColumnRef column) const;

    uint64_t WriteSequence() const {
        return m_writes;
    }

    Ordinal Stamp() const {
        return m_stamp;
    }

    void SetStamp(Ordinal stamp) {
        m_stamp = stamp;
    }

    StampTarget Stamps(ColumnRef column);
    void StampWorld(uint32_t world);
    Ordinal WorldStamp(uint32_t world) const;

    void Stage(ColumnRef column, uint32_t first, std::span<const std::byte> cells);
    void Fill(ColumnRef column, uint32_t first, uint32_t count, std::span<const std::byte> cell);
    void Override(ColumnRef column, uint32_t row, uint64_t owner,
                  std::span<const std::byte> authored, std::span<const std::byte> value);
    void ClearOverride(ColumnRef column, uint32_t row, uint64_t owner);
    bool Overridden(ColumnRef column, uint32_t row) const;
    std::vector<std::byte> Authored(ColumnRef column, uint32_t row) const;
    void Reassert(ColumnRef column, std::span<const uint32_t> rows);
    void DropOverrides(SchemaId table, uint32_t row);
    void CloneRows(SchemaId table, std::span<const uint32_t> sources, uint32_t first,
                   uint32_t copies);

    ArrayRef Array(ColumnRef column, uint32_t row) const;
    bool Exclusive(ColumnRef column, uint32_t row) const;
    void SetArray(ColumnRef column, uint32_t row, const ArrayRef& array);
    std::vector<uint32_t> ArrayRows(ColumnRef column, Ordinal since, Ordinal until) const;
    void LogArrayRows(ColumnRef column, std::span<const uint32_t> rows);
    void TrimArrayWrites(Ordinal through);

    void SetCounts(uint32_t prims, uint32_t worlds);

    void SetTextureBindings(uint64_t address) {
        m_textureBindings = address;
    }

    void Flush();
    AuthoredData Data() const;
    uint64_t DeviceData();

    void Prepare() {
        Kernels();
    }

    device::Download ReadRowsAsync(const Ref<device::Buffer>& source, uint32_t stride,
                                   std::span<const uint32_t> rows);

    template <class T>
    std::vector<T> ReadRows(const Ref<device::Buffer>& source, std::span<const uint32_t> rows) {
        static_assert(sizeof(T) % sizeof(uint32_t) == 0);
        std::vector<T> values(rows.size());
        const device::Download download = ReadRowsAsync(source, sizeof(T), rows);
        download.Wait();
        if (!values.empty() && download.Bytes().size() == values.size() * sizeof(T)) {
            std::memcpy(values.data(), download.Bytes().data(), values.size() * sizeof(T));
        }
        return values;
    }

    template <class Params>
    device::Operation Dispatch(const char* entry, const Params& params, uint64_t threads) {
        device::Queue& queue = m_device.GetQueue();
        if (threads == 0) {
            return queue.Mark();
        }
        device::CommandEncoder encoder = queue.CreateEncoder();
        {
            const device::ZoneScope zone(
                encoder, profiler::ZoneId::Named(std::string("ColumnStore/") + entry));
            encoder.Dispatch(Kernels().Entry(entry), params,
                             {.x = device::GroupCount(threads, kCellsGroup)});
        }
        return queue.Submit(encoder.Finish());
    }

  private:
    struct Entry {
        std::unique_ptr<ColumnTable> table;
        std::vector<uint32_t> stamps;
        std::vector<uint64_t> writes;
        uint32_t pins = 0;
    };

    struct OverrideLayer {
        uint64_t owner = 0;
        std::vector<std::byte> value;
    };

    struct OverrideCell {
        std::vector<std::byte> authored;
        std::vector<OverrideLayer> layers;
    };

    struct AuthoredRead {
        uint64_t key = 0;
        device::Download read;
    };

    struct ArrayWrite {
        Ordinal stamp = 0;
        uint32_t row = 0;
    };

    const Entry& Lookup(SchemaId schema) const;
    void Place(std::byte* data, uint32_t offset, uint64_t address) const;
    void FillWords(ColumnTable& table, uint32_t column, uint32_t first, uint32_t count,
                   std::span<const std::byte> cell);
    void StampRows(ColumnRef column, uint32_t first, uint32_t count);
    void StageLive(ColumnRef column, uint32_t first, std::span<const std::byte> cells);
    bool OverridesBetween(ColumnRef column, uint32_t first, uint32_t count) const;
    void LandAuthored(bool wait) const;
    device::Program& Kernels();
    static uint64_t ArrayKey(ColumnRef column, uint32_t row);

    device::Device& m_device;
    DeviceHeap m_heap;
    std::vector<Entry> m_entries;
    ColumnTable m_prims;
    ColumnTable m_slots;
    ColumnTable m_worlds;
    std::map<uint32_t, std::unique_ptr<ArrayPool>> m_arrayHeaps;
    std::vector<std::byte> m_payload;
    std::unordered_map<uint64_t, ArrayRef> m_arrays;
    std::map<uint64_t, std::vector<ArrayWrite>> m_arrayWrites;
    mutable std::map<uint64_t, OverrideCell> m_overrides;
    mutable std::vector<AuthoredRead> m_authoredReads;
    std::vector<Ordinal> m_worldStamps = std::vector<Ordinal>(1, 0);
    Ref<device::Program> m_program;
    Ref<device::Buffer> m_deviceData;
    std::vector<std::byte> m_uploaded;
    uint64_t m_textureBindings = 0;
    uint64_t m_writes = 0;
    Ordinal m_stamp = 1;
    uint32_t m_primCount = 0;
    uint32_t m_worldCount = 0;
    std::vector<Attachment> m_attachments;
};

}
