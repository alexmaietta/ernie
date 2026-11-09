// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/algorithms/radix_sort.h"
#include "ernie/algorithms/scan.h"
#include "ernie/device/buffer.h"
#include "ernie/device/commands.h"
#include "ernie/device/program.h"
#include "ernie/overlay/kernels/composite.h"
#include "ernie/overlay/kernels/groups.h"
#include "ernie/overlay/kernels/overlay.h"
#include "ernie/overlay/kernels/selection.h"
#include "ernie/overlay/kernels/shapes.h"
#include "ernie/overlay/overlay.h"
#include "ernie/stage/types.h"

namespace ernie {

struct OverlayData {
    struct Batch {
        OverlayKind kind = OverlayKind::Lines;
        OverlayStyle style;
        device::Tensor points;
        device::Tensor ends;
        device::Tensor colors;
        device::Tensor sizes;
        device::Tensor worlds;
        device::Tensor itemCount;
        device::Tensor rotations;
        device::Tensor scalars;
        device::Tensor extras;
        device::Tensor chars;
        device::Tensor prims;
        device::Tensor groups;
        device::Tensor pixels;
        uint32_t groupCount = 0;
        OverlayPrimShape primShape = OverlayPrimShape::Frames;
        float length = 0;
        float fade = 0;
        std::optional<float> vectorScale;
        std::optional<ScreenPlacement> placement;
        OverlayNumberFormat format;
        std::shared_ptr<const void> owner;
        std::vector<Float3> hostPoints;
        std::vector<Float3> hostEnds;
        std::vector<uint32_t> hostColors;
        std::vector<float> hostSizes;
        std::vector<uint32_t> hostWorlds;
        std::vector<Float4> hostRotations;
        std::vector<float> hostScalars;
        std::vector<uint32_t> hostExtras;
        std::vector<uint8_t> hostCodes;
        std::vector<uint32_t> hostPrims;
        std::vector<uint32_t> hostGroups;
        std::vector<uint32_t> hostIds;
        uint32_t maxChars = 0;
    };

    std::vector<Batch> batches;

    bool Outlined() const;
};

struct OverlayTarget {
    uint32_t camera = kNoPrim;
    uint32_t product = kNoPrim;
    uint32_t layer = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    float missDistance = 0;
    uint64_t overlay = 0;
    uint64_t depth = 0;
    uint64_t overlays = ~uint64_t(0);
    uint64_t primIds = 0;
};

struct OverlayPickRequest {
    uint32_t target = 0;
    Float2 pixel;
    float radius = 0;
    Ref<device::Buffer> readback;
};

struct OverlayBatchSource {
    uint32_t overlay = 0;
    uint32_t batch = 0;
};

struct OverlayPickState {
    std::mutex mutex;
    std::condition_variable served;
    bool recorded = false;
    device::Operation done;
    Ref<device::Buffer> readback;
    std::vector<std::string> names;
    std::vector<OverlayBatchSource> sources;
    std::string product;
    uint32_t layer = 0;
    Float2 pixel;
    float radius = 0;
};

struct OverlayCameraSource {
    uint32_t camera = kNoPrim;
    device::Tensor pixels;
};

using OverlayContents = std::shared_ptr<const OverlayData>;

struct OverlayOutput {
    std::string_view product;
    const std::vector<std::string>* layers = nullptr;
    const device::Tensor* image = nullptr;
    std::vector<OverlayTarget> targets;
};

struct OverlayFont {
    std::array<OverlayGlyph, 256> glyphs{};
    std::vector<int8_t> kerning;
    std::vector<uint8_t> field;
    float textHeight = 0;
};

struct OverlaysImpl {
    static OverlaysImpl& Get(Overlays& overlays) {
        return *overlays.m_impl;
    }

    bool Outlined() const;
    std::vector<std::shared_ptr<OverlayPickState>> TakePicks(std::span<const std::string> products);

    device::Device& device;
    mutable std::mutex mutex;
    std::map<std::string, OverlayContents, std::less<>> layers;
    std::vector<std::shared_ptr<OverlayPickState>> picks;
};

const OverlayFont& OverlayTextFont();
std::vector<uint8_t> OverlayCodes(std::string_view text);

class OverlayPass {
  public:
    explicit OverlayPass(device::Device& device);

    static OverlayContents Share(const Overlay& overlay);
    static void Serve(std::span<const std::shared_ptr<OverlayPickState>> picks,
                      const device::Operation& done);

    std::vector<OverlayContents> Draw(device::CommandEncoder& encoder, const Snapshot& snapshot,
                                      uint32_t worldCount, const OverlaysImpl& overlays,
                                      std::span<const OverlayOutput> outputs,
                                      std::span<const OverlayCameraSource> cameraImages,
                                      std::span<const std::shared_ptr<OverlayPickState>> picks);
    static void CheckDevice(const OverlayData& overlay, device::Device& device);

    void Record(device::CommandEncoder& encoder, const Snapshot& snapshot, uint32_t worldCount,
                std::span<const OverlayTarget> targets, std::span<const OverlayContents> overlays,
                std::span<const OverlayCameraSource> cameraImages = {},
                std::span<const OverlayPickRequest> picks = {});

    const std::vector<OverlayBatchSource>& BatchSources() const {
        return m_batchSources;
    }

    uint32_t PairCapacity() const {
        return m_pairCapacity;
    }

  private:
    struct Uploaded {
        std::weak_ptr<const OverlayData> contents;
        std::vector<Ref<device::Buffer>> buffers;
    };

    struct Grouping {
        Ref<device::Buffer> keys;
        Ref<device::Buffer> order;
        Ref<device::Buffer> starts;
    };

    using GroupKey = std::pair<uint32_t, uint32_t>;

    void AddBatches(const OverlayContents& overlay, uint32_t index, uint32_t worldCount,
                    std::vector<OverlayBatch>& batches, uint32_t& firstItem);
    void Unite(OverlayBatch& gpu, uint32_t rows, uint32_t groups, uint64_t prims,
               uint32_t primStride, uint64_t groupIds, uint32_t groupStride);
    void RecordUnions(device::CommandEncoder& encoder, const Snapshot& snapshot);
    void Group(OverlayBatch& gpu, GroupKey key, uint32_t rows, uint32_t worldCount, uint64_t worlds,
               uint32_t worldStride, uint64_t itemCounts);
    void RecordGroups(device::CommandEncoder& encoder);
    void Reserve(Ref<device::Buffer>& buffer, uint64_t bytes, const char* label);

    device::Device& m_device;
    Ref<device::Program> m_program;
    Ref<device::Program> m_maskProgram;
    Ref<device::Buffer> m_unused;
    std::optional<algorithms::Scan> m_scan;
    std::optional<algorithms::RadixSort> m_sort;
    uint32_t m_workCapacity = 0;
    uint32_t m_pairCapacity = 0;
    std::map<const OverlayData*, Uploaded> m_uploads;
    Ref<device::Buffer> m_views;
    Ref<device::Buffer> m_batches;
    Ref<device::Buffer> m_records;
    Ref<device::Buffer> m_tileCounts;
    Ref<device::Buffer> m_offsets;
    Ref<device::Buffer> m_keys;
    Ref<device::Buffer> m_values;
    Ref<device::Buffer> m_occupied;
    Ref<device::Buffer> m_pairCount;
    Ref<device::Buffer> m_pairsNeeded;
    std::optional<algorithms::RadixSort> m_groupSort;
    uint32_t m_groupSortCapacity = 0;
    std::vector<Grouping> m_groupings;
    std::vector<OverlayGroupParams> m_groupParams;
    std::vector<GroupKey> m_groupKeys;
    std::vector<GroupKey> m_measuredKeys;
    std::map<GroupKey, uint32_t> m_groupStrides;
    Ref<device::Buffer> m_groupNeeds;
    Ref<device::Buffer> m_groupNeedsRead;
    Ref<device::Buffer> m_fontField;
    Ref<device::Buffer> m_fontGlyphs;
    Ref<device::Buffer> m_fontKerning;
    std::vector<OverlayUniteParams> m_uniteParams;
    std::vector<Ref<device::Buffer>> m_boxes;
    std::vector<OverlayOutline> m_outlines;
    Ref<device::Buffer> m_outlineBuffer;
    Ref<device::Buffer> m_outlineMasks;
    Ref<device::Buffer> m_outlineDepths;
    Ref<device::Buffer> m_fills;
    Ref<device::Buffer> m_fillCount;
    Ref<device::Buffer> m_cameraImages;
    Ref<device::Buffer> m_pickQueries;
    Ref<device::Buffer> m_pickResults;
    std::vector<OverlayBatchSource> m_batchSources;
};

}
