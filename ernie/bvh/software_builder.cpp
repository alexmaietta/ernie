// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/bvh/software_builder.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "ernie/bvh/common.h"
#include "ernie/bvh/kernels.h"
#include "ernie/bvh/kernels/instances.h"
#include "ernie/bvh/kernels/refit.h"
#include "ernie/bvh/lbvh.h"
#include "ernie/device/device.h"
#include "ernie/device/program.h"
#include "ernie/device/queue.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie::bvh {

using device::Buffer;
using device::BufferRange;
using device::CommandEncoder;
using device::Device;
using device::Traversal;

namespace {

class SoftwareBuilder final : public Builder {
  public:
    explicit SoftwareBuilder(Device& device) : Builder(device) {}

    Traversal Format() const override {
        return Traversal::Software;
    }

    void Prepare() override {
        Fit(1, 1);
        Updating();
    }

    Ref<Blas> CreateBlas(const Geometry& geometry) override {
        return Ref<Blas>(new SoftwareBlas(m_device, geometry));
    }

    Ref<Tlas> CreateTlas(std::span<const uint32_t> capacities) override {
        return Ref<Tlas>(new SoftwareTlas(m_device, capacities));
    }

    void Build(CommandEncoder& encoder, std::span<const Ref<Blas>> blases) override {
        ERNIE_ZONE("Bvh/SoftwareBlases");
        static const profiler::ZoneId zone = profiler::ZoneId::Named("Bvh/BlasBuild");
        if (blases.empty()) {
            return;
        }
        Log(LogLevel::Debug, "building {} software BLASes", blases.size());
        const device::ZoneScope scope(encoder, zone);
        for (const uint32_t kind : {kBvhTriangles, kBvhBoxes}) {
            std::vector<Geometry> geometries;
            std::vector<BufferRange> outputs;
            uint64_t primitives = 0;
            for (const Ref<Blas>& blas : blases) {
                const auto& software = RequireFormat<SoftwareBlas>("Build", *blas);
                const Geometry& geometry = software.GetGeometry();
                if (Kind(geometry) == kind) {
                    geometries.push_back(geometry);
                    outputs.emplace_back(software.Storage());
                    primitives += PrimitiveCount(geometry);
                }
            }
            if (!geometries.empty()) {
                Fit(primitives, geometries.size()).Build(encoder, geometries, outputs);
            }
        }
    }

    void Refit(CommandEncoder& encoder, std::span<const Ref<Blas>> blases) override {
        ERNIE_ZONE("Bvh/SoftwareBlases");
        static const profiler::ZoneId zone = profiler::ZoneId::Named("Bvh/BlasRefit");
        if (blases.empty()) {
            return;
        }
        Log(LogLevel::Debug, "refitting {} software BLASes", blases.size());
        const device::ZoneScope scope(encoder, zone);
        for (const uint32_t kind : {kBvhTriangles, kBvhBoxes}) {
            std::vector<RefitSegment> segments;
            for (const Ref<Blas>& blas : blases) {
                const auto& software = RequireFormat<SoftwareBlas>("Refit", *blas);
                const Geometry& geometry = software.GetGeometry();
                RequireRefittable("Refit", ToAccelInput(geometry));
                if (Kind(geometry) == kind) {
                    segments.push_back({software.Handle(), 0,
                                        LbvhEntry(Updating(), "Refit", geometry),
                                        PrimitiveCount(geometry)});
                }
            }
            RefitSegments(encoder, kind, 0, segments);
        }
    }

    void Build(CommandEncoder& encoder, Tlas& tlas, BufferRange instances,
               std::optional<BufferRange> counts, std::span<const uint32_t> worlds) override {
        ERNIE_ZONE("Bvh/SoftwareTlas");
        static const profiler::ZoneId zone = profiler::ZoneId::Named("Bvh/TlasBuild");
        const auto& software = RequireFormat<SoftwareTlas>("Build", tlas);
        RequireInstances("Build", tlas, instances);
        const std::vector<uint32_t> selected = SelectWorlds("Build", tlas, worlds);
        Log(LogLevel::Debug, "building {} of {} software TLAS worlds", selected.size(),
            tlas.Count());
        const device::ZoneScope scope(encoder, zone);
        std::vector<BufferRange> outputs;
        std::vector<uint32_t> capacities;
        for (const uint32_t w : selected) {
            outputs.push_back(software.World(w));
            capacities.push_back(tlas.Capacities()[w]);
        }
        if (selected.size() == tlas.Count()) {
            Fit(Total(capacities), capacities.size())
                .Build(encoder, instances, capacities, counts, outputs);
            return;
        }

        const std::vector<uint32_t> offsets = Offsets(tlas.Capacities());
        const uint64_t total = std::max<uint64_t>(Total(capacities), 1);
        Buffer& compact =
            Reserve(m_device, m_compact,
                    {.bytes = total * sizeof(InstanceRecord), .label = "tlas world instances"});
        Buffer& compactCounts =
            Reserve(m_device, m_compactCounts,
                    {.bytes = selected.size() * sizeof(uint32_t), .label = "tlas world counts"});
        std::vector<uint32_t> sourceOffsets;
        for (const uint32_t w : selected) {
            sourceOffsets.push_back(offsets[w]);
        }
        Tables tables;
        const uint64_t sourceStart = tables.Add(sourceOffsets);
        const uint64_t offsetsStart = tables.Add(Offsets(capacities));
        const uint64_t worldsStart = tables.Add(selected);
        const Ref<Buffer> table = tables.Upload(m_device, "tlas world gather");
        const GatherWorldsParams params = {.source = {instances.Address()},
                                           .destination = {compact.Address()},
                                           .sourceOffsets = {table->Address() + sourceStart},
                                           .offsets = {table->Address() + offsetsStart},
                                           .worlds = {table->Address() + worldsStart},
                                           .counts = {counts ? counts->Address() : 0},
                                           .gatheredCounts = {compactCounts.Address()},
                                           .total = uint32_t(Total(capacities)),
                                           .selected = uint32_t(selected.size())};
        encoder.Barrier();
        encoder.Dispatch(Updating().Entry("GatherWorlds"), params,
                         {.x = device::GroupCount(std::max<uint64_t>(params.total, params.selected),
                                                  kGatherWorldsThreads)});
        encoder.Barrier();
        std::optional<BufferRange> compactRange;
        if (counts) {
            compactRange = compactCounts.Slice(0, selected.size() * sizeof(uint32_t));
        }
        Fit(Total(capacities), capacities.size())
            .Build(encoder, compact.Slice(0, total * sizeof(InstanceRecord)), capacities,
                   compactRange, outputs);
    }

    void Refit(CommandEncoder& encoder, Tlas& tlas, BufferRange instances,
               std::optional<BufferRange>, std::span<const uint32_t> worlds) override {
        ERNIE_ZONE("Bvh/SoftwareTlas");
        static const profiler::ZoneId zone = profiler::ZoneId::Named("Bvh/TlasRefit");
        RequireFormat<SoftwareTlas>("Refit", tlas);
        RequireInstances("Refit", tlas, instances);
        const std::vector<uint32_t> offsets = Offsets(tlas.Capacities());
        const std::vector<uint32_t> selected = SelectWorlds("Refit", tlas, worlds);
        Log(LogLevel::Debug, "refitting {} of {} software TLAS worlds", selected.size(),
            tlas.Count());
        const device::ZoneScope scope(encoder, zone);
        std::vector<RefitSegment> segments;
        for (const uint32_t w : selected) {
            segments.push_back({tlas.Handle(w), offsets[w], {}, tlas.Capacities()[w]});
        }
        RefitSegments(encoder, kBvhInstances, instances.Address(), segments);
    }

  private:
    struct RefitSegment {
        uint64_t header = 0;
        uint32_t instanceOffset = 0;
        LbvhGeometry geometry{};
        uint32_t primitives = 0;
    };

    device::Program& Updating() {
        if (!m_updating) {
            m_updating = m_device.CreateProgram(
                {.modules = {"ernie.bvh.kernels.refit", "ernie.bvh.kernels.instances"},
                 .library = &kBvhKernels,
                 .entries = {"RefitLeaves", "RefitNodes", "GatherWorlds"},
                 .label = "bvh updates"});
        }
        return *m_updating;
    }

    void RefitSegments(CommandEncoder& encoder, uint32_t kind, uint64_t instances,
                       std::span<const RefitSegment> segments) {
        if (segments.empty()) {
            return;
        }
        std::vector<uint64_t> headers;
        std::vector<LbvhGeometry> geometries;
        std::vector<uint32_t> instanceOffsets;
        std::vector<uint32_t> rooms;
        for (const RefitSegment& segment : segments) {
            headers.push_back(segment.header);
            geometries.push_back(segment.geometry);
            instanceOffsets.push_back(segment.instanceOffset);
            rooms.push_back(std::max(segment.primitives, 1u));
        }
        const std::vector<uint32_t> nodeOffsets = Offsets(rooms);
        const uint32_t nodes = nodeOffsets.back();

        Tables tables;
        const uint64_t headersStart = tables.Add(headers);
        const uint64_t geometriesStart = tables.Add(geometries);
        const uint64_t instanceOffsetsStart = tables.Add(instanceOffsets);
        const uint64_t nodeOffsetsStart = tables.Add(nodeOffsets);
        const Ref<Buffer> table = tables.Upload(m_device, "bvh refit tables");

        const uint64_t bytes = uint64_t(nodes) * 16 * sizeof(uint32_t);
        const uint64_t base =
            Reserve(m_device, m_refitScratch, {.bytes = bytes, .label = "bvh refit scratch"})
                .Address();
        const RefitParams params = {.headers = {table->Address() + headersStart},
                                    .instances = {instances},
                                    .instanceOffsets = {table->Address() + instanceOffsetsStart},
                                    .geometries = {table->Address() + geometriesStart},
                                    .nodeOffsets = {table->Address() + nodeOffsetsStart},
                                    .parents = {base},
                                    .arrivals = {base + uint64_t(nodes) * sizeof(uint32_t)},
                                    .boxes = {base + uint64_t(nodes) * 2 * sizeof(uint32_t)},
                                    .leafBoxes = {base + uint64_t(nodes) * 8 * sizeof(uint32_t)},
                                    .segments = uint32_t(segments.size()),
                                    .nodes = nodes,
                                    .kind = kind};

        const device::Size3 groups{.x = device::GroupCount(nodes, kRefitThreads)};
        device::Program& program = Updating();
        encoder.Barrier();
        encoder.Dispatch(program.Entry("RefitLeaves"), params, groups);
        encoder.Barrier();
        encoder.Dispatch(program.Entry("RefitNodes"), params, groups);
    }

    LbvhBuilder& Fit(uint64_t primitives, size_t segments) {
        if (primitives > LbvhBuilder::kMaxPrimitives) {
            throw Error("Build of {} primitives; a build takes at most {}", primitives,
                        LbvhBuilder::kMaxPrimitives);
        }
        if (segments > std::numeric_limits<uint32_t>::max()) {
            throw Error("Build of {} BVHs; a build takes at most {}", segments,
                        std::numeric_limits<uint32_t>::max());
        }

        if (!m_lbvh || m_lbvh->Capacity() < primitives || m_lbvh->Segments() < segments) {
            uint64_t capacity = primitives;
            uint64_t most = segments;
            if (m_lbvh) {
                capacity = std::max(capacity, uint64_t(m_lbvh->Capacity()) * 3 / 2);
                most = std::max(most, uint64_t(m_lbvh->Segments()) * 3 / 2);
            }
            Log(LogLevel::Debug, "the LBVH builder grows to {} primitives in {} BVHs", capacity,
                most);
            m_lbvh.emplace(
                m_device, uint32_t(std::min<uint64_t>(capacity, LbvhBuilder::kMaxPrimitives)),
                uint32_t(std::min<uint64_t>(most, std::numeric_limits<uint32_t>::max())));
        }
        return *m_lbvh;
    }

    std::optional<LbvhBuilder> m_lbvh;
    Ref<device::Program> m_updating;
    Ref<Buffer> m_refitScratch;
    Ref<Buffer> m_compact;
    Ref<Buffer> m_compactCounts;
};

}

SoftwareBlas::SoftwareBlas(Device& device, const Geometry& geometry)
    : Blas(kFormat), m_geometry(geometry), m_buffers(GeometryBuffers(geometry)),
      m_storage(device.CreateBuffer(
          {.bytes = StorageBytes(PrimitiveCount(geometry), Kind(geometry)), .label = "blas"})) {}

SoftwareTlas::SoftwareTlas(Device& device, std::span<const uint32_t> capacities)
    : Tlas(kFormat, capacities) {
    std::vector<uint64_t> handles;
    uint64_t bytes = 0;
    for (const uint32_t capacity : capacities) {
        handles.push_back(bytes);
        bytes += StorageBytes(capacity, kBvhInstances);
    }
    m_storage = device.CreateBuffer({.bytes = bytes, .label = "tlas"});
    for (uint64_t& handle : handles) {
        handle += m_storage->Address();
    }
    SetHandles(device, std::move(handles));
}

BufferRange SoftwareTlas::World(uint32_t world) const {
    return m_storage->Slice(Handle(world) - m_storage->Address(),
                            StorageBytes(Capacities()[world], kBvhInstances));
}

std::unique_ptr<Builder> CreateSoftwareBuilder(Device& device) {
    return std::make_unique<SoftwareBuilder>(device);
}

}
