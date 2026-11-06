// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "ernie/bvh/common.h"
#include "ernie/bvh/kernels.h"
#include "ernie/bvh/kernels/instances.h"
#include "ernie/device/accel.h"
#include "ernie/device/device.h"
#include "ernie/device/kernels/shared.h"
#include "ernie/device/program.h"
#include "ernie/device/queue.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie::bvh {

using device::Accel;
using device::Buffer;
using device::BufferRange;
using device::BufferUsage;
using device::CommandEncoder;
using device::Device;
using device::Traversal;

namespace {

class HardwareBlas final : public Blas {
  public:
    static constexpr Traversal kFormat = Traversal::Hardware;

    HardwareBlas(Device& device, const Geometry& geometry)
        : Blas(kFormat), m_input(ToAccelInput(geometry)), m_buffers(GeometryBuffers(geometry)),
          m_accel(device.CreateAccel(m_input, "blas")) {}

    uint64_t Handle() const override {
        return m_accel->Address();
    }

    const device::AccelInput& GetInput() const {
        return m_input;
    }

    Accel& GetAccel() const {
        return *m_accel;
    }

  private:
    device::AccelInput m_input;
    std::vector<Ref<Buffer>> m_buffers;
    Ref<Accel> m_accel;
};

class HardwareTlas final : public Tlas {
  public:
    static constexpr Traversal kFormat = Traversal::Hardware;

    HardwareTlas(Device& device, std::span<const uint32_t> capacities) : Tlas(kFormat, capacities) {
        const std::vector<uint32_t> offsets = Offsets(capacities);
        m_instances = device.CreateBuffer(
            {.bytes = std::max<uint64_t>(offsets.back(), 1) * sizeof(InstanceRecord),
             .usage = BufferUsage::AccelBuildInput,
             .label = "tlas instances"});
        m_offsets = device.CreateBuffer(
            {.bytes = offsets.size() * sizeof(uint32_t), .label = "tlas offsets"});
        device.GetQueue().Write(*m_offsets, offsets.data());

        std::vector<uint64_t> handles;
        for (uint32_t w = 0; w < capacities.size(); ++w) {
            const device::AccelInstances world = {
                .instances = m_instances->Slice(uint64_t(offsets[w]) * sizeof(InstanceRecord),
                                                uint64_t(capacities[w]) * sizeof(InstanceRecord)),
                .count = capacities[w]};
            m_accels.push_back(device.CreateAccel(world, "tlas"));
            m_worlds.push_back(world);
            handles.push_back(m_accels.back()->Address());
        }

        SetHandles(device, std::move(handles));
    }

    Accel& GetAccel(uint32_t world) const {
        return *m_accels[world];
    }

    const device::AccelInstances& GetWorld(uint32_t world) const {
        return m_worlds[world];
    }

    Buffer& Instances() const {
        return *m_instances;
    }

    Buffer& InstanceOffsets() const {
        return *m_offsets;
    }

  private:
    std::vector<Ref<Accel>> m_accels;
    std::vector<device::AccelInstances> m_worlds;
    Ref<Buffer> m_instances;
    Ref<Buffer> m_offsets;
};

class HardwareBuilder final : public Builder {
  public:
    explicit HardwareBuilder(Device& device) : Builder(device) {}

    Traversal Format() const override {
        return Traversal::Hardware;
    }

    void Prepare() override {
        Masking();
    }

    Ref<Blas> CreateBlas(const Geometry& geometry) override {
        return Ref<Blas>(new HardwareBlas(m_device, geometry));
    }

    Ref<Tlas> CreateTlas(std::span<const uint32_t> capacities) override {
        return Ref<Tlas>(new HardwareTlas(m_device, capacities));
    }

    void Build(CommandEncoder& encoder, std::span<const Ref<Blas>> blases) override {
        BuildBlases("Build", encoder, blases, false);
    }

    void Refit(CommandEncoder& encoder, std::span<const Ref<Blas>> blases) override {
        BuildBlases("Refit", encoder, blases, true);
    }

    void Build(CommandEncoder& encoder, Tlas& tlas, BufferRange instances,
               std::optional<BufferRange> counts, std::span<const uint32_t> worlds) override {
        BuildWorlds("Build", encoder, tlas, instances, counts, worlds, false);
    }

    void Refit(CommandEncoder& encoder, Tlas& tlas, BufferRange instances,
               std::optional<BufferRange> counts, std::span<const uint32_t> worlds) override {
        BuildWorlds("Refit", encoder, tlas, instances, counts, worlds, true);
    }

  private:
    void BuildBlases(const char* what, CommandEncoder& encoder, std::span<const Ref<Blas>> blases,
                     bool update) {
        ERNIE_ZONE("Bvh/HardwareBlases");
        static const profiler::ZoneId buildZone = profiler::ZoneId::Named("Bvh/BlasBuild");
        static const profiler::ZoneId refitZone = profiler::ZoneId::Named("Bvh/BlasRefit");
        std::vector<const HardwareBlas*> hardware;
        std::vector<uint64_t> sizes;
        for (const Ref<Blas>& blas : blases) {
            hardware.push_back(&RequireFormat<HardwareBlas>(what, *blas));
            if (update) {
                RequireRefittable(what, hardware.back()->GetInput());
            }
            const device::AccelSizes& accel = hardware.back()->GetAccel().Sizes();
            sizes.push_back(update ? accel.updateScratch : accel.scratch);
        }
        if (hardware.empty()) {
            return;
        }

        Log(LogLevel::Debug, "{} {} hardware BLASes", update ? "refitting" : "building",
            hardware.size());
        const std::vector<BufferRange> scratch = Scratches(sizes);
        const device::ZoneScope zone(encoder, update ? refitZone : buildZone);
        encoder.Barrier();
        encoder.Concurrent(true);
        for (uint32_t b = 0; b < hardware.size(); ++b) {
            encoder.BuildAccel(hardware[b]->GetAccel(), hardware[b]->GetInput(), scratch[b],
                               update);
        }
        encoder.Concurrent(false);
    }

    void BuildWorlds(const char* what, CommandEncoder& encoder, const Tlas& structure,
                     BufferRange instances, std::optional<BufferRange> counts,
                     std::span<const uint32_t> worlds, bool update) {
        ERNIE_ZONE("Bvh/HardwareTlas");
        static const profiler::ZoneId buildZone = profiler::ZoneId::Named("Bvh/TlasBuild");
        static const profiler::ZoneId refitZone = profiler::ZoneId::Named("Bvh/TlasRefit");
        const auto& tlas = RequireFormat<HardwareTlas>(what, structure);
        RequireInstances(what, tlas, instances);
        const std::vector<uint32_t> selected = SelectWorlds(what, tlas, worlds);
        const std::vector<uint32_t> offsets = Offsets(tlas.Capacities());
        const uint64_t total = Total(tlas.Capacities());
        Log(LogLevel::Debug, "{} {} of {} hardware TLAS worlds", update ? "refitting" : "building",
            selected.size(), tlas.Count());
        const device::ZoneScope zone(encoder, update ? refitZone : buildZone);
        if (total > 0) {
            encoder.Barrier();
            if (counts) {
                encoder.Dispatch(Masking().Entry("MaskInstances"),
                                 MaskInstancesParams{.source = {instances.Address()},
                                                     .destination = {tlas.Instances().Address()},
                                                     .offsets = {tlas.InstanceOffsets().Address()},
                                                     .counts = {counts->Address()},
                                                     .total = uint32_t(total),
                                                     .worlds = tlas.Count()},
                                 {.x = device::GroupCount(total, kMaskInstancesThreads)});
            } else if (selected.size() == tlas.Count()) {
                const uint64_t bytes = total * sizeof(InstanceRecord);
                encoder.Copy(instances.Slice(0, bytes), tlas.Instances().Slice(0, bytes));
            } else {
                for (const uint32_t w : selected) {
                    const uint64_t offset = uint64_t(offsets[w]) * sizeof(InstanceRecord);
                    const uint64_t bytes = uint64_t(tlas.Capacities()[w]) * sizeof(InstanceRecord);
                    if (bytes > 0) {
                        encoder.Copy(instances.Slice(offset, bytes),
                                     tlas.Instances().Slice(offset, bytes));
                    }
                }
            }
        }

        std::vector<uint64_t> sizes;
        for (const uint32_t w : selected) {
            const device::AccelSizes& accel = tlas.GetAccel(w).Sizes();
            sizes.push_back(update ? accel.updateScratch : accel.scratch);
        }

        const std::vector<BufferRange> scratch = Scratches(sizes);
        encoder.Barrier();
        encoder.Concurrent(true);
        for (uint32_t i = 0; i < selected.size(); ++i) {
            encoder.BuildAccel(tlas.GetAccel(selected[i]), tlas.GetWorld(selected[i]), scratch[i],
                               update);
        }
        encoder.Concurrent(false);
    }

    std::vector<BufferRange> Scratches(std::span<const uint64_t> sizes) {
        std::vector<uint64_t> offsets;
        std::vector<uint64_t> aligned;
        uint64_t total = 0;
        for (const uint64_t size : sizes) {
            offsets.push_back(total);
            aligned.push_back(RoundUp(std::max<uint64_t>(size, 1), device::kBufferAlignment));
            total += aligned.back();
        }

        Buffer& buffer = Reserve(m_device, m_scratch,
                                 {.bytes = std::max<uint64_t>(total, 256),
                                  .usage = BufferUsage::AccelStorage,
                                  .label = "accel scratch"});

        std::vector<BufferRange> ranges;
        for (uint32_t i = 0; i < sizes.size(); ++i) {
            ranges.push_back(buffer.Slice(offsets[i], aligned[i]));
        }
        return ranges;
    }

    device::Program& Masking() {
        if (!m_masking) {
            m_masking = m_device.CreateProgram({.modules = {"ernie.bvh.kernels.instances"},
                                                .library = &kBvhKernels,
                                                .entries = {"MaskInstances"},
                                                .label = "tlas instances"});
        }
        return *m_masking;
    }

    Ref<Buffer> m_scratch;
    Ref<device::Program> m_masking;
};

}

std::unique_ptr<Builder> CreateHardwareBuilder(Device& device) {
    return std::make_unique<HardwareBuilder>(device);
}

}
