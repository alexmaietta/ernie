// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/commands.h"
#include "ernie/device/operation.h"
#include "ernie/device/ref.h"
#include "ernie/device/stream.h"
#include "ernie/device/texture.h"
#include "ernie/export.h"

namespace ernie::device {

class Device;
class Queue;

class ERNIE_API Queue {
  public:
    virtual ~Queue();
    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    Device& GetDevice() const {
        return m_device;
    }

    CommandEncoder CreateEncoder();

    Operation Submit(CommandBuffer&& commands);

    Operation Submit(const CommandBuffer& commands);

    Operation Mark();

    void BeginBatch();

    Operation EndBatch();

    void Follow(Stream stream);

    void Follow(const Operation& operation);

    void Exclusive(const std::function<void()>& use);

    void Write(BufferRange dst, const void* data);
    BufferRange Upload(const void* data, uint64_t bytes);

    // Uploads one whole level of dst from Texture::LevelBytes of tightly packed texels.
    void Write(Texture& dst, uint32_t level, const void* data);

    Download ReadAsync(BufferRange src);

    void Read(BufferRange src, void* data);

    void Retire(std::shared_ptr<void> retired, const Operation& until);

    uint64_t Submitted() const;

    uint64_t Completed() const;

    void WaitIdle();

  protected:
    explicit Queue(Device& device, bool runsKernels = true);

    static std::shared_ptr<void>& Recording(const CommandBuffer& commands) {
        return commands.m_recorded.native;
    }

    class Timestamps {
      public:
        virtual ~Timestamps() = default;
        virtual bool Read(std::span<uint64_t> ns) const = 0;
    };

    virtual void Execute(const CommandBuffer& queued, const CommandBuffer& commands,
                         const Timestamps* timestamps, uint64_t value) = 0;

    virtual std::shared_ptr<Timestamps> CreateTimestamps(uint32_t count) = 0;

    virtual uint64_t Finished() const = 0;

    virtual void WaitFor(uint64_t value) const = 0;

    virtual bool SharesStreams() {
        return false;
    }

    virtual void* NativeStream() {
        return nullptr;
    }

    virtual void ExecuteNative(const CommandBuffer& queued, uint64_t value, const void* native);

    virtual void FollowQueue(Queue& other, uint64_t value);

    Operation RunNative(const void* native);

    std::unique_lock<std::mutex> LockSubmissions();

    virtual void FollowStream(void* stream);
    virtual void StreamWait(void* stream, uint64_t value);

  private:
    friend class CommandBuffer;
    friend class Device;
    friend struct Operation;
    struct State;

    void OpenRecording();
    void CloseRecording();

    Operation SubmitCommands(const CommandBuffer& commands);
    uint64_t Run(const CommandBuffer& commands, const void* native = nullptr);
    uint64_t Launch(const CommandBuffer& queued, const CommandBuffer& commands, uint64_t value,
                    const void* native);
    uint64_t RunEach(const CommandBuffer& commands);
    CommandBuffer Isolate(const CommandBuffer& commands, const Command& command);
    std::shared_ptr<Timestamps> TimestampsFor(const CommandBuffer& commands);
    void CollectZones(uint64_t finished) const;

    void Wait(uint64_t value);
    void Reach(uint64_t value);
    void Collect(uint64_t finished) const;
    void Retire(Resource& resource);
    void Retire(std::shared_ptr<void> retired, uint64_t value);
    void Close() noexcept;

    Device& m_device;
    bool m_runsKernels;
    std::unique_ptr<State> m_state;
};

}
