// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/adapter.h"
#include "ernie/device/resource.h"
#include "ernie/export.h"

namespace ernie {
class DiskCache;
struct KernelLibrary;
struct DeviceGlobals;
}

namespace ernie::device {

struct KernelCode;
class Program;

struct Size3 {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;

    bool operator==(const Size3&) const = default;
};

inline constexpr uint32_t kMaxGroupsPerAxis = 65535;

constexpr Size3 FoldGroups(uint32_t count) {
    const uint64_t plane = uint64_t(kMaxGroupsPerAxis) * kMaxGroupsPerAxis;
    const uint32_t z = uint32_t((count + plane - 1) / plane);
    const uint64_t perLayer = (uint64_t(count) + z - 1) / z;
    const uint32_t y = uint32_t((perLayer + kMaxGroupsPerAxis - 1) / kMaxGroupsPerAxis);
    const uint32_t x = uint32_t((perLayer + y - 1) / y);
    return {x, y, z};
}

constexpr uint32_t GroupCount(uint64_t threadCount, uint32_t threadsPerGroup) {
    return uint32_t((threadCount + threadsPerGroup - 1) / threadsPerGroup);
}

ERNIE_API void RegisterFile(std::string path, std::string source);

struct ProgramDesc {
    std::vector<std::string> modules;
    const KernelLibrary* library = nullptr;
    std::vector<std::string> entries;
    bool loadEntries = true;
    std::string label = "program";
    Traversal traversal = Traversal::Auto;
};

struct KernelStat {
    std::string name;
    std::string value;
};

struct KernelStats {
    double compileMs = 0;
    bool cached = false;
    std::optional<uint32_t> registers;
    std::optional<uint32_t> spillBytes;
    std::optional<uint32_t> sharedBytes;
    std::vector<KernelStat> driver;
};

class ERNIE_API Kernel {
  public:
    virtual ~Kernel();
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    const std::string& Name() const {
        return m_name;
    }

    Program& GetProgram() const {
        return *m_program;
    }

    bool Ready() const;

    void Wait() const;

    Size3 GroupSize() const;

    uint32_t ParamBytes() const;

    bool Foldable() const;

    const KernelStats& Stats() const;

  protected:
    Kernel();

    virtual KernelStats Load(const KernelCode& code, const DiskCache& cache,
                             const DeviceGlobals& globals) = 0;

  private:
    friend class Program;
    struct State;

    std::string m_name;
    Program* m_program = nullptr;
    std::unique_ptr<State> m_state;
};

ERNIE_API void WaitForBuilds();

class ERNIE_API Program : public Resource {
  public:
    ~Program() override;

    Kernel& Entry(std::string_view name);

  private:
    friend class Device;
    struct State;

    Program(Device& device, const ProgramDesc& desc,
            std::shared_ptr<const std::map<std::string, std::string, std::less<>>> sources);

    Kernel& Request(std::string_view name, bool wanted);

    void QueueLoad(Kernel& kernel);

    void Load(Kernel& kernel, const KernelCode& code, const DiskCache& cache,
              std::chrono::steady_clock::time_point start);

    void Finish(Kernel& kernel, const std::string& label);

    std::unique_ptr<State> m_state;
};

}
