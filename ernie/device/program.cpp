// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/program.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ernie/device/device.h"
#include "ernie/device/slang/compiler.h"
#include "ernie/device/slang/debug.h"
#include "ernie/util/disk_cache.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/thread_pool.h"

namespace ernie::device {
namespace {

struct Registry {
    std::mutex mutex;
    std::shared_ptr<const SourceFiles> files = std::make_shared<SourceFiles>();
};

Registry& Files() {
    static Registry registry;
    return registry;
}

std::shared_ptr<const SourceFiles> RegisteredFiles() {
    Registry& registry = Files();
    const std::lock_guard lock(registry.mutex);
    return registry.files;
}

ThreadPool& Pool() {
    static ThreadPool& pool =
        *new ThreadPool(std::clamp(std::thread::hardware_concurrency() / 2, 1u, 4u));
    return pool;
}

struct Builds {
    std::mutex mutex;
    std::condition_variable done;
    uint64_t pending = 0;
    uint64_t queued = 0;
    uint64_t finished = 0;
    uint64_t cached = 0;
    std::multiset<std::string> building;
};

Builds& PendingBuilds() {
    static Builds& builds = *new Builds;
    return builds;
}

void FinishBuild(const std::string& name = {}, bool cached = false) {
    Builds& builds = PendingBuilds();
    {
        const std::lock_guard lock(builds.mutex);
        --builds.pending;
        ++builds.finished;
        builds.cached += cached ? 1 : 0;
        if (const auto found = builds.building.find(name); found != builds.building.end()) {
            builds.building.erase(found);
        }
    }
    builds.done.notify_all();
}

std::string ResourceUsage(const KernelStats& stats) {
    std::string usage;
    if (stats.registers) {
        usage += fmt::format(", {} registers", *stats.registers);
    }
    if (stats.spillBytes) {
        usage += fmt::format(", {} bytes of scratch memory", *stats.spillBytes);
    }
    return usage;
}

}

void RegisterFile(std::string path, std::string source) {
    Registry& registry = Files();
    const std::lock_guard lock(registry.mutex);
    auto files = std::make_shared<SourceFiles>(*registry.files);
    (*files)[std::move(path)] = std::move(source);
    registry.files = std::move(files);
}

namespace {

struct BuildJob {
    enum class Stage { Compiling, Compiled, Loading, Done, Abandoned };

    std::mutex mutex;
    std::condition_variable loaded;
    Stage stage = Stage::Compiling;
    bool wanted = false;
    std::optional<KernelCode> code;
};

}

struct Kernel::State {
    std::shared_ptr<BuildJob> job = std::make_shared<BuildJob>();
    std::promise<KernelShape> promise;
    std::shared_future<KernelShape> compiled = promise.get_future().share();
    KernelStats stats;

    const KernelShape& Get(const Kernel& kernel) const {
        if (compiled.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            if (const std::function<void()> queued = Pool().Take(&kernel)) {
                queued();
            }
        }
        return compiled.get();
    }
};

Kernel::Kernel() : m_state(std::make_unique<State>()) {}

Kernel::~Kernel() = default;

bool Kernel::Ready() const {
    return m_state->compiled.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

void Kernel::Wait() const {
    m_state->Get(*this);
}

Size3 Kernel::GroupSize() const {
    return m_state->Get(*this).groupSize;
}

uint32_t Kernel::ParamBytes() const {
    return m_state->Get(*this).paramBytes;
}

bool Kernel::Foldable() const {
    return m_state->Get(*this).foldable;
}

const KernelStats& Kernel::Stats() const {
    m_state->Get(*this);
    return m_state->stats;
}

struct Device::ProgramCache {
    using Key =
        std::tuple<const KernelLibrary*, std::vector<std::string>, Traversal, const SourceFiles*>;

    std::mutex mutex;
    std::map<Key, Ref<Program>> programs;
};

struct Program::State {
    const KernelLibrary* library = nullptr;
    std::shared_ptr<const SourceFiles> sources;
    std::vector<std::string> modules;
    Traversal traversal = Traversal::Software;
    std::mutex mutex;
    std::map<std::string, std::unique_ptr<Kernel>, std::less<>> kernels;
};

Program::Program(Device& device, const ProgramDesc& desc,
                 std::shared_ptr<const SourceFiles> sources)
    : Resource(device, desc.label), m_state(std::make_unique<State>()) {
    m_state->library = desc.library;
    m_state->sources = std::move(sources);
    for (const std::string& name : desc.modules) {
        if (!HasModule(m_state->library, *m_state->sources, name)) {
            throw Error("program \"{}\" names module {}, which is neither registered nor in "
                        "Ernie's kernel library",
                        desc.label, name);
        }
    }
    m_state->modules = desc.modules;

    m_state->traversal = desc.traversal == Traversal::Auto ? device.GetTraversal() : desc.traversal;
    if (m_state->traversal == Traversal::Hardware && device.GetTraversal() != Traversal::Hardware) {
        throw UnsupportedError("program \"{}\" traverses rays in hardware, which {} doesn't",
                               desc.label, device.Adapter().alias);
    }
}

Program::~Program() {
    for (const auto& [name, kernel] : m_state->kernels) {
        if (Pool().Cancel(kernel.get())) {
            FinishBuild();
            continue;
        }
        BuildJob& job = *kernel->m_state->job;
        std::unique_lock lock(job.mutex);
        if (job.stage == BuildJob::Stage::Compiling) {
            job.stage = BuildJob::Stage::Abandoned;
            continue;
        }
        if (job.stage == BuildJob::Stage::Compiled) {
            continue;
        }
        job.loaded.wait(lock, [&] { return job.stage == BuildJob::Stage::Done; });
    }
}

Kernel& Program::Entry(std::string_view name) {
    return Request(name, true);
}

Kernel& Program::Request(std::string_view name, bool wanted) {
    const std::lock_guard lock(m_state->mutex);
    if (const auto found = m_state->kernels.find(name); found != m_state->kernels.end()) {
        Kernel& kernel = *found->second;
        if (wanted) {
            BuildJob& job = *kernel.m_state->job;
            const std::lock_guard staging(job.mutex);
            job.wanted = true;
            if (job.stage == BuildJob::Stage::Compiled) {
                job.stage = BuildJob::Stage::Loading;
                QueueLoad(kernel);
            }
        }
        return kernel;
    }

    std::unique_ptr<Kernel> made = GetDevice().NewKernel();
    made->m_name = name;
    made->m_program = this;
    made->m_state->job->wanted = wanted;
    Kernel& kernel = *m_state->kernels.emplace(std::string(name), std::move(made)).first->second;
    {
        Builds& builds = PendingBuilds();
        const std::lock_guard pending(builds.mutex);
        ++builds.pending;
        ++builds.queued;
    }
    const AdapterInfo& adapter = GetDevice().Adapter();
    const std::string label = Label() + ": " + kernel.Name();
    Pool().Add(&kernel,
               [this, &kernel, label, job = kernel.m_state->job, backend = adapter.backend,
                traversal = m_state->traversal, sampling = GetDevice().GetSampling(),
                debugInfo = GetDevice().m_kernelDebugInfo, library = m_state->library,
                sources = m_state->sources, modules = m_state->modules, entry = kernel.Name(),
                program = Label(), directory = GetDevice().KernelCacheDirectory(),
                narrow = adapter.features.waveSizeMin < 8] {
                   {
                       Builds& builds = PendingBuilds();
                       const std::lock_guard building(builds.mutex);
                       builds.building.insert(label);
                   }
                   const auto start = std::chrono::steady_clock::now();
                   const DiskCache slang(directory / "slang");
                   const DiskCache native(directory / BackendName(backend));
                   std::optional<KernelCode> code;
                   std::exception_ptr failure;
                   try {
                       code = Compile(backend, traversal, sampling, debugInfo, library, *sources,
                                      modules, entry, program, slang, narrow);
                   } catch (...) {
                       failure = std::current_exception();
                   }
                   {
                       const std::lock_guard staging(job->mutex);
                       if (job->stage == BuildJob::Stage::Abandoned) {
                           FinishBuild(label, false);
                           return;
                       }
                       if (!failure && !job->wanted) {
                           const bool cached = code->cached;
                           job->code = std::move(code);
                           job->stage = BuildJob::Stage::Compiled;
                           FinishBuild(label, cached);
                           return;
                       }
                       job->stage = BuildJob::Stage::Loading;
                   }
                   if (failure) {
                       kernel.m_state->promise.set_exception(failure);
                   } else {
                       Load(kernel, *code, native, start);
                   }
                   Finish(kernel, label);
               });
    return kernel;
}

void Program::QueueLoad(Kernel& kernel) {
    {
        Builds& builds = PendingBuilds();
        const std::lock_guard pending(builds.mutex);
        ++builds.pending;
        ++builds.queued;
    }
    Pool().Add(&kernel, [this, &kernel, label = Label() + ": " + kernel.Name(),
                         directory = GetDevice().KernelCacheDirectory() /
                                     BackendName(GetDevice().Adapter().backend)] {
        {
            Builds& builds = PendingBuilds();
            const std::lock_guard building(builds.mutex);
            builds.building.insert(label);
        }
        BuildJob& job = *kernel.m_state->job;
        Load(kernel, *job.code, DiskCache(directory), std::chrono::steady_clock::now());
        job.code.reset();
        Finish(kernel, label);
    });
}

void Program::Finish(Kernel& kernel, const std::string& label) {
    const bool cached = kernel.Ready() && kernel.m_state->stats.cached;
    BuildJob& job = *kernel.m_state->job;
    {
        const std::lock_guard staging(job.mutex);
        job.stage = BuildJob::Stage::Done;
    }
    job.loaded.notify_all();
    FinishBuild(label, cached);
}

void Program::Load(Kernel& kernel, const KernelCode& code, const DiskCache& cache,
                   std::chrono::steady_clock::time_point start) {
    try {
        const AdapterInfo& adapter = GetDevice().Adapter();
        if (code.shape.paramBytes > adapter.limits.maxParamBytes) {
            throw Error("kernel {} of program \"{}\" takes {} bytes of parameters; {} takes at "
                        "most {}",
                        kernel.Name(), Label(), code.shape.paramBytes, adapter.alias,
                        adapter.limits.maxParamBytes);
        }

        GetDevice().Debug().AddKernel(kernel.Name(), code.strings);
        const auto loading = std::chrono::steady_clock::now();
        KernelStats stats = kernel.Load(code, cache, GetDevice().Globals());
        const auto loaded = std::chrono::steady_clock::now();
        stats.compileMs = std::chrono::duration<double, std::milli>(loaded - start).count();
        stats.cached = stats.cached && code.cached;
        Log(LogLevel::Debug,
            "kernel {} of program \"{}\" for {} {} in {:.1f} ms, {:.1f} ms of it in {}{}",
            kernel.Name(), Label(), adapter.alias,
            stats.cached ? "loaded from the cache" : "compiled", stats.compileMs,
            std::chrono::duration<double, std::milli>(loaded - loading).count(),
            BackendName(adapter.backend), ResourceUsage(stats));

        kernel.m_state->stats = std::move(stats);
        kernel.m_state->promise.set_value(code.shape);
    } catch (...) {
        kernel.m_state->promise.set_exception(std::current_exception());
    }
}

BuildProgress CurrentBuilds() {
    Builds& builds = PendingBuilds();
    const std::lock_guard lock(builds.mutex);
    return {builds.queued, builds.finished, builds.cached,
            std::vector<std::string>(builds.building.begin(), builds.building.end())};
}

void WaitForBuilds() {
    Builds& builds = PendingBuilds();
    std::unique_lock lock(builds.mutex);
    builds.done.wait(lock, [&] { return builds.pending == 0; });
}

Ref<Program> Device::CreateProgram(const ProgramDesc& desc) {
    static std::mutex creating;
    std::shared_ptr<ProgramCache> cache;
    {
        const std::lock_guard lock(creating);
        if (!m_programs) {
            m_programs = std::make_shared<ProgramCache>();
        }
        cache = m_programs;
    }
    const std::shared_ptr<const SourceFiles> sources = RegisteredFiles();
    const Traversal traversal = desc.traversal == Traversal::Auto ? m_traversal : desc.traversal;
    const ProgramCache::Key key{desc.library, desc.modules, traversal, sources.get()};
    Ref<Program> program;
    {
        const std::lock_guard lock(cache->mutex);
        if (const auto found = cache->programs.find(key); found != cache->programs.end()) {
            program = found->second;
        } else {
            program = Ref<Program>(new Program(*this, desc, sources));
            Adopt(*program);
            cache->programs.emplace(key, program);
        }
    }
    for (const std::string& entry : desc.entries) {
        program->Request(entry, desc.loadEntries);
    }
    return program;
}

}
