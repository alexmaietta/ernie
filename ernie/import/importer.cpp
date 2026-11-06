// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/import/importer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <utility>

#include "ernie/import/obj.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/text.h"
#include "ernie/util/thread_pool.h"

namespace ernie {
namespace {

struct Registry {
    std::mutex mutex;
    std::map<std::string, std::shared_ptr<Importer>, std::less<>> importers;
};

ThreadPool& Readers() {
    static ThreadPool pool(std::clamp(std::thread::hardware_concurrency() / 2, 1u, 4u));
    return pool;
}

Registry& Importers() {
    static Registry registry{.importers = {{".obj", CreateObjImporter()}}};
    return registry;
}

std::string NormalName(std::string_view name) {
    return std::filesystem::path(name).lexically_normal().generic_string();
}

}

std::filesystem::path ImportSource::Locate(std::string_view name) const {
    return (file.parent_path() / std::filesystem::path(name)).lexically_normal();
}

FileBytes ImportSource::Find(std::string_view name) const {
    const auto found = files.find(NormalName(name));
    return found == files.end() ? nullptr : found->second;
}

FileBytes ImportSource::Read(std::string_view name) const {
    if (FileBytes bytes = Find(name)) {
        return bytes;
    }
    const std::filesystem::path located = Locate(name);
    std::ifstream stream(located, std::ios::binary);
    if (!stream) {
        throw Error("Reading \"{}\": it isn't in memory and can't be opened", located.string());
    }
    const std::vector<char> text((std::istreambuf_iterator<char>(stream)),
                                 std::istreambuf_iterator<char>());
    const auto bytes = std::as_bytes(std::span(text));
    return std::make_shared<const std::vector<std::byte>>(bytes.begin(), bytes.end());
}

FileBytes ImportSource::ReadEntry() const {
    return Read(file.filename().generic_string());
}

Importer::~Importer() = default;

class ImportState : public std::enable_shared_from_this<ImportState> {
  public:
    ImportState(std::string reader = {}, std::string source = {}, std::string path = {},
                const ImportDesc& desc = {})
        : m_reader(std::move(reader)), m_source(std::move(source)), m_path(std::move(path)),
          m_callback(desc.progress), m_up(desc.up), m_metersPerUnit(desc.metersPerUnit) {}

    std::optional<UpAxis> Up() const {
        return m_up;
    }

    std::optional<float> MetersPerUnit() const {
        return m_metersPerUnit;
    }

    void Report(std::string_view stage, uint64_t done, uint64_t total) {
        if (m_cancelled) {
            throw CancelledFailure();
        }
        ImportProgress progress{std::string(stage), done, total};
        {
            const std::lock_guard lock(m_mutex);
            m_progress = progress;
        }
        if (m_callback) {
            m_callback(progress);
        }
    }

    bool Cancelled() const {
        return m_cancelled;
    }

    ImportProgress Progress() const {
        const std::lock_guard lock(m_mutex);
        return m_progress;
    }

    void Cancel() {
        if (Finished()) {
            return;
        }
        m_cancelled = true;
        if (Readers().Cancel(this)) {
            Finish({}, std::make_exception_ptr(CancelledFailure()));
        }
        Log(LogLevel::Info, "cancelled importing {} at \"{}\"", m_source, m_path);
    }

    void Read(const ImportReader& reader) {
        const auto start = std::chrono::steady_clock::now();
        try {
            ImportContext context(shared_from_this());
            ImportPlan plan = reader.read(context);
            if (m_cancelled) {
                throw CancelledFailure();
            }
            m_readMs = Milliseconds(start);
            Log(LogLevel::Debug, "{} read {} into {} prims in {:.1f} ms", m_reader, m_source,
                plan.PrimCount(), m_readMs);
            Finish(std::move(plan), nullptr);
        } catch (...) {
            Finish({}, m_cancelled ? std::make_exception_ptr(CancelledFailure())
                                   : std::current_exception());
        }
    }

    bool Finished() const {
        const std::lock_guard lock(m_mutex);
        return m_finished;
    }

    bool Applied() const {
        return m_applied;
    }

    Prim Apply(Scene& scene) {
        {
            std::unique_lock lock(m_mutex);
            m_done.wait(lock, [this] { return m_finished; });
        }
        if (!m_applied) {
            m_applied = true;
            try {
                if (m_error) {
                    std::rethrow_exception(m_error);
                }
                const auto start = std::chrono::steady_clock::now();
                m_root = m_plan->Apply(scene, m_path);
                Log(LogLevel::Info,
                    "{} imported {} at \"{}\" as {} prims, read in {:.1f} ms and defined in "
                    "{:.1f} ms",
                    m_reader, m_source, m_path, m_plan->PrimCount(), m_readMs, Milliseconds(start));
            } catch (...) {
                m_error = std::current_exception();
            }
            m_plan.reset();
        }
        if (m_error) {
            std::rethrow_exception(m_error);
        }
        return m_root;
    }

  private:
    CancelledError CancelledFailure() const {
        return CancelledError("Importing {} at \"{}\": cancelled", m_source, m_path);
    }

    static double Milliseconds(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    }

    void Finish(std::optional<ImportPlan> plan, std::exception_ptr error) {
        {
            const std::lock_guard lock(m_mutex);
            m_plan = std::move(plan);
            m_error = std::move(error);
            m_finished = true;
        }
        m_done.notify_all();
    }

    std::string m_reader;
    std::string m_source;
    std::string m_path;
    ImportProgressCallback m_callback;
    std::optional<UpAxis> m_up;
    std::optional<float> m_metersPerUnit;
    std::atomic<bool> m_cancelled = false;
    mutable std::mutex m_mutex;
    std::condition_variable m_done;
    ImportProgress m_progress;
    bool m_finished = false;
    std::optional<ImportPlan> m_plan;
    std::exception_ptr m_error;
    double m_readMs = 0;
    bool m_applied = false;
    Prim m_root;
};

CancelledError::~CancelledError() = default;

ImportContext::ImportContext() : m_state(std::make_shared<ImportState>()) {}

ImportContext::ImportContext(std::shared_ptr<ImportState> state) : m_state(std::move(state)) {}

void ImportContext::Report(std::string_view stage, uint64_t done, uint64_t total) {
    m_state->Report(stage, done, total);
}

std::optional<UpAxis> ImportContext::Up() const {
    return m_state->Up();
}

std::optional<float> ImportContext::MetersPerUnit() const {
    return m_state->MetersPerUnit();
}

bool ImportContext::Cancelled() const {
    return m_state->Cancelled();
}

void ImportPlan::Add(std::string path, DefineStep define) {
    m_steps.push_back({std::move(path), std::move(define)});
}

std::string ImportPlan::AddFile(std::string_view name, FileBytes bytes) {
    static std::atomic<uint64_t> plans = 0;
    if (m_id == 0) {
        m_id = ++plans;
    }
    std::string path = std::format("memory://{}/{}", m_id, name);
    m_files.emplace_back(path, std::move(bytes));
    return path;
}

uint32_t ImportPlan::PrimCount() const {
    return uint32_t(m_steps.size());
}

Prim ImportPlan::Apply(Scene& scene, std::string_view root) const {
    bool definedRoot = false;
    try {
        for (const Step& step : m_steps) {
            step.define(scene, step.path);
            definedRoot = definedRoot || step.path == root;
        }
        const std::optional<Prim> prim = scene.Find(root);
        if (!prim) {
            throw Error("Importing at \"{}\": the importer defined nothing there", root);
        }
        for (const auto& [path, bytes] : m_files) {
            scene.AddFile(path, bytes, *prim);
        }
        return *prim;
    } catch (...) {
        if (const std::optional<Prim> prim = scene.Find(root); definedRoot && prim) {
            scene.Remove(*prim);
        }
        throw;
    }
}

bool ImportJob::Ready() const {
    return !m_state || m_state->Finished();
}

ImportProgress ImportJob::Progress() const {
    return m_state ? m_state->Progress() : ImportProgress{};
}

void ImportJob::Cancel() {
    if (m_state) {
        m_state->Cancel();
    }
}

Prim ImportJob::Wait() {
    if (!m_state) {
        throw Error("Waiting on an import job that was never started");
    }
    return m_state->Apply(*m_scene);
}

void RegisterImporter(std::shared_ptr<Importer> importer) {
    if (!importer) {
        throw Error("Registering an importer: the importer is null");
    }
    const std::string name = importer->Name();
    const std::vector<std::string> extensions = importer->Extensions();
    if (extensions.empty()) {
        throw Error("Registering the {} importer: it reads no extensions", name);
    }
    for (const std::string& extension : extensions) {
        if (extension.size() < 2 || extension.front() != '.') {
            throw Error("Registering the {} importer: \"{}\" isn't an extension like \".obj\"",
                        name, extension);
        }
    }
    std::vector<std::pair<std::string, std::shared_ptr<Importer>>> replaced;
    {
        Registry& registry = Importers();
        const std::lock_guard lock(registry.mutex);
        for (const std::string& extension : extensions) {
            const std::string key = Lowercase(extension);
            std::shared_ptr<Importer>& slot = registry.importers[key];
            if (slot && slot != importer) {
                replaced.emplace_back(key, std::move(slot));
            }
            slot = importer;
        }
    }
    for (const auto& [extension, previous] : replaced) {
        Log(LogLevel::Info, "the {} importer replaces the {} importer for \"{}\" files", name,
            previous->Name(), extension);
    }
    Log(LogLevel::Debug, "registered the {} importer for {}", name, Join(extensions));
}

void UnregisterImporter(std::string_view extension) {
    std::shared_ptr<Importer> removed;
    {
        Registry& registry = Importers();
        const std::lock_guard lock(registry.mutex);
        const auto found = registry.importers.find(Lowercase(extension));
        if (found == registry.importers.end()) {
            return;
        }
        removed = std::move(found->second);
        registry.importers.erase(found);
    }
    Log(LogLevel::Debug, "unregistered the {} importer for \"{}\" files", removed->Name(),
        Lowercase(extension));
}

std::shared_ptr<Importer> FindImporter(std::string_view extension) {
    Registry& registry = Importers();
    const std::lock_guard lock(registry.mutex);
    const auto found = registry.importers.find(Lowercase(extension));
    return found == registry.importers.end() ? nullptr : found->second;
}

std::vector<std::string> ImporterExtensions() {
    Registry& registry = Importers();
    const std::lock_guard lock(registry.mutex);
    std::vector<std::string> extensions;
    for (const auto& [extension, importer] : registry.importers) {
        extensions.push_back(extension);
    }
    return extensions;
}

ImportJob Scene::Import(std::string_view path, const std::filesystem::path& file) {
    return Import(path, ImportSource{.file = file}, {});
}

ImportJob Scene::Import(std::string_view path, const std::filesystem::path& file,
                        const ImportDesc& desc) {
    return Import(path, ImportSource{.file = file}, desc);
}

ImportJob Scene::Import(std::string_view path, const ImportSource& source) {
    return Import(path, source, {});
}

ImportJob Scene::Import(std::string_view path, const ImportSource& source, const ImportDesc& desc) {
    const std::string extension = source.file.extension().string();
    const bool inMemory = source.Find(source.file.filename().generic_string()) != nullptr;
    Log(LogLevel::Info, "importing \"{}\"{} at \"{}\" as a \"{}\" file", source.file.string(),
        inMemory ? " from memory" : "", path, extension);
    std::shared_ptr<Importer> importer = desc.importer;
    if (importer) {
        Log(LogLevel::Debug, "the {} importer was given for \"{}\"", importer->Name(),
            source.file.string());
    } else {
        importer = FindImporter(extension);
        if (!importer) {
            const std::vector<std::string> extensions = ImporterExtensions();
            throw UnsupportedError(
                "Importing \"{}\" at \"{}\": no importer reads \"{}\" files; the registered "
                "importers read {}",
                source.file.string(), path, extension,
                extensions.empty() ? "nothing" : Join(extensions));
        }
        Log(LogLevel::Debug, "the {} importer reads \"{}\" files", importer->Name(), extension);
    }
    return StartImport(
        path,
        ImportReader{.name = std::format("the {} importer", importer->Name()),
                     .read =
                         [importer, root = std::string(path), source](ImportContext& context) {
                             return importer->Read(root, source, context);
                         }},
        desc, std::format("\"{}\"", source.file.string()));
}

ImportJob Scene::Import(std::string_view path, const ImportReader& reader, const ImportDesc& desc) {
    Log(LogLevel::Info, "importing from {} at \"{}\"", reader.name, path);
    return StartImport(path, reader, desc, std::format("from {}", reader.name));
}

ImportJob Scene::StartImport(std::string_view path, const ImportReader& reader,
                             const ImportDesc& desc, std::string source) {
    if (!reader.read) {
        throw Error("Importing at \"{}\": the reader has no read function", path);
    }
    auto state =
        std::make_shared<ImportState>(reader.name, std::move(source), std::string(path), desc);
    if (desc.synchronous) {
        state->Read(reader);
        state->Apply(*this);
        return ImportJob(Ref<Scene>(this), std::move(state));
    }
    Readers().Add(state.get(), [reader, state] { state->Read(reader); });
    m_impl->AddCommitStep([this, state] {
        if (!state->Finished()) {
            return false;
        }
        if (!state->Applied()) {
            try {
                state->Apply(*this);
            } catch (const CancelledError&) {
            } catch (const std::exception& error) {
                Log(LogLevel::Error, "{}", error.what());
            }
        }
        return true;
    });
    return ImportJob(Ref<Scene>(this), std::move(state));
}

}
