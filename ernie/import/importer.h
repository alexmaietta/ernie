// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ernie/export.h"
#include "ernie/scene/prim.h"
#include "ernie/scene/scene.h"
#include "ernie/util/error.h"

namespace ernie {

class ERNIE_API CancelledError : public Error {
  public:
    using Error::Error;
    ~CancelledError() override;
};

struct ERNIE_API ImportProgress {
    std::string stage;
    uint64_t done = 0;
    uint64_t total = 0;
};

using ImportProgressCallback = std::function<void(const ImportProgress& progress)>;

class ImportState;

enum class UpAxis : uint8_t { Y, Z };

class ERNIE_API ImportContext {
  public:
    ImportContext();
    explicit ImportContext(std::shared_ptr<ImportState> state);

    void Report(std::string_view stage, uint64_t done, uint64_t total);
    bool Cancelled() const;
    std::optional<UpAxis> Up() const;
    std::optional<float> MetersPerUnit() const;

  private:
    std::shared_ptr<ImportState> m_state;
};

struct ERNIE_API ImportSource {
    std::filesystem::path file;
    std::map<std::string, FileBytes, std::less<>> files;

    std::filesystem::path Locate(std::string_view name) const;
    FileBytes Find(std::string_view name) const;
    FileBytes Read(std::string_view name) const;
    FileBytes ReadEntry() const;
};

class ERNIE_API ImportPlan {
  public:
    using DefineStep = std::function<void(Scene& scene, std::string_view path)>;

    template <class T> void Define(std::string path, typename T::Desc desc = {}) {
        Add(std::move(path), [desc = std::move(desc)](Scene& scene, std::string_view at) {
            scene.Define<T>(at, desc);
        });
    }

    void Add(std::string path, DefineStep define);
    std::string AddFile(std::string_view name, FileBytes bytes);
    uint32_t PrimCount() const;
    Prim Apply(Scene& scene, std::string_view root) const;

  private:
    struct Step {
        std::string path;
        DefineStep define;
    };

    std::vector<Step> m_steps;
    std::vector<std::pair<std::string, FileBytes>> m_files;
    uint64_t m_id = 0;
};

class ERNIE_API Importer {
  public:
    virtual ~Importer();

    virtual std::string Name() const = 0;
    virtual std::vector<std::string> Extensions() const = 0;
    virtual ImportPlan Read(std::string_view path, const ImportSource& source,
                            ImportContext& context) = 0;
};

struct ERNIE_API ImportReader {
    std::string name;
    std::function<ImportPlan(ImportContext& context)> read;
};

struct ERNIE_API ImportDesc {
    std::shared_ptr<Importer> importer;
    ImportProgressCallback progress;
    bool synchronous = false;
    std::optional<UpAxis> up;
    std::optional<float> metersPerUnit;
};

class ERNIE_API ImportJob {
  public:
    ImportJob() = default;

    bool Ready() const;
    ImportProgress Progress() const;
    void Cancel();
    Prim Wait();

  private:
    friend class Scene;

    ImportJob(Ref<Scene> scene, std::shared_ptr<ImportState> state)
        : m_scene(std::move(scene)), m_state(std::move(state)) {}

    Ref<Scene> m_scene;
    std::shared_ptr<ImportState> m_state;
};

ERNIE_API void RegisterImporter(std::shared_ptr<Importer> importer);
ERNIE_API void UnregisterImporter(std::string_view extension);
ERNIE_API std::shared_ptr<Importer> FindImporter(std::string_view extension);
ERNIE_API std::vector<std::string> ImporterExtensions();

}
