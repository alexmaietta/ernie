// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/usd_import.h"

#include "ernie/scene/prim_type.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/value.h"
#include "ernie/usd/plugin_api.h"
#include "ernie/util/profile.h"
#include "ernie/util/tensor.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#else
#include <dlfcn.h>
#endif

namespace ernie {
namespace {

#if defined(__APPLE__)
constexpr const char* kPluginName = "libernie_usd.dylib";
#else
constexpr const char* kPluginName = "libernie_usd.so";
#endif

struct Plugin {
    ErnieUsdImportFn import = nullptr;
    std::string usdVersion;
    std::string error;
};

#if defined(_WIN32)
Plugin LoadPlugin() {
    Plugin plugin;
    plugin.error = "The USD importer is not built for Windows yet";
    return plugin;
}
#else
/// Where to look for the plugin, first to last.
std::vector<std::filesystem::path> Candidates() {
    std::vector<std::filesystem::path> candidates;
    if (const char* path = std::getenv("ERNIE_USD_PLUGIN");
        path != nullptr && *path != '\0') {
        candidates.emplace_back(path);
        return candidates;
    }
    // Beside the binary this code is in: libernie, the Python module,
    // or a program linking Ernie statically.
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&UsdImportAvailable), &info) !=
            0 &&
        info.dli_fname != nullptr) {
        const std::filesystem::path self =
            std::filesystem::path(info.dli_fname).parent_path();
        candidates.push_back(self / kPluginName);
        candidates.push_back(self / ".." / "lib" / kPluginName);
    }
#ifdef ERNIE_USD_PLUGIN_BUILD_PATH
    candidates.emplace_back(ERNIE_USD_PLUGIN_BUILD_PATH);
#endif
    return candidates;
}

Plugin LoadPlugin() {
    Plugin plugin;
    std::string tried;
    for (const std::filesystem::path& path : Candidates()) {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            tried += "\n  " + path.string() + " (missing)";
            continue;
        }
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            const std::string error = dlerror();
            plugin.error = "Cannot load the USD importer " +
                           path.string() + ": " + error;
            if (error.find("pxrInternal") != std::string::npos) {
                plugin.error +=
                    "\nThe process already has another OpenUSD "
                    "loaded; build ernie_usd against it "
                    "(ERNIE_USD_ROOT) so that only one is.";
            }
            return plugin;
        }
        const auto abi = reinterpret_cast<ErnieUsdAbiVersionFn>(
            dlsym(handle, usd_plugin::kAbiVersionSymbol));
        const auto version = reinterpret_cast<ErnieUsdVersionFn>(
            dlsym(handle, usd_plugin::kVersionSymbol));
        const auto import = reinterpret_cast<ErnieUsdImportFn>(
            dlsym(handle, usd_plugin::kImportSymbol));
        if (abi == nullptr || version == nullptr ||
            import == nullptr) {
            plugin.error =
                path.string() + " is not Ernie's USD importer";
            return plugin;
        }
        if (abi() != usd_plugin::kAbiVersion) {
            plugin.error =
                path.string() + " is for another Ernie (ABI " +
                std::to_string(abi()) + ", not " +
                std::to_string(usd_plugin::kAbiVersion) + ")";
            return plugin;
        }
        plugin.import = import;
        plugin.usdVersion = version();
        return plugin;
    }
    plugin.error =
        "Ernie's USD importer (" + std::string(kPluginName) +
        ") is not installed; build Ernie with -DERNIE_USD=ON. Looked "
        "for:" +
        tried;
    return plugin;
}
#endif

const Plugin& GetPlugin() {
    static std::once_flag once;
    static Plugin plugin;
    std::call_once(once, [] {
        ERNIE_ZONE("USD plugin load");
        plugin = LoadPlugin();
    });
    return plugin;
}

/// Authors what the plugin translates into a Scene.
class SceneSink final : public usd_plugin::Sink {
  public:
    SceneSink(Scene& scene, UsdImportResult& result)
        : scene_(scene), result_(result) {}

    void Define(const char* path, const char* type) override {
        scene_.Define(Path(path), type);
    }

    void SetBool(const char* path, const char* attribute,
                 bool value) override {
        Set(path, attribute, Value(value));
    }

    void SetInt(const char* path, const char* attribute,
                int value) override {
        Set(path, attribute, Value(value));
    }

    void SetNumbers(const char* path, const char* attribute,
                    const double* values,
                    std::size_t count) override {
        Set(path, attribute, NumbersValue(attribute, values, count));
    }

    void SetNumbersAt(const char* path, const char* attribute,
                      double time, const double* values,
                      std::size_t count) override {
        Set(path, attribute, NumbersValue(attribute, values, count),
            time);
    }

    bool Declares(const char* path, const char* attribute) override {
        const Path prim(path);
        if (!scene_.Has(prim)) {
            return false;
        }
        const PrimTypeDef* type = PrimTypeOf(scene_, prim);
        if (type == nullptr) {
            return false;
        }
        const std::string_view name(attribute);
        if (type->Find(name) != nullptr) {
            return true;
        }
        for (const PrimTypeDef* level = type; level != nullptr;
             level = level->base) {
            for (const std::string_view open :
                 level->openNamespaces) {
                if (name.starts_with(open) &&
                    name.size() > open.size() &&
                    name[open.size()] == ':') {
                    return true;
                }
            }
        }
        return false;
    }

    void SetTokenAt(const char* path, const char* attribute,
                    double time, const char* value) override {
        Set(path, attribute, Value(std::string(value)), time);
    }

    void WriteFloatArrayAt(const char* path, const char* attribute,
                           double time, const float* values,
                           const std::size_t* shape,
                           std::size_t rank) override {
        std::vector<std::int64_t> dims;
        std::size_t count = 1;
        for (std::size_t i = 0; i < rank; ++i) {
            dims.push_back(static_cast<std::int64_t>(shape[i]));
            count *= shape[i];
        }
        scene_.WriteAt(
            Path(path), attribute, time,
            Tensor::Copy(std::span(values, count), std::move(dims)));
    }

    void SetToken(const char* path, const char* attribute,
                  const char* value) override {
        Set(path, attribute, Value(std::string(value)));
    }

    void SetTargets(const char* path, const char* attribute,
                    const char* const* targets, std::size_t count,
                    bool list) override {
        if (!list && count == 1) {
            Set(path, attribute, Value(Path(targets[0])));
            return;
        }
        std::vector<Path> paths;
        paths.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            paths.emplace_back(targets[i]);
        }
        Set(path, attribute, Value(std::move(paths)));
    }

    void WriteFloats(const char* path, const char* attribute,
                     const float* values, std::size_t rows,
                     std::size_t columns) override {
        std::vector<std::int64_t> shape{
            static_cast<std::int64_t>(rows)};
        if (columns != 1) {
            shape.push_back(static_cast<std::int64_t>(columns));
        }
        scene_.Write(Path(path), attribute,
                     Tensor::Copy(std::span(values, rows * columns),
                                  std::move(shape)));
    }

    void WriteInts(const char* path, const char* attribute,
                   const std::int32_t* values,
                   std::size_t count) override {
        scene_.Write(
            Path(path), attribute,
            Tensor::Copy(std::span(values, count),
                         {static_cast<std::int64_t>(count)}));
    }

    void SetTokens(const char* path, const char* attribute,
                   const char* const* values,
                   std::size_t count) override {
        std::vector<std::string> tokens(values, values + count);
        Set(path, attribute, Value(std::move(tokens)));
    }

    void WriteFloatArray(const char* path, const char* attribute,
                         const float* values,
                         const std::size_t* shape,
                         std::size_t rank) override {
        WriteArray(path, attribute, values, shape, rank);
    }

    void WriteIntArray(const char* path, const char* attribute,
                       const std::int32_t* values,
                       const std::size_t* shape,
                       std::size_t rank) override {
        WriteArray(path, attribute, values, shape, rank);
    }

    void SetTimeRange(double start, double end,
                      double timeCodesPerSecond,
                      double framesPerSecond) override {
        result_.timeCodeRange = glm::dvec2(start, end);
        result_.timeCodesPerSecond = timeCodesPerSecond;
        result_.framesPerSecond = framesPerSecond;
    }

    void MapSource(const char* source, const char* path) override {
        result_.prims.insert_or_assign(source, Path(path));
    }

    void Warn(const char* message) override {
        result_.warnings.emplace_back(message);
    }

  private:
    template <typename T>
    void WriteArray(const char* path, const char* attribute,
                    const T* values, const std::size_t* shape,
                    std::size_t rank) {
        std::vector<std::int64_t> dims;
        std::size_t count = 1;
        for (std::size_t i = 0; i < rank; ++i) {
            dims.push_back(static_cast<std::int64_t>(shape[i]));
            count *= shape[i];
        }
        scene_.Write(
            Path(path), attribute,
            Tensor::Copy(std::span(values, count), std::move(dims)));
    }

    /// One number, a 2-, 3- or 4-vector, or a 4x4 matrix (column
    /// after column) as a value.
    static Value NumbersValue(const char* attribute,
                              const double* values,
                              std::size_t count) {
        switch (count) {
        case 1:
            return values[0];
        case 2:
            return glm::dvec2(values[0], values[1]);
        case 3:
            return glm::dvec3(values[0], values[1], values[2]);
        case 4:
            return glm::dvec4(values[0], values[1], values[2],
                              values[3]);
        case 16: {
            glm::dmat4 matrix(1.0);
            for (int column = 0; column < 4; ++column) {
                for (int row = 0; row < 4; ++row) {
                    matrix[column][row] =
                        values[static_cast<std::size_t>(4 * column +
                                                        row)];
                }
            }
            return matrix;
        }
        default:
            throw std::invalid_argument(std::string(attribute) +
                                        ": " + std::to_string(count) +
                                        " numbers are not a value");
        }
    }

    /// Sets `value`, first converted to the attribute's declared type
    /// (a double to a float, a dvec3 to a colour, ...) as the JSON
    /// loader does; with `time`, as its time sample there.
    void Set(const char* path, std::string_view attribute,
             Value value, std::optional<double> time = std::nullopt) {
        const Path prim(path);
        if (const PrimTypeDef* type = PrimTypeOf(scene_, prim)) {
            if (const AttributeDef* def = type->Find(attribute)) {
                try {
                    value = def->Coerce(value);
                } catch (const std::invalid_argument&) {
                    // Scene::Set reports the mismatch.
                }
            }
        }
        if (time) {
            scene_.SetAt(prim, attribute, *time, std::move(value));
        } else {
            scene_.Set(prim, attribute, std::move(value));
        }
    }

    Scene& scene_;
    UsdImportResult& result_;
};

char UpAxisOf(const Scene& scene) {
    const Value* value = scene.Get(Path::Root(), tokens::kUpAxis);
    const std::string* axis =
        value != nullptr ? value->Get<std::string>() : nullptr;
    return axis != nullptr && *axis == tokens::kY ? 'Y' : 'Z';
}

double MetersPerUnitOf(const Scene& scene) {
    const Value* value =
        scene.Get(Path::Root(), tokens::kMetersPerUnit);
    const double* meters =
        value != nullptr ? value->Get<double>() : nullptr;
    return meters != nullptr && *meters > 0.0 ? *meters : 1.0;
}

}  // namespace

bool UsdImportAvailable(std::string* reason) {
    const Plugin& plugin = GetPlugin();
    if (reason != nullptr) {
        *reason = plugin.error;
    }
    return plugin.import != nullptr;
}

UsdImportResult ImportUsd(Scene& scene, const Path& root,
                          const std::filesystem::path& file,
                          const UsdImportOptions& options) {
    ERNIE_ZONE("USD import");
    ERNIE_ZONE_TEXT(file.string());
    const Plugin& plugin = GetPlugin();
    if (plugin.import == nullptr) {
        throw std::runtime_error(plugin.error);
    }
    std::vector<usd_plugin::VariantSelection> variants;
    variants.reserve(options.variants.size());
    for (const UsdVariantSelection& variant : options.variants) {
        variants.push_back({variant.prim.c_str(), variant.set.c_str(),
                            variant.selection.c_str()});
    }
    const std::string fileText = file.string();
    usd_plugin::Request request;
    request.file = fileText.c_str();
    request.root = root.String().c_str();
    request.hasTime = options.time.has_value();
    request.time = options.time.value_or(0.0);
    request.variants = variants.data();
    request.variantCount = variants.size();
    request.proxy = options.proxy;
    request.guide = options.guide;
    request.upAxis = UpAxisOf(scene);
    request.metersPerUnit = MetersPerUnitOf(scene);

    UsdImportResult result;
    result.root = root;
    SceneSink sink(scene, result);
    std::array<char, 4096> error{};
    if (!plugin.import(&request, &sink, error.data(), error.size())) {
        throw std::runtime_error(error.data());
    }
    // The file's animation range for a scene without one.
    Stage stage(scene, Path::Root());
    if (result.timeCodeRange &&
        !(stage.GetEndTimeCode() > stage.GetStartTimeCode())) {
        stage.SetStartTimeCode(result.timeCodeRange->x)
            .SetEndTimeCode(result.timeCodeRange->y)
            .SetTimeCodesPerSecond(result.timeCodesPerSecond)
            .SetFramesPerSecond(result.framesPerSecond);
    }
    return result;
}

}  // namespace ernie
