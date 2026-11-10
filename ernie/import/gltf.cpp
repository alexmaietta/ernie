// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/import/gltf.h"

#include <memory>
#include <string>

#include <cgltf.h>

#include "ernie/util/error.h"
#include "ernie/util/log.h"

namespace ernie {
namespace {

using GltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

GltfData Parse(const ImportSource& source) {
    const FileBytes bytes = source.ReadEntry();
    const cgltf_options options{};
    cgltf_data* data = nullptr;
    if (const cgltf_result result = cgltf_parse(&options, bytes->data(), bytes->size(), &data);
        result != cgltf_result_success) {
        throw Error("Reading glTF \"{}\": cgltf couldn't parse it (result {})",
                    source.file.string(), int(result));
    }
    return {data, cgltf_free};
}

void LoadBuffers(const cgltf_data&, const ImportSource&) {}

void DefineMaterials(ImportPlan&, std::string_view, const cgltf_data&) {}

void DefineMeshes(ImportPlan&, std::string_view, const cgltf_data&) {}

void DefineNodes(ImportPlan&, std::string_view, const cgltf_data&) {}

void DefineCameras(ImportPlan&, std::string_view, const cgltf_data&) {}

void DefineLights(ImportPlan&, std::string_view, const cgltf_data&) {}

class GltfImporter final : public Importer {
  public:
    std::string Name() const override {
        return "glTF";
    }

    std::vector<std::string> Extensions() const override {
        return {".gltf", ".glb"};
    }

    ImportPlan Read(std::string_view path, const ImportSource& source,
                    ImportContext& context) override {
        context.Report("parsing", 0, 1);
        const GltfData data = Parse(source);
        LoadBuffers(*data, source);
        context.Report("parsing", 1, 1);

        ImportPlan plan;
        plan.Define<Xform>(std::string(path));
        DefineMaterials(plan, path, *data);
        DefineMeshes(plan, path, *data);
        DefineNodes(plan, path, *data);
        DefineCameras(plan, path, *data);
        DefineLights(plan, path, *data);
        Log(LogLevel::Warning,
            "the glTF importer is a placeholder, so \"{}\" imports as an empty Xform",
            source.file.string());
        return plan;
    }
};

}

std::shared_ptr<Importer> CreateGltfImporter() {
    return std::make_shared<GltfImporter>();
}

}
