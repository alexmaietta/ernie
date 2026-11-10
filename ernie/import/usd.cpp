// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/import/usd.h"

#include <memory>
#include <string>
#include <vector>

#include <tinyusd/tinyusd.hpp>

#include "ernie/util/error.h"
#include "ernie/util/log.h"

namespace ernie {
namespace {

tinyusd::Stage Open(const ImportSource& source) {
    try {
        return tinyusd::Stage::open(source.file.string());
    } catch (const tinyusd::Error& error) {
        throw Error("Reading USD \"{}\": tinyusd couldn't open it ({})", source.file.string(),
                    error.what());
    }
}

void DefineMaterials(ImportPlan&, std::string_view, const tinyusd::Stage&) {}

void DefineMeshes(ImportPlan&, std::string_view, const tinyusd::Stage&) {}

void DefineXforms(ImportPlan&, std::string_view, const tinyusd::Stage&) {}

void DefineInstancers(ImportPlan&, std::string_view, const tinyusd::Stage&) {}

void DefineCameras(ImportPlan&, std::string_view, const tinyusd::Stage&) {}

void DefineLights(ImportPlan&, std::string_view, const tinyusd::Stage&) {}

class UsdImporter final : public Importer {
  public:
    std::string Name() const override {
        return "USD";
    }

    std::vector<std::string> Extensions() const override {
        return {".usd", ".usda", ".usdc", ".usdz"};
    }

    ImportPlan Read(std::string_view path, const ImportSource& source,
                    ImportContext& context) override {
        context.Report("opening", 0, 1);
        const tinyusd::Stage stage = Open(source);
        context.Report("opening", 1, 1);

        ImportPlan plan;
        plan.Define<Xform>(std::string(path));
        DefineMaterials(plan, path, stage);
        DefineMeshes(plan, path, stage);
        DefineXforms(plan, path, stage);
        DefineInstancers(plan, path, stage);
        DefineCameras(plan, path, stage);
        DefineLights(plan, path, stage);
        Log(LogLevel::Warning,
            "the USD importer is a placeholder, so \"{}\" imports as an empty Xform",
            source.file.string());
        return plan;
    }
};

}

std::shared_ptr<Importer> CreateUsdImporter() {
    return std::make_shared<UsdImporter>();
}

}
