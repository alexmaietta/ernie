// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/io/image.h"
#include "ernie/scene/access/attribute.h"
#include "ernie/scene/access/view.h"
#include "ernie/scene/catalog/catalog.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/store/store.h"
#include "ernie/scene/systems/textures.h"
#include "ernie/scene/timeline/timeline.h"
#include "support/devices.h"

namespace ernie {
namespace {

std::string WriteImage(const std::string& name, Rgba color) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "ernie_scene_textures";
    std::filesystem::create_directories(directory);
    const std::filesystem::path path = directory / name;
    WritePng(path, Image(4, 4, color));
    return path.string();
}

struct Scene {
    Store& store;
    Catalog& catalog;
    Timeline& timeline;
    Textures& textures;
    Ordinal ordinal = 0;

    void Publish() {
        timeline.Publish(++ordinal).Wait();
    }

    void Settle() {
        textures.Finish();
        EXPECT_TRUE(timeline.Pending());
        Publish();
    }

    uint32_t Material(const std::string& path, const std::string& texture) {
        const uint32_t prim =
            catalog.Define(SchemaId::UsdPreviewSurface, std::vector<std::string>{path}).firstPrim;
        Texture(prim, texture);
        return prim;
    }

    void Texture(uint32_t material, const std::string& texture) {
        View(catalog, SchemaId::UsdPreviewSurface, std::vector<uint32_t>{material})
            .Get<Token>(kUsdPreviewSurfaceDiffuseColorTexture)
            .Set(catalog.GetTokens().Intern(texture));
    }

    uint32_t Bindings(uint32_t material) {
        const uint32_t row = catalog.GetPrims().Row(material, SchemaId::Material);
        return Items<uint32_t>(
                   store.Schema(SchemaId::Material).Read(kMaterialTextures.index, row, 1))
            .front();
    }
};

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        Store store(device);
        Catalog catalog(store);
        Timeline timeline(catalog);
        Textures textures(catalog);
        timeline.Add(textures);
        Scene scene{store, catalog, timeline, textures};
        test(scene);
    });
}

TEST(TexturesTest, ImagesLoadInTheBackgroundAndBindOnceReady) {
    const std::string red = WriteImage("red.png", {1, 0, 0, 1});
    ForEachScene([&](Scene& s) {
        const uint32_t look = s.Material("/Look", red);
        s.Publish();
        EXPECT_EQ(s.textures.Handle(look, "diffuseColor"), 0u);
        EXPECT_EQ(s.Bindings(look), kNoTextures);
        s.Settle();
        EXPECT_NE(s.textures.Handle(look, "diffuseColor"), 0u);
        EXPECT_NE(s.Bindings(look), kNoTextures);
        EXPECT_EQ(s.textures.Stats().loaded, 1u);
        EXPECT_EQ(s.textures.Stats().loading, 0u);
        EXPECT_FALSE(s.timeline.Pending());
    });
}

TEST(TexturesTest, AnImageSharedByMaterialsLoadsOnce) {
    const std::string red = WriteImage("red.png", {1, 0, 0, 1});
    ForEachScene([&](Scene& s) {
        const uint32_t a = s.Material("/A", red);
        const uint32_t b = s.Material("/B", red);
        s.Publish();
        s.Settle();
        EXPECT_EQ(s.textures.Stats().images, 1u);
        EXPECT_EQ(s.textures.Stats().loaded, 1u);
        EXPECT_NE(s.textures.Handle(a, "diffuseColor"), 0u);
        EXPECT_NE(s.textures.Handle(b, "diffuseColor"), 0u);
    });
}

TEST(TexturesTest, MissingFilesFailWithoutABinding) {
    ForEachScene([&](Scene& s) {
        const uint32_t look = s.Material("/Look", "/no/such/image.png");
        s.Publish();
        s.Settle();
        EXPECT_EQ(s.textures.Stats().failed, 1u);
        EXPECT_EQ(s.textures.Handle(look, "diffuseColor"), 0u);
        EXPECT_EQ(s.Bindings(look), kNoTextures);
    });
}

TEST(TexturesTest, ANewPathRebindsAndReleasesTheOldImage) {
    const std::string red = WriteImage("red.png", {1, 0, 0, 1});
    const std::string blue = WriteImage("blue.png", {0, 0, 1, 1});
    ForEachScene([&](Scene& s) {
        const uint32_t look = s.Material("/Look", red);
        s.Publish();
        s.Settle();
        s.Texture(look, blue);
        s.Publish();
        EXPECT_EQ(s.textures.Stats().images, 1u);
        EXPECT_EQ(s.textures.Stats().loading, 1u);
        s.Settle();
        EXPECT_EQ(s.textures.Stats().loaded, 1u);
        EXPECT_NE(s.textures.Handle(look, "diffuseColor"), 0u);
        EXPECT_EQ(s.textures.Stats().images, 1u);
    });
}

TEST(TexturesTest, CheckerboardPlanesBindTheChecker) {
    ForEachScene([&](Scene& s) {
        s.catalog.Define<Plane>(std::vector<std::string>{"/Ground"}, {.checkerboard = true});
        s.Publish();
        s.Settle();
        EXPECT_NE(s.textures.Checker(), 0u);
    });
}

TEST(TexturesTest, PublishingWithNothingToLoadDoesNothing) {
    const std::string red = WriteImage("red.png", {1, 0, 0, 1});
    ForEachScene([&](Scene& s) {
        s.Material("/Look", red);
        s.Publish();
        s.Settle();
        const uint32_t runs = s.textures.Runs();
        s.Publish();
        s.catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/Ball"});
        s.Publish();
        EXPECT_EQ(s.textures.Runs(), runs + 1);
        EXPECT_EQ(s.textures.Stats().loaded, 0u);
        EXPECT_EQ(s.textures.Stats().images, 1u);
    });
}

}
}
