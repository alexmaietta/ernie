// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/io/image.h"
#include "ernie/scene/builders/textures.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
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
    ColumnStore& store;
    Structure& structure;
    Publisher& publisher;
    TextureBuilder& textures;
    Ordinal ordinal = 0;

    void Publish() {
        publisher.Publish(++ordinal).Wait();
    }

    void Settle() {
        textures.Finish();
        EXPECT_TRUE(publisher.Pending());
        Publish();
    }

    uint32_t Material(const std::string& path, const std::string& texture) {
        const uint32_t prim =
            structure.Define(SchemaId::UsdPreviewSurface, std::vector<std::string>{path}).firstPrim;
        Texture(prim, texture);
        return prim;
    }

    void Texture(uint32_t material, const std::string& texture) {
        View(structure, SchemaId::UsdPreviewSurface, std::vector<uint32_t>{material})
            .Get<Token>(kUsdPreviewSurfaceDiffuseColorTexture)
            .Set(structure.GetTokens().Intern(texture));
    }

    uint32_t Bindings(uint32_t material) {
        return textures.Record(material).textures;
    }
};

class TextureProbe : public TableBuilder {
  public:
    std::string_view Name() const override {
        return "texture probe";
    }

    std::span<const ColumnRef> Inputs() const override {
        return {};
    }

    std::span<const TableId> TableInputs() const override {
        static constexpr TableId kTables[] = {TableId::Textures};
        return kTables;
    }

    bool Structural() const override {
        return false;
    }

    void Run(const ChangeSet& changes, TableWrites&) override {
        wrote = changes.Wrote(TableId::Textures);
        worlds = changes.worlds;
    }

    bool wrote = false;
    std::vector<uint32_t> worlds;
};

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        ColumnStore store(device);
        Structure structure(store);
        Publisher publisher(structure);
        TextureBuilder textures(structure);
        publisher.Add(textures);
        Scene scene{store, structure, publisher, textures};
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
        EXPECT_FALSE(s.publisher.Pending());
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
        s.structure.Define<Plane>(std::vector<std::string>{"/Ground"}, {.checkerboard = true});
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
        s.structure.Define(SchemaId::Sphere, std::vector<std::string>{"/Ball"});
        s.Publish();
        EXPECT_EQ(s.textures.Runs(), runs + 1);
        EXPECT_EQ(s.textures.Stats().loaded, 0u);
        EXPECT_EQ(s.textures.Stats().images, 1u);
    });
}

TEST(TexturesTest, DomeRecordsCarryTheirImageSizeAndRebindsAreLoggedPerWorld) {
    const std::string red = WriteImage("red.png", {1, 0, 0, 1});
    test::ForEachDevice([&](device::Device& device) {
        ColumnStore store(device);
        Structure structure(store);
        Publisher publisher(structure);
        TextureBuilder textures(structure);
        TextureProbe probe;
        publisher.Add(textures);
        publisher.Add(probe);
        Ordinal ordinal = 0;
        structure.Define(SchemaId::World, std::vector<std::string>{"/Env"});
        const uint32_t dome =
            structure.Define(SchemaId::DomeLight, std::vector<std::string>{"/Env/Sky"}).firstPrim;
        View(structure, SchemaId::DomeLight, std::vector<uint32_t>{dome})
            .Get<Token>(kDomeLightTextureFile)
            .Set(structure.GetTokens().Intern(red));
        publisher.Publish(++ordinal).Wait();
        EXPECT_EQ(textures.Record(dome).textures, kNoTextures);

        textures.Finish();
        probe.wrote = false;
        publisher.Publish(++ordinal).Wait();
        EXPECT_TRUE(probe.wrote);
        EXPECT_EQ(probe.worlds, (std::vector<uint32_t>{0}));
        const TextureRecord record = textures.Record(dome);
        EXPECT_NE(record.textures, kNoTextures);
        EXPECT_EQ(record.width, 4u);
        EXPECT_EQ(record.height, 4u);

        structure.Remove(dome);
        publisher.Publish(++ordinal).Wait();
        EXPECT_EQ(textures.Record(dome).textures, kNoTextures);
        EXPECT_EQ(textures.Record(dome).width, 0u);
    });
}

}
}
