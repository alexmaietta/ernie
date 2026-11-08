// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/shell/editor.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/app/viewer/shell/history.h"
#include "ernie/app/viewer/shell/selection.h"
#include "ernie/device/device.h"
#include "ernie/scene/scene.h"

namespace ernie::app {
namespace {

class EditorTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        s_device = device::Device::Create();
    }

    static void TearDownTestSuite() {
        s_device = nullptr;
    }

    static Ref<device::Device> s_device;
};

Ref<device::Device> EditorTest::s_device;

struct Session {
    Ref<Scene> scene;
    Selection selection;
    History history;
    Editor editor{scene, selection, history};

    explicit Session(device::Device& device) : scene(Scene::Create(device)) {}

    bool Has(const std::string& path) const {
        return scene->Find(path).has_value();
    }
};

TEST(EditorClassifyTest, ClassifiesPrimsIntoGroups) {
    EXPECT_EQ(Classify(SchemaId::Xform), PrimGroup::Organize);
    EXPECT_EQ(Classify(SchemaId::Sphere), PrimGroup::Geometry);
    EXPECT_EQ(Classify(SchemaId::DistantLight), PrimGroup::Lights);
    EXPECT_EQ(Classify(SchemaId::Camera), PrimGroup::Cameras);
}

TEST_F(EditorTest, CreatesUniqueChildrenOfTheSelectionWithUndo) {
    {
        Session session(*s_device);
        const std::string group = session.editor.Create(SchemaId::Xform, {}, {});
        EXPECT_EQ(group, "/Xform");
        const std::string first = session.editor.Create(SchemaId::Sphere, {1, 2, 3}, {});
        EXPECT_EQ(first, "/Xform/Sphere");
        session.selection.Set(group);
        EXPECT_EQ(session.editor.Create(SchemaId::Sphere, {}, {}), "/Xform/Sphere_1");
        EXPECT_EQ(session.selection.paths, std::vector<std::string>{"/Xform/Sphere_1"});
        EXPECT_EQ(session.scene->Find(first)->Get(Xformable::kTranslate), (Double3{1, 2, 3}));

        session.history.Undo();
        EXPECT_FALSE(session.Has("/Xform/Sphere_1"));
        EXPECT_EQ(session.selection.paths, std::vector<std::string>{group});
        session.history.Redo();
        EXPECT_TRUE(session.Has("/Xform/Sphere_1"));
    }
}

TEST_F(EditorTest, RenamesAndReparentsWithUndo) {
    {
        Session session(*s_device);
        session.scene->Define<Xform>("/A");
        session.scene->Define<Xform>("/A/Child");
        session.scene->Define<Xform>("/B");
        session.selection.Set("/A/Child");

        EXPECT_TRUE(session.editor.Rename("/A", "Renamed"));
        EXPECT_TRUE(session.Has("/Renamed/Child"));
        EXPECT_EQ(session.selection.paths, std::vector<std::string>{"/Renamed/Child"});
        EXPECT_FALSE(session.editor.Rename("/Renamed", "B"));
        session.history.Undo();
        EXPECT_TRUE(session.Has("/A/Child"));
        EXPECT_EQ(session.selection.paths, std::vector<std::string>{"/A/Child"});

        EXPECT_FALSE(session.editor.Reparent("/A", "/A/Child"));
        EXPECT_TRUE(session.editor.Reparent("/A", "/B"));
        EXPECT_TRUE(session.Has("/B/A/Child"));
        session.history.Undo();
        EXPECT_TRUE(session.Has("/A/Child"));
        EXPECT_FALSE(session.Has("/B/A"));
    }
}

TEST_F(EditorTest, DuplicatesAndDeletesTheSelectionWithUndo) {
    {
        Session session(*s_device);
        session.scene->Define<Xform>("/Group");
        session.scene->Define<Sphere>("/Group/Ball", {.radius = 2});
        session.selection.paths = {"/Group", "/Group/Ball"};

        session.editor.Duplicate();
        EXPECT_TRUE(session.Has("/Group_1/Ball"));
        EXPECT_EQ(session.scene->Find("/Group_1/Ball")->Get(Sphere::kRadius), 2);
        EXPECT_EQ(session.selection.paths, std::vector<std::string>{"/Group_1"});
        session.history.Undo();
        EXPECT_FALSE(session.Has("/Group_1"));

        session.selection.Set("/Group");
        session.editor.Delete();
        EXPECT_FALSE(session.Has("/Group"));
        session.history.Undo();
        EXPECT_TRUE(session.Has("/Group/Ball"));
        EXPECT_EQ(session.scene->Find("/Group/Ball")->Get(Sphere::kRadius), 2);

        session.selection.Clear();
        session.editor.Delete();
        session.editor.Duplicate();
        EXPECT_EQ(session.history.NextUndo(), nullptr);
        ASSERT_NE(session.history.NextRedo(), nullptr);
        EXPECT_EQ(*session.history.NextRedo(), "Delete Group");
    }
}

TEST_F(EditorTest, SetsPosesAndVisibilityWithUndo) {
    {
        Session session(*s_device);
        session.scene->Define<Sphere>("/Ball");
        const Prim ball = *session.scene->Find("/Ball");
        ball.Set(Xformable::kTranslate, Double3{4, 5, 6});
        ball.Set(Xformable::kScale, Float3{2, 2, 2});
        session.editor.SetPose("move", "/Ball", {}, {}, {1, 1, 1}, {4, 5, 6}, {}, {2, 2, 2});
        session.history.Undo();
        EXPECT_EQ(ball.Get(Xformable::kTranslate), (Double3{0, 0, 0}));
        EXPECT_EQ(ball.Get(Xformable::kScale), (Float3{1, 1, 1}));
        session.history.Redo();
        EXPECT_EQ(ball.Get(Xformable::kTranslate), (Double3{4, 5, 6}));
        EXPECT_EQ(ball.Get(Xformable::kScale), (Float3{2, 2, 2}));
        session.history.Clear();

        session.editor.SetVisible("/Ball", false);
        ASSERT_NE(session.history.NextUndo(), nullptr);
        EXPECT_EQ(*session.history.NextUndo(), "Hide Ball");
        session.history.Undo();
        EXPECT_EQ(session.history.NextUndo(), nullptr);
    }
}

TEST_F(EditorTest, SetsCellsAcrossPrimsAndRestoresEachOnUndo) {
    {
        Session session(*s_device);
        session.scene->Define<Sphere>("/A", {.radius = 1});
        session.scene->Define<Sphere>("/B", {.radius = 2});
        const std::vector<std::string> paths = {"/A", "/B"};
        const auto cells = [](std::vector<float> values) {
            std::vector<std::byte> bytes(values.size() * sizeof(float));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            return bytes;
        };
        session.editor.SetCells(
            "Set Radius",
            [scene = session.scene, paths] {
                return scene->Select(Span<const std::string>(paths), SchemaId::Sphere);
            },
            Sphere::kRadius.Column(), cells({1, 2}), cells({5}));
        EXPECT_EQ(session.scene->Find("/A")->Get(Sphere::kRadius), 5);
        EXPECT_EQ(session.scene->Find("/B")->Get(Sphere::kRadius), 5);
        session.history.Undo();
        EXPECT_EQ(session.scene->Find("/A")->Get(Sphere::kRadius), 1);
        EXPECT_EQ(session.scene->Find("/B")->Get(Sphere::kRadius), 2);
        session.history.Redo();
        EXPECT_EQ(session.scene->Find("/B")->Get(Sphere::kRadius), 5);
    }
}

TEST_F(EditorTest, SetsRelationshipTargetsWithUndo) {
    {
        Session session(*s_device);
        session.scene->DefineMaterial("/Red");
        session.scene->DefineMaterial("/Blue");
        session.scene->Define<Sphere>("/Ball", {.materialBinding = {"/Red"}});
        const auto target = [scene = session.scene] { return scene->Select("/Ball"); };
        const ColumnRef binding = Sphere::kMaterialBinding.Column();
        session.editor.SetTargets("Set Material", target, binding, {{"/Red"}}, {"/Blue"});
        EXPECT_EQ(session.scene->Find("/Ball")->Get(Sphere::kMaterialBinding),
                  std::vector<std::string>{"/Blue"});
        session.history.Undo();
        EXPECT_EQ(session.scene->Find("/Ball")->Get(Sphere::kMaterialBinding),
                  std::vector<std::string>{"/Red"});
        session.editor.SetTargets("Clear Material", target, binding, {{"/Red"}}, {});
        EXPECT_TRUE(session.scene->Find("/Ball")->Get(Sphere::kMaterialBinding).empty());
        EXPECT_EQ(*session.history.NextUndo(), "Clear Material");
    }
}

}
}
