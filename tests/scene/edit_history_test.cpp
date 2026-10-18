// SPDX-License-Identifier: BSD-3-Clause
/// @file edit_history_test.cpp
/// @brief Tests of undo and redo: attribute edits, drags, groups and subtrees.
#include "ernie/scene/edit_history.h"
#include "ernie/scene/json.h"
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/schemas/mesh.h"
#include "ernie/scene/schemas/shade.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace ernie {
namespace {

double Radius(const Scene& scene) {
    return Sphere(scene, "/Ball").GetRadius();
}

TEST(EditHistory, AttributeEditsUndoAndRedo) {
    Scene scene;
    Sphere::Define(scene, "/Ball");
    EditHistory history;
    history.Set(scene, "/Ball", Sphere::kRadius, 2.0);
    history.Set(scene, "/Ball", Sphere::kRadius, 3.0);
    EXPECT_EQ(history.UndoLabel(), "/Ball radius");
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(Radius(scene), 2.0);
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_FALSE(scene.IsAuthored(*scene.Find("/Ball"), Sphere::kRadius)) << "unset again";
    EXPECT_FALSE(history.Undo(scene));
    ASSERT_TRUE(history.Redo(scene));
    EXPECT_EQ(Radius(scene), 2.0);

    history.Clear(scene, "/Ball", Sphere::kRadius);
    EXPECT_FALSE(history.CanRedo()) << "an edit drops the undone steps";
    history.Undo(scene);
    EXPECT_EQ(Radius(scene), 2.0);
    EXPECT_THROW(history.Set(scene, "/Ball", Sphere::kRadius, -1.0), std::invalid_argument);
}

TEST(EditHistory, DragsMergeIntoOneStepAndGroupsUndoTogether) {
    Scene scene;
    Sphere::Define(scene, "/Ball");
    EditHistory history;
    for (const double radius : {1.5, 2.0, 2.5}) {
        history.Set(scene, "/Ball", Sphere::kRadius, radius, true);
    }
    history.Close();
    history.Set(scene, "/Ball", Sphere::kRadius, 4.0, true);
    EXPECT_EQ(history.Size(), 2u) << "a drag, then a new one after Close";
    history.Undo(scene);
    history.Undo(scene);
    EXPECT_FALSE(scene.IsAuthored(*scene.Find("/Ball"), Sphere::kRadius));

    history.BeginGroup();
    history.Define(scene, "/Box", "Cube");
    history.Set(scene, "/Box", Cube::kSize, 5.0);
    history.EndGroup();
    history.Undo(scene);
    EXPECT_FALSE(scene.Has("/Box")) << "the group undoes whole";
    history.Redo(scene);
    EXPECT_EQ(Cube(scene, "/Box").GetSize(), 5.0);
}

TEST(EditHistory, RemovedSubtreesComeBackWhole) {
    Scene scene;
    LoadScene(scene, R"({"prims": {
        "/World": {"typeName": "Xform", "apiSchemas": ["MaterialBindingAPI"],
                   "material:binding": "/Looks/Rock", "xform": {"translate": [1, 2, 3]}},
        "/World/Cloud": {"typeName": "Points", "points": [[0, 0, 0], [1, 1, 1]],
                         "widths.timeSamples": {"0": [0.1], "10": [0.3]}}}})");
    const std::string before = SaveScene(scene);
    const Tensor* points = scene.Read(*scene.Find("/World/Cloud"), PointBased::kPoints);
    const void* data = points->Data();

    EditHistory history;
    ASSERT_TRUE(history.Remove(scene, "/World"));
    EXPECT_FALSE(history.Remove(scene, "/World"));
    EXPECT_FALSE(scene.Has("/World/Cloud"));
    EXPECT_EQ(history.UndoLabel(), "Remove /World");
    history.Undo(scene);
    EXPECT_EQ(SaveScene(scene), before) << "types, API schemas, values and samples";
    EXPECT_EQ(scene.Read(*scene.Find("/World/Cloud"), PointBased::kPoints)->Data(), data)
        << "arrays shared, not copied";
    history.Redo(scene);
    EXPECT_FALSE(scene.Has("/World"));
}

TEST(EditHistory, PrimsAuthoredApartAreAddedAsOneStep) {
    Scene scratch;
    Sphere::Define(scratch, "/World/Ball").SetRadius(7.0);
    Scene scene;
    EditHistory history;
    history.Add(scene, scratch, "/World");
    EXPECT_EQ(Sphere(scene, "/World/Ball").GetRadius(), 7.0);
    EXPECT_THROW(history.Add(scene, scratch, "/World"), std::invalid_argument);
    history.Undo(scene);
    EXPECT_FALSE(scene.Has("/World"));
}

TEST(EditHistory, EditsOfRemovedPrimsAreSkipped) {
    Scene scene;
    Sphere::Define(scene, "/Ball");
    EditHistory history;
    history.Set(scene, "/Ball", Sphere::kRadius, 2.0);
    scene.Remove("/Ball");
    EXPECT_TRUE(history.Undo(scene));
    EXPECT_FALSE(scene.Has("/Ball"));
}

}  // namespace
}  // namespace ernie
