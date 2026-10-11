// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/edit_history.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

using ernie::EditHistory;
using ernie::Path;
using ernie::Scene;
namespace tokens = ernie::tokens;

TEST(EditHistory, UndoesAndRedoesSetsAndClears) {
    Scene scene;
    ernie::Cube::Define(scene, "/A").SetSize(1.0);
    EditHistory history;
    EXPECT_FALSE(history.CanUndo());
    history.Set(scene, "/A", tokens::kSize, 2.0);
    history.Set(scene, "/A", tokens::kDisplayColor,
                glm::vec3(1.0f, 0.0f, 0.0f));
    history.Clear(scene, "/A", tokens::kSize);
    EXPECT_EQ(scene.Get(Path("/A"), tokens::kSize), nullptr);
    EXPECT_EQ(history.UndoLabel(), "/A size");

    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.Get<double>("/A", tokens::kSize, 0.0), 2.0);
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.Get(Path("/A"), tokens::kDisplayColor), nullptr)
        << "unauthored again";
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.Get<double>("/A", tokens::kSize, 0.0), 1.0);
    EXPECT_FALSE(history.Undo(scene));

    ASSERT_TRUE(history.Redo(scene));
    EXPECT_EQ(scene.Get<double>("/A", tokens::kSize, 0.0), 2.0);
    // A new edit drops what was undone.
    history.Set(scene, "/A", tokens::kSize, 5.0);
    EXPECT_FALSE(history.CanRedo());
    EXPECT_EQ(history.Size(), 2u);
}

TEST(EditHistory, OpenEditsMergeUntilClosed) {
    Scene scene;
    ernie::Cube::Define(scene, "/A").SetSize(1.0);
    EditHistory history;
    for (double size = 1.5; size <= 3.0; size += 0.5) {
        history.Set(scene, "/A", tokens::kSize, size, true);
    }
    EXPECT_EQ(history.Size(), 1u) << "one drag";
    history.Close();
    history.Set(scene, "/A", tokens::kSize, 4.0, true);
    EXPECT_EQ(history.Size(), 2u) << "a new drag after closing";
    history.Set(scene, "/A", tokens::kDisplayColor, glm::vec3(1.0f),
                true);
    EXPECT_EQ(history.Size(), 3u) << "another attribute";
    history.Undo(scene);
    history.Undo(scene);
    EXPECT_EQ(scene.Get<double>("/A", tokens::kSize, 0.0), 3.0);
    history.Undo(scene);
    EXPECT_EQ(scene.Get<double>("/A", tokens::kSize, 0.0), 1.0)
        << "the whole drag";
}

TEST(EditHistory, DefinesAndRemovesUndoWithTheirSubtrees) {
    Scene scene;
    EditHistory history;
    history.BeginGroup();
    history.Define(scene, "/World/Box", tokens::kCube);
    history.Set(scene, "/World/Box", tokens::kSize, 3.0);
    history.EndGroup();
    EXPECT_EQ(history.UndoLabel(), "Create /World/Box");
    EXPECT_THROW(history.Define(scene, "/World/Box", tokens::kCube),
                 std::invalid_argument);
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_FALSE(scene.Has("/World/Box"))
        << "the group undoes together";
    ASSERT_TRUE(history.Redo(scene));
    EXPECT_EQ(scene.Get<double>("/World/Box", tokens::kSize, 0.0),
              3.0);

    const std::array<float, 3> points{1, 2, 3};
    ernie::Mesh::Define(scene, "/World/Box/Child")
        .SetPoints(ernie::Tensor::Copy<float>(points, {1, 3}));
    ASSERT_TRUE(history.Remove(scene, "/World"));
    EXPECT_FALSE(scene.Has("/World/Box"));
    EXPECT_FALSE(history.Remove(scene, "/World"));
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.TypeOf("/World/Box"), tokens::kCube);
    EXPECT_EQ(scene.Get<double>("/World/Box", tokens::kSize, 0.0),
              3.0);
    ASSERT_NE(scene.Read("/World/Box/Child", tokens::kPoints),
              nullptr);
    EXPECT_EQ(
        scene.Read("/World/Box/Child", tokens::kPoints)->Elements(),
        3);
    ASSERT_TRUE(history.Redo(scene));
    EXPECT_FALSE(scene.Has("/World"));
}

TEST(EditHistory, AddsPrimsAuthoredInAnotherScene) {
    Scene scratch;
    const std::array<float, 3> points{1, 2, 3};
    scratch.Define("/World/Mesh", tokens::kMesh)
        .Set(tokens::kDisplayColor, glm::vec3(1.0f))
        .Write(tokens::kPoints,
               ernie::Tensor::Copy<float>(points, {1, 3}));
    ernie::Cube::Define(scratch, "/World/Mesh/Child");
    Scene scene;
    EditHistory history;
    history.Add(scene, scratch, "/World/Mesh");
    EXPECT_EQ(scene.TypeOf("/World/Mesh/Child"), tokens::kCube);
    ASSERT_NE(scene.Read("/World/Mesh", tokens::kPoints), nullptr);
    EXPECT_EQ(history.UndoLabel(), "Create /World/Mesh");
    EXPECT_THROW(history.Add(scene, scratch, "/World/Mesh"),
                 std::invalid_argument);
    history.Undo(scene);
    EXPECT_FALSE(scene.Has("/World/Mesh"));
    history.Redo(scene);
    EXPECT_EQ(scene.Get<glm::vec3>("/World/Mesh",
                                   tokens::kDisplayColor,
                                   glm::vec3(0.0f)),
              glm::vec3(1.0f));
}

TEST(EditHistory, SkipsPrimsRemovedSince) {
    Scene scene;
    ernie::Cube::Define(scene, "/A");
    ernie::Cube::Define(scene, "/B");
    EditHistory history;
    history.Set(scene, "/A", tokens::kSize, 2.0);
    history.Set(scene, "/B", tokens::kSize, 3.0);
    scene.Remove("/B");
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.Get(Path("/A"), tokens::kSize), nullptr);
    EXPECT_FALSE(scene.Has("/B"));
}
