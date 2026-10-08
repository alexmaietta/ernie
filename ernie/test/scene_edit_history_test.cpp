#include "ernie/scene/edit_history.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

using ernie::EditHistory;
using ernie::Path;
using ernie::Scene;
namespace tokens = ernie::tokens;

TEST(EditHistory, UndoesAndRedoesSetsAndClears) {
    Scene scene;
    scene.Define("/A", tokens::kCube).Set(tokens::kSize, 1.0);
    EditHistory history;
    EXPECT_FALSE(history.CanUndo());
    history.Set(scene, "/A", tokens::kSize, 2.0);
    history.Set(scene, "/A", tokens::kDisplayColor, glm::vec3(1.0f, 0.0f, 0.0f));
    history.Clear(scene, "/A", tokens::kSize);
    EXPECT_EQ(scene.Get(Path("/A"), tokens::kSize), nullptr);
    EXPECT_EQ(history.UndoLabel(), "/A size");

    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.Get<double>("/A", tokens::kSize, 0.0), 2.0);
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.Get(Path("/A"), tokens::kDisplayColor), nullptr) << "unauthored again";
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
    scene.Define("/A", tokens::kCube).Set(tokens::kSize, 1.0);
    EditHistory history;
    for (double size = 1.5; size <= 3.0; size += 0.5) {
        history.Set(scene, "/A", tokens::kSize, size, true);
    }
    EXPECT_EQ(history.Size(), 1u) << "one drag";
    history.Close();
    history.Set(scene, "/A", tokens::kSize, 4.0, true);
    EXPECT_EQ(history.Size(), 2u) << "a new drag after closing";
    history.Set(scene, "/A", tokens::kDisplayColor, glm::vec3(1.0f), true);
    EXPECT_EQ(history.Size(), 3u) << "another attribute";
    history.Undo(scene);
    history.Undo(scene);
    EXPECT_EQ(scene.Get<double>("/A", tokens::kSize, 0.0), 3.0);
    history.Undo(scene);
    EXPECT_EQ(scene.Get<double>("/A", tokens::kSize, 0.0), 1.0) << "the whole drag";
}

TEST(EditHistory, SkipsPrimsRemovedSince) {
    Scene scene;
    scene.Define("/A", tokens::kCube);
    scene.Define("/B", tokens::kCube);
    EditHistory history;
    history.Set(scene, "/A", tokens::kSize, 2.0);
    history.Set(scene, "/B", tokens::kSize, 3.0);
    scene.Remove("/B");
    ASSERT_TRUE(history.Undo(scene));
    EXPECT_EQ(scene.Get(Path("/A"), tokens::kSize), nullptr);
    EXPECT_FALSE(scene.Has("/B"));
}
