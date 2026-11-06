// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/viewer/shell/history.h"

#include <string>

#include <gtest/gtest.h>

namespace ernie::app {
namespace {

Edit Adding(int& value, int amount) {
    return {"add " + std::to_string(amount), [&value, amount] { value -= amount; },
            [&value, amount] { value += amount; }};
}

TEST(HistoryTest, UndoesAndRedoesInOrder) {
    History history;
    int value = 0;
    EXPECT_FALSE(history.Undo());
    history.Do(Adding(value, 1));
    history.Do(Adding(value, 10));
    EXPECT_EQ(value, 11);
    EXPECT_EQ(*history.NextUndo(), "add 10");

    EXPECT_TRUE(history.Undo());
    EXPECT_EQ(value, 1);
    EXPECT_EQ(*history.NextRedo(), "add 10");
    EXPECT_TRUE(history.Redo());
    EXPECT_EQ(value, 11);
    EXPECT_FALSE(history.Redo());

    history.Undo();
    history.Do(Adding(value, 100));
    EXPECT_EQ(value, 101);
    EXPECT_EQ(history.NextRedo(), nullptr);
}

TEST(HistoryTest, KeepsOnlyTheNewestEdits) {
    History history;
    int value = 0;
    for (size_t i = 0; i < History::kKept + 5; ++i) {
        history.Do(Adding(value, 1));
    }
    size_t undone = 0;
    while (history.Undo()) {
        ++undone;
    }
    EXPECT_EQ(undone, History::kKept);
    EXPECT_EQ(value, 5);
    history.Clear();
    EXPECT_EQ(history.NextUndo(), nullptr);
    EXPECT_EQ(history.NextRedo(), nullptr);
}

}
}
