// SPDX-License-Identifier: BSD-3-Clause
/// @file path_test.cpp
/// @brief Tests of prim paths, interned names and the path table.
#include "ernie/scene/path.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace ernie {
namespace {

TEST(Path, ParsesAbsolutePathsOfIdentifiers) {
    EXPECT_EQ(Path("/World/Sun_2").String(), "/World/Sun_2");
    EXPECT_TRUE(Path("/").IsRoot());
    EXPECT_TRUE(Path().IsEmpty());
    for (const char* bad : {"", "World", "/World/", "//a", "/2a", "/a b", "/a-b", "/a/./b"}) {
        EXPECT_THROW(Path{bad}, std::invalid_argument) << bad;
    }
    EXPECT_TRUE(Path::IsValidName("_x9"));
    EXPECT_FALSE(Path::IsValidName(""));
}

TEST(Path, NavigatesNamesParentsAndChildren) {
    const Path sun("/World/Sun");
    EXPECT_EQ(sun.Name(), "Sun");
    EXPECT_EQ(sun.Parent(), Path("/World"));
    EXPECT_EQ(Path("/World").Parent(), Path::Root());
    EXPECT_TRUE(Path::Root().Parent().IsEmpty());
    EXPECT_TRUE(Path().Parent().IsEmpty());
    EXPECT_EQ(Path::Root().Name(), "");
    EXPECT_EQ(Path::Root().Child("World"), Path("/World"));
    EXPECT_EQ(Path("/World").Child("Sun"), sun);
    EXPECT_THROW(Path().Child("x"), std::invalid_argument);
    EXPECT_THROW(sun.Child("a/b"), std::invalid_argument);
}

TEST(Path, PrefixesRespectNameBoundaries) {
    EXPECT_TRUE(Path("/World/Sun").HasPrefix(Path("/World")));
    EXPECT_TRUE(Path("/World").HasPrefix(Path("/World")));
    EXPECT_TRUE(Path("/World").HasPrefix(Path::Root()));
    EXPECT_FALSE(Path("/Worlds").HasPrefix(Path("/World")));
    EXPECT_FALSE(Path("/World").HasPrefix(Path()));
    EXPECT_FALSE(Path().HasPrefix(Path::Root()));
}

TEST(Path, OrdersDescendantsRightAfterTheirAncestorAndHashes) {
    EXPECT_LT(Path("/a"), Path("/a/b"));
    EXPECT_LT(Path("/a/z"), Path("/a0"));
    EXPECT_LT(Path("/a/z"), Path("/a_"));
    const std::unordered_set<Path> set{Path("/a"), Path("/a"), Path("/b")};
    EXPECT_EQ(set.size(), 2u);
}

TEST(TokenTable, InternsEachNameOnce) {
    TokenTable tokens;
    const Token sun = tokens.Intern("Sun");
    EXPECT_NE(sun, 0u) << "0 is no name";
    EXPECT_EQ(tokens.Intern("Sun"), sun);
    EXPECT_NE(tokens.Intern("Moon"), sun);
    EXPECT_EQ(tokens.Text(sun), "Sun");
    EXPECT_EQ(tokens.Find("Sun"), sun);
    EXPECT_FALSE(tokens.Find("Earth"));
}

TEST(PathTable, InternsPathsAsNodesWithChildrenInOrder) {
    PathTable paths;
    const PathId wheel = paths.Intern("/World/Rover/Wheel_FL");
    const PathId rover = paths.Parent(wheel);
    EXPECT_EQ(paths.ToPath(rover), Path("/World/Rover"));
    EXPECT_EQ(paths.Intern("/World/Rover/Wheel_FL"), wheel);
    EXPECT_EQ(paths.Find("/World/Rover"), rover);
    EXPECT_FALSE(paths.Find("/World/Lander"));
    EXPECT_EQ(paths.ToPath(kRootPath), Path::Root());
    EXPECT_EQ(paths.Find("/"), kRootPath);

    const PathId fr = paths.Child(rover, "Wheel_FR");
    std::vector<PathId> children;
    for (PathId child = paths.FirstChild(rover); child != kNoPath;
         child = paths.NextSibling(child)) {
        children.push_back(child);
    }
    EXPECT_EQ(children, (std::vector<PathId>{wheel, fr}));
    EXPECT_TRUE(paths.HasPrefix(wheel, rover));
    EXPECT_TRUE(paths.HasPrefix(wheel, kRootPath));
    EXPECT_FALSE(paths.HasPrefix(rover, wheel));
    EXPECT_THROW(paths.Intern(Path()), std::invalid_argument);
}

}  // namespace
}  // namespace ernie
