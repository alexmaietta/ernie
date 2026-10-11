// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/scene.h"
#include "ernie/scene/scene_json.h"
#include "ernie/scene/stage_schemas.h"
#include "ernie/scene/tokens.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using ernie::Path;
using ernie::Scene;
using ernie::Stage;

TEST(Stage, ItsFallbacksAreWhatErnieDraws) {
    const Scene scene;
    const Stage stage(scene);
    EXPECT_EQ(stage.GetUpAxis(), "Z");
    EXPECT_EQ(stage.GetMetersPerUnit(), 1.0);
    EXPECT_EQ(stage.GetKilogramsPerUnit(), 1.0);
    EXPECT_EQ(stage.GetTimeCodesPerSecond(), 24.0);
    EXPECT_EQ(stage.GetFramesPerSecond(), 24.0);
    EXPECT_EQ(stage.GetTimeCode(), 0.0);
    EXPECT_EQ(stage.GetDefaultPrim(), "");
    EXPECT_EQ(scene.TypeOf(Path::Root()), "")
        << "the root stays untyped";
    EXPECT_EQ(scene.SchemaOf(Path::Root()), &Stage::Definition());
}

TEST(Stage, IsTheRootsAttributesCheckedAndTracked) {
    Scene scene;
    Stage stage(scene);
    const std::uint64_t before = scene.Version();
    stage.SetUpAxis("Y").SetMetersPerUnit(0.01).SetTimeCodesPerSecond(
        60.0);
    EXPECT_GT(scene.Version(), before);
    EXPECT_EQ(*scene.Get(Path::Root(), ernie::tokens::kUpAxis)
                   ->Get<std::string>(),
              "Y");
    EXPECT_EQ(Stage(scene).GetMetersPerUnit(), 0.01);

    // The time code is the animation's clock: 2.5 s at 60 per second.
    stage.SetTime(2.5);
    EXPECT_EQ(stage.GetTimeCode(), 150.0);
    EXPECT_EQ(stage.GetTime(), 2.5);

    EXPECT_THROW(scene.Set(Path::Root(), ernie::tokens::kUpAxis,
                           ernie::Value(3.0)),
                 std::invalid_argument)
        << "a token, not a number";
    EXPECT_THROW(
        scene.Set(Path::Root(), "metresPerUnit", ernie::Value(1.0)),
        std::invalid_argument)
        << "a misspelt name";
    EXPECT_THROW(Stage(scene, Path("/World")), std::invalid_argument);
    scene.Define(Path("/World"), "Xform");
    EXPECT_THROW(Stage(scene, Path("/World")), std::invalid_argument)
        << "only the root holds it";
}

TEST(Stage, SceneFilesKeepItUnderTheRoot) {
    Scene scene;
    Stage(scene)
        .SetUpAxis("Y")
        .SetTimeCodesPerSecond(30.0)
        .SetTimeCode(12.0);
    scene.Define(Path("/World"), "Xform");
    const std::string text = ernie::SaveScene(scene);
    EXPECT_NE(text.find("\"/\""), std::string::npos) << text;

    Scene loaded;
    ernie::LoadScene(loaded, text);
    const Stage stage(loaded);
    EXPECT_EQ(stage.GetUpAxis(), "Y");
    EXPECT_EQ(stage.GetTimeCodesPerSecond(), 30.0);
    EXPECT_EQ(stage.GetTimeCode(), 12.0);
    EXPECT_TRUE(loaded.Has(Path("/World")));

    // Unauthored, a file has no root entry.
    EXPECT_EQ(ernie::SaveScene(Scene()).find("\"/\""),
              std::string::npos);
}
