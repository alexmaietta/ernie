// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <string>

// clang-format off
#define ERNIE_STAGE_ATTRIBUTES(X)                                                                  \
    X(std::string, UpAxis, ::ernie::tokens::kUpAxis, ::ernie::tokens::kZ,                          \
      (::ernie::AttributeMeta{.label = "Up axis",                                                  \
                              .doc = "The axis that points up. Ernie draws z up.",                 \
                              .group = "Units",                                                    \
                              .options = {::ernie::tokens::kY, ::ernie::tokens::kZ}}))             \
    X(double, MetersPerUnit, ::ernie::tokens::kMetersPerUnit, 1.0,                                 \
      (::ernie::AttributeMeta{.label = "Metres per unit",                                          \
                              .doc = "Metres in one scene unit. Ernie draws in metres.",           \
                              .group = "Units", .min = 1e-9, .max = 1e9}))                         \
    X(int, SubdivisionRefineLevel, ::ernie::tokens::kSubdivisionRefineLevel, 1,                    \
      (::ernie::AttributeMeta{.label = "Subdivision level",                                        \
                              .doc = "Refinement of subdivision surfaces without their own "       \
                                     "ernie:refineLevel: 2^level segments along each control "     \
                                     "edge (usdview's complexity low to very high is 0 to 3).",    \
                              .group = "Subdivision", .min = 0.0, .max = 8.0}))                    \
    X(double, KilogramsPerUnit, ::ernie::tokens::kKilogramsPerUnit, 1.0,                           \
      (::ernie::AttributeMeta{.label = "Kilograms per unit",                                       \
                              .doc = "Kilograms in one unit of mass.",                             \
                              .group = "Units", .min = 1e-9, .max = 1e9}))                         \
    X(double, TimeCodesPerSecond, ::ernie::tokens::kTimeCodesPerSecond, 24.0,                      \
      (::ernie::AttributeMeta{.label = "Time codes per second",                                    \
                              .doc = "Time codes in one second of animation.",                     \
                              .group = "Time", .min = 1e-6, .max = 1e9}))                          \
    X(double, FramesPerSecond, ::ernie::tokens::kFramesPerSecond, 24.0,                            \
      (::ernie::AttributeMeta{.label = "Frames per second",                                        \
                              .doc = "The rate animation is meant to play at.",                    \
                              .group = "Time", .min = 1e-6, .max = 1e9}))                          \
    X(double, StartTimeCode, ::ernie::tokens::kStartTimeCode, 0.0,                                 \
      (::ernie::AttributeMeta{.label = "Start time code",                                          \
                              .doc = "Where animation starts.", .group = "Time"}))                 \
    X(double, EndTimeCode, ::ernie::tokens::kEndTimeCode, 0.0,                                     \
      (::ernie::AttributeMeta{.label = "End time code",                                            \
                              .doc = "Where animation ends.", .group = "Time"}))                   \
    X(double, TimeCode, ::ernie::tokens::kTimeCode, 0.0,                                           \
      (::ernie::AttributeMeta{.label = "Time code",                                                \
                              .doc = "The time code animation is at, apart from a renderer's "     \
                                     "sensor clock.",                                              \
                              .group = "Time"}))                                                   \
    X(std::string, DefaultPrim, ::ernie::tokens::kDefaultPrim, "",                                 \
      (::ernie::AttributeMeta{.label = "Default prim",                                             \
                              .doc = "The root prim a reference to the scene brings in, by name.", \
                              .group = "Composition"}))
// clang-format on

namespace ernie {

/// A scene's stage metadata, as UsdStage keeps it on its pseudo-root:
/// its units (the up axis, metres and kilograms per unit), its time
/// (time codes and frames per second, the range animation plays
/// over, and the time code it is at) and its default prim. They are
/// attributes of the scene's root prim `/`, which stays untyped: the
/// root's schema is this one, so `Scene::Set` checks them, edits are
/// tracked and undone as any other, and scene files keep them under
/// "/". The view is not a prim type (`PrimTypes`): no prim is
/// defined as one.
///
/// Fallbacks are what Ernie draws with: z up and metres, which USD's
/// own fallbacks (y up, centimetres) are not, so a scene from USD
/// states its units. Ernie records other units for interchange
/// without converting them yet.
///
/// The time code is the animation's clock, which an application sets
/// as ovstage's `apply_usd_time` does (`SetTime`, in seconds); a
/// render delegate's sensor clock (`RenderDelegate::Time`), which
/// its steps advance, is apart from it, so animation can hold or
/// loop while sensors run on.
///
/// References:
/// - Pixar, "UsdStage" (stage metadata),
///   https://openusd.org/release/api/class_usd_stage.html
/// - Pixar, "Encoding Stage UpAxis" and "Encoding Stage Linear
///   Units",
///   https://openusd.org/release/api/group___usd_geom_up_axis__group.html
/// - NVIDIA, ovstage `ovstage_population_apply_usd_time`.
class Stage : public SchemaView {
    ERNIE_SCHEMA_BODY(Stage, SchemaView, ERNIE_STAGE_ATTRIBUTES)

    /// The metadata of `scene`: a view of its root prim.
    explicit Stage(const Scene& scene) : Stage(scene, Path::Root()) {}
    explicit Stage(Scene& scene) : Stage(scene, Path::Root()) {}

    /// The animation's time, seconds: the time code over the time
    /// codes per second.
    double GetTime() const {
        return GetTimeCode() / GetTimeCodesPerSecond();
    }

    /// Sets the time code to `seconds` of animation.
    Stage& SetTime(double seconds) {
        return SetTimeCode(seconds * GetTimeCodesPerSecond());
    }
};

}  // namespace ernie
