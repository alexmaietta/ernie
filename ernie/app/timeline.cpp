// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/timeline.h"

#include "ernie/scene/stage_schemas.h"

#include <algorithm>
#include <cmath>

namespace ernie::app {

namespace {

/// Time codes Kit's new stages run for.
constexpr double kDefaultLength = 100.0;

}  // namespace

std::pair<double, double> Timeline::Range(const Scene& scene) {
    const Stage stage(scene);
    const double start = stage.GetStartTimeCode();
    const double end = stage.GetEndTimeCode();
    return {start, end > start ? end : start + kDefaultLength};
}

double Timeline::FrameTimeCodes(const Scene& scene) {
    const Stage stage(scene);
    return stage.GetTimeCodesPerSecond() / stage.GetFramesPerSecond();
}

double Timeline::TimeCode(const Scene& scene) {
    return Stage(scene).GetTimeCode();
}

void Timeline::Stop(Scene& scene) {
    playing_ = false;
    Seek(scene, Range(scene).first);
}

void Timeline::Seek(Scene& scene, double timeCode) {
    if (TimeCode(scene) != timeCode) {
        Stage(scene).SetTimeCode(timeCode);
    }
}

void Timeline::StepFrames(Scene& scene, int frames) {
    playing_ = false;
    const auto [start, end] = Range(scene);
    Seek(scene,
         std::clamp(TimeCode(scene) + frames * FrameTimeCodes(scene),
                    start, end));
}

double Timeline::Advance(Scene& scene, double seconds) {
    if (!playing_) {
        return 0.0;
    }
    const Stage stage(scene);
    const double perSecond = stage.GetTimeCodesPerSecond();
    const auto [start, end] = Range(scene);
    double now = TimeCode(scene);
    if (now < start || now > end) {
        now = start;
    }
    const double step = everyFrame_
                            ? FrameTimeCodes(scene)
                            : std::max(seconds, 0.0) * perSecond;
    double next = now + step;
    if (next > end) {
        if (looping_) {
            next = start + std::fmod(next - start, end - start);
        } else {
            next = end;
            playing_ = false;
        }
    }
    Seek(scene, next);
    return step / perSecond;
}

}  // namespace ernie::app
