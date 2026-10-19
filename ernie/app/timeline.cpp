// SPDX-License-Identifier: BSD-3-Clause
/// @file timeline.cpp
/// @brief Playing, stepping and seeking the scene's time code.
#include "ernie/app/timeline.h"

#include <algorithm>
#include <cmath>

namespace ernie::app {

void Timeline::SetRange(double start, double end) {
    start_ = start;
    end_ = end > start ? end : start + 1.0;
}

void Timeline::SetFramesPerSecond(double fps) {
    fps_ = std::max(fps, 1e-3);
}

void Timeline::Stop(Scene& scene) {
    playing_ = false;
    Seek(scene, start_);
}

void Timeline::Seek(Scene& scene, double timeCode) {
    if (scene.Time() != timeCode) {
        scene.SetTime(timeCode);
    }
}

void Timeline::StepFrames(Scene& scene, int frames) {
    playing_ = false;
    Seek(scene, std::clamp(std::round(scene.Time()) + frames, start_, end_));
}

double Timeline::Advance(Scene& scene, double seconds) {
    if (!playing_) {
        return 0.0;
    }
    double now = scene.Time();
    if (now < start_ || now > end_) {
        now = start_;
    }
    const double step = everyFrame ? 1.0 : std::max(seconds, 0.0) * fps_;
    double next = now + step;
    if (next > end_) {
        if (looping) {
            next = start_ + std::fmod(next - start_, end_ - start_);
        } else {
            next = end_;
            playing_ = false;
        }
    }
    Seek(scene, next);
    return step / fps_;
}

}  // namespace ernie::app
