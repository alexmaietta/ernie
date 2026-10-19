// SPDX-License-Identifier: BSD-3-Clause
/// @file timeline.h
/// @brief The animation clock, after Omniverse Kit's timeline.
#pragma once

#include "ernie/scene/scene.h"

namespace ernie::app {

/// @brief Plays the scene's time code over a range: in real time or a frame each step, looping or
/// stopping. It sets the scene's clock (`Scene::SetTime`), outside the undo history. Time codes
/// count frames, as USD's do when time codes per second equal frames per second.
///
/// References:
/// - NVIDIA, Omniverse Kit `omni.timeline` (play, pause, stop, looping, play every frame).
class Timeline {
  public:
    double Start() const {
        return start_;
    }
    double End() const {
        return end_;
    }
    /// @brief Sets the range; an end not past the start is a frame past it.
    void SetRange(double start, double end);
    double FramesPerSecond() const {
        return fps_;
    }
    void SetFramesPerSecond(double fps);

    void Play() {
        playing_ = true;
    }
    void Pause() {
        playing_ = false;
    }
    /// @brief Pauses at the range's start.
    void Stop(Scene& scene);
    bool Playing() const {
        return playing_;
    }
    /// Whether play restarts at the end rather than stopping; on by default.
    bool looping = true;
    /// Whether each step advances one frame, so every frame is drawn, as captures need.
    bool everyFrame = false;

    /// @brief Sets the time code, playing or not.
    void Seek(Scene& scene, double timeCode);
    /// @brief Pauses and steps whole frames within the range; negative steps back.
    void StepFrames(Scene& scene, int frames);
    /// @brief Advances while playing, from the start when outside the range, looping or stopping
    /// past the end.
    /// @param seconds Real time elapsed, unless playing every frame.
    /// @return Seconds of animation advanced; 0 while paused.
    double Advance(Scene& scene, double seconds);

  private:
    double start_ = 0.0;
    double end_ = 100.0;
    double fps_ = 24.0;
    bool playing_ = false;
};

}  // namespace ernie::app
