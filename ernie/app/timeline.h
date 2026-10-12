// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/scene.h"

#include <utility>

namespace ernie::app {

/// The animation's clock, as Omniverse Kit's timeline: plays the
/// scene's time code (`Stage::GetTimeCode`) over the stage's range,
/// from its start time code to its end, in real time or a frame at a
/// time, looping or stopping at the end. It writes the time code into
/// the scene's root, as an application sets it, outside the undo
/// history; the renderer's sensor clock is apart from it.
///
/// Time codes run at the stage's time codes per second; a frame is
/// the stage's frames per second's share of a second of them.
///
/// Reference: NVIDIA, Omniverse Kit `omni.timeline` (play, pause,
/// stop, looping, play every frame).
class Timeline {
  public:
    /// The range the timeline plays: the stage's start and end time
    /// codes, or, while the end is not past the start, 100 time codes
    /// from the start, as Kit's new stages hold.
    static std::pair<double, double> Range(const Scene& scene);

    /// Time codes in one frame: the stage's time codes per second
    /// over its frames per second.
    static double FrameTimeCodes(const Scene& scene);

    /// The time code the scene's animation is at.
    static double TimeCode(const Scene& scene);

    void Play() {
        playing_ = true;
    }
    void Pause() {
        playing_ = false;
    }
    /// Pauses, back at the range's start.
    void Stop(Scene& scene);
    bool IsPlaying() const {
        return playing_;
    }

    /// Whether playing past the range's end starts over at its start;
    /// otherwise it stops at the end. On by default.
    bool IsLooping() const {
        return looping_;
    }
    void SetLooping(bool looping) {
        looping_ = looping;
    }

    /// Whether playing advances a frame each step, whatever time the
    /// step took, so every frame is drawn once (capture, slow
    /// renders); otherwise playback keeps to real time. Off by
    /// default.
    bool PlaysEveryFrame() const {
        return everyFrame_;
    }
    void SetPlayEveryFrame(bool everyFrame) {
        everyFrame_ = everyFrame;
    }

    /// Sets the time code to `timeCode`, playing or not.
    void Seek(Scene& scene, double timeCode);

    /// Pauses and moves `frames` frames on (back for negative ones),
    /// within the range.
    void StepFrames(Scene& scene, int frames);

    /// While playing, advances the time code by `seconds` of real
    /// time (a frame when playing every frame), from the range's
    /// start if it is outside it; past the end, loops or stops there.
    /// Returns the seconds of animation it advanced: 0 while paused.
    double Advance(Scene& scene, double seconds);

  private:
    bool playing_ = false;
    bool looping_ = true;
    bool everyFrame_ = false;
};

}  // namespace ernie::app
