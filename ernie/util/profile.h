#pragma once

// CPU zones for the Tracy profiler in a profiling build (configure with -DERNIE_TRACY=ON, run
// Ernie, and connect the Tracy profiler; nothing is collected until it connects). Without
// ERNIE_TRACY the macros compile to nothing. GPU zones are in ernie/device/gpu_profile.h.
//
//     ERNIE_ZONE("Sync");               // the rest of the scope, one per scope
//     ERNIE_ZONE_TEXT(path.String());   // text attached to the scope's zone
//     ERNIE_PLOT("views", 256);         // a value plotted over time
//     ERNIE_FRAME_MARK("Step");         // the end of a frame of a named frame set
//
// References:
// - B. Taudul, "Tracy Profiler". https://github.com/wolfpld/tracy

#if ERNIE_TRACY
#include <tracy/Tracy.hpp>

#define ERNIE_ZONE(name) ZoneScopedN(name)
#define ERNIE_ZONE_TEXT(text)                                                                      \
    do {                                                                                           \
        const auto& ernieZoneText = (text);                                                        \
        ZoneText(ernieZoneText.data(), ernieZoneText.size());                                      \
    } while (false)
#define ERNIE_PLOT(name, value) TracyPlot(name, static_cast<double>(value))
#define ERNIE_FRAME_MARK(name) FrameMarkNamed(name)
#else
#define ERNIE_ZONE(name)
#define ERNIE_ZONE_TEXT(text)
#define ERNIE_PLOT(name, value)
#define ERNIE_FRAME_MARK(name)
#endif
