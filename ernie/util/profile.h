// SPDX-License-Identifier: BSD-3-Clause
/// @file profile.h
/// @brief CPU profiling zones for Ernie's profiler and Tracy.
/// Zones feed the viewer's Profiler panel (profiler.h) and, with
/// -DERNIE_TRACY=ON, Tracy. GPU zones are in device/gpu_profile.h.
///
///     // The rest of the scope, one per scope; a string literal.
///     ERNIE_ZONE("Sync");
///     // Text attached to the scope's zone (Tracy only).
///     ERNIE_ZONE_TEXT(path.String());
///     // A value plotted over time (Tracy only).
///     ERNIE_PLOT("views", 256);
///     // The end of a frame of a named frame set (Tracy only).
///     ERNIE_FRAME_MARK("Step");
///
/// References:
/// - B. Taudul, "Tracy Profiler". https://github.com/wolfpld/tracy
#pragma once

#include "ernie/util/profiler.h"

#define ERNIE_PROFILE_CONCAT_(a, b) a##b
#define ERNIE_PROFILE_CONCAT(a, b) ERNIE_PROFILE_CONCAT_(a, b)
#define ERNIE_OWN_ZONE(name)                                                                       \
    const ::ernie::profiler::ScopedZone ERNIE_PROFILE_CONCAT(ernieZone, __LINE__)(name)

#if ERNIE_TRACY
#include <tracy/Tracy.hpp>

#define ERNIE_ZONE(name)                                                                           \
    ZoneScopedN(name);                                                                             \
    ERNIE_OWN_ZONE(name)
#define ERNIE_ZONE_TEXT(text)                                                                      \
    do {                                                                                           \
        const auto& ernieZoneText = (text);                                                        \
        ZoneText(ernieZoneText.data(), ernieZoneText.size());                                      \
    } while (false)
#define ERNIE_PLOT(name, value) TracyPlot(name, static_cast<double>(value))
#define ERNIE_FRAME_MARK(name) FrameMarkNamed(name)
#else
#define ERNIE_ZONE(name) ERNIE_OWN_ZONE(name)
#define ERNIE_ZONE_TEXT(text)
#define ERNIE_PLOT(name, value)
#define ERNIE_FRAME_MARK(name)
#endif
