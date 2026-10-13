// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/profile_report.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>

namespace ernie::profiler {

namespace {

/// Adds `zones`, ordered as they started, to the tree under `roots`,
/// each under the zone one level out that ran last; `ms` and `calls`
/// gather sums (nanoseconds) to be averaged.
void Gather(std::vector<ReportNode>& roots,
            std::vector<const Zone*> zones) {
    std::sort(zones.begin(), zones.end(),
              [](const Zone* a, const Zone* b) {
                  return a->start != b->start ? a->start < b->start
                                              : a->depth < b->depth;
              });
    std::vector<ReportNode*> path;
    for (const Zone* zone : zones) {
        path.resize(std::min<std::size_t>(zone->depth, path.size()));
        std::vector<ReportNode>& siblings =
            path.empty() ? roots : path.back()->children;
        auto node = std::find_if(siblings.begin(), siblings.end(),
                                 [&](const ReportNode& n) {
                                     return n.name == zone->name;
                                 });
        if (node == siblings.end()) {
            siblings.emplace_back().name = zone->name;
            node = siblings.end() - 1;
        }
        node->ms += static_cast<double>(zone->end - zone->start);
        node->calls += 1.0;
        path.push_back(&*node);
    }
}

/// Turns `Gather`'s sums over `frames` frames into means and shares
/// of `totalMs`.
void Average(std::vector<ReportNode>& nodes, double frames,
             double totalMs) {
    for (ReportNode& node : nodes) {
        node.ms = node.ms * 1e-6 / frames;
        node.calls /= frames;
        node.percent =
            totalMs > 0.0 ? 100.0 * node.ms / totalMs : 0.0;
        Average(node.children, frames, totalMs);
    }
}

void AddRows(const std::vector<ReportNode>& nodes,
             const ReportFilter& filter, std::uint32_t depth,
             std::vector<ReportRow>& rows) {
    // Siblings as they ran, or longest first (ties as they ran).
    std::vector<const ReportNode*> order;
    order.reserve(nodes.size());
    for (const ReportNode& node : nodes) {
        order.push_back(&node);
    }
    if (filter.byTime) {
        std::stable_sort(
            order.begin(), order.end(),
            [](const ReportNode* a, const ReportNode* b) {
                return a->ms > b->ms;
            });
    }
    for (const ReportNode* node : order) {
        if (node->ms < filter.minMs) {
            continue;
        }
        const std::size_t at = rows.size();
        rows.push_back({node, depth, false});
        if (depth < filter.depth) {
            AddRows(node->children, filter, depth + 1, rows);
        }
        rows[at].parent = rows.size() > at + 1;
    }
}

nlohmann::json ToJson(const std::vector<ReportNode>& nodes) {
    nlohmann::json zones = nlohmann::json::array();
    for (const ReportNode& node : nodes) {
        zones.push_back({{"name", node.name},
                         {"ms", node.ms},
                         {"calls", node.calls},
                         {"percent", node.percent},
                         {"children", ToJson(node.children)}});
    }
    return zones;
}

std::string ThreadName(const std::vector<std::string>& names,
                       std::uint32_t thread) {
    return thread < names.size() ? names[thread]
                                 : "Thread " + std::to_string(thread);
}

}  // namespace

Report Summarize(std::span<const Frame> frames,
                 const std::vector<std::string>& threadNames) {
    Report report;
    std::map<std::uint32_t, std::vector<ReportNode>> threads;
    for (const Frame& frame : frames) {
        report.cpuMs += static_cast<double>(frame.end - frame.start);
        ++report.cpuFrames;
        std::map<std::uint32_t, std::vector<const Zone*>> byThread;
        for (const Zone& zone : frame.cpu) {
            byThread[zone.thread].push_back(&zone);
        }
        for (auto& [thread, zones] : byThread) {
            Gather(threads[thread], std::move(zones));
        }
        if (frame.gpu.empty()) {
            continue;
        }
        ++report.gpuFrames;
        std::vector<const Zone*> zones;
        for (const Zone& zone : frame.gpu) {
            zones.push_back(&zone);
            if (zone.depth == 0) {
                report.gpuMs +=
                    static_cast<double>(zone.end - zone.start);
            }
        }
        Gather(report.gpu, std::move(zones));
    }
    if (report.gpuFrames > 0) {
        const double count = static_cast<double>(report.gpuFrames);
        report.gpuMs = report.gpuMs * 1e-6 / count;
        Average(report.gpu, count, report.gpuMs);
    }
    if (report.cpuFrames > 0) {
        const double count = static_cast<double>(report.cpuFrames);
        report.cpuMs = report.cpuMs * 1e-6 / count;
        for (auto& [thread, roots] : threads) {
            Average(roots, count, report.cpuMs);
            ReportNode node;
            node.name = ThreadName(threadNames, thread);
            for (const ReportNode& root : roots) {
                node.ms += root.ms;
            }
            node.calls = 1.0;
            node.percent = report.cpuMs > 0.0
                               ? 100.0 * node.ms / report.cpuMs
                               : 0.0;
            node.children = std::move(roots);
            report.cpu.push_back(std::move(node));
        }
    }
    return report;
}

std::vector<ReportRow> Rows(const std::vector<ReportNode>& nodes,
                            const ReportFilter& filter) {
    std::vector<ReportRow> rows;
    AddRows(nodes, filter, 0, rows);
    return rows;
}

std::string ReportText(const Report& report,
                       const ReportFilter& filter) {
    std::string text;
    char line[256];
    const auto section = [&](const char* title,
                             const std::vector<ReportNode>& nodes,
                             double totalMs, std::size_t frames) {
        // The section's total heads it; its zones below, indented.
        std::snprintf(line, sizeof(line), "%-56s %12s %10s\n", "",
                      "Duration (ms)", "Percent");
        text += line;
        std::snprintf(
            line, sizeof(line), "%-56s %12.3f   (%zu frames)\n",
            (std::string(title) + " total").c_str(), totalMs, frames);
        text += line;
        for (const ReportRow& row : Rows(nodes, filter)) {
            const std::string name =
                std::string(2 * (row.depth + 1), ' ') +
                row.node->name;
            std::snprintf(line, sizeof(line),
                          "%-56s %12.3f %9.2f%%\n", name.c_str(),
                          row.node->ms, row.node->percent);
            text += line;
        }
        text += "\n";
    };
    section("GPU", report.gpu, report.gpuMs, report.gpuFrames);
    section("CPU", report.cpu, report.cpuMs, report.cpuFrames);
    return text;
}

std::string ReportJson(const Report& report) {
    const nlohmann::json json = {{"gpu",
                                  {{"ms", report.gpuMs},
                                   {"frames", report.gpuFrames},
                                   {"zones", ToJson(report.gpu)}}},
                                 {"cpu",
                                  {{"ms", report.cpuMs},
                                   {"frames", report.cpuFrames},
                                   {"zones", ToJson(report.cpu)}}}};
    return json.dump(2);
}

std::string TraceJson(std::span<const Frame> frames,
                      const std::vector<std::string>& threadNames) {
    constexpr int kCpu = 1;
    constexpr int kGpu = 2;
    nlohmann::json events = nlohmann::json::array();
    events.push_back({{"name", "process_name"},
                      {"ph", "M"},
                      {"pid", kCpu},
                      {"args", {{"name", "CPU"}}}});
    events.push_back({{"name", "process_name"},
                      {"ph", "M"},
                      {"pid", kGpu},
                      {"args", {{"name", "GPU"}}}});
    events.push_back({{"name", "thread_name"},
                      {"ph", "M"},
                      {"pid", kGpu},
                      {"tid", 0},
                      {"args", {{"name", "Queue"}}}});
    if (frames.empty()) {
        return nlohmann::json{{"traceEvents", events}}.dump();
    }
    const std::uint64_t origin = frames.front().start;
    const auto us = [](double ns) { return ns * 1e-3; };
    std::map<std::uint32_t, bool> threads;
    for (const Frame& frame : frames) {
        events.push_back(
            {{"name", "Frame " + std::to_string(frame.index)},
             {"ph", "X"},
             {"pid", kCpu},
             {"tid", "frames"},
             {"ts", us(static_cast<double>(frame.start - origin))},
             {"dur",
              us(static_cast<double>(frame.end - frame.start))}});
        for (const Zone& zone : frame.cpu) {
            threads[zone.thread] = true;
            events.push_back(
                {{"name", zone.name},
                 {"ph", "X"},
                 {"pid", kCpu},
                 {"tid", zone.thread},
                 {"ts", us(static_cast<double>(zone.start - origin))},
                 {"dur",
                  us(static_cast<double>(zone.end - zone.start))}});
        }
        std::uint64_t base = UINT64_MAX;
        for (const Zone& zone : frame.gpu) {
            base = std::min(base, zone.start);
        }
        for (const Zone& zone : frame.gpu) {
            events.push_back(
                {{"name", zone.name},
                 {"ph", "X"},
                 {"pid", kGpu},
                 {"tid", 0},
                 {"ts", us(static_cast<double>(frame.start - origin) +
                           static_cast<double>(zone.start - base))},
                 {"dur",
                  us(static_cast<double>(zone.end - zone.start))}});
        }
    }
    for (const auto& [thread, used] : threads) {
        events.push_back(
            {{"name", "thread_name"},
             {"ph", "M"},
             {"pid", kCpu},
             {"tid", thread},
             {"args", {{"name", ThreadName(threadNames, thread)}}}});
    }
    return nlohmann::json{{"traceEvents", events},
                          {"displayTimeUnit", "ms"}}
        .dump();
}

}  // namespace ernie::profiler
