// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/sensor_chain.h"

#include "ernie/scene/render_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"

#include <algorithm>
#include <array>
#include <exception>
#include <optional>
#include <utility>
#include <vector>

namespace ernie::app {

namespace {

/// `base`, or it with the first number after it that names no prim.
Path FreePath(const Scene& scene, const std::string& base) {
    Path path(base);
    for (int n = 1; scene.Has(path); ++n) {
        path = Path(base + std::to_string(n));
    }
    return path;
}

/// Whether any RenderProduct or RenderVar's chain names `filter`.
bool Referenced(const Scene& scene, const Path& filter) {
    const auto names = [&](const std::vector<Path>& chain) {
        return std::ranges::find(chain, filter) != chain.end();
    };
    for (const Path& path : scene.Prims(tokens::kRenderProduct)) {
        if (names(RenderProduct(scene, path).GetFilters())) {
            return true;
        }
    }
    for (const Path& path : scene.Prims(tokens::kRenderVar)) {
        if (names(RenderVar(scene, path).GetFilters())) {
            return true;
        }
    }
    return false;
}

/// What filter `type` reads and what it leaves: `qe` reads the
/// spectrum and leaves electrons, `noise` reads and leaves electrons;
/// none for a type neither built in nor in `registry`.
std::optional<std::pair<FilterDomain, FilterDomain>>
DomainsOf(std::string_view type,
          const SensorFilterRegistry& registry) {
    using enum FilterDomain;
    if (const std::optional<SensorFilter> builtin =
            SensorFilterNamed(type)) {
        switch (*builtin) {
        case SensorFilter::Lens:
            return std::pair{Spectrum, Spectrum};
        case SensorFilter::Qe:
            return std::pair{Spectrum, Electrons};
        case SensorFilter::Noise:
            return std::pair{Electrons, Electrons};
        case SensorFilter::Adc:
            return std::pair{Electrons, Digital};
        case SensorFilter::Cie:
            return std::pair{Spectrum, Rgb};
        case SensorFilter::Exposure:
            return std::pair{Rgb, Display};
        }
    }
    const SensorFilterModel* model = registry.Find(type);
    if (model == nullptr) {
        return std::nullopt;
    }
    if (model->kind == FilterKind::Integrator) {
        return std::pair{Spectrum, model->domain};
    }
    return std::pair{model->domain, model->domain};
}

/// Whether the SensorFilter prims `filters` make a valid chain.
bool Valid(const Scene& scene, const std::vector<Path>& filters,
           const SensorFilterRegistry& registry) {
    try {
        CheckFilterChain(SensorFilterStepsOf(scene, filters),
                         registry);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

Path CustomizeChain(Scene& scene, const Path& camera) {
    const Path product = FreePath(
        scene, "/Render/" + std::string(camera.Name()) + "Sensor");
    const Path folder(product.String() + "Chain");
    std::vector<Path> filters;
    for (const SensorFilter filter : DefaultFilterChain()) {
        if (filter == SensorFilter::Lens) {
            continue;  // always first
        }
        const Path path = folder.Child(NameOf(filter));
        SensorFilterPrim::Define(scene, path)
            .SetFilterType(std::string(NameOf(filter)));
        filters.push_back(path);
    }
    const Path var = product.Child("LdrColor");
    RenderVar::Define(scene, var).SetSourceName("LdrColor");
    RenderProduct::Define(scene, product)
        .SetCamera(camera)
        .SetOrderedVars({var})
        .SetFilters(filters);
    return product;
}

Path AddChainFilter(Scene& scene, const Path& product,
                    const std::string& type,
                    const SensorFilterRegistry& registry) {
    RenderProduct view(scene, product);
    const std::vector<Path> filters = view.GetFilters();
    const Path folder = filters.empty()
                            ? Path(product.String() + "Chain")
                            : filters.front().Parent();
    const Path added = FreePath(scene, folder.Child(type).String());
    SensorFilterPrim::Define(scene, added).SetFilterType(type);
    const auto with = [&](std::size_t at) {
        std::vector<Path> candidate = filters;
        candidate.insert(candidate.begin() +
                             static_cast<std::ptrdiff_t>(at),
                         added);
        return candidate;
    };
    // Right after the last filter leaving what it reads (a mosaic
    // after the noise, white balance after cie), when that is valid;
    // else the latest valid place; else the end.
    std::optional<std::size_t> at;
    if (const auto domains = DomainsOf(type, registry)) {
        std::size_t after = 0;
        for (std::size_t i = 0; i < filters.size(); ++i) {
            const auto other = DomainsOf(
                SensorFilterPrim(scene, filters[i]).GetFilterType(),
                registry);
            if (other && other->second == domains->first) {
                after = i + 1;
            }
        }
        if (Valid(scene, with(after), registry)) {
            at = after;
        }
    }
    for (std::size_t i = filters.size() + 1; !at && i-- > 0;) {
        if (Valid(scene, with(i), registry)) {
            at = i;
        }
    }
    view.SetFilters(with(at.value_or(filters.size())));
    return added;
}

void MoveChainFilter(Scene& scene, const Path& product,
                     std::size_t index, int offset) {
    RenderProduct view(scene, product);
    std::vector<Path> filters = view.GetFilters();
    const auto target = static_cast<std::ptrdiff_t>(index) + offset;
    if (index >= filters.size() || target < 0 ||
        target >= static_cast<std::ptrdiff_t>(filters.size())) {
        return;
    }
    std::swap(filters[index],
              filters[static_cast<std::size_t>(target)]);
    view.SetFilters(filters);
}

void RemoveChainFilter(Scene& scene, const Path& product,
                       std::size_t index) {
    RenderProduct view(scene, product);
    std::vector<Path> filters = view.GetFilters();
    if (index >= filters.size()) {
        return;
    }
    const Path removed = filters[index];
    filters.erase(filters.begin() +
                  static_cast<std::ptrdiff_t>(index));
    view.SetFilters(filters);
    if (scene.Has(removed) && !Referenced(scene, removed)) {
        scene.Remove(removed);
    }
}

std::string ChainFlowOf(std::string_view type,
                        const SensorFilterRegistry& registry) {
    if (const std::optional<SensorFilter> builtin =
            SensorFilterNamed(type)) {
        switch (*builtin) {
        case SensorFilter::Lens:
            return "spectrum";
        case SensorFilter::Qe:
            return "spectrum > electrons";
        case SensorFilter::Noise:
            return "electrons";
        case SensorFilter::Adc:
            return "electrons > digital";
        case SensorFilter::Cie:
            return "spectrum > rgb";
        case SensorFilter::Exposure:
            return "rgb > display";
        }
    }
    const SensorFilterModel* model = registry.Find(type);
    if (model == nullptr) {
        return "unknown";
    }
    static constexpr std::array<const char*, 5> kDomains{
        "spectrum", "electrons", "digital", "rgb", "display"};
    const std::string domain =
        kDomains[static_cast<std::size_t>(model->domain)];
    return model->kind == FilterKind::Integrator
               ? "spectrum > " + domain
               : domain;
}

}  // namespace ernie::app
