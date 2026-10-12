// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/reflectance_library.h"

#include "ernie/util/reflectance_table.h"

#include <iterator>
#include <utility>
#include <vector>

namespace ernie {
namespace {

const std::vector<MeasuredReflectance>& Library() {
    static const std::vector<MeasuredReflectance> library = [] {
        std::vector<MeasuredReflectance> entries;
        for (const reflectance_table::Entry& entry :
             reflectance_table::kEntries) {
            std::vector<double> wavelengths;
            std::vector<double> values;
            for (int i = entry.first; i < entry.first + entry.count;
                 ++i) {
                wavelengths.push_back(
                    reflectance_table::kPairs[2 * i]);
                values.push_back(
                    reflectance_table::kPairs[2 * i + 1]);
            }
            entries.push_back({entry.id, entry.name, entry.source,
                               Spectrum(std::move(wavelengths),
                                        std::move(values))});
        }
        return entries;
    }();
    return library;
}

}  // namespace

std::span<const MeasuredReflectance> ReflectanceLibrary() {
    return Library();
}

const MeasuredReflectance* FindReflectance(std::string_view id) {
    for (const MeasuredReflectance& entry : Library()) {
        if (entry.id == id) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace ernie
