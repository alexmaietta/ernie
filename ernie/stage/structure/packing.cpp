// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/stage/structure/packing.h"

#include <algorithm>
#include <cstring>
#include <numeric>

#include "ernie/util/error.h"

namespace ernie {

namespace {

void CheckPrimvar(const Primvar& primvar) {
    if (primvar.name.empty()) {
        throw Error("a primvar needs a name");
    }
    if (primvar.elementSize == 0) {
        throw Error("primvar {} has an elementSize of 0", primvar.name);
    }
    const uint32_t stride = PrimvarComponents(primvar.type) * primvar.elementSize;
    if (primvar.words.size() % stride != 0) {
        throw Error("primvar {} has {} words, not a multiple of its {} per value", primvar.name,
                    primvar.words.size(), stride);
    }
    for (const int32_t index : primvar.indices) {
        if (index < 0 || uint32_t(index) >= primvar.ValueCount()) {
            throw Error("primvar {} indexes value {} of {}", primvar.name, index,
                        primvar.ValueCount());
        }
    }
}

}

std::vector<int32_t> PackPrimvars(Tokens& tokens, std::span<const Primvar> primvars) {
    if (primvars.empty()) {
        return {};
    }

    std::vector<uint32_t> names(primvars.size());
    for (size_t i = 0; i < primvars.size(); ++i) {
        CheckPrimvar(primvars[i]);
        names[i] = tokens.Intern(primvars[i].name).id;
    }
    std::vector<uint32_t> order(primvars.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return names[a] < names[b]; });
    for (size_t i = 1; i < order.size(); ++i) {
        if (names[order[i]] == names[order[i - 1]]) {
            throw Error("primvar {} is given twice", primvars[order[i]].name);
        }
    }

    std::vector<int32_t> table(1 + primvars.size() * kPrimvarEntryWords);
    table[0] = int32_t(primvars.size());
    for (size_t slot = 0; slot < order.size(); ++slot) {
        const Primvar& primvar = primvars[order[slot]];
        PrimvarEntry entry{};
        entry.name = names[order[slot]];
        entry.type = uint32_t(primvar.type);
        entry.interpolation = uint32_t(primvar.interpolation);
        entry.elementSize = primvar.elementSize;
        entry.values = uint32_t(table.size());
        entry.valueCount = primvar.ValueCount();
        table.insert(table.end(), primvar.words.begin(), primvar.words.end());
        entry.indices = uint32_t(table.size());
        entry.indexCount = uint32_t(primvar.indices.size());
        table.insert(table.end(), primvar.indices.begin(), primvar.indices.end());
        std::memcpy(table.data() + 1 + slot * kPrimvarEntryWords, &entry, sizeof(entry));
    }
    return table;
}

std::vector<Primvar> UnpackPrimvars(const Tokens& tokens, std::span<const int32_t> table) {
    if (table.empty()) {
        return {};
    }
    const uint32_t count = uint32_t(table[0]);
    if (table.size() < 1 + size_t(count) * kPrimvarEntryWords) {
        throw Error("a primvar table of {} words can't hold {} entries", table.size(), count);
    }
    std::vector<Primvar> primvars;
    primvars.reserve(count);
    for (uint32_t slot = 0; slot < count; ++slot) {
        PrimvarEntry entry;
        std::memcpy(&entry, table.data() + 1 + slot * kPrimvarEntryWords, sizeof(entry));
        Primvar primvar;
        primvar.name = std::string(tokens.Text(Token{entry.name}));
        primvar.type = PrimvarType(entry.type);
        primvar.interpolation = Interpolation(entry.interpolation);
        primvar.elementSize = entry.elementSize;
        const uint32_t words =
            entry.valueCount * PrimvarComponents(primvar.type) * entry.elementSize;
        if (size_t(entry.values) + words > table.size() ||
            size_t(entry.indices) + entry.indexCount > table.size()) {
            throw Error("primvar {} runs past the end of its table", primvar.name);
        }
        const auto values = table.subspan(entry.values, words);
        primvar.words.assign(values.begin(), values.end());
        const auto indices = table.subspan(entry.indices, entry.indexCount);
        primvar.indices.assign(indices.begin(), indices.end());
        primvars.push_back(std::move(primvar));
    }
    return primvars;
}

std::vector<uint32_t> InternTokens(Tokens& tokens, std::span<const std::string> texts) {
    std::vector<uint32_t> ids;
    ids.reserve(texts.size());
    for (const std::string& text : texts) {
        ids.push_back(tokens.Intern(text).id);
    }
    return ids;
}

}
