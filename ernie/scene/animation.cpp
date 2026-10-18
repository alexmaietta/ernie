// SPDX-License-Identifier: BSD-3-Clause
/// @file animation.cpp
/// @brief Time-sampled attributes: their sparse samples per prim, and the clock that evaluates
/// them into the columns.
#include "ernie/scene/scene.h"

#include <algorithm>

namespace ernie {

void Scene::SetAt(PrimId prim, std::string_view attribute, double time, const Value& value) {
    TimeSample sample{time, {}, {}};
    Author(prim, attribute, value, &sample.value);
    AddSample(prim, attribute, std::move(sample));
}

void Scene::WriteAt(PrimId prim, std::string_view attribute, double time, Tensor array) {
    CheckArray(prim, attribute, array);
    AddSample(prim, attribute, {time, {}, std::move(array)});
}

void Scene::AddSample(PrimId prim, std::string_view attribute, TimeSample sample) {
    const AttributeId token = store_.Tokens().Intern(attribute);
    Animated& animated = animated_[prim.slot];
    animated.prim = prim;
    std::vector<TimeSample>& samples = animated.samples[token];
    // Samples are all values or all arrays: one of the other kind replaces them.
    if (!samples.empty() && samples.front().tensor.Empty() != sample.tensor.Empty()) {
        samples.clear();
    }
    const auto at = std::ranges::lower_bound(samples, sample.time, {}, &TimeSample::time);
    if (at != samples.end() && at->time == sample.time) {
        *at = std::move(sample);
    } else {
        samples.insert(at, std::move(sample));
    }
    Evaluate(prim, token, samples);
}

std::span<const TimeSample> Scene::Samples(PrimId prim, std::string_view attribute) const {
    const auto animated = animated_.find(prim.slot);
    const std::optional<Token> token = store_.Tokens().Find(attribute);
    if (animated == animated_.end() || animated->second.prim != prim || !token) {
        return {};
    }
    const auto samples = animated->second.samples.find(*token);
    return samples == animated->second.samples.end() ? std::span<const TimeSample>()
                                                     : std::span(samples->second);
}

std::optional<Value> Scene::GetAt(PrimId prim, std::string_view attribute, double time) const {
    const std::span<const TimeSample> samples = Samples(prim, attribute);
    if (samples.empty()) {
        return Get(prim, attribute);
    }
    Value value = ernie::Evaluate(samples, time).value;
    return value.Empty() ? std::nullopt : std::optional(std::move(value));
}

bool Scene::ClearSamples(PrimId prim, std::string_view attribute) {
    if (Samples(prim, attribute).empty()) {
        return false;
    }
    const auto animated = animated_.find(prim.slot);
    animated->second.samples.erase(*store_.Tokens().Find(attribute));
    if (animated->second.samples.empty()) {
        animated_.erase(animated);
    }
    return true;
}

void Scene::SetTime(double timeCode) {
    time_ = timeCode;
    for (const auto& [slot, animated] : animated_) {
        for (const auto& [attribute, samples] : animated.samples) {
            Evaluate(animated.prim, attribute, samples);
        }
    }
}

void Scene::Evaluate(PrimId prim, AttributeId attribute, std::span<const TimeSample> samples) {
    const std::string_view name = store_.Tokens().Text(attribute);
    TimeSample now = ernie::Evaluate(samples, time_);
    if (!now.tensor.Empty()) {
        const Tensor* current = Read(prim, name);
        if (current == nullptr || current->Data() != now.tensor.Data()) {
            store_.Set(prim, attribute, std::move(now.tensor));
        }
    } else if (Get(prim, name) != now.value) {
        Author(prim, name, now.value);
    }
}

}  // namespace ernie
