// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/render/render_output.h"

#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "ernie/device/buffer.h"
#include "ernie/device/queue.h"
#include "ernie/io/image.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/span.h"
#include "ernie/util/text.h"

namespace ernie {

const RenderVarOutput& FrameOutput::RenderVar(std::string_view path) const {
    const auto found = renderVars.find(path);
    if (found == renderVars.end()) {
        throw Error("The frame has no render var {}", path);
    }
    return found->second;
}

const RenderVarOutput& FrameOutput::Source(std::string_view sourceName) const {
    for (const auto& [path, var] : renderVars) {
        if (var.sourceName == sourceName) {
            return var;
        }
    }
    throw Error("The frame has no render var whose source is {}", sourceName);
}

IdTexts DecodeIdMap(const device::Tensor& map) {
    constexpr size_t kRecordWords = 6;
    const std::vector<std::byte> bytes = map.ToHostBytes();
    uint32_t count = 0;
    if (bytes.size() < sizeof(count)) {
        throw Error("An id map holds at least its {}-byte count, not {} bytes", sizeof(count),
                    bytes.size());
    }
    const size_t texts = bytes.size() - sizeof(count);
    std::memcpy(&count, bytes.data() + texts, sizeof(count));
    if (size_t(count) * kRecordWords * sizeof(uint32_t) > texts) {
        throw Error("An id map of {} bytes can't hold {} entries", bytes.size(), count);
    }
    IdTexts entries;
    for (uint32_t entry = 0; entry < count; ++entry) {
        uint32_t record[kRecordWords];
        std::memcpy(record, bytes.data() + entry * sizeof(record), sizeof(record));
        const size_t size = record[4];
        const size_t offset = record[5];
        if (offset > texts || size > texts - offset) {
            throw Error("Id map entry {} runs past the map's {} bytes", entry, bytes.size());
        }
        entries[{record[0], record[1], record[2], record[3]}] =
            std::string(reinterpret_cast<const char*>(bytes.data()) + offset, size);
    }
    return entries;
}

namespace {

void WriteLabels(const std::filesystem::path& path, const IdMap& map, bool single) {
    const auto escaped = [](std::string_view text) {
        std::string result;
        for (const char c : text) {
            if (c == '"' || c == '\\') {
                result += '\\';
                result += c;
            } else if (uint8_t(c) < 0x20) {
                result += fmt::format("\\u{:04x}", int(c));
            } else {
                result += c;
            }
        }
        return result;
    };
    std::string text = "{";
    const char* separator = "\n";
    for (const auto& [id, label] : map) {
        const std::array<uint8_t, 4> color =
            IdColor(Span<const uint32_t>(id.data(), single ? 1 : id.size()));
        const std::string key = single
                                    ? std::to_string(id[0])
                                    : fmt::format("({}, {}, {}, {})", id[0], id[1], id[2], id[3]);
        std::string fields;
        if (!label.path.empty()) {
            fields += fmt::format("\"path\": \"{}\", ", escaped(label.path));
        }
        fields += "\"labels\": {";
        const char* comma = "";
        for (const auto& [taxonomy, names] : label.labels) {
            fields += fmt::format("{}\"{}\": [", comma, escaped(taxonomy));
            for (size_t i = 0; i < names.size(); ++i) {
                fields += fmt::format("{}\"{}\"", i > 0 ? ", " : "", escaped(names[i]));
            }
            fields += "]";
            comma = ", ";
        }
        text += fmt::format("{}    \"{}\": {{{}}}, \"color\": [{}, {}, {}, {}]}}", separator, key,
                            fields, color[0], color[1], color[2], color[3]);
        separator = ",\n";
    }
    text += "\n}\n";
    std::ofstream file(path, std::ios::binary);
    file << text;
    if (!file) {
        throw Error("Cannot write {}", path.string());
    }
}

template <class T> T Viewable(std::string_view source, device::DType type, const T& image) {
    if (source == "DistanceToCamera" || source == "DistanceToImagePlane") {
        return ColorizeScalars(image, kDepthColors);
    }
    if (source == "Intensity") {
        return ColorizeScalars(image);
    }
    if (source == "RadialVelocity") {
        return ColorizeScalars(image, {.colormap = Colormap::Turbo, .centered = true});
    }
    if (source == "Normal") {
        return ColorizeNormals(image);
    }
    if (type == device::DType::UInt32) {
        return ColorizeIds(image);
    }
    return image;
}

std::filesystem::path WriteVar(std::filesystem::path path, const RenderVarOutput& renderVar,
                               const Pixels& image) {
    if (renderVar.idMap && image.type == device::DType::UInt32) {
        WriteLabels(std::filesystem::path(path).replace_extension(".json"), *renderVar.idMap,
                    image.channels == 1);
    }
    if (Lowercase(path.extension().string()) == ".exr") {
        return WriteImage(std::move(path), image);
    }
    if (!path.has_extension()) {
        path += ".png";
    }
    return WriteImage(std::move(path), Viewable(renderVar.sourceName, image.type, image));
}

std::function<Pixels()> Fetch(const device::Tensor& tensor) {
    const Span<const int64_t> shape = tensor.Shape();
    const Span<const int64_t> strides = tensor.Strides();
    bool compact = shape.size() == 2 || shape.size() == 3;
    int64_t expected = 1;
    for (size_t axis = shape.size(); compact && axis-- > 0;) {
        compact = shape[axis] <= 1 || strides[axis] == expected;
        expected *= shape[axis];
    }
    if (!compact) {
        return [pixels = Pixels(tensor)] { return pixels; };
    }
    const uint64_t bytes = uint64_t(expected) * device::ElementBytes(tensor.Type());
    device::Buffer& buffer = tensor.GetBuffer();
    const device::Download download =
        buffer.GetDevice().GetQueue().ReadAsync(buffer.Slice(tensor.Offset(), bytes));
    return [download, tensor] {
        download.Wait();
        const Span<const int64_t> sides = tensor.Shape();
        return Pixels(download.Bytes(), tensor.Type(), sides[0], sides[1],
                      sides.size() == 3 ? sides[2] : 1);
    };
}

}

std::filesystem::path WriteRenderVar(std::filesystem::path path, const RenderVarOutput& renderVar,
                                     int64_t camera) {
    return WriteVar(std::move(path), renderVar, Pixels(renderVar.tensor.Select(camera)));
}

namespace {

template <class Write>
std::vector<std::filesystem::path> WriteVars(const std::filesystem::path& directory,
                                             const FrameOutput& frame, bool raw, Write write) {
    std::filesystem::create_directories(directory);
    std::vector<std::filesystem::path> written;
    for (const auto& [path, renderVar] : frame.renderVars) {
        const Span<const int64_t> shape = renderVar.tensor.Shape();
        if (shape.size() < 3) {
            continue;
        }
        const std::string name = path.substr(path.rfind('/') + 1);
        const bool exr = raw && renderVar.tensor.Type() != device::DType::UInt8;
        for (int64_t camera = 0; camera < shape[0]; ++camera) {
            const std::filesystem::path file =
                directory / fmt::format("{}_{}{}", name, camera, exr ? ".exr" : ".png");
            write(file, renderVar, camera);
            written.push_back(file);
        }
    }
    if (!frame.overlay.Empty()) {
        const RenderVarOutput overlay{"overlay", "Overlay", frame.overlay, nullptr};
        for (int64_t camera = 0; camera < frame.overlay.Shape()[0]; ++camera) {
            const std::filesystem::path file = directory / fmt::format("overlay_{}.png", camera);
            write(file, overlay, camera);
            written.push_back(file);
        }
    }
    return written;
}

}

std::vector<std::filesystem::path> WriteFrame(const std::filesystem::path& directory,
                                              const FrameOutput& frame, bool raw) {
    return WriteVars(directory, frame, raw,
                     [](const std::filesystem::path& file, const RenderVarOutput& renderVar,
                        int64_t camera) { WriteRenderVar(file, renderVar, camera); });
}

struct ImageWriter::State {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::function<void()>> jobs;
    std::vector<std::thread> threads;
    uint32_t maxPending = 0;
    size_t pending = 0;
    std::exception_ptr error;
    bool stopping = false;

    void Run() {
        std::unique_lock lock(mutex);
        while (true) {
            changed.wait(lock, [&] { return stopping || !jobs.empty(); });
            if (jobs.empty()) {
                return;
            }
            std::function<void()> job = std::move(jobs.front());
            jobs.pop_front();
            lock.unlock();
            std::exception_ptr failed;
            try {
                job();
            } catch (...) {
                failed = std::current_exception();
            }
            job = nullptr;
            lock.lock();
            if (failed && !error) {
                error = failed;
            }
            --pending;
            changed.notify_all();
        }
    }

    void Submit(std::function<void()> job) {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return pending < maxPending; });
        ++pending;
        jobs.push_back(std::move(job));
        changed.notify_all();
    }

    void Drain() {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return pending == 0; });
    }
};

ImageWriter::ImageWriter(uint32_t threads, uint32_t maxPending)
    : m_state(std::make_unique<State>()) {
    if (threads == 0 || maxPending == 0) {
        throw Error(
            "An image writer needs at least one thread and one pending write, not {} and {}",
            threads, maxPending);
    }
    m_state->maxPending = maxPending;
    for (uint32_t i = 0; i < threads; ++i) {
        m_state->threads.emplace_back([state = m_state.get()] { state->Run(); });
    }
}

ImageWriter::~ImageWriter() {
    m_state->Drain();
    {
        const std::lock_guard lock(m_state->mutex);
        m_state->stopping = true;
    }
    m_state->changed.notify_all();
    for (std::thread& thread : m_state->threads) {
        thread.join();
    }
    if (m_state->error) {
        try {
            std::rethrow_exception(m_state->error);
        } catch (const std::exception& error) {
            Log(LogLevel::Error, "an image writer dropped a failed write: {}", error.what());
        }
    }
}

void ImageWriter::WriteImage(std::filesystem::path path, Pixels image) {
    m_state->Submit(
        [path = std::move(path), image = std::move(image)] { ernie::WriteImage(path, image); });
}

void ImageWriter::WriteImage(std::filesystem::path path, const device::Tensor& image) {
    m_state->Submit(
        [path = std::move(path), fetch = Fetch(image)] { ernie::WriteImage(path, fetch()); });
}

void ImageWriter::WriteRenderVar(std::filesystem::path path, const RenderVarOutput& renderVar,
                                 int64_t camera) {
    const device::Tensor image = renderVar.tensor.Select(camera);
    device::Tensor shown = image;
    if (Lowercase(path.extension().string()) != ".exr") {
        if (!path.has_extension()) {
            path += ".png";
        }
        shown = Viewable(renderVar.sourceName, image.Type(), image);
    }
    const Span<const int64_t> shape = image.Shape();
    const bool single = shape.size() == 2 || shape[2] == 1;
    std::shared_ptr<const IdMap> labels =
        image.Type() == device::DType::UInt32 ? renderVar.idMap : nullptr;
    m_state->Submit([path = std::move(path), labels = std::move(labels), single,
                     fetch = Fetch(shown)] {
        if (labels) {
            WriteLabels(std::filesystem::path(path).replace_extension(".json"), *labels, single);
        }
        ernie::WriteImage(path, fetch());
    });
}

std::vector<std::filesystem::path> ImageWriter::WriteFrame(const std::filesystem::path& directory,
                                                           const FrameOutput& frame, bool raw) {
    return WriteVars(directory, frame, raw,
                     [this](const std::filesystem::path& file, const RenderVarOutput& renderVar,
                            int64_t camera) { WriteRenderVar(file, renderVar, camera); });
}

void ImageWriter::Wait() {
    m_state->Drain();
    const std::lock_guard lock(m_state->mutex);
    if (m_state->error) {
        std::exception_ptr error = std::exchange(m_state->error, nullptr);
        std::rethrow_exception(error);
    }
}

size_t ImageWriter::Pending() const {
    const std::lock_guard lock(m_state->mutex);
    return m_state->pending;
}

const ProductOutput& RenderOutput::operator[](std::string_view product) const {
    const auto found = m_products.find(product);
    if (found == m_products.end()) {
        throw Error("The step rendered no product {}", product);
    }
    return found->second;
}

}
