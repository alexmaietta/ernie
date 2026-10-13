// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/pds.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ernie {

namespace {

namespace fs = std::filesystem;

std::string Trim(std::string text) {
    const auto space = [](unsigned char c) {
        return std::isspace(c) != 0;
    };
    text.erase(text.begin(),
               std::find_if_not(text.begin(), text.end(), space));
    text.erase(
        std::find_if_not(text.rbegin(), text.rend(), space).base(),
        text.end());
    return text;
}

std::string Upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::toupper(c));
                   });
    return text;
}

/// The label's KEY = value pairs up to END, comments dropped and
/// values that run over lines (quoted text, sets, sequences) joined;
/// the first of a key repeated in several objects kept.
std::map<std::string, std::string>
ParseLabel(const std::string& text) {
    std::map<std::string, std::string> label;
    std::istringstream lines(text);
    std::string line;
    std::string key;
    std::string value;
    int open = 0;  // Unclosed quotes (1) or brackets.
    const auto finish = [&] {
        if (!key.empty()) {
            label.emplace(key, Trim(value));
        }
        key.clear();
        value.clear();
    };
    while (std::getline(lines, line)) {
        // Comments.
        for (std::size_t at;
             (at = line.find("/*")) != std::string::npos;) {
            const std::size_t end = line.find("*/", at + 2);
            line.erase(at, end == std::string::npos
                               ? std::string::npos
                               : end + 2 - at);
        }
        if (open > 0) {
            value += ' ' + Trim(line);
        } else {
            const std::string trimmed = Trim(line);
            if (trimmed == "END") {
                break;
            }
            const std::size_t equals = trimmed.find('=');
            if (equals == std::string::npos) {
                continue;
            }
            key = Upper(Trim(trimmed.substr(0, equals)));
            value = Trim(trimmed.substr(equals + 1));
        }
        open = 0;
        bool quoted = false;
        for (const char c : value) {
            if (c == '"') {
                quoted = !quoted;
            } else if (!quoted && (c == '(' || c == '{')) {
                ++open;
            } else if (!quoted && (c == ')' || c == '}')) {
                --open;
            }
        }
        open = std::max(open, 0) + (quoted ? 1 : 0);
        if (open == 0) {
            finish();
        }
    }
    finish();
    return label;
}

/// A value's text without quotes, and a number's units (`<deg>`).
std::string Bare(std::string value) {
    if (const std::size_t units = value.find('<');
        units != std::string::npos && value.front() != '"') {
        value.erase(units);
    }
    value = Trim(value);
    if (value.size() >= 2 &&
        (value.front() == '"' || value.front() == '\'') &&
        value.back() == value.front()) {
        value = value.substr(1, value.size() - 2);
    }
    return Trim(value);
}

/// The file in `directory` named `name`, in any case (PDS volumes
/// name files in capitals that disks often hold in small letters).
fs::path FindBeside(const fs::path& directory,
                    const std::string& name) {
    const fs::path exact = directory / name;
    if (fs::exists(exact)) {
        return exact;
    }
    std::error_code error;
    for (const fs::directory_entry& entry :
         fs::directory_iterator(directory, error)) {
        if (Upper(entry.path().filename().string()) == Upper(name)) {
            return entry.path();
        }
    }
    return exact;
}

}  // namespace

PdsImage::PdsImage(const fs::path& path) {
    const std::string name = path.string();
    const std::string extension = Upper(path.extension().string());
    // The label: this file if it is one, else one beside the data,
    // else attached at the data's start.
    fs::path labelPath;
    if (extension == ".LBL") {
        labelPath = path;
    } else {
        const fs::path beside = FindBeside(
            path.parent_path(), path.stem().string() + ".lbl");
        if (fs::exists(beside)) {
            labelPath = beside;
        }
    }
    std::string text;
    bool attached = false;
    if (!labelPath.empty()) {
        std::ifstream file(labelPath, std::ios::binary);
        text.assign(std::istreambuf_iterator<char>(file), {});
    } else {
        std::ifstream file(path, std::ios::binary);
        text.resize(1 << 16);
        file.read(text.data(),
                  static_cast<std::streamsize>(text.size()));
        text.resize(static_cast<std::size_t>(file.gcount()));
        attached = true;
    }
    if (text.find("PDS_VERSION_ID") == std::string::npos) {
        throw std::runtime_error(name + " has no PDS3 label");
    }
    label_ = ParseLabel(text);
    const auto number = [&](const std::string& key,
                            std::optional<double> fallback =
                                std::nullopt) -> double {
        const std::optional<std::string> value = Label(key);
        if (!value) {
            if (fallback) {
                return *fallback;
            }
            throw std::runtime_error(name +
                                     ": the PDS label has no " + key);
        }
        try {
            return std::stod(*value);
        } catch (const std::exception&) {
            throw std::runtime_error(name + ": the PDS label's " +
                                     key + " is not a number");
        }
    };
    width_ = static_cast<std::uint32_t>(number("LINE_SAMPLES"));
    height_ = static_cast<std::uint32_t>(number("LINES"));
    if (number("BANDS", 1.0) != 1.0) {
        throw std::runtime_error(name +
                                 ": PDS images of several bands "
                                 "are not read");
    }
    const auto bits =
        static_cast<std::uint32_t>(number("SAMPLE_BITS"));
    const std::string sampleType =
        Upper(Label("SAMPLE_TYPE").value_or(""));
    const bool real = sampleType.find("REAL") != std::string::npos;
    const bool unsignedType =
        sampleType.find("UNSIGNED") != std::string::npos;
    bigEndian_ = sampleType.rfind("LSB", 0) != 0 &&
                 sampleType.rfind("PC_", 0) != 0 &&
                 sampleType.rfind("VAX", 0) != 0;
    if (real && bits == 32) {
        type_ = RasterSampleType::Float32;
    } else if (!real && bits == 8) {
        type_ = RasterSampleType::UInt8;
    } else if (!real && bits == 16) {
        type_ = unsignedType ? RasterSampleType::UInt16
                             : RasterSampleType::Int16;
    } else if (!real && bits == 32) {
        type_ = unsignedType ? RasterSampleType::UInt32
                             : RasterSampleType::Int32;
    } else {
        throw std::runtime_error(
            name + ": PDS samples " + sampleType + " of " +
            std::to_string(bits) + " bits are not read");
    }
    scale_ = number("SCALING_FACTOR", 1.0);
    offset_ = number("OFFSET", 0.0);
    if (Label("MISSING_CONSTANT")) {
        noData_ = number("MISSING_CONSTANT");
    }

    // The data: ^IMAGE names a file beside the label ("NAME.IMG", or
    // ("NAME.IMG", record)), or a record or byte of this one.
    const std::string pointer = label_.count("^IMAGE") != 0
                                    ? label_.at("^IMAGE")
                                    : std::string();
    fs::path dataPath = path;
    std::uint64_t record = 1;
    bool bytes = false;
    if (const std::size_t quote = pointer.find('"');
        quote != std::string::npos) {
        const std::size_t end = pointer.find('"', quote + 1);
        dataPath =
            FindBeside(path.parent_path(),
                       pointer.substr(quote + 1, end - quote - 1));
        if (const std::size_t comma = pointer.find(',', end);
            comma != std::string::npos) {
            const std::string rest = pointer.substr(comma + 1);
            record = std::stoull(rest);
            bytes = Upper(rest).find("<BYTES>") != std::string::npos;
        }
    } else if (!pointer.empty()) {
        record = std::stoull(pointer);
        bytes = Upper(pointer).find("<BYTES>") != std::string::npos;
    } else if (!attached) {
        throw std::runtime_error(name +
                                 ": the PDS label has no ^IMAGE");
    }
    const auto recordBytes =
        static_cast<std::uint64_t>(number("RECORD_BYTES", 0.0));
    start_ = bytes ? record - 1 : (record - 1) * recordBytes;
    file_ = std::make_shared<const MappedFile>(dataPath);
    const std::uint64_t needed =
        start_ + std::uint64_t{width_} * height_ * SampleBytes();
    if (file_->Bytes().size() < needed) {
        throw std::runtime_error(
            dataPath.string() +
            " is shorter than its PDS label says");
    }

    // Simple cylindrical grids.
    const std::string projection =
        Upper(Label("MAP_PROJECTION_TYPE").value_or(""));
    if ((projection == "SIMPLE CYLINDRICAL" ||
         projection == "EQUIRECTANGULAR") &&
        number("CENTER_LATITUDE", 0.0) == 0.0 &&
        Label("WESTERNMOST_LONGITUDE") && Label("MAXIMUM_LATITUDE")) {
        LatLongGrid grid;
        grid.west = number("WESTERNMOST_LONGITUDE");
        grid.north = number("MAXIMUM_LATITUDE");
        if (Label("MAP_RESOLUTION")) {
            grid.degreesAcross = grid.degreesDown =
                1.0 / number("MAP_RESOLUTION");
        } else {
            grid.degreesAcross =
                (number("EASTERNMOST_LONGITUDE") - grid.west) /
                width_;
            grid.degreesDown =
                (grid.north - number("MINIMUM_LATITUDE")) / height_;
        }
        grid_ = grid;
    }
}

std::optional<std::string>
PdsImage::Label(const std::string& key) const {
    const auto it = label_.find(Upper(key));
    if (it == label_.end()) {
        return std::nullopt;
    }
    return Bare(it->second);
}

void PdsImage::Read(std::uint32_t x, std::uint32_t y, std::uint32_t w,
                    std::uint32_t h, std::span<std::byte> out) const {
    if (x > width_ || w > width_ - x || y > height_ ||
        h > height_ - y) {
        throw std::out_of_range("PDS: a rectangle past the image");
    }
    const std::uint32_t bytes = SampleBytes();
    if (out.size() < std::size_t{w} * h * bytes) {
        throw std::out_of_range(
            "PDS: too little room for the rectangle");
    }
    const std::span<const std::byte> data = file_->Bytes();
    const bool swap =
        bytes > 1 &&
        bigEndian_ != (std::endian::native == std::endian::big);
    for (std::uint32_t row = 0; row < h; ++row) {
        std::byte* line = out.data() + std::size_t{row} * w * bytes;
        std::memcpy(line,
                    data.data() + start_ +
                        (std::uint64_t{y + row} * width_ + x) * bytes,
                    std::size_t{w} * bytes);
        if (swap) {
            for (std::size_t i = 0; i < std::size_t{w} * bytes;
                 i += bytes) {
                std::reverse(line + i, line + i + bytes);
            }
        }
    }
}

}  // namespace ernie
