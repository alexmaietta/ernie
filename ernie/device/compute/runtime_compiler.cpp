// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/compute/runtime_compiler.h"

#include <deque>
#include <format>
#include <optional>
#include <vector>

#include "ernie/util/disk_cache.h"
#include "ernie/util/error.h"
#include "ernie/util/library.h"
#include "ernie/util/log.h"
#include "ernie/util/text.h"
#include "ernie/util/version.h"

namespace ernie::device::compute {

namespace {

using Program = RuntimeCompiler::Program;

void Check(const RuntimeCompiler& compiler, int result, const char* call) {
    if (result == 0) {
        return;
    }
    const char* text = compiler.getErrorString(result);
    throw ApiError(std::format("{}: {}", compiler.name, call), text != nullptr ? text : "unknown",
                   result);
}

Program CreateProgram(const RuntimeCompiler& compiler, std::string_view source,
                      std::span<const SourceHeader> headers) {
    std::vector<std::string> texts;
    std::vector<const char*> sources;
    std::vector<const char*> names;
    for (const SourceHeader& header : headers) {
        texts.emplace_back(header.text);
    }
    for (size_t i = 0; i < headers.size(); ++i) {
        sources.push_back(texts[i].c_str());
        names.push_back(headers[i].name);
    }

    Program program = nullptr;
    const std::string text(source);
    Check(compiler,
          compiler.createProgram(&program, text.c_str(), "kernel.cu", int(headers.size()),
                                 sources.data(), names.data()),
          "createProgram");
    return program;
}

std::string ReadOutput(const RuntimeCompiler& compiler, Program program,
                       int (*size)(Program, size_t*), int (*fill)(Program, char*),
                       const char* call) {
    size_t bytes = 0;
    Check(compiler, size(program, &bytes), call);
    std::string output(bytes, '\0');
    Check(compiler, fill(program, output.data()), call);
    return output;
}

}

std::string RuntimeCompiler::Compile(std::string_view source, std::span<const SourceHeader> headers,
                                     std::span<const std::string> options,
                                     std::string_view what) const {
    Program program = CreateProgram(*this, source, headers);

    struct Destroy {
        const RuntimeCompiler& compiler;
        Program& program;

        ~Destroy() {
            compiler.destroyProgram(&program);
        }
    } destroy{*this, program};

    std::vector<const char*> arguments;
    for (const std::string& option : options) {
        arguments.push_back(option.c_str());
    }
    if (compileProgram(program, int(arguments.size()), arguments.data()) != 0) {
        throw Error("Compiling {} with {} {} failed:\n{}", what, name, version.ToString(),
                    ReadOutput(*this, program, getLogSize, getLog, "getProgramLog"));
    }
    return ReadOutput(*this, program, getCodeSize, getCode, "getCode");
}

RuntimeCompiler::Compiled RuntimeCompiler::CompileCached(std::string_view source,
                                                         std::span<const SourceHeader> headers,
                                                         std::span<const std::string> options,
                                                         std::string_view what,
                                                         const DiskCache& cache) const {
    std::string key = std::format("{} {} {}", name, version.ToString(),
                                  Join({options.begin(), options.end()}, " "));
    for (const SourceHeader& header : headers) {
        key += std::format(" {} {}", header.name, Digest(header.text));
    }
    key += " source " + Digest(source);

    if (std::optional<std::string> code = cache.Load(key)) {
        return {*std::move(code), true};
    }

    Compiled compiled{Compile(source, headers, options, what), false};
    cache.Store(key, compiled.code);
    return compiled;
}

Loaded<RuntimeCompiler> LoadRuntimeCompiler(std::string_view name, std::string_view prefix,
                                            std::string_view code,
                                            std::initializer_list<const char*> names,
                                            std::initializer_list<const char*> paths,
                                            Version minimum) {
    const Library library = Library::Load(names, paths);
    if (!library) {
        return std::format("no {} was found", name);
    }

    RuntimeCompiler compiler;
    compiler.name = name;
    compiler.library = library.Name();
    SymbolResolver resolve(library);
    std::deque<std::string> symbols;
    const auto call = [&](std::string_view suffix, auto& function) {
        resolve(symbols.emplace_back(std::format("{}{}", prefix, suffix)).c_str(), function);
    };
    call("Version", compiler.getVersion);
    call("GetErrorString", compiler.getErrorString);
    call("CreateProgram", compiler.createProgram);
    call("DestroyProgram", compiler.destroyProgram);
    call("CompileProgram", compiler.compileProgram);
    call("GetProgramLogSize", compiler.getLogSize);
    call("GetProgramLog", compiler.getLog);
    call(std::format("Get{}Size", code), compiler.getCodeSize);
    call(std::format("Get{}", code), compiler.getCode);
    if (resolve.Missing() != nullptr) {
        return std::format("{} lacks {}", compiler.library, resolve.Missing());
    }

    int major = 0;
    int minor = 0;
    compiler.getVersion(&major, &minor);
    compiler.version = {major, minor};
    if (compiler.version < minimum) {
        return TooOld(name, compiler.version, minimum);
    }

    Log(LogLevel::Debug, "{} {} from {}", name, compiler.version.ToString(), compiler.library);
    return compiler;
}

}
