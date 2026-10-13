// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// The boundary between Ernie and its USD importer plugin,
/// `libernie_usd`, the only library that links OpenUSD. Ernie loads
/// the plugin when a USD file is first imported (`ImportUsd`,
/// ernie/scene/usd_import.h) and hands it a `Sink` over its Scene;
/// the plugin composes the stage and authors prims through it.
///
/// Only plain types cross: Ernie's Scene code is linked into both
/// libernie and the Python module, so the plugin cannot link it too
/// without a second copy of its registries. The plugin links the
/// OpenUSD the host process uses (Isaac Sim's, say), so that a
/// process never loads two.

#include <cstddef>
#include <cstdint>

namespace ernie::usd_plugin {

/// Bumped whenever `Sink` or `Request` change.
inline constexpr std::uint32_t kAbiVersion = 3;

/// Where the plugin authors what it translates; Ernie implements it
/// over a Scene, coercing values to each attribute's declared type.
/// Paths are Ernie paths. Calls may throw (std::exception) when Ernie
/// rejects a prim or value; the plugin reports those as warnings.
class Sink {
  public:
    /// Defines the prim at `path` as `type`, replacing its type.
    virtual void Define(const char* path, const char* type) = 0;
    virtual void SetBool(const char* path, const char* attribute,
                         bool value) = 0;
    virtual void SetInt(const char* path, const char* attribute,
                        int value) = 0;
    /// One number, a 2-, 3- or 4-vector, or a 4x4 matrix (16 values,
    /// column after column, glm's layout).
    virtual void SetNumbers(const char* path, const char* attribute,
                            const double* values,
                            std::size_t count) = 0;
    /// A token, string or (resolved) asset path.
    virtual void SetToken(const char* path, const char* attribute,
                          const char* value) = 0;
    /// A relationship: one target, or a list.
    virtual void SetTargets(const char* path, const char* attribute,
                            const char* const* targets,
                            std::size_t count, bool list) = 0;
    /// An array of `rows` rows of `columns` floats.
    virtual void WriteFloats(const char* path, const char* attribute,
                             const float* values, std::size_t rows,
                             std::size_t columns) = 0;
    virtual void WriteInts(const char* path, const char* attribute,
                           const std::int32_t* values,
                           std::size_t count) = 0;
    /// A list of tokens, such as joint names.
    virtual void SetTokens(const char* path, const char* attribute,
                           const char* const* values,
                           std::size_t count) = 0;
    /// An array of floats of `rank` dimensions `shape`, row-major.
    virtual void WriteFloatArray(const char* path,
                                 const char* attribute,
                                 const float* values,
                                 const std::size_t* shape,
                                 std::size_t rank) = 0;
    /// An array of ints of `rank` dimensions `shape`, row-major.
    virtual void WriteIntArray(const char* path,
                               const char* attribute,
                               const std::int32_t* values,
                               const std::size_t* shape,
                               std::size_t rank) = 0;
    /// Time samples (`Scene::SetAt`, `Scene::WriteAt`): as
    /// `SetNumbers`, `SetToken` and `WriteFloatArray`, at time code
    /// `time`.
    virtual void SetNumbersAt(const char* path, const char* attribute,
                              double time, const double* values,
                              std::size_t count) = 0;
    virtual void SetTokenAt(const char* path, const char* attribute,
                            double time, const char* value) = 0;
    virtual void WriteFloatArrayAt(const char* path,
                                   const char* attribute, double time,
                                   const float* values,
                                   const std::size_t* shape,
                                   std::size_t rank) = 0;
    /// The stage's authored time code range and rates.
    virtual void SetTimeRange(double start, double end,
                              double timeCodesPerSecond,
                              double framesPerSecond) = 0;
    /// Records that the USD prim `source` became the Ernie prim
    /// `path`.
    virtual void MapSource(const char* source, const char* path) = 0;
    /// Something the plugin skipped or approximated.
    virtual void Warn(const char* message) = 0;

  protected:
    ~Sink() = default;
};

/// One variant selection, applied on the session layer before the
/// stage is translated.
struct VariantSelection {
    const char* prim;  ///< USD path of the prim with the variant set.
    const char* set;
    const char* selection;
};

struct Request {
    std::uint32_t abiVersion = kAbiVersion;
    const char* file = nullptr;
    /// The Ernie path the file's default prim becomes (the way a USD
    /// reference places it), or under which its root prims go when it
    /// has none.
    const char* root = nullptr;
    /// The time code to read; default time when `hasTime` is false.
    double time = 0.0;
    bool hasTime = false;
    const VariantSelection* variants = nullptr;
    std::size_t variantCount = 0;
    /// Translate prims of purpose "proxy" and "guide" too ("default"
    /// and "render" always are).
    bool proxy = false;
    bool guide = false;
    /// The Ernie stage's units, which the root's transform converts
    /// the file's into.
    char upAxis = 'Z';
    double metersPerUnit = 1.0;
};

}  // namespace ernie::usd_plugin

extern "C" {
/// `kAbiVersion` of the plugin.
using ErnieUsdAbiVersionFn = std::uint32_t (*)();
/// The OpenUSD version the plugin was built against, e.g. "0.26.8".
using ErnieUsdVersionFn = const char* (*)();
/// Imports `request->file` through `sink`; returns false with a
/// message in `error` (NUL-terminated, cut to `errorSize`) on
/// failure.
using ErnieUsdImportFn =
    bool (*)(const ernie::usd_plugin::Request* request,
             ernie::usd_plugin::Sink* sink, char* error,
             std::size_t errorSize);
}

namespace ernie::usd_plugin {
inline constexpr const char* kAbiVersionSymbol =
    "ernie_usd_abi_version";
inline constexpr const char* kVersionSymbol = "ernie_usd_version";
inline constexpr const char* kImportSymbol = "ernie_usd_import";
}  // namespace ernie::usd_plugin
