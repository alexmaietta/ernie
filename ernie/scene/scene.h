// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ernie/device/device.h"
#include "ernie/device/operation.h"
#include "ernie/device/ref.h"
#include "ernie/device/stream.h"
#include "ernie/export.h"
#include "ernie/scene/defines.h"
#include "ernie/scene/prim.h"
#include "ernie/scene/records/render_vars.h"
#include "ernie/scene/schema/schema.h"
#include "ernie/scene/structure/tokens.h"
#include "ernie/scene/types.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/poses.h"
#include "ernie/scene/views/view.h"
#include "ernie/util/span.h"

namespace ernie {

struct ERNIE_API SceneDesc {
    device::DeviceDesc device;
    device::Traversal traversal = device::Traversal::Auto;
    uint32_t maxRefits = 16;
    uint32_t pointsPerBox = 1;
    unsigned textureThreads = 4;
    bool waitForKernels = true;
};

struct PublishDesc {
    std::optional<Ordinal> ordinal;
    std::optional<double> dt;
    bool wait = false;
};

class SceneImpl;
class ImportJob;
struct ImportReader;
struct ImportDesc;
struct ImportSource;

using ShaderLoader =
    std::function<std::vector<std::string>(Scene& scene, const std::filesystem::path& file)>;

ERNIE_API void SetShaderLoader(ShaderLoader loader);
ERNIE_API uint32_t RetypePrims(Scene& scene, SchemaId from, SchemaId to);
ERNIE_API ShaderPrim Define(Scene& scene, std::string_view path, const Shader::Desc& desc);
ERNIE_API PrimRange Define(Scene& scene, Span<const std::string> paths, const Shader::Desc& desc,
                           const BatchDesc& batch = {});

class ERNIE_API PrimCopy {
  public:
    PrimCopy();
    ~PrimCopy();
    PrimCopy(const PrimCopy& other);
    PrimCopy(PrimCopy&& other) noexcept;
    PrimCopy& operator=(const PrimCopy& other);
    PrimCopy& operator=(PrimCopy&& other) noexcept;

    bool Empty() const;
    std::string_view Name() const;
    uint32_t PrimCount() const;

  private:
    friend class Scene;
    struct State;
    std::shared_ptr<const State> m_state;
};

class ERNIE_API Overrides {
  public:
    Overrides();
    ~Overrides();
    Overrides(const Overrides&) = delete;
    Overrides& operator=(const Overrides&) = delete;
    Overrides(Overrides&& other) noexcept;
    Overrides& operator=(Overrides&& other) noexcept;

    void Set(Prim prim, ColumnRef column, Span<const std::byte> cell);

    template <class T> void Set(Prim prim, Field<T> field, const T& value) {
        const AttributeCell<T> cell = value;
        Set(prim, field.Column(), AsBytes(Span<const AttributeCell<T>>(&cell, 1)));
    }

    void Clear(Prim prim, ColumnRef column);
    void Clear();

    bool Empty() const {
        return m_cells.empty();
    }

  private:
    friend class Scene;

    Overrides(Ref<Scene> scene, uint64_t owner);

    Ref<Scene> m_scene;
    uint64_t m_owner = 0;
    std::vector<std::pair<Prim, ColumnRef>> m_cells;
};

class ERNIE_API ComposedWorld {
  public:
    ComposedWorld();
    ~ComposedWorld();
    ComposedWorld(const ComposedWorld&) = delete;
    ComposedWorld& operator=(const ComposedWorld&) = delete;
    ComposedWorld(ComposedWorld&& other) noexcept;
    ComposedWorld& operator=(ComposedWorld&& other) noexcept;

    void Set(const ComposedWorldDesc& desc);

    uint32_t World() const {
        return m_world;
    }

  private:
    friend class Scene;

    ComposedWorld(Ref<Scene> scene, uint32_t world);
    void Release();

    Ref<Scene> m_scene;
    uint32_t m_world = kGlobalWorld;
};

class ERNIE_API Scene : public RefCounted {
  public:
    static Ref<Scene> Create(const SceneDesc& desc = {});
    static Ref<Scene> Create(device::Device& device, const SceneDesc& desc = {});
    ~Scene() override;

    template <class T>
    typename T::PrimType Define(std::string_view path, const typename T::Desc& desc = {}) {
        return typename T::PrimType(ernie::Define(*this, path, desc));
    }

    template <class T>
    PrimRange Define(Span<const std::string> paths, const typename T::Desc& desc = {},
                     const BatchDesc& batch = {}) {
        return ernie::Define(*this, paths, desc, batch);
    }

    Prim Define(std::string_view path, SchemaId schema);
    PrimRange Define(Span<const std::string> paths, SchemaId schema, const BatchDesc& batch = {});
    void Remove(Prim prim);
    Overrides CreateOverrides();
    bool Overridden(Prim prim, ColumnRef column) const;
    std::vector<std::byte> ReadAuthored(Prim prim, ColumnRef column) const;
    void MarkSession(Prim prim);
    ComposedWorld ComposeWorld(Prim world, const ComposedWorldDesc& desc = {});

    template <class T> void Apply(Prim prim, const typename T::Desc& desc = {}) {
        ernie::Apply(*this, Span<const Prim>(&prim, 1), desc);
    }

    template <class T> void Apply(Span<const Prim> prims, const typename T::Desc& desc = {}) {
        ernie::Apply(*this, prims, desc);
    }

    template <class T> void Apply(const PrimRange& range, const typename T::Desc& desc = {}) {
        ernie::Apply(*this, Handles(range), desc);
    }

    template <class T>
    void Apply(Prim prim, std::string_view instance, const typename T::Desc& desc = {}) {
        ernie::Apply(*this, Span<const Prim>(&prim, 1), instance, desc);
    }

    template <class T>
    void Apply(Span<const Prim> prims, std::string_view instance,
               const typename T::Desc& desc = {}) {
        ernie::Apply(*this, prims, instance, desc);
    }

    template <class T>
    void Apply(const PrimRange& range, std::string_view instance,
               const typename T::Desc& desc = {}) {
        ernie::Apply(*this, Handles(range), instance, desc);
    }

    template <class T> void Unapply(Prim prim, std::string_view instance) {
        Unapply(Span<const Prim>(&prim, 1), T::kSchema, instance);
    }

    template <class T> void Unapply(Span<const Prim> prims, std::string_view instance) {
        Unapply(prims, T::kSchema, instance);
    }

    template <class T> void Unapply(Prim prim) {
        Unapply(Span<const Prim>(&prim, 1), T::kSchema);
    }

    template <class T> void Unapply(Span<const Prim> prims) {
        Unapply(prims, T::kSchema);
    }

    template <class T> void Unapply(const PrimRange& range) {
        Unapply(Handles(range), T::kSchema);
    }

    void Apply(Span<const Prim> prims, SchemaId api);
    void Apply(Span<const Prim> prims, SchemaId api, std::string_view instance);
    void Unapply(Span<const Prim> prims, SchemaId api);
    void Unapply(Span<const Prim> prims, SchemaId api, std::string_view instance);
    std::vector<Prim> Handles(const PrimRange& range) const;
    Prim Move(Prim prim, std::string_view path);
    Prim Rename(Prim prim, std::string_view name);
    PrimCopy Copy(Prim prim);
    Prim Paste(const PrimCopy& copy, std::string_view path);
    PrimRange Clone(Prim source, Span<const std::string> targets, const CloneDesc& desc = {});
    ImportJob Import(std::string_view path, const std::filesystem::path& file);
    ImportJob Import(std::string_view path, const std::filesystem::path& file,
                     const ImportDesc& desc);
    ImportJob Import(std::string_view path, const ImportSource& source);
    ImportJob Import(std::string_view path, const ImportSource& source, const ImportDesc& desc);
    ImportJob Import(std::string_view path, const ImportReader& reader, const ImportDesc& desc);

    std::optional<Prim> Find(std::string_view path) const;
    Prim ByIndex(uint32_t index) const;
    View Select(std::string_view pattern, SchemaId schema = SchemaId::None) const;
    View Select(Span<const std::string> paths, SchemaId schema = SchemaId::None) const;
    View Select(const PrimRange& range, SchemaId schema = SchemaId::None, uint32_t index = 0) const;
    View Select(Prim prim) const;
    View Select(std::string_view pattern, SchemaId api, std::string_view instance) const;
    View Select(Prim prim, SchemaId api, std::string_view instance) const;

    template <class T> typename T::ViewType Select(std::string_view pattern) const {
        return typename T::ViewType(Select(pattern, T::kSchema));
    }

    template <class T>
    typename T::ViewType Select(std::string_view pattern, std::string_view instance) const {
        return typename T::ViewType(Select(pattern, T::kSchema, instance));
    }

    uint32_t PrimCount() const;
    uint32_t WorldCount() const;
    std::optional<Box> Bounds(uint32_t world = kGlobalWorld) const;
    device::Download ReadBoundsAsync(uint32_t world = kGlobalWorld) const;
    device::Download ReadAllBoundsAsync() const;
    device::Download ReadWorldTransformsAsync(Span<const std::string> paths) const;
    std::vector<ernie::WorldTransform> WorldTransforms(Span<const std::string> paths) const;
    device::Download ReadPrimBoundsAsync(Span<const std::string> paths) const;
    std::vector<Box> PrimBounds(Span<const std::string> paths) const;

    device::Operation Publish(const PublishDesc& desc = {}, Stream producer = Stream::None());
    double Time() const;
    void SetTime(double time);
    bool Pending() const;
    Ordinal Published() const;
    uint64_t StructureVersion() const;
    Snapshot Acquire(Stream consumer = Stream::None());
    using RefCounted::Release;
    void Release(const Snapshot& snapshot, Stream consumer = Stream::None());
    void Require(Span<const ernie::Token> taxonomies);

    void AddFile(std::string path, FileBytes bytes, std::optional<Prim> owner = std::nullopt);
    void RemoveFile(std::string_view path);

    ernie::Token Token(std::string_view text);
    std::string_view Text(ernie::Token token) const;
    device::Device& GetDevice() const;

  private:
    friend class SceneImpl;

    ImportJob StartImport(std::string_view path, const ImportReader& reader, const ImportDesc& desc,
                          std::string source);

    Scene(device::Device& device, const SceneDesc& desc);

    std::unique_ptr<SceneImpl> m_impl;
};

}
