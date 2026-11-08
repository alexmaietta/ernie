// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/scene.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ernie/device/program.h"
#include "ernie/scene/primvar.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/scene/views/view_state.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

struct CopiedValue {
    SchemaId owner = SchemaId::None;
    uint32_t index = 0;
    std::vector<std::byte> bytes;
    std::string instance;
    std::vector<std::string> strings;
    std::vector<Primvar> primvars;

    ColumnRef Column() const {
        return {owner, &Columns(owner)[index], index};
    }
};

struct CopiedPrim {
    std::string relative;
    SchemaId schema = SchemaId::None;
    uint64_t apis = 0;
    std::vector<std::pair<SchemaId, std::string>> instances;
    std::vector<CopiedValue> values;
    std::vector<std::pair<std::string, FileBytes>> files;
};

bool IsToken(ColumnType type) {
    return type == ColumnType::Token || type == ColumnType::Asset;
}

std::string Rebased(const std::string& target, const std::string& from, std::string_view to) {
    if (target == from) {
        return std::string(to);
    }
    if (target.starts_with(from + "/")) {
        return std::string(to) + target.substr(from.size());
    }
    return target;
}

}

SceneImpl::SceneImpl(Scene& scene, device::Device& device, const SceneDesc& desc)
    : m_scene(scene), m_device(&device), m_waitForKernels(desc.waitForKernels), m_store(device),
      m_structure(m_store), m_publisher(m_structure),
      m_builder(bvh::Builder::Create(device, desc.traversal)), m_transforms(m_structure),
      m_meshes(m_structure, *m_builder),
      m_points(m_structure, *m_builder, desc.maxRefits, desc.pointsPerBox),
      m_bounds(m_structure, m_instances), m_tangents(m_structure, m_meshes),
      m_textures(m_structure, desc.textureThreads), m_semantics(m_structure),
      m_sensors(m_structure), m_products(scene, m_structure), m_plans(scene, m_products),
      m_captures(scene, m_clock, m_products), m_materials(m_structure, m_textures),
      m_spectra(m_structure), m_instances(m_structure, *m_builder, desc.maxRefits),
      m_surfaces(m_structure, m_instances), m_lights(m_structure, m_instances) {
    m_publisher.Add(m_transforms);
    m_publisher.Add(m_meshes);
    m_publisher.Add(m_points);
    m_publisher.Add(m_tangents);
    m_publisher.Add(m_textures);
    m_publisher.Add(m_semantics);
    m_publisher.Add(m_sensors);
    m_publisher.Add(m_products);
    m_publisher.Add(m_spectra);
    m_publisher.Add(m_materials);
    m_publisher.Add(m_instances);
    m_publisher.Add(m_surfaces);
    m_publisher.Add(m_bounds);
    m_publisher.Add(m_lights);
    m_store.Prepare();
    m_builder->Prepare();
    for (Table* system : std::initializer_list<Table*>{
             &m_transforms, &m_meshes, &m_points, &m_bounds, &m_tangents, &m_textures, &m_semantics,
             &m_sensors, &m_materials, &m_instances, &m_surfaces, &m_lights}) {
        system->Prepare();
    }
    if (desc.waitForKernels) {
        device::WaitForBuilds();
    }
}

Prim SceneImpl::Handle(uint32_t prim) const {
    return Prim(m_scene, prim, m_structure.GetPrims().Generation(prim));
}

PrimRange SceneImpl::Range(const PrimBlock& block) const {
    return {block.firstPrim, block.count, block.copies, block.firstWorld, block.worldsPerCopy};
}

std::vector<uint32_t> SceneImpl::Indices(Span<const Prim> prims) const {
    std::vector<uint32_t> indices;
    indices.reserve(prims.size());
    for (const Prim& prim : prims) {
        Check(prim);
        indices.push_back(prim.Index());
    }
    return indices;
}

void SceneImpl::Check(const Prim& prim) const {
    if (&prim.GetScene() != &m_scene) {
        throw Error("Prim {} belongs to another scene", prim.Index());
    }
    if (!prim.Valid()) {
        throw Error("Prim {} was removed", prim.Index());
    }
}

namespace {

ShaderLoader& Loader() {
    static ShaderLoader loader;
    return loader;
}

SchemaId DefinedShader(Scene& scene, const Shader::Desc& desc) {
    if (desc.implementationSource != ImplementationSource::SourceAsset) {
        const SchemaId schema = ShaderSchema(desc.id);
        if (schema == SchemaId::None) {
            throw Error("no shader has info:id \"{}\"", desc.id);
        }
        return schema;
    }
    const std::string& file = desc.slangSourceAsset;
    const std::string& name = desc.slangSourceAssetSubIdentifier;
    if (file.empty() || name.empty()) {
        throw Error("A shader loaded from a source asset needs info:slang:sourceAsset and "
                    "info:slang:sourceAsset:subIdentifier");
    }
    if (const SchemaId known = ShaderSchema(name); known != SchemaId::None) {
        return known;
    }
    if (!Loader()) {
        throw Error("{} can't load {}, because nothing has registered a shader loader", name, file);
    }
    Loader()(scene, file);
    const SchemaId loaded = ShaderSchema(name);
    if (loaded == SchemaId::None) {
        throw Error("{} has no material node called {}", file, name);
    }
    return loaded;
}

void WriteShader(RowWriter& writer, SchemaId schema, const Shader::Desc& desc) {
    WriteFallbacks(schema, writer);
    writer.Set(SchemaId::Connectable, Shader::kConnectionsSources.index, desc.connectionsSources);
    writer.Set(SchemaId::Connectable, Shader::kConnectionsInputs.index, desc.connectionsInputs);
    writer.Set(SchemaId::Connectable, Shader::kConnectionsOutputs.index, desc.connectionsOutputs);
    writer.Set(SchemaId::Shader, Shader::kId.index, std::string(Info(schema).name));
    if (desc.implementationSource == ImplementationSource::SourceAsset) {
        writer.Set(SchemaId::Shader, Shader::kImplementationSource.index,
                   desc.implementationSource);
        writer.Set(SchemaId::Shader, Shader::kSlangSourceAsset.index, desc.slangSourceAsset);
        writer.Set(SchemaId::Shader, Shader::kSlangSourceAssetSubIdentifier.index,
                   desc.slangSourceAssetSubIdentifier);
    }
}

}

void SetShaderLoader(ShaderLoader loader) {
    Loader() = std::move(loader);
}

uint32_t RetypePrims(Scene& scene, SchemaId from, SchemaId to) {
    SceneImpl& impl = SceneImpl::Get(scene);
    Structure& structure = impl.GetStructure();
    ColumnStore& store = structure.GetColumnStore();
    const Prims& prims = structure.GetPrims();
    const Span<const ColumnInfo> before = Columns(from);
    const Span<const ColumnInfo> after = Columns(to);
    uint32_t moved = 0;
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim) || prims.Record(prim).schema != from) {
            continue;
        }
        std::vector<std::pair<uint32_t, std::vector<std::byte>>> kept;
        const uint32_t row = prims.Row(prim, from);
        for (uint32_t column = 0; column < after.size(); ++column) {
            for (uint32_t old = 0; old < before.size(); ++old) {
                if (before[old].field == after[column].field &&
                    before[old].type == after[column].type && !IsArray(before[old].type)) {
                    kept.emplace_back(column, store.Schema(from).Read(old, row, 1));
                }
            }
        }
        structure.Retype(prim, to);
        const uint32_t retyped = prims.Row(prim, to);
        for (const auto& [column, cell] : kept) {
            store.Stage({to, &after[column], column}, retyped, cell);
        }
        ++moved;
    }
    if (moved > 0) {
        Log(LogLevel::Debug, "moved {} from the old {} to its new inputs", Counted(moved, "prim"),
            Info(to).name);
    }
    return moved;
}

ShaderPrim Define(Scene& scene, std::string_view path, const Shader::Desc& desc) {
    const SchemaId schema = DefinedShader(scene, desc);
    Log(LogLevel::Debug, "{} is a {} shader", path, Info(schema).name);
    return ShaderPrim(SceneImpl::Get(scene).Define(
        path, schema, [&](RowWriter& writer) { WriteShader(writer, schema, desc); }));
}

PrimRange Define(Scene& scene, Span<const std::string> paths, const Shader::Desc& desc,
                 const BatchDesc& batch) {
    const SchemaId schema = DefinedShader(scene, desc);
    return SceneImpl::Get(scene).Define(
        paths, schema, [&](RowWriter& writer) { WriteShader(writer, schema, desc); }, batch);
}

Ref<Scene> Scene::Create(const SceneDesc& desc) {
    const Ref<device::Device> device = device::Device::Create(desc.device);
    return Create(*device, desc);
}

Ref<Scene> Scene::Create(device::Device& device, const SceneDesc& desc) {
    return Ref<Scene>(new Scene(device, desc));
}

Scene::Scene(device::Device& device, const SceneDesc& desc)
    : m_impl(std::make_unique<SceneImpl>(*this, device, desc)) {}

Scene::~Scene() = default;

Prim Scene::Define(std::string_view path, SchemaId schema) {
    return m_impl->Define(path, schema, [&](RowWriter& writer) { WriteFallbacks(schema, writer); });
}

PrimRange Scene::Define(Span<const std::string> paths, SchemaId schema, const BatchDesc& batch) {
    return m_impl->Define(
        paths, schema, [&](RowWriter& writer) { WriteFallbacks(schema, writer); }, batch);
}

Prim Scene::Move(Prim prim, std::string_view path) {
    m_impl->Check(prim);
    Structure& structure = m_impl->GetStructure();
    structure.Move(prim.Index(), path);
    std::vector<std::string> xformable;
    for (const uint32_t moved : structure.GetPaths().Subtree(prim.Index())) {
        if (IsA(structure.GetPrims().Record(moved).schema, SchemaId::Xformable)) {
            xformable.push_back(structure.GetPaths().Path(moved));
        }
    }
    if (!xformable.empty()) {
        Select(xformable, SchemaId::Xformable).Get(Xformable::kTranslate.Column()).MarkChanged();
    }
    return ByIndex(prim.Index());
}

Prim Scene::Rename(Prim prim, std::string_view name) {
    m_impl->Check(prim);
    const std::optional<Prim> parent = prim.Parent();
    return Move(prim, (parent ? parent->Path() : std::string()) + "/" + std::string(name));
}

void Scene::Remove(Prim prim) {
    m_impl->Check(prim);
    m_impl->GetStructure().Remove(prim.Index());
}

namespace {

bool SameColumn(ColumnRef a, ColumnRef b) {
    return a.schema == b.schema && a.index == b.index;
}

}

Overrides::Overrides() = default;

Overrides::Overrides(Ref<Scene> scene, uint64_t owner)
    : m_scene(std::move(scene)), m_owner(owner) {}

Overrides::~Overrides() {
    if (m_scene) {
        try {
            Clear();
        } catch (const std::exception& error) {
            Log(LogLevel::Warning, "Clearing overrides: {}", error.what());
        }
    }
}

Overrides::Overrides(Overrides&& other) noexcept
    : m_scene(std::move(other.m_scene)), m_owner(other.m_owner), m_cells(std::move(other.m_cells)) {
}

Overrides& Overrides::operator=(Overrides&& other) noexcept {
    if (this != &other) {
        if (m_scene) {
            try {
                Clear();
            } catch (const std::exception& error) {
                Log(LogLevel::Warning, "Clearing overrides: {}", error.what());
            }
        }
        m_scene = std::move(other.m_scene);
        m_owner = other.m_owner;
        m_cells = std::move(other.m_cells);
    }
    return *this;
}

void Overrides::Set(Prim prim, ColumnRef column, Span<const std::byte> cell) {
    if (!m_scene) {
        throw Error("Overrides from no scene can't set {}", column.column->field);
    }
    if (IsArray(column.column->type)) {
        throw Error("{} holds arrays, which overrides don't", column.column->field);
    }
    if (column.schema == SchemaId::Shader && column.index == Shader::kId.index) {
        throw Error("A shader's info:id picks its node, so overrides can't change it");
    }
    SceneImpl& impl = SceneImpl::Get(*m_scene);
    impl.Check(prim);
    const uint32_t row = impl.GetStructure().GetPrims().Row(prim.Index(), column.schema);
    ColumnStore& store = impl.GetColumnStore();
    std::vector<std::byte> authored;
    if (!store.Overridden(column, row)) {
        authored = m_scene->Select(prim).Get(column).ReadHost();
    }
    store.Override(column, row, m_owner, authored, cell);
    store.StampWorld(prim.World());
    const bool held = std::ranges::any_of(m_cells, [&](const auto& entry) {
        return entry.first.Index() == prim.Index() && SameColumn(entry.second, column);
    });
    if (!held) {
        m_cells.emplace_back(prim, column);
    }
}

void Overrides::Clear(Prim prim, ColumnRef column) {
    const auto found = std::ranges::find_if(m_cells, [&](const auto& entry) {
        return entry.first.Index() == prim.Index() && SameColumn(entry.second, column);
    });
    if (found == m_cells.end()) {
        return;
    }
    if (prim.Valid()) {
        SceneImpl& impl = SceneImpl::Get(*m_scene);
        const uint32_t row = impl.GetStructure().GetPrims().FindRow(prim.Index(), column.schema);
        if (row != kNoRow) {
            impl.GetColumnStore().ClearOverride(column, row, m_owner);
            impl.GetColumnStore().StampWorld(prim.World());
        }
    }
    m_cells.erase(found);
}

void Overrides::Clear() {
    while (!m_cells.empty()) {
        const auto [prim, column] = m_cells.back();
        Clear(prim, column);
    }
}

Overrides Scene::CreateOverrides() {
    static std::atomic<uint64_t> next{1};
    return Overrides(Ref<Scene>(this), next++);
}

bool Scene::Overridden(Prim prim, ColumnRef column) const {
    m_impl->Check(prim);
    const uint32_t row = m_impl->GetStructure().GetPrims().FindRow(prim.Index(), column.schema);
    return row != kNoRow && m_impl->GetColumnStore().Overridden(column, row);
}

std::vector<std::byte> Scene::ReadAuthored(Prim prim, ColumnRef column) const {
    if (Overridden(prim, column)) {
        const uint32_t row = m_impl->GetStructure().GetPrims().Row(prim.Index(), column.schema);
        return m_impl->GetColumnStore().Authored(column, row);
    }
    return Select(prim).Get(column).ReadHost();
}

ComposedWorld::ComposedWorld() = default;

ComposedWorld::ComposedWorld(Ref<Scene> scene, uint32_t world)
    : m_scene(std::move(scene)), m_world(world) {}

ComposedWorld::~ComposedWorld() {
    Release();
}

ComposedWorld::ComposedWorld(ComposedWorld&& other) noexcept
    : m_scene(std::move(other.m_scene)), m_world(std::exchange(other.m_world, kGlobalWorld)) {}

ComposedWorld& ComposedWorld::operator=(ComposedWorld&& other) noexcept {
    if (this != &other) {
        Release();
        m_scene = std::move(other.m_scene);
        m_world = std::exchange(other.m_world, kGlobalWorld);
    }
    return *this;
}

void ComposedWorld::Release() {
    if (m_scene && m_world != kGlobalWorld) {
        SceneImpl::Get(*m_scene).GetStructure().Decompose(m_world);
        SceneImpl::Get(*m_scene).GetColumnStore().StampWorld(m_world);
    }
    m_world = kGlobalWorld;
}

void ComposedWorld::Set(const ComposedWorldDesc& desc) {
    if (!m_scene) {
        throw Error("An empty world view shows nothing");
    }
    Structure& structure = SceneImpl::Get(*m_scene).GetStructure();
    const uint32_t worlds = structure.GetPrims().WorldCount();
    if (!desc.offsets.empty() && desc.offsets.size() != desc.worlds.size()) {
        throw Error("A world view has {} offsets for {} worlds", desc.offsets.size(),
                    desc.worlds.size());
    }
    for (const uint32_t world : desc.worlds) {
        if (world >= worlds || world == m_world) {
            throw Error("A world view can't show world {} of {}", world, worlds);
        }
    }
    if (structure.Compose(m_world, desc)) {
        SceneImpl::Get(*m_scene).GetColumnStore().StampWorld(m_world);
    }
}

ComposedWorld Scene::ComposeWorld(Prim world, const ComposedWorldDesc& desc) {
    m_impl->Check(world);
    if (!world.IsA(SchemaId::World) || !world.IsSession()) {
        throw Error("{} isn't a session World, which only can show other worlds", world.Path());
    }
    if (m_impl->GetStructure().Compositions().contains(world.World())) {
        throw Error("{} already shows other worlds", world.Path());
    }
    ComposedWorld view(Ref<Scene>(this), world.World());
    view.Set(desc);
    return view;
}

void Scene::MarkSession(Prim prim) {
    m_impl->Check(prim);
    m_impl->GetStructure().MarkSession(prim.Index());
}

void Scene::Apply(Span<const Prim> prims, SchemaId api) {
    m_impl->Apply(prims, api, [&](RowWriter& writer) { WriteFallbacks(api, writer); });
}

void Scene::Apply(Span<const Prim> prims, SchemaId api, std::string_view instance) {
    m_impl->Apply(prims, api, instance, [&](RowWriter& writer) { WriteFallbacks(api, writer); });
}

void Scene::Unapply(Span<const Prim> prims, SchemaId api) {
    m_impl->GetStructure().Unapply(m_impl->Indices(prims), api);
}

void Scene::Unapply(Span<const Prim> prims, SchemaId api, std::string_view instance) {
    m_impl->GetStructure().Unapply(m_impl->Indices(prims), api, instance);
}

std::vector<Prim> Scene::Handles(const PrimRange& range) const {
    std::vector<Prim> prims;
    prims.reserve(range.Size());
    for (uint32_t prim = range.firstPrim; prim < range.firstPrim + range.Size(); ++prim) {
        prims.push_back(m_impl->Handle(prim));
    }
    return prims;
}

PrimRange Scene::Clone(Prim source, Span<const std::string> targets, const CloneDesc& desc) {
    m_impl->Check(source);
    return m_impl->Range(m_impl->GetStructure().Clone(
        source.Index(), std::span(targets.data(), targets.size()), desc));
}

std::optional<Prim> Scene::Find(std::string_view path) const {
    const uint32_t prim = m_impl->GetStructure().Find(path);
    if (prim == kNoPrim) {
        return std::nullopt;
    }
    return m_impl->Handle(prim);
}

Prim Scene::ByIndex(uint32_t index) const {
    if (!m_impl->GetStructure().GetPrims().Alive(index)) {
        throw Error("No prim {}", index);
    }
    return m_impl->Handle(index);
}

View Scene::Select(std::string_view pattern, SchemaId schema) const {
    Structure& structure = const_cast<SceneImpl&>(*m_impl).GetStructure();
    return View(structure, schema, structure.Select(pattern, schema));
}

View Scene::Select(Span<const std::string> paths, SchemaId schema) const {
    Structure& structure = const_cast<SceneImpl&>(*m_impl).GetStructure();
    std::vector<uint32_t> prims;
    for (const std::string& path : paths) {
        const uint32_t prim = structure.Find(path);
        if (prim == kNoPrim) {
            throw Error("No prim at \"{}\"", path);
        }
        prims.push_back(prim);
    }
    return View(structure, schema, std::move(prims));
}

View Scene::Select(const PrimRange& range, SchemaId schema, uint32_t index) const {
    Structure& structure = const_cast<SceneImpl&>(*m_impl).GetStructure();
    std::vector<uint32_t> prims;
    for (uint32_t copy = 0; copy < range.copies; ++copy) {
        prims.push_back(range.Index(copy, index));
    }
    return View(structure, schema, std::move(prims));
}

View Scene::Select(Prim prim) const {
    m_impl->Check(prim);
    Structure& structure = const_cast<SceneImpl&>(*m_impl).GetStructure();
    return View(structure, prim.Schema(), std::vector<uint32_t>{prim.Index()});
}

View Scene::Select(std::string_view pattern, SchemaId api, std::string_view instance) const {
    Structure& structure = const_cast<SceneImpl&>(*m_impl).GetStructure();
    if (!IsMultipleApi(api)) {
        throw Error("{} isn't applied many times, so it has no instances", Info(api).name);
    }
    const uint32_t token = structure.GetTokens().Intern(instance).id;
    std::vector<uint32_t> prims = structure.Select(pattern, api);
    std::erase_if(prims, [&](uint32_t prim) {
        return structure.GetPrims().InstanceRow(prim, api, token) == kNoRow;
    });
    return View(structure, api, std::move(prims), instance);
}

View Scene::Select(Prim prim, SchemaId api, std::string_view instance) const {
    m_impl->Check(prim);
    Structure& structure = const_cast<SceneImpl&>(*m_impl).GetStructure();
    if (!IsMultipleApi(api)) {
        throw Error("{} isn't applied many times, so it has no instances", Info(api).name);
    }
    if (structure.GetPrims().InstanceRow(prim.Index(), api,
                                         structure.GetTokens().Intern(instance).id) == kNoRow) {
        throw Error("{} has no {} {}", prim.Path(), Info(api).name, instance);
    }
    return View(structure, api, std::vector<uint32_t>{prim.Index()}, instance);
}

uint32_t Scene::PrimCount() const {
    return m_impl->GetStructure().GetPrims().LiveCount();
}

uint32_t Scene::WorldCount() const {
    return m_impl->GetStructure().GetPrims().WorldCount();
}

device::Operation Scene::Publish(const PublishDesc& desc, Stream producer) {
    {
        ERNIE_ZONE("Scene/CommitSteps");
        m_impl->RunCommitSteps();
    }
    if (desc.dt) {
        SimulationClock& clock = m_impl->GetClock();
        clock.start = clock.end;
        clock.end += *desc.dt;
        ++clock.steps;
    }
    device::Operation done =
        m_impl->GetPublisher().Publish(desc.ordinal.value_or(Published() + 1), producer);
    while (desc.wait && Pending()) {
        m_impl->GetPublisher().Finish();
        m_impl->RunCommitSteps();
        done = m_impl->GetPublisher().Publish(Published() + 1, producer);
    }
    m_impl->GetCaptures().Schedule();
    return done;
}

double Scene::Time() const {
    return m_impl->GetClock().end;
}

void Scene::SetTime(double time) {
    SimulationClock& clock = m_impl->GetClock();
    Log(LogLevel::Debug, "simulation time set from {} to {}", clock.end, time);
    clock.start = time;
    clock.end = time;
    ++clock.steps;
    ++clock.resets;
}

namespace {

std::vector<uint32_t> PathPrims(const Scene& scene, Span<const std::string> paths) {
    std::vector<uint32_t> prims;
    for (const std::string& path : paths) {
        const std::optional<Prim> prim = scene.Find(path);
        prims.push_back(prim ? prim->Index() : kNoPrim);
    }
    return prims;
}

std::vector<uint32_t> BoundableRows(const Prims& prims, std::vector<uint32_t> indices) {
    for (uint32_t& prim : indices) {
        prim = prims.Alive(prim) && IsA(prims.Record(prim).schema, SchemaId::Boundable) ? prim
                                                                                        : kNoRow;
    }
    return indices;
}

}

device::Download Scene::ReadWorldTransformsAsync(Span<const std::string> paths) const {
    const std::vector<uint32_t> rows =
        m_impl->GetStructure().GetPrims().Rows(PathPrims(*this, paths), SchemaId::Xformable);
    return m_impl->GetColumnStore().ReadRowsAsync(m_impl->GetTransforms().Transforms(),
                                                  sizeof(ernie::WorldTransform), rows);
}

std::vector<ernie::WorldTransform> Scene::WorldTransforms(Span<const std::string> paths) const {
    const std::vector<uint32_t> rows =
        m_impl->GetStructure().GetPrims().Rows(PathPrims(*this, paths), SchemaId::Xformable);
    return m_impl->GetColumnStore().ReadRows<ernie::WorldTransform>(
        m_impl->GetTransforms().Transforms(), rows);
}

static_assert(sizeof(Box) == sizeof(BoundableBox));

device::Download Scene::ReadPrimBoundsAsync(Span<const std::string> paths) const {
    const std::vector<uint32_t> rows =
        BoundableRows(m_impl->GetStructure().GetPrims(), PathPrims(*this, paths));
    return m_impl->GetColumnStore().ReadRowsAsync(m_impl->GetBounds().PrimBounds(),
                                                  sizeof(BoundableBox), rows);
}

std::vector<Box> Scene::PrimBounds(Span<const std::string> paths) const {
    const std::vector<uint32_t> rows =
        BoundableRows(m_impl->GetStructure().GetPrims(), PathPrims(*this, paths));
    return m_impl->GetColumnStore().ReadRows<Box>(m_impl->GetBounds().PrimBounds(), rows);
}

std::optional<Box> Scene::Bounds(uint32_t world) const {
    const std::optional<BvhBox> box = m_impl->GetBounds().World(world);
    if (!box) {
        return std::nullopt;
    }
    return Box{{box->lower[0], box->lower[1], box->lower[2]},
               {box->upper[0], box->upper[1], box->upper[2]}};
}

device::Download Scene::ReadBoundsAsync(uint32_t world) const {
    return m_impl->GetBounds().ReadWorldAsync(world);
}

device::Download Scene::ReadAllBoundsAsync() const {
    return m_impl->GetBounds().ReadWorldsAsync();
}

bool Scene::Pending() const {
    return m_impl->CommitStepsPending() || m_impl->GetPublisher().Pending();
}

uint64_t Scene::StructureVersion() const {
    return m_impl->GetStructure().Epoch();
}

Ordinal Scene::Published() const {
    return const_cast<SceneImpl&>(*m_impl).GetPublisher().Published();
}

Snapshot Scene::Acquire(Stream consumer) {
    return m_impl->GetPublisher().Acquire(consumer);
}

void Scene::Release(const Snapshot& snapshot, Stream consumer) {
    m_impl->GetPublisher().Release(snapshot, consumer);
}

void Scene::Require(Span<const ernie::Token> taxonomies) {
    m_impl->GetSemantics().Require(std::span(taxonomies.data(), taxonomies.size()));
}

void Scene::AddFile(std::string path, FileBytes bytes, std::optional<Prim> owner) {
    std::optional<uint32_t> index;
    if (owner) {
        m_impl->Check(*owner);
        index = owner->Index();
    }
    m_impl->GetStructure().AddFile(std::move(path), std::move(bytes), index);
}

void Scene::RemoveFile(std::string_view path) {
    m_impl->GetStructure().RemoveFile(path);
}

ernie::Token Scene::Token(std::string_view text) {
    return m_impl->GetStructure().GetTokens().Intern(text);
}

std::string_view Scene::Text(ernie::Token token) const {
    return m_impl->GetStructure().GetTokens().Text(token);
}

device::Device& Scene::GetDevice() const {
    return m_impl->GetDevice();
}

struct PrimCopy::State {
    std::string root;
    std::vector<CopiedPrim> prims;
};

PrimCopy::PrimCopy() = default;
PrimCopy::~PrimCopy() = default;
PrimCopy::PrimCopy(const PrimCopy& other) = default;
PrimCopy::PrimCopy(PrimCopy&& other) noexcept = default;
PrimCopy& PrimCopy::operator=(const PrimCopy& other) = default;
PrimCopy& PrimCopy::operator=(PrimCopy&& other) noexcept = default;

bool PrimCopy::Empty() const {
    return !m_state || m_state->prims.empty();
}

std::string_view PrimCopy::Name() const {
    if (Empty()) {
        return {};
    }
    return std::string_view(m_state->root).substr(m_state->root.rfind('/') + 1);
}

uint32_t PrimCopy::PrimCount() const {
    return m_state ? uint32_t(m_state->prims.size()) : 0;
}

PrimCopy Scene::Copy(Prim prim) {
    ERNIE_ZONE("Scene/Copy");
    m_impl->Check(prim);
    Structure& structure = m_impl->GetStructure();
    auto state = std::make_shared<PrimCopy::State>();
    state->root = prim.Path();

    struct Pending {
        CopiedValue* value;
        ColumnType type;
        device::Download read;
    };

    std::vector<Pending> reads;
    for (const uint32_t index : structure.GetPaths().Subtree(prim.Index())) {
        const PrimRecord& record = structure.GetPrims().Record(index);
        const std::string path = structure.GetPaths().Path(index);
        CopiedPrim& copied = state->prims.emplace_back();
        copied.relative = path.substr(state->root.size());
        if (copied.relative.starts_with('/')) {
            copied.relative.erase(0, 1);
        }
        copied.schema = record.schema;
        copied.apis = record.apis;
        copied.files = structure.OwnedFiles(index);
        for (const SchemaId table : PrimTables(record.schema, record.apis)) {
            std::vector<std::string> names = {""};
            if (IsMultipleApi(table)) {
                names = ByIndex(index).Instances(table);
                for (const std::string& name : names) {
                    copied.instances.emplace_back(table, name);
                }
            }
            const Span<const ColumnInfo> columns = Columns(table);
            for (const std::string& name : names) {
                for (uint32_t column = 0; column < columns.size(); ++column) {
                    if (!Has(columns[column].flags, ColumnFlags::Internal)) {
                        copied.values.push_back({table, column, {}, name});
                    }
                }
            }
        }
        const View view = Select(ByIndex(index));
        for (CopiedValue& value : copied.values) {
            const ColumnInfo& info = *value.Column().column;
            const UntypedAttribute attribute =
                value.instance.empty()
                    ? view.Get(value.Column())
                    : Select(ByIndex(index), value.owner, value.instance).Get(value.Column());
            if (info.type == ColumnType::PrimArray) {
                value.strings = attribute.GetTargets(0);
            } else if (info.type == ColumnType::TokenArray) {
                value.strings = attribute.GetTokens(0);
            } else if (info.type == ColumnType::IntArray && info.field == "primvars") {
                value.primvars = attribute.GetPrimvars(0);
            } else if (IsArray(info.type)) {
                value.bytes = attribute.GetArray(0);
            } else {
                reads.push_back({&value, info.type, attribute.ReadHostAsync()});
            }
        }
    }
    for (Pending& pending : reads) {
        pending.read.Wait();
        const Span<const std::byte> bytes = pending.read.Bytes();
        if (IsToken(pending.type)) {
            uint32_t id = 0;
            std::memcpy(&id, bytes.data(), std::min<size_t>(bytes.size(), sizeof(id)));
            pending.value->strings = {std::string(Text(ernie::Token{id}))};
        } else {
            pending.value->bytes.assign(bytes.begin(), bytes.end());
        }
    }
    PrimCopy copy;
    copy.m_state = std::move(state);
    return copy;
}

Prim Scene::Paste(const PrimCopy& copy, std::string_view path) {
    ERNIE_ZONE("Scene/Paste");
    if (copy.Empty()) {
        throw Error("Pasting an empty copy at \"{}\"", path);
    }
    const PrimCopy::State& state = *copy.m_state;
    const auto pastedPath = [&](const CopiedPrim& copied) {
        return copied.relative.empty() ? std::string(path)
                                       : std::string(path) + "/" + copied.relative;
    };

    std::vector<Prim> pasted;
    try {
        for (const CopiedPrim& copied : state.prims) {
            const Prim prim = Define(pastedPath(copied), copied.schema);
            pasted.push_back(prim);
            for (const SchemaId api : kApiSchemas) {
                if (HasApi(copied.apis, api) && !IsMultipleApi(api) && !prim.HasApi(api)) {
                    Apply(Span<const Prim>(&prim, 1), api);
                }
            }
            for (const auto& [api, name] : copied.instances) {
                const std::vector<std::string> present = prim.Instances(api);
                if (std::ranges::find(present, name) == present.end()) {
                    Apply(Span<const Prim>(&prim, 1), api, name);
                }
            }
        }
    } catch (...) {
        if (!pasted.empty()) {
            Remove(pasted.front());
        }
        throw;
    }

    for (size_t at = 0; at < state.prims.size(); ++at) {
        const CopiedPrim& copied = state.prims[at];
        const Prim prim = ByIndex(pasted[at].Index());
        const View view = Select(prim);
        for (const CopiedValue& value : copied.values) {
            const ColumnInfo& info = *value.Column().column;
            UntypedAttribute attribute =
                value.instance.empty()
                    ? view.Get(value.Column())
                    : Select(prim, value.owner, value.instance).Get(value.Column());
            if (info.type == ColumnType::PrimArray) {
                std::vector<std::string> targets;
                for (const std::string& target : value.strings) {
                    targets.push_back(Rebased(target, state.root, path));
                }
                attribute.SetTargets(0, targets);
            } else if (info.type == ColumnType::TokenArray) {
                attribute.SetTokens(0, value.strings);
            } else if (info.type == ColumnType::IntArray && info.field == "primvars") {
                attribute.SetPrimvars(0, value.primvars);
            } else if (IsArray(info.type)) {
                attribute.SetArray(0, value.bytes);
            } else if (IsToken(info.type)) {
                const uint32_t id = value.strings.empty() || value.strings.front().empty()
                                        ? 0
                                        : Token(value.strings.front()).id;
                attribute.Write(AsBytes(Span<const uint32_t>(&id, 1)));
            } else if (!value.bytes.empty()) {
                attribute.Write(value.bytes);
            }
        }
        for (const auto& [file, bytes] : copied.files) {
            AddFile(file, bytes, prim);
        }
    }
    return ByIndex(pasted.front().Index());
}

}
