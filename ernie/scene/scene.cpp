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
#include <variant>
#include <vector>

#include "ernie/device/program.h"
#include "ernie/scene/access/view_state.h"
#include "ernie/scene/primvar.h"
#include "ernie/scene/scene_impl.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

SchemaId MaterialSchema(std::string_view id) {
    for (uint32_t index = 0; index < uint32_t(SchemaId::Count); ++index) {
        const SchemaId schema = SchemaId(index);
        if (Info(schema).name == id && Info(schema).kind == SchemaKind::Concrete &&
            IsA(schema, SchemaId::Material)) {
            return schema;
        }
    }
    throw Error("no material is called {}", id);
}

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

ColumnRef InputColumn(SchemaId schema, std::string_view id, std::string_view name) {
    const ColumnRef column = FindColumn(schema, name);
    const bool owned =
        column && column.schema != SchemaId::Material &&
        (IsA(column.schema, SchemaId::Material) || Info(column.schema).kind == SchemaKind::Api);
    if (!owned || Has(column.column->flags, ColumnFlags::Derived)) {
        throw Error("{} has no input {}", id, name);
    }
    return column;
}

uint32_t EnumValue(ColumnRef column, std::string_view id, std::string_view text) {
    const std::span<const std::string_view> tokens = column.column->tokens;
    const auto found = std::ranges::find(tokens, text);
    if (found == tokens.end()) {
        throw Error("{}'s {} can't be {}", id, column.column->field, text);
    }
    return uint32_t(found - tokens.begin());
}

void SetInput(RowWriter& writer, SchemaId schema, std::string_view id, const MaterialInput& input) {
    const ColumnRef column = InputColumn(schema, id, input.name);
    const ColumnType type = column.column->type;
    const auto mismatch = [&] {
        return Error("{}'s {} can't take that kind of value", id, input.name);
    };
    std::visit(
        [&]<class T>(const T& value) {
            if constexpr (std::is_same_v<T, std::string>) {
                if (type == ColumnType::Enum) {
                    writer.Set(column.schema, column.index, EnumValue(column, id, value));
                } else if (type == ColumnType::Token || type == ColumnType::Asset) {
                    writer.Set(column.schema, column.index, value);
                } else {
                    throw mismatch();
                }
            } else if constexpr (std::is_same_v<T, int32_t>) {
                if (type == ColumnType::Int) {
                    writer.Set(column.schema, column.index, value);
                } else if (type == ColumnType::Float) {
                    writer.Set(column.schema, column.index, float(value));
                } else {
                    throw mismatch();
                }
            } else {
                constexpr ColumnType expected = std::is_same_v<T, bool>     ? ColumnType::Bool
                                                : std::is_same_v<T, float>  ? ColumnType::Float
                                                : std::is_same_v<T, Float2> ? ColumnType::Float2
                                                : std::is_same_v<T, Float3> ? ColumnType::Float3
                                                                            : ColumnType::Float4;
                if (type != expected) {
                    throw mismatch();
                }
                writer.Set(column.schema, column.index, value);
            }
        },
        input.value);
}

}

void SetInputs(RowWriter& writer, SchemaId schema, std::span<const MaterialInput> inputs) {
    const std::string_view id = Info(schema).name;
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (inputs[j].name == inputs[i].name) {
                throw Error("{}'s {} is given twice", id, inputs[i].name);
            }
        }
        SetInput(writer, schema, id, inputs[i]);
    }
}

SceneImpl::SceneImpl(Scene& scene, device::Device& device, const SceneDesc& desc)
    : m_scene(scene), m_device(&device), m_store(device), m_catalog(m_store), m_timeline(m_catalog),
      m_builder(bvh::Builder::Create(device, desc.traversal)), m_hierarchy(m_catalog),
      m_geometry(m_catalog, *m_builder), m_bounds(m_catalog, m_instances), m_tangents(m_catalog),
      m_textures(m_catalog, desc.textureThreads), m_semantics(m_catalog),
      m_instances(m_catalog, *m_builder, desc.maxRefits), m_lights(m_catalog, m_instances) {
    m_timeline.Add(m_hierarchy);
    m_timeline.Add(m_geometry);
    m_timeline.Add(m_tangents);
    m_timeline.Add(m_textures);
    m_timeline.Add(m_semantics);
    m_timeline.Add(m_instances);
    m_timeline.Add(m_bounds);
    m_timeline.Add(m_lights);
    m_store.Prepare();
    m_builder->Prepare();
    for (System* system :
         std::initializer_list<System*>{&m_hierarchy, &m_geometry, &m_bounds, &m_tangents,
                                        &m_textures, &m_semantics, &m_instances, &m_lights}) {
        system->Prepare();
    }
    if (desc.waitForKernels) {
        device::WaitForBuilds();
    }
}

Prim SceneImpl::Handle(uint32_t prim) const {
    return Prim(m_scene, prim, m_catalog.GetPrims().Generation(prim));
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

Prim Define(Scene& scene, std::string_view path, const Material::Desc& desc) {
    const SchemaId schema = MaterialSchema(desc.id);
    return SceneImpl::Get(scene).Define(path, schema, Material::Apis(desc), [&](RowWriter& writer) {
        WriteFallbacks(schema, writer);
        Material::Write(writer, desc);
        SetInputs(writer, schema, desc.inputs);
    });
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
    Catalog& catalog = m_impl->GetCatalog();
    catalog.Move(prim.Index(), path);
    std::vector<std::string> xformable;
    for (const uint32_t moved : catalog.GetPaths().Subtree(prim.Index())) {
        if (IsA(catalog.GetPrims().At(moved).schema, SchemaId::Xformable)) {
            xformable.push_back(catalog.GetPaths().Path(moved));
        }
    }
    if (!xformable.empty()) {
        Select(xformable, SchemaId::Xformable).Get(Xformable::kTranslate.Column()).MarkChanged();
    }
    return At(prim.Index());
}

Prim Scene::Rename(Prim prim, std::string_view name) {
    m_impl->Check(prim);
    const std::optional<Prim> parent = prim.Parent();
    return Move(prim, (parent ? parent->Path() : std::string()) + "/" + std::string(name));
}

void Scene::Remove(Prim prim) {
    m_impl->Check(prim);
    m_impl->GetCatalog().Remove(prim.Index());
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
    SceneImpl& impl = SceneImpl::Get(*m_scene);
    impl.Check(prim);
    const uint32_t row = impl.GetCatalog().GetPrims().Row(prim.Index(), column.schema);
    Store& store = impl.GetStore();
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
        const uint32_t row = impl.GetCatalog().GetPrims().FindRow(prim.Index(), column.schema);
        if (row != kNoRow) {
            impl.GetStore().ClearOverride(column, row, m_owner);
            impl.GetStore().StampWorld(prim.World());
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
    const uint32_t row = m_impl->GetCatalog().GetPrims().FindRow(prim.Index(), column.schema);
    return row != kNoRow && m_impl->GetStore().Overridden(column, row);
}

std::vector<std::byte> Scene::ReadAuthored(Prim prim, ColumnRef column) const {
    if (Overridden(prim, column)) {
        const uint32_t row = m_impl->GetCatalog().GetPrims().Row(prim.Index(), column.schema);
        return m_impl->GetStore().Authored(column, row);
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
        SceneImpl::Get(*m_scene).GetCatalog().Decompose(m_world);
        SceneImpl::Get(*m_scene).GetStore().StampWorld(m_world);
    }
    m_world = kGlobalWorld;
}

void ComposedWorld::Set(const ComposedWorldDesc& desc) {
    if (!m_scene) {
        throw Error("An empty world view shows nothing");
    }
    Catalog& catalog = SceneImpl::Get(*m_scene).GetCatalog();
    const uint32_t worlds = catalog.GetPrims().WorldCount();
    if (!desc.offsets.empty() && desc.offsets.size() != desc.worlds.size()) {
        throw Error("A world view has {} offsets for {} worlds", desc.offsets.size(),
                    desc.worlds.size());
    }
    for (const uint32_t world : desc.worlds) {
        if (world >= worlds || world == m_world) {
            throw Error("A world view can't show world {} of {}", world, worlds);
        }
    }
    if (catalog.Compose(m_world, desc)) {
        SceneImpl::Get(*m_scene).GetStore().StampWorld(m_world);
    }
}

ComposedWorld Scene::ComposeWorld(Prim world, const ComposedWorldDesc& desc) {
    m_impl->Check(world);
    if (!world.IsA(SchemaId::World) || !world.IsSession()) {
        throw Error("{} isn't a session World, which only can show other worlds", world.Path());
    }
    if (m_impl->GetCatalog().Compositions().contains(world.World())) {
        throw Error("{} already shows other worlds", world.Path());
    }
    ComposedWorld view(Ref<Scene>(this), world.World());
    view.Set(desc);
    return view;
}

void Scene::MarkSession(Prim prim) {
    m_impl->Check(prim);
    m_impl->GetCatalog().MarkSession(prim.Index());
}

void Scene::Apply(Span<const Prim> prims, SchemaId api) {
    m_impl->Apply(prims, api, [&](RowWriter& writer) { WriteFallbacks(api, writer); });
}

void Scene::Apply(Span<const Prim> prims, SchemaId api, std::string_view instance) {
    m_impl->Apply(prims, api, instance, [&](RowWriter& writer) { WriteFallbacks(api, writer); });
}

void Scene::Unapply(Span<const Prim> prims, SchemaId api) {
    m_impl->GetCatalog().Unapply(m_impl->Indices(prims), api);
}

void Scene::Unapply(Span<const Prim> prims, SchemaId api, std::string_view instance) {
    m_impl->GetCatalog().Unapply(m_impl->Indices(prims), api, instance);
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
    return m_impl->Range(m_impl->GetCatalog().Clone(
        source.Index(), std::span(targets.data(), targets.size()), desc));
}

std::optional<Prim> Scene::Find(std::string_view path) const {
    const uint32_t prim = m_impl->GetCatalog().Find(path);
    if (prim == kNoPrim) {
        return std::nullopt;
    }
    return m_impl->Handle(prim);
}

Prim Scene::At(uint32_t index) const {
    if (!m_impl->GetCatalog().GetPrims().Alive(index)) {
        throw Error("No prim {}", index);
    }
    return m_impl->Handle(index);
}

View Scene::Select(std::string_view pattern, SchemaId schema) const {
    Catalog& catalog = const_cast<SceneImpl&>(*m_impl).GetCatalog();
    return View(catalog, schema, catalog.Select(pattern, schema));
}

View Scene::Select(Span<const std::string> paths, SchemaId schema) const {
    Catalog& catalog = const_cast<SceneImpl&>(*m_impl).GetCatalog();
    std::vector<uint32_t> prims;
    for (const std::string& path : paths) {
        const uint32_t prim = catalog.Find(path);
        if (prim == kNoPrim) {
            throw Error("No prim at \"{}\"", path);
        }
        prims.push_back(prim);
    }
    return View(catalog, schema, std::move(prims));
}

View Scene::Select(const PrimRange& range, SchemaId schema, uint32_t index) const {
    Catalog& catalog = const_cast<SceneImpl&>(*m_impl).GetCatalog();
    std::vector<uint32_t> prims;
    for (uint32_t copy = 0; copy < range.copies; ++copy) {
        prims.push_back(range.Index(copy, index));
    }
    return View(catalog, schema, std::move(prims));
}

View Scene::Select(Prim prim) const {
    m_impl->Check(prim);
    Catalog& catalog = const_cast<SceneImpl&>(*m_impl).GetCatalog();
    return View(catalog, prim.Schema(), std::vector<uint32_t>{prim.Index()});
}

View Scene::Select(std::string_view pattern, SchemaId api, std::string_view instance) const {
    Catalog& catalog = const_cast<SceneImpl&>(*m_impl).GetCatalog();
    if (!IsMultipleApi(api)) {
        throw Error("{} isn't applied many times, so it has no instances", Info(api).name);
    }
    const uint32_t token = catalog.GetTokens().Intern(instance).id;
    std::vector<uint32_t> prims = catalog.Select(pattern, api);
    std::erase_if(prims, [&](uint32_t prim) {
        return catalog.GetPrims().InstanceRow(prim, api, token) == kNoRow;
    });
    return View(catalog, api, std::move(prims), instance);
}

View Scene::Select(Prim prim, SchemaId api, std::string_view instance) const {
    m_impl->Check(prim);
    Catalog& catalog = const_cast<SceneImpl&>(*m_impl).GetCatalog();
    if (!IsMultipleApi(api)) {
        throw Error("{} isn't applied many times, so it has no instances", Info(api).name);
    }
    if (catalog.GetPrims().InstanceRow(prim.Index(), api,
                                       catalog.GetTokens().Intern(instance).id) == kNoRow) {
        throw Error("{} has no {} {}", prim.Path(), Info(api).name, instance);
    }
    return View(catalog, api, std::vector<uint32_t>{prim.Index()}, instance);
}

uint32_t Scene::PrimCount() const {
    return m_impl->GetCatalog().GetPrims().LiveCount();
}

uint32_t Scene::WorldCount() const {
    return m_impl->GetCatalog().GetPrims().WorldCount();
}

device::Operation Scene::Publish(const PublishDesc& desc, Stream producer) {
    {
        ERNIE_ZONE("Scene/CommitSteps");
        m_impl->RunCommitSteps();
    }
    device::Operation done =
        m_impl->GetTimeline().Publish(desc.ordinal.value_or(Published() + 1), producer);
    while (desc.wait && Pending()) {
        m_impl->GetTimeline().Finish();
        m_impl->RunCommitSteps();
        done = m_impl->GetTimeline().Publish(Published() + 1, producer);
    }
    return done;
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
    return m_impl->CommitStepsPending() || m_impl->GetTimeline().Pending();
}

uint64_t Scene::StructureVersion() const {
    return m_impl->GetCatalog().Epoch();
}

Ordinal Scene::Published() const {
    return const_cast<SceneImpl&>(*m_impl).GetTimeline().Published();
}

Snapshot Scene::Acquire(Stream consumer) {
    Snapshot snapshot = m_impl->GetTimeline().Acquire(consumer);
    const Ref<device::Buffer>& layout = m_impl->GetInstances().Layout();
    snapshot.instances = layout ? layout->Address() : 0;
    const Ref<device::Buffer>& lights = m_impl->GetLights().Layout();
    snapshot.lights = lights ? lights->Address() : 0;
    snapshot.semantics = m_impl->GetSemantics().Data();
    const Ref<device::Buffer>& bounds = m_impl->GetBounds().Boxes();
    snapshot.bounds = bounds ? bounds->Address() : 0;
    return snapshot;
}

void Scene::Release(const Snapshot& snapshot, Stream consumer) {
    m_impl->GetTimeline().Release(snapshot, consumer);
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
    m_impl->GetCatalog().AddFile(std::move(path), std::move(bytes), index);
}

void Scene::RemoveFile(std::string_view path) {
    m_impl->GetCatalog().RemoveFile(path);
}

ernie::Token Scene::Token(std::string_view text) {
    return m_impl->GetCatalog().GetTokens().Intern(text);
}

std::string_view Scene::Text(ernie::Token token) const {
    return m_impl->GetCatalog().GetTokens().Text(token);
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
    Catalog& catalog = m_impl->GetCatalog();
    auto state = std::make_shared<PrimCopy::State>();
    state->root = prim.Path();

    struct Pending {
        CopiedValue* value;
        ColumnType type;
        device::Download read;
    };

    std::vector<Pending> reads;
    for (const uint32_t index : catalog.GetPaths().Subtree(prim.Index())) {
        const PrimRecord& record = catalog.GetPrims().At(index);
        const std::string path = catalog.GetPaths().Path(index);
        CopiedPrim& copied = state->prims.emplace_back();
        copied.relative = path.substr(state->root.size());
        if (copied.relative.starts_with('/')) {
            copied.relative.erase(0, 1);
        }
        copied.schema = record.schema;
        copied.apis = record.apis;
        copied.files = catalog.OwnedFiles(index);
        for (const SchemaId table : PrimTables(record.schema, record.apis)) {
            std::vector<std::string> names = {""};
            if (IsMultipleApi(table)) {
                names = At(index).Instances(table);
                for (const std::string& name : names) {
                    copied.instances.emplace_back(table, name);
                }
            }
            const Span<const ColumnInfo> columns = Columns(table);
            for (const std::string& name : names) {
                for (uint32_t column = 0; column < columns.size(); ++column) {
                    if (!Has(columns[column].flags, ColumnFlags::Derived)) {
                        copied.values.push_back({table, column, {}, name});
                    }
                }
            }
        }
        const View view = Select(At(index));
        for (CopiedValue& value : copied.values) {
            const ColumnInfo& info = *value.Column().column;
            const UntypedAttribute attribute =
                value.instance.empty()
                    ? view.Get(value.Column())
                    : Select(At(index), value.owner, value.instance).Get(value.Column());
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
    const auto pathOf = [&](const CopiedPrim& copied) {
        return copied.relative.empty() ? std::string(path)
                                       : std::string(path) + "/" + copied.relative;
    };

    std::vector<Prim> pasted;
    try {
        for (const CopiedPrim& copied : state.prims) {
            const Prim prim = Define(pathOf(copied), copied.schema);
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
        const Prim prim = At(pasted[at].Index());
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
    return At(pasted.front().Index());
}

}
