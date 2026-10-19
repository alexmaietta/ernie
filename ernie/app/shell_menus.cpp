// SPDX-License-Identifier: BSD-3-Clause
/// @file shell_menus.cpp
/// @brief The shell's menu bar and the edits its menus and shortcuts make: files opened and
/// saved, undo and redo, prims created, duplicated and deleted, each edit one undo step.
#include "ernie/app/create.h"
#include "ernie/app/menu.h"
#include "ernie/app/ui.h"

#include "ernie/scene/json.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/render.h"
#include "ernie/util/profiler.h"

#include <imgui_stdlib.h>

#include <array>
#include <exception>
#include <string>
#include <utility>

namespace ernie::app {
namespace {

constexpr const char* kDocumentation = "https://alexmaietta.github.io/ernie/";
constexpr const char* kRepository = "https://github.com/alexmaietta/ernie";

/// @return Whether a prim of `type` groups others: new prims go under a selected one.
bool IsGroup(std::string_view type) {
    return type == Xform::kName || type == Scope::kName;
}

/// @return A camera the scene's render settings name, for new render prims; empty for none.
Path SettingsCamera(const Scene& scene) {
    for (const Path& settings : scene.Prims(RenderSettings::kName)) {
        const Path camera = RenderSettings(scene, settings).GetCamera();
        if (!camera.IsEmpty()) {
            return camera;
        }
    }
    return {};
}

}  // namespace

template <typename Edit> void Shell::Attempt(const char* what, Edit&& edit) {
    try {
        edit();
    } catch (const std::exception& error) {
        log_.Error(std::string(what) + ": " + error.what());
    }
}

void Shell::DrawMenuBar() {
    if (!ImGui::BeginMainMenuBar()) {
        return;
    }
    const bool selected = !selection_.Empty();
    Menu("File", [&] {
        windows_.open |= MenuAction("Open Scene...", "Ctrl+O");
        windows_.save |= MenuAction("Save Scene As...", "Ctrl+Shift+S");
        ImGui::Separator();
        choices_.quit |= MenuAction("Quit", "Ctrl+Q");
    });
    Menu("Edit", [&] {
        const std::string undo = history_.CanUndo() ? "Undo " + history_.UndoLabel() : "Undo";
        const std::string redo = history_.CanRedo() ? "Redo " + history_.RedoLabel() : "Redo";
        choices_.undo |= MenuAction(undo.c_str(), "Ctrl+Z", history_.CanUndo());
        choices_.redo |= MenuAction(redo.c_str(), "Ctrl+Shift+Z", history_.CanRedo());
        ImGui::Separator();
        choices_.duplicate |= MenuAction("Duplicate", "Ctrl+D", selected);
        choices_.remove |= MenuAction("Delete", "Del", selected);
        ImGui::Separator();
        choices_.frame |= MenuAction(selected ? "Frame Selection" : "Frame Scene", "F");
        choices_.clearSelection |= MenuAction("Clear Selection", nullptr, selected);
    });
    Menu("Create", [&] { DrawCreateMenu(); });
    Menu("Window", [&] {
        for (const std::unique_ptr<Panel>& panel : panels_) {
            bool visible = panel->Visible();
            MenuCheck(panel->Title().c_str(), visible);
            panel->SetVisible(visible);
        }
        ImGui::Separator();
        choices_.resetLayout |= MenuAction("Reset Layout");
        MenuCheck("ImGui Demo", windows_.demo);
    });
    Menu("Help", [&] { DrawHelpMenu(); });
    ImGui::EndMainMenuBar();
}

void Shell::DrawCreateMenu() {
    std::string_view group;
    for (const CreatableType& creatable : CreatableTypes()) {
        if (group != creatable.group) {
            if (!group.empty()) {
                ImGui::Separator();
            }
            group = creatable.group;
            MenuSection(creatable.group);
        }
        if (MenuAction(creatable.label)) {
            choices_.create = creatable.type;
        }
    }
}

void Shell::DrawHelpMenu() {
    const std::string docs = kDocumentation;
    const std::array<std::pair<const char*, std::string>, 5> links{{
        {"Documentation", docs},
        {"Viewer Guide", docs + "getting-started/viewer/"},
        {"Demos", docs + "getting-started/demos/"},
        {"Python API", docs + "api/python/scene/"},
        {"C++ API", docs + "api/cpp/"},
    }};
    const auto open = [&](const std::string& url) {
        if (!OpenExternally(url)) {
            log_.Error("Could not open " + url);
        }
    };
    for (const auto& [label, url] : links) {
        if (MenuAction(label)) {
            open(url);
        }
    }
    ImGui::Separator();
    if (MenuAction("Report an Issue")) {
        open(std::string(kRepository) + "/issues");
    }
    if (MenuAction("Source Code")) {
        open(kRepository);
    }
    ImGui::Separator();
    MenuCheck("Keyboard and Mouse", windows_.controls);
    MenuCheck("About Ernie", windows_.about);
}

void Shell::DrawFileWindow() {
    if (!windows_.open && !windows_.save) {
        return;
    }
    const char* title = windows_.open ? "Open Scene" : "Save Scene As";
    bool shown = true;
    ImGui::SetNextWindowSize(ImVec2(32.0f * ImGui::GetFontSize(), 0.0f), ImGuiCond_Appearing);
    if (ImGui::Begin(title, &shown, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking)) {
        ImGui::TextUnformatted("Scene file (.json):");
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
        }
        const bool entered =
            ImGui::InputText("##path", &filePath_, ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::Button(windows_.open ? "Open" : "Save") || entered) {
            if (windows_.open) {
                Attempt("Opening a scene", [&] { OpenScene(filePath_); });
            } else {
                Attempt("Saving the scene", [&] { SaveScene(filePath_); });
            }
            shown = false;
        }
        ImGui::SameLine();
        shown &= !ImGui::Button("Cancel");
    }
    ImGui::End();
    if (!shown) {
        windows_.open = windows_.save = false;
    }
}

void Shell::TakeShortcuts() {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) {
        return;
    }
    const auto pressed = [](ImGuiKey key) { return ImGui::IsKeyPressed(key, false); };
    const bool plain = !io.KeyCtrl && !io.KeyAlt;
    choices_.quit |= io.KeyCtrl && pressed(ImGuiKey_Q);
    choices_.frame |= plain && pressed(ImGuiKey_F);
    choices_.undo |= io.KeyCtrl && !io.KeyShift && pressed(ImGuiKey_Z);
    choices_.redo |= io.KeyCtrl && ((io.KeyShift && pressed(ImGuiKey_Z)) || pressed(ImGuiKey_Y));
    choices_.duplicate |= io.KeyCtrl && pressed(ImGuiKey_D);
    choices_.remove |= plain && pressed(ImGuiKey_Delete);
    windows_.open |= io.KeyCtrl && pressed(ImGuiKey_O);
    windows_.save |= io.KeyCtrl && io.KeyShift && pressed(ImGuiKey_S);
}

void Shell::Apply() {
    open_ = open_ && !choices_.quit;
    layoutBuilt_ = layoutBuilt_ && !choices_.resetLayout;
    if (choices_.clearSelection) {
        selection_.Clear();
    }
    if (choices_.undo) {
        Attempt("Undoing", [&] { history_.Undo(scene_); });
    }
    if (choices_.redo) {
        Attempt("Redoing", [&] { history_.Redo(scene_); });
    }
    if (choices_.duplicate) {
        Attempt("Duplicating", [&] { DuplicateSelection(); });
    }
    if (choices_.remove) {
        Attempt("Deleting", [&] { DeleteSelection(); });
    }
    if (!choices_.create.empty()) {
        Attempt("Creating", [&] { Create(choices_.create); });
    }
    // Undone prims leave the selection.
    std::vector<Path> kept;
    for (const Path& path : selection_.Paths()) {
        if (scene_.Find(path)) {
            kept.push_back(path);
        }
    }
    if (kept.size() != selection_.Paths().size()) {
        selection_.Clear();
        for (const Path& path : kept) {
            selection_.Toggle(path);
        }
    }
    if (choices_.frame) {
        Frame();
    }
}

Path Shell::Create(std::string_view type) {
    const CreatableType* creatable = FindCreatable(type);
    if (creatable == nullptr) {
        throw std::invalid_argument("The viewer cannot create a '" + std::string(type) + "'");
    }
    Path parent(creatable->parent);
    const std::optional<PrimId> selected =
        selection_.Empty() ? std::nullopt : scene_.Find(selection_.Paths().front());
    if (selected && IsGroup(scene_.TypeName(*selected))) {
        parent = selection_.Paths().front();
    }
    const Path path = FreeChildPath(scene_, parent, creatable->name);
    const std::optional<PrimId> parentPrim = scene_.Find(parent);
    // Authored in a scratch scene, then copied in as one step with any missing parents.
    Scene scratch;
    const CreateContext context{.target = camera_.target,
                                .scale = std::max(0.05 * camera_.distance, 1e-3),
                                .parentWorld = parentPrim ? scene_.WorldTransform(*parentPrim)
                                                          : glm::dmat4(1.0),
                                .view = camera_.Transform(),
                                .verticalFov = camera_.verticalFov,
                                .camera = SettingsCamera(scene_)};
    AuthorDefaultPrim(scratch, path, type, context);
    // Missing ancestors become Xforms, outermost first, in the same step.
    std::vector<Path> missing;
    for (Path at = parent; !at.IsRoot() && !scene_.Find(at); at = at.Parent()) {
        missing.insert(missing.begin(), at);
    }
    history_.BeginGroup();
    for (const Path& at : missing) {
        history_.Define(scene_, at, Xform::kName);
    }
    history_.Add(scene_, scratch, path);
    history_.EndGroup();
    selection_.Set(path);
    log_.Info("Created " + path.String());
    return path;
}

void Shell::DuplicateSelection() {
    std::vector<Path> copies;
    history_.BeginGroup();
    for (const Path& path : selection_.Paths()) {
        if (!scene_.Find(path)) {
            continue;
        }
        const Path copy = FreeChildPath(scene_, path.Parent(), path.Name());
        history_.Copy(scene_, path, copy);
        copies.push_back(copy);
    }
    history_.EndGroup();
    selection_.Clear();
    for (const Path& copy : copies) {
        selection_.Toggle(copy);
    }
}

void Shell::DeleteSelection() {
    history_.BeginGroup();
    for (const Path& path : selection_.Paths()) {
        history_.Remove(scene_, path);
    }
    history_.EndGroup();
    selection_.Clear();
}

void Shell::OpenScene(const std::filesystem::path& path) {
    // Read once aside, so a bad file leaves the scene as it was.
    Scene checked;
    LoadSceneFile(checked, path);
    for (const Path& root : scene_.Children(Path::Root())) {
        scene_.Remove(root);
    }
    LoadSceneFile(scene_, path);
    history_.Reset();
    selection_.Clear();
    log_.Info("Opened " + path.string());
    Frame();
}

void Shell::SaveScene(const std::filesystem::path& path) {
    SaveSceneFile(scene_, path);
    log_.Info("Saved " + path.string());
}

}  // namespace ernie::app
