import math
import os
import tempfile

import pytest

import ernie
from ernie import tokens

app = pytest.importorskip("ernie.app", reason="Ernie was built without its app", exc_type=ImportError)


def cube_scene(lit=True):
    scene = ernie.Scene()
    scene.define("/World/Cube", tokens.CUBE).set(tokens.SIZE, 2.0)
    if lit:
        scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
            tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 1.0))
        )
    return scene


@pytest.fixture
def viewer():
    scene = cube_scene()
    try:
        return app.Viewer(scene, hidden=True, width=480, height=320)
    except RuntimeError as error:
        pytest.skip(f"No Vulkan device: {error}")


def test_opens_with_its_own_camera_and_lights_the_scene(viewer):
    assert viewer.active_camera == app.Viewer.perspective_camera
    assert viewer.cameras[0] == app.Viewer.perspective_camera
    assert str(app.Viewer.session_root) == "/__session"
    assert viewer.lighting == app.Lighting.STAGE
    assert viewer.scene.has(app.Viewer.perspective_camera)
    assert viewer.device_name


def test_step_runs_frame_callbacks_and_close_ends_run(viewer):
    frames = []

    def tick(v, info):
        frames.append(info.frame)
        if len(frames) == 3:
            v.close()

    callback = viewer.on_frame(tick)
    viewer.run()
    assert frames == [0, 1, 2]
    assert not viewer.is_open
    assert not viewer.step()
    viewer.remove_callback(callback)


def test_python_exceptions_in_callbacks_propagate(viewer):
    def fail(v, info):
        raise ValueError("boom")

    callback = viewer.on_frame(fail)
    with pytest.raises(ValueError, match="boom"):
        viewer.step()
    viewer.remove_callback(callback)
    assert viewer.step()


def test_cameras_lighting_and_selection(viewer):
    viewer.camera_from_view("/Cameras/Saved")
    viewer.active_camera = "/Cameras/Saved"
    assert viewer.active_camera == ernie.Path("/Cameras/Saved")
    viewer.lighting = app.Lighting.CAMERA
    assert viewer.step()
    assert viewer.wireframe is None
    viewer.wireframe = ernie.Wireframe.HIDDEN_LINE
    assert viewer.step()
    assert viewer.scene.get(viewer.session_root.child("RenderSettings"), ernie.tokens.WIREFRAME) == (
        ernie.tokens.HIDDEN_LINE
    )
    viewer.wireframe = None
    assert "PrimId" in viewer.view_vars and "SpectralRadiance" in viewer.view_vars
    viewer.view_var = "Normal"
    assert viewer.step()
    viewer.view_var = "SpectralRadiance"
    viewer.view_band = 3
    assert viewer.step()
    viewer.view_band = -1
    viewer.view_var = "LdrColor"
    seen = []
    viewer.on_selection(lambda v, paths: seen.append(paths))
    viewer.selection = [ernie.Path("/World/Cube")]
    assert seen == [[ernie.Path("/World/Cube")]]
    viewer.frame([ernie.Path("/World/Cube")])
    with pytest.raises(ValueError):
        viewer.active_camera = "/World/Cube"


def test_widget_panels_and_menus(viewer):
    panel = viewer.add_panel("Sim")
    slider = panel.slider("Sun", 0.0, 90.0, 10.0, on_change=lambda value: None)
    check = panel.checkbox("Shadows", True)
    combo = panel.combo("Mode", ["a", "b"], 1)
    text = panel.text("ready")
    panel.plot("History", lambda: [1.0, 2.0, 3.0])
    panel.set(slider, 45.0)
    panel.set(check, False)
    panel.set(combo, 0)
    panel.set(text, "running")
    with pytest.raises(ValueError):
        panel.set(slider, True)
    viewer.add_menu_item("Tools", "Reset", lambda v: None)
    assert viewer.step()

    # The viewport's eye menu, heads-up display and settings menu take items too.
    switched = []
    toggle = viewer.add_viewport_toggle("Contacts", False, "contacts",
                                        lambda v, on: switched.append(on))
    hud = viewer.add_hud_item("Physics", lambda v: "1 ms")
    viewer.add_viewport_menu_item("Pause", lambda v: None, lambda v: True)
    display = viewer.viewport_display
    display.grid = False
    display.hud_triangles = True
    viewer.viewport_display = display
    assert viewer.viewport_display == display
    viewer.set_viewport_toggle_shown(toggle, True)
    assert switched == [True] and viewer.viewport_toggle_shown(toggle)
    assert viewer.step()
    viewer.set_hud_item_shown(hud, False)
    assert not viewer.hud_item_shown(hud)
    viewer.remove_viewport_item(toggle)
    with pytest.raises(ValueError):
        viewer.viewport_toggle_shown(toggle)
    assert panel.title == "Sim"
    viewer.remove_panel("Sim")


def test_capture_writes_a_png(viewer, tmp_path):
    viewer.step()
    path = tmp_path / "view.png"
    viewer.capture(path)
    viewer.step()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_layouts_and_themes(viewer, tmp_path):
    viewer.step()
    layout = tmp_path / "layout.ini"
    viewer.save_layout(layout)
    assert "[Docking]" in layout.read_text()
    viewer.load_layout(layout)
    theme = app.light_theme()
    theme.accent = (0.9, 0.3, 0.1, 1.0)
    viewer.theme = theme
    assert viewer.theme.accent == pytest.approx((0.9, 0.3, 0.1, 1.0))
    assert viewer.step()


def test_an_unlit_scene_opens_with_the_default_light():
    try:
        viewer = app.Viewer(cube_scene(lit=False), hidden=True, width=160, height=120)
    except RuntimeError as error:
        pytest.skip(f"No Vulkan device: {error}")
    assert viewer.lighting == app.Lighting.DEFAULT


def test_pre_and_post_render_callbacks_run_in_order(viewer):
    order = []
    viewer.on_post_render(lambda v, info: order.append(("post", info.frame)))
    viewer.on_pre_render(lambda v, info: order.append(("pre", info.frame)))
    viewer.on_frame(lambda v, info: order.append(("frame", info.frame)))
    viewer.step()
    assert order == [("frame", 0), ("pre", 0), ("post", 0)]


def test_panels_take_a_dock(viewer):
    panel = viewer.add_panel("Left", app.Dock.LEFT)
    assert panel.dock == app.Dock.LEFT
    assert viewer.add_panel("Default").dock == app.Dock.RIGHT
    assert viewer.step()


def test_renderer_is_an_option_and_switchable(viewer):
    assert "raster" in app.Viewer.renderers
    assert viewer.renderer == "raster"
    with pytest.raises(ValueError):
        viewer.renderer = "no_such_backend"
    viewer.renderer = "raster"
    viewer.sync()
    assert viewer.step()


def test_clicks_pick_prims_and_edits_undo(viewer):
    viewer.step()
    viewer.step()
    # The default view frames the cube, so the image's centre shows it.
    assert viewer.pick(240, 160) == ernie.Path("/World/Cube")
    assert viewer.pick(2, 2) == ernie.Path()
    assert viewer.pick(10000, 10000) == ernie.Path()

    viewer.selection = [ernie.Path("/World/Cube")]
    viewer.step()  # draws the Stage and Properties panels and the selection's outline
    assert "__selection" in viewer.scene.overlay_names()

    viewer.edit("/World/Cube", tokens.SIZE, 3.0)
    viewer.edit("/World/Cube", tokens.DISPLAY_COLOR, (1.0, 0.0, 0.0))
    assert viewer.scene.get("/World/Cube", tokens.SIZE) == 3.0
    assert viewer.undo()
    assert viewer.scene.get("/World/Cube", tokens.DISPLAY_COLOR) is None
    assert viewer.undo()
    assert viewer.scene.get("/World/Cube", tokens.SIZE) == 2.0
    assert viewer.redo()
    assert viewer.scene.get("/World/Cube", tokens.SIZE) == 3.0
    viewer.clear_attribute("/World/Cube", tokens.SIZE)
    assert viewer.scene.get("/World/Cube", tokens.SIZE) is None
    viewer.step()

    assert viewer.gizmo == app.Gizmo.TRANSLATE
    viewer.gizmo = app.Gizmo.ROTATE
    viewer.gizmo_local = True
    viewer.step()


def test_create_makes_every_type_and_undoes(viewer):
    types = app.Viewer.creatable_types()
    assert "Cube" in types and "SphereLight" in types and "Camera" in types
    before = set(str(p) for p in viewer.scene.prims())
    created = []
    for kind in types:
        path = viewer.create(kind)
        assert viewer.scene.type_of(path) == kind
        assert viewer.selection == [path]
        created.append(path)
        viewer.step()  # each renders without an error
        assert not any(level == "error" for _, level, _ in viewer.log_entries), viewer.log_entries
    assert created[0] != created[1]
    assert viewer.create("Cube") != viewer.create("Cube"), "unique names"
    with pytest.raises(ValueError):
        viewer.create("Teapot")

    viewer.delete([ernie.Path("/World")])
    assert not viewer.scene.has("/World/Cube")
    viewer.undo()
    assert viewer.scene.has("/World/Cube")
    for _ in range(len(types) + 2):
        viewer.undo()
    created_paths = set(str(p) for p in viewer.scene.prims() if not str(p).startswith("/__session"))
    assert created_paths <= before | {"/World", "/Looks", "/Render", "/Render/Vars"}

    viewer.log("hello", level="warning")
    assert viewer.log_entries[-1][1:] == ("warning", "hello")
    viewer.clear_log()
    assert viewer.log_entries == []


def test_auto_exposure_keys_bright_scenes_to_middle_grey():
    # A grey cube under 100,000 lux: through the viewer's camera (exposure 1, f/1, 1 s) the
    # camera's own exposure is white; automatic exposure brings it to the middle.
    image_module = pytest.importorskip("PIL.Image")
    scene = cube_scene(lit=False)
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.0, -0.3, 1.0))
    ).set(tokens.PHOTOMETRIC_ILLUMINANCE, 100000.0)
    try:
        viewer = app.Viewer(scene, hidden=True, width=320, height=240)
    except RuntimeError as error:
        pytest.skip(f"No Vulkan device: {error}")
    assert viewer.exposure == app.Exposure.CAMERA and not viewer.auto_exposing, "off by default"
    viewer.exposure = app.Exposure.AUTO
    display = viewer.viewport_display  # the sun's gizmo would cover the centre
    display.grid = display.lights = False
    viewer.viewport_display = display
    for _ in range(30):  # long enough to adapt, at the hidden viewer's frame times
        viewer.step()
    assert viewer.metered_ev == pytest.approx(math.log2(0.5 * 100000.0 / math.pi * 8.0), abs=1.0)

    def centre():
        path = tempfile.mktemp(suffix=".png")
        viewer.capture(path)
        viewer.step()
        with image_module.open(path) as image:
            value = image.getpixel((image.width // 2, image.height // 2))[1]
        os.remove(path)
        return value

    automatic = centre()
    assert 60 < automatic < 220, automatic
    viewer.exposure = app.Exposure.CAMERA
    assert not viewer.auto_exposing
    viewer.step()
    assert centre() == 255, "the camera's own exposure: blown out"
