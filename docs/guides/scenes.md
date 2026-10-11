# Describing scenes

An Ernie scene is a tree of prims, each at a path such as `/World/Rover/Mast/Camera`. A prim has a type (`SphereLight`, `Mesh`, `Camera`) and attributes named as in OpenUSD (`inputs:radius`, `focalLength`, `points`). Every prim type Ernie draws is declared once, in C++, as a list of its attributes with their types, defaults, units and limits. From that one declaration Ernie builds typed C++ views (`SphereLight::Define(scene, path).SetRadius(0.1)`), typed Python classes (`scene.define(path, ernie.SphereLight, radius=0.1)`), Python stubs for your IDE, the viewer's property panels, validation of every attribute you set, and the JSON scene loader.

This guide covers describing scenes with those typed prims, in Python and in C++. The examples are short and runnable: paste a Python block after the ones before it, or run it alone where it builds its own scene.

## A first scene

```python
import ernie
from ernie import tokens

scene = ernie.Scene()
scene.define("/World/Ground", ernie.GroundPlane, checker=True)
rock = scene.define("/World/Rock", ernie.Sphere, radius=0.5, translate=(0, 0, 0.5))
sun = scene.define("/World/Sun", ernie.DistantLight, angle=0.53, photometric_illuminance=128000.0)
sun.set(tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7)))
camera = scene.define("/World/Camera", ernie.Camera, focal_length=35.0)
camera.set(tokens.TRANSFORM, ernie.look_at_transform((0, -8, 3), (0, 0, 0.5)))
scene.define("/Render/Settings", ernie.RenderSettings, camera=camera, resolution=(640, 480))

image = ernie.Renderer().render(scene)  # a (480, 640, 4) uint8 NumPy array
```

The same scene in C++:

```cpp
#include <ernie/scene/camera_schemas.h>
#include <ernie/scene/geometry_schemas.h>
#include <ernie/scene/light_schemas.h>
#include <ernie/scene/render_schemas.h>
#include <ernie/scene/schema.h>
#include <ernie/session/renderer.h>

int main() {
    using namespace ernie;
    Scene scene;
    GroundPlane::Define(scene, "/World/Ground").SetChecker(true);
    Sphere::Define(scene, "/World/Rock")
        .SetRadius(0.5)
        .SetXform({.translate = glm::dvec3(0.0, 0.0, 0.5)});
    DistantLight::Define(scene, "/World/Sun")
        .SetAngle(0.53)
        .SetPhotometricIlluminance(128000.0)
        .SetTransform(DistantLightTransform({0.4, -0.6, 0.7}));
    Camera::Define(scene, "/World/Camera")
        .SetFocalLength(35.0)
        .SetTransform(LookAtTransform({0.0, -8.0, 3.0}, {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0}));
    RenderSettings::Define(scene, "/Render/Settings")
        .SetCamera("/World/Camera")
        .SetResolution({640, 480});

    Renderer renderer;
    Tensor image = renderer.Render(scene);
}
```

## Prim types

Ernie's types follow OpenUSD's schemas (UsdGeom, UsdLux, UsdShade, UsdRender), with Ernie's own where USD has none (`GroundPlane`, `Terrain`, `StarField`, `SensorFilter`). Each type derives from a base and has its base's attributes too: every prim has `visibility` and `world`, every Gprim a `display_color` and `material_binding`, every light an `intensity` and `color_temperature`. Abstract types, marked *(abstract)*, are bases only: you can look prims up as them but not define them.

```text
Imageable (abstract)              visibility, world, quaternion_order
├── Xform                         a transform grouping its children
├── Scope                         a grouping without a transform
├── Boundable (abstract)          extent
│   ├── Gprim (abstract)          display_color, material_binding
│   │   ├── Cube, Sphere
│   │   ├── AxisShape (abstract)  axis
│   │   │   └── Cylinder, Cone, Capsule, Plane
│   │   ├── PointBased (abstract) points
│   │   │   └── Mesh, Points
│   │   ├── Volume
│   │   └── Terrain
│   └── PointInstancer
├── GroundPlane
├── Light (abstract)              intensity, exposure, color, color_temperature, spectrum, ...
│   ├── BoundableLightBase (abstract)      power, photometric_power, ...
│   │   └── SphereLight, RectLight, DiskLight
│   └── NonboundableLightBase (abstract)   photometric_illuminance
│       └── DistantLight, DomeLight
├── Camera
├── Material, Shader
├── RenderSettingsBase (abstract) camera, resolution, ...
│   └── RenderSettings, RenderProduct
├── RenderVar
├── SensorFilter
└── StarField
```

In Python each type is a class of the same name in the `ernie` module, and the classes derive from each other as the types do, so `isinstance(lamp, ernie.Light)` holds for a `SphereLight`. In C++ each is a view class in a header per family: `base_schemas.h` (Imageable, Xform, Scope, Boundable, Gprim), `geometry_schemas.h` (shapes, Mesh, Points, PointInstancer, Volume, GroundPlane), `light_schemas.h`, `camera_schemas.h`, `material_schemas.h`, `render_schemas.h` and `terrain_schemas.h` (Terrain, StarField), all in `ernie/scene/`. Three C++ class names differ from their type names: `BoundableLight` (`BoundableLightBase`), `NonboundableLight` (`NonboundableLightBase`) and `SensorFilterPrim` (`SensorFilter`).

### Listing types and attributes

The registry answers what types there are and what they take, so you rarely need to read the headers:

```python
[t.name for t in ernie.prim_types()]          # every type, bases first
ernie.SphereLight.prim_type.base.name         # 'BoundableLightBase'
[a.py_name for a in ernie.SphereLight.attributes()]   # its own and its bases', root first

radius = ernie.SphereLight.radius             # on the class, an attribute's definition
radius.name, radius.py_name                   # ('inputs:radius', 'radius')
radius.doc, radius.default, radius.group      # ('Metres.', 0.5, 'Shape')
radius.kind, radius.min, radius.max           # (AttributeKind.DOUBLE, 0.0, 1000.0)
ernie.SphereLight.physical_illuminant.options # ['white', 'blackbody', 'illuminant D', 'custom']
```

The generated stubs carry each attribute's doc and default too, so an IDE shows them as you type.

`ernie.find_prim_type("SphereLight")` gives the same `PrimType` by name, with `is_a("Light")`, `find("inputs:radius")` (by USD name) and `all_attributes()`. In C++ the registry is `PrimTypes()`, `FindPrimType(name)`, `GetPrimType(name)` (which throws for an unknown name) and `PrimTypeOf(scene, path)`, in `ernie/scene/prim_type.h`; each `PrimTypeDef` has its `name`, `base`, `abstract`, `doc` and `attributes`, with `AllAttributes()`, `Find(usdName)`, `FindByPyName(name)` and `IsA(other)`. The [Schemas and lights](../api/python/schemas.md) API page lists the types with their attributes.

## Defining prims

`scene.define(path, cls, **attributes)` defines a prim of the class's type and returns it as an instance of the class. Keyword arguments set attributes by their Python names, the snake_case of their C++ names (`color_temperature`, `focal_length`, `material_binding`). On the instance each attribute is a property: reading it gives its value as the declared type, and setting it converts and checks the value.

```python
scene = ernie.Scene()
lamp = scene.define("/World/Lamp", ernie.SphereLight, radius=0.1, intensity=40.0)
lamp.radius                       # 0.1
lamp.color_temperature = 3000.0
lamp.enable_color_temperature = True
lamp.path, lamp.type              # (Path('/World/Lamp'), 'SphereLight')
```

The class method form, `ernie.SphereLight.define(scene, "/World/Lamp", radius=0.1)`, is the same. Defining an abstract type raises `TypeError: Light is abstract; define one of the types derived from it`.

Defining a prim defines its missing ancestors as untyped prims, as `UsdStage::DefinePrim` does. Defining a prim that exists with the same type returns it unchanged; defining it with another type retypes it and drops its attributes.

`scene.define(path, "Sphere")` with a type name instead of a class returns a plain `Prim`, still taking keyword attributes for a registered type. A typed instance has everything a `Prim` has (`path`, `type`, `get`, `set`, `read`, `write`, `set_xform`, `define`, `get_prim`) and its `.prim` is the `Prim` itself.

### Defaults, and whether a value is set

An attribute you have not set reads as its declared default: the value Ernie renders it with, which is UsdGeom's, UsdLux's or Ernie's own. So `lamp.color_temperature` is `6500.0` before you set it, and `ernie.SphereLight.color_temperature.default` says so. The defaults live in the declarations alone; nothing is written into the scene until you set it.

To tell a default from a value you set, ask the prim by USD name: `prim.get(name)` is `None` when unset (and `prim.read(name)` for arrays), and `scene.clear(path, name)` unsets it again.

```python
lamp.get(ernie.SphereLight.exposure.name) is None   # True: never set, reads as 0.0
scene.clear(lamp.path, "inputs:radius")
lamp.radius                                          # 0.5, the default again
```

In C++ every attribute has `Get<Name>()` (the default when unset), `Has<Name>()` (whether it is set, as a value or an array), `Set<Name>(value)` and `Clear<Name>()`:

```cpp
SphereLight lamp = SphereLight::Define(scene, "/World/Lamp").SetRadius(0.1);
lamp.GetColorTemperature();   // 6500, the default
lamp.HasColorTemperature();   // false
lamp.ClearRadius();           // GetRadius() is 0.5 again
```

### Children and relative paths

`define` on a prim defines a child, by name or by a relative path; the parent can be a typed instance or a `Prim`:

```python
scene = ernie.Scene()
rover = scene.define("/World/Rover", ernie.Xform)
rover.define("Body", ernie.Cube, size=1.0, display_color=(0.8, 0.8, 0.8))
mast_camera = rover.define("Mast/Camera", ernie.Camera, focal_length=12.0)
mast_camera.path                  # Path('/World/Rover/Mast/Camera'); Mast is untyped
scene.children("/World/Rover")    # [Path('/World/Rover/Body'), Path('/World/Rover/Mast')]
```

In C++, `Prim::Define(relative, type)` does the same, and a view's `GetPrim()` gives its `Prim`; `Path::Child(name)` builds a child's path.

## Transforms

A prim's transform is a stack of USD transform ops. The keywords `translate`, `rotate` (Euler angles in degrees), `rotation_order` (`"XYZ"` by default), `orient` (a quaternion, in place of `rotate`) and `scale` (three numbers or one) set them, on `define` or on `set`. Points are scaled, then rotated, then moved.

```python
scene = ernie.Scene()
rover = scene.define("/World/Rover", ernie.Xform, translate=(1, 2, 0), rotate=(0, 0, 90))
rover.define("Wheel", ernie.Cylinder, radius=0.2, height=0.1, translate=(0.5, 0.4, 0.2),
             orient=ernie.Quat.from_axis_angle((1, 0, 0), 90.0), scale=1.0)
rover.set(translate=(5, 0, 0))    # replaces the ops
scene.local_transform("/World/Rover")    # 4 x 4
scene.world_transform("/World/Rover/Wheel")
```

`ernie.Quat(w, x, y, z)` puts the real part first, as USD writes `xformOp:orient`; `ernie.Quat.from_xyzw` takes the order SciPy, ROS and Warp use, and `Quat.from_axis_angle` and `Quat.from_euler` build one. `prim.set_xform(...)` takes the same keywords.

A whole matrix is one op, `tokens.TRANSFORM` (`xformOp:transform`): a 4 x 4 array or rows of four, in the math convention (column vectors, translation in the last column). `ernie.look_at_transform(eye, target)` and `ernie.distant_light_transform(towards_light)` build the usual ones:

```python
camera = scene.define("/World/Camera", ernie.Camera)
camera.set(tokens.TRANSFORM, ernie.look_at_transform((0, -8, 3), (0, 0, 1)))
```

In C++ a view has `SetXform(XformOps)`, with designated initializers for the parts you set, and `SetTransform(glm::dmat4)`; `GetLocalTransform()` reads it back:

```cpp
Xform::Define(scene, "/World/Rover")
    .SetXform({.translate = glm::dvec3(1.0, 2.0, 0.0), .rotate = glm::dvec3(0.0, 0.0, 90.0)});
Camera::Define(scene, "/World/Camera")
    .SetTransform(LookAtTransform({0.0, -8.0, 3.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}));
```

See [Transforms](../concepts/transforms.md) for frames and the floating origin.

## Looking up existing prims

`ernie.X.at(scene, path)` gives the prim at `path` as an `X`, and `ernie.X.at(prim, relative)` one below a prim. It raises `LookupError` if there is no prim there and `TypeError` if the prim is not an `X` or a type derived from it, so it also checks what you expect:

```python
scene = ernie.Scene()
scene.define("/World/Lamp", ernie.SphereLight, radius=0.1)
rover = scene.define("/World/Rover", ernie.Xform)
rover.define("Mast/Camera", ernie.Camera)

ernie.SphereLight.at(scene, "/World/Lamp").radius   # 0.1
ernie.Light.at(scene, "/World/Lamp")                # Light('/World/Lamp'): any light will do
ernie.Camera.at(rover, "Mast/Camera")
ernie.Sphere.at(scene, "/World/Lamp")   # TypeError: '/World/Lamp' is a SphereLight, not a Sphere
```

`ernie.typed(prim)` gives a `Prim` as its own most derived class, or the `Prim` itself when its type has no class (untyped prims):

```python
ernie.typed(scene.get_prim("/World/Lamp"))      # SphereLight('/World/Lamp')
[ernie.typed(scene.get_prim(p)) for p in scene.prims("SphereLight")]
```

In C++, construct the view on the path: `SphereLight lamp(scene, "/World/Lamp")` throws `std::invalid_argument` if there is no prim there or it is not a `SphereLight`. `Light light(scene, path)` takes any light.

## Relationships

An attribute that names other prims is a relationship: a Gprim's `material_binding`, a RenderSettings' `camera`, `products` and `lights`, a PointInstancer's `prototypes`, a RenderProduct's `ordered_vars` and `filters`. Set one with a path (a string or `ernie.Path`), a `Prim` or a typed prim; a list relationship also takes a list of them, and one target given alone is a list of one. A single relationship reads as a `Path` (or `None`), a list as a list of `Path`s.

```python
scene = ernie.Scene()
basalt = scene.define("/Looks/Basalt", ernie.Material, brdf="hapke", albedo=(0.12, 0.12, 0.12))
rock = scene.define("/World/Rock", ernie.Sphere, material_binding=basalt)
rock.material_binding             # Path('/Looks/Basalt')

sun = scene.define("/World/Sun", ernie.DistantLight)
camera = scene.define("/World/Camera", ernie.Camera)
product = scene.define("/Render/Product", ernie.RenderProduct, camera=camera)
settings = scene.define("/Render/Settings", ernie.RenderSettings, camera="/World/Camera",
                        products=[product], lights=[sun])
settings.camera                   # Path('/World/Camera')
settings.cameras                  # [Path('/World/Camera')]
product.cameras = [camera, "/World/Camera2"]   # a batch, one view per camera
```

A render product's or settings' `camera` reads its first target (None when unset), and `cameras` reads them all; set either to one camera or a list.

In C++ a single relationship's setter takes a `Path` and a list's takes one target or a list (`.SetLights({Path("/World/Sun"), Path("/World/Lamp")})`); getters return a `Path` (empty when unset) or a `std::vector<Path>`.

## Arrays

Bulk data (a mesh's `points` and topology, a terrain's `heights`, a PointInstancer's `positions` and `orientations`, a material's maps, a star field's catalogue, a camera's batch `poses`) are array attributes, of kind `AttributeKind.TENSOR`. Set them with any array that speaks DLPack or the buffer protocol: NumPy, PyTorch, JAX, CuPy or Warp, on the CPU or a CUDA device. Each attribute's doc gives the dtype and shape it expects.

```python
import numpy as np

scene = ernie.Scene()
points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
mesh = scene.define("/World/Tri", ernie.Mesh, points=points,
                    face_vertex_counts=np.array([3], np.int32),
                    face_vertex_indices=np.array([0, 1, 2], np.int32))
mesh.points                       # the (3, 3) float32 array
np.shares_memory(mesh.points, points)    # True: written without a copy
```

Writing an array keeps it, sharing its memory, so writing never copies. Changing the array in place afterwards is not seen, though: write it again, or map it, which tells the renderer it changed:

```python
with scene.map(mesh.path, tokens.POINTS) as p:
    p[:, 2] += 1.0                # written in place, seen after the block
```

Simulations that update many prims each step from one GPU array (`scene.write` of many paths, `ernie.PrimSet`, CUDA streams) are covered in [GPU interop](gpu-interop.md). An attribute named on its class (`ernie.PointInstancer.positions`) is its USD name, so it can be passed wherever an attribute name goes, and a class's `write_rows` writes row i of each array, by Python name, to prim i, with `translate`, `orient` and `scale` for the transform ops:

```python
swarms = ernie.PrimSet([f"/World/Swarm{i}" for i in range(64)])
scene.write(swarms, ernie.PointInstancer.positions, positions)    # (64, n, 3)
ernie.PointInstancer.write_rows(scene, swarms, positions=positions, orientations=orientations,
                                translate=centres)               # (64, 3)
```

A write to many prims checks the attribute once for each prim type among them, before writing any row. In C++ an array attribute's setter takes a `Tensor` (`Tensor::Copy<float>(values, {3, 3})`, or one from DLPack), and its getter returns the written `const Tensor*`, null when unset; arrays have no default. An attribute that takes an array in place of its value has both: `mesh.GetDisplayColor()` for the one colour, and `mesh.ReadDisplayColor()` and `mesh.WriteDisplayColor(tensor)` for colours per vertex.

## Validation

Every attribute you set on a prim of a registered type is checked against its declaration, whether by a property, a keyword, `prim.set`, `scene.set`, a loader, or C++'s `Scene::Set` and `Scene::Write`. A misspelt name is an error, with the closest declared names suggested; so is a value that cannot be the declared type.

```python
lamp.radus = 1.0
# AttributeError: SphereLight has no attribute 'radus'; did you mean 'radius'?
scene.define("/World/Other", ernie.SphereLight, radus=1.0)
# TypeError: SphereLight has no attribute 'radus'; did you mean 'radius'?
lamp.set("inputs:radus", 1.0)
# ValueError: '/World/Lamp' is a SphereLight, which has no attribute 'inputs:radus'; did you mean 'inputs:radius'?
lamp.radius = "big"
# ValueError: 'inputs:radius' must be a number
```

C++ throws `std::invalid_argument` with the same messages, such as `'/World/Lamp' (SphereLight): 'inputs:radius' must be a number`. Values convert as their readers would read them: any number for a number, an int for a bool, a tuple or list for a vector, a string for a relationship.

An array written in place of a value (`scene.write`, `Scene::Write`) is taken only by array attributes and by those declared to take one (`AttributeDef.takes_array`, C++'s `AttributeMeta::array`): `display_color` with a colour per vertex, `widths` with one per point, `extent` as (2, 3), and transform ops with a pose per prim. Setting such an attribute to nested lists writes them as a float32 array:

```python
mesh.display_color = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]   # a (3, 3) float32 array
mesh.display_color.shape                                  # (3, 3)
```

Transform ops are declared too, for every prim, untyped ones included (C++'s `FindXformOp`): `xformOp:translate` and the Euler rotations take 3 numbers, `xformOp:scale` 3 or one, `xformOp:rotateX` (Y, Z) one angle, `xformOp:orient` a `Quat` (four plain numbers are refused, since w may come first or last) and `xformOp:transform` a 4 x 4 matrix, suffixed ops (`xformOp:orient:pivot`) as their op. A keyword `define` defines the prim before it sets the attributes, so after a failed one the prim is there, with the attributes set before the bad one.

Some names stay open, for data Ernie does not declare:

| Allowed on any typed prim | Example |
| --- | --- |
| Attributes in a namespace the type does not declare attributes in: your own data | `user:batteryLevel`, `sim:mass` |
| Primvars | `primvars:temperature` |
| Transform ops (checked by their kind, above) and their order | `xformOp:translate:pivot`, `xformOpOrder` |
| UsdShade connections | `outputs:surface.connect` |
| Any name in the type's open namespaces | a Material's or Shader's `inputs:*` and `outputs:*`, a SensorFilter's `inputs:*` |
| Any name at all on a grouping prim | `Xform` and `Scope` |

What is caught is a name without a namespace (`radus`), and a name in a namespace the type uses but does not open (`inputs:radus` on a light, `ernie:sensor:noize` on a camera). So put your own data in a namespace of your own, not in `inputs:` or `ernie:`. Untyped prims, and prims of types that are not registered, take anything but transform op values of the wrong kind.

## Loading scenes from a dict or JSON

`ernie.load_scene(description, scene=None)` builds prims from a description: a dict, JSON text, or the path of a `.json` file. It adds them to `scene`, or to a new Scene, and returns the scene. Its C++ counterpart is `LoadScene(scene, json)`, with `LoadSceneFile(scene, path)` for a file, in `ernie/scene/scene_json.h`; it reads the same format and throws `std::invalid_argument` naming the prim and key (or the line and column of malformed JSON).

```cpp
#include <ernie/scene/scene_json.h>

ernie::Scene scene;
ernie::LoadSceneFile(scene, "moon_base.json");
ernie::LoadScene(scene, R"({"/World/Rock": {"type": "Sphere", "radius": 0.5}})");
```

```python
scene = ernie.load_scene({
    "/World": {
        "type": "Xform",
        "Ground": {"type": "GroundPlane", "checker": True},
        "Rock": {"type": "Sphere", "radius": 0.5, "translate": [0, 0, 0.5],
                 "material_binding": "/Looks/Basalt"},
        "Sun": {"type": "DistantLight", "inputs:angle": 0.53,
                "photometric_illuminance": 128000.0, "rotate": [-40, 0, 30]},
        "Mast": {"type": "Xform", "translate": [2, 0, 1], "orient": [0.7071068, 0, 0, 0.7071068]},
        "Lander": {"type": "Mesh",
                   "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                   "face_vertex_counts": [3], "face_vertex_indices": [0, 1, 2]},
        "Camera": {"type": "Camera", "focal_length": 35.0,
                   "transform": [[1, 0, 0, 0], [0, 1, 0, -8], [0, 0, 1, 3], [0, 0, 0, 1]]},
    },
    "/Looks/Basalt": {"type": "Material", "brdf": "hapke", "albedo": [0.12, 0.12, 0.12]},
    "/Render/Settings": {"type": "RenderSettings", "camera": "/World/Camera",
                         "resolution": [640, 480]},
})
```

The format:

| Key or value | Meaning |
| --- | --- |
| Top-level keys | Prim paths. A key without a leading `/` is taken from the root (`"World"` is `/World`). |
| `"type"` | The prim's type name, as in `ernie.prim_types()`. Leave it out for an untyped prim, or to add attributes to a prim that exists (as when loading into a scene). |
| A dict value | A child prim, by name, described the same way. |
| Any other value | An attribute, by Python name (`"radius"`) or USD name (`"inputs:angle"`). USD names reach attributes Ernie does not declare too, within the rules of [Validation](#validation). |
| `"translate"`, `"rotate"`, `"rotation_order"`, `"scale"` | Transform ops, as the keywords: 3 numbers, Euler angles in degrees, an order such as `"ZYX"`, one number or 3. |
| `"orient"` | A quaternion as `[w, x, y, z]`, the real part first as in `ernie.Quat(w, x, y, z)` (in a Python dict, a `Quat` too). |
| `"transform"` | A 4 x 4 matrix as rows of four (column vectors, translation in the last column). |
| Lists for array attributes | Become arrays: int32 for attributes of whole numbers (`face_vertex_counts`, `face_vertex_indices`, `proto_indices`; `AttributeDef.is_integer`), float32 for the rest (`points`, `heights`), whatever the numbers look like. |
| Nested lists for an attribute that takes an array | Become a float32 array: a `display_color` per vertex as rows of 3. |
| Strings for relationships | Target paths; lists of them for list relationships. |

Each value is read as its attribute's declared kind (a transform op's too, `xformOp:orient:pivot` as a quaternion), not by how it looks, and checked as setting it is, so a misspelt name or a wrong value raises as above. Attributes may be given by Python name or USD name alike. Only undeclared attributes are read by their look: 2 to 4 numbers as a vector, other lists as float arrays, nested lists as a float32 array. Python's `load_scene` hands the description to C++'s `LoadScene` (a dict written as JSON first, its `Quat`s, `Path`s and prims included), so the two read every description the same way. Arrays in a dict (NumPy, or any DLPack array, CUDA ones included) go over beside the JSON rather than through it, each as `{"$array": i}` in the text and the array itself in `LoadScene`'s `arrays`: where an array goes (`points`, `heights`, per-vertex `display_color`) it is written as it is, shared and not copied, its dtype kept, as `scene.write` writes it; elsewhere (`translate`, a radius) it is read as the numbers it holds. So a dict of large arrays loads without the arrays being turned into lists.

## The C++ API

Each concrete type's view class has a static `Define(scene, path)` that defines the prim and returns its view, and a constructor `X(scene, path)` that views an existing one (and checks its type). A view is a cheap handle: a scene pointer, the path and the type, and copies name the same prim. Setters return the view, so they chain:

```cpp
#include <ernie/scene/geometry_schemas.h>
#include <ernie/scene/light_schemas.h>
#include <ernie/scene/material_schemas.h>

using namespace ernie;
Scene scene;
Material basalt = Material::Define(scene, "/Looks/Basalt")
                      .SetBrdf("hapke")
                      .SetAlbedo({0.12, 0.12, 0.12});
Sphere rock = Sphere::Define(scene, "/World/Rock")
                  .SetRadius(0.5)
                  .SetMaterialBinding(basalt.GetPath())
                  .SetXform({.translate = glm::dvec3(0.0, 0.0, 0.5)})
                  .Set("user:mass", Value(120.0));  // any attribute by USD name
```

Per attribute a view has `Get<Name>`, `Has<Name>`, `Set<Name>` and `Clear<Name>`. Per prim it has `SetTransform`, `SetXform`, `GetLocalTransform`, `Set(name, Value)` and `Write(name, Tensor)` for attributes the declarations do not cover (both validated), `GetPath`, `GetScene`, `GetPrim` (the `Prim`, for `Define` of children and raw `Get`), and `PrimType()` (its `PrimTypeDef`, which may be a type derived from the view's).

A view made from a `const Scene&` reads only: its getters work, and its setters, `GetPrim` and the raw `Set` throw `std::logic_error`. Pass `const Scene&` to code that should only read the scene:

```cpp
double Brightness(const Scene& scene) {
    return SphereLight(scene, "/World/Lamp").GetIntensity();
}
```

The views read and write the scene's attributes by USD name, through `Scene::Set`, `Scene::Write` and `Scene::Get`, so they change nothing about storage or change tracking: code using `scene.Set(path, tokens::kRadius, Value(0.1))` and code using `SphereLight(scene, path).SetRadius(0.1)` see the same scene.

## When to use raw tokens

The typed attributes cover what Ernie declares. Set by USD name, with `prim.set(token, value)` in Python or `Set(token, Value)` on a C++ view, for:

- transform ops directly: `tokens.TRANSFORM`, `tokens.XFORM_OP_ORDER`, `xformOp:orient` arrays written per step;
- primvars, UsdShade inputs and connections (`inputs:*` on a Shader, `outputs:surface.connect` on a Material) and filter inputs on a SensorFilter, which depend on the shader or filter type;
- your own data in your own namespace (`sim:mass`);
- untyped prims, which have no class.

`ernie.tokens` holds the names Ernie reads (`tokens.TRANSFORM`, `tokens.POINTS`, `tokens.MATERIAL_BINDING`), as `ernie::tokens` does in C++ (`tokens::kTransform`), and `ernie.X.attr.name` gives an attribute's USD name. A typed instance's `set(token, value)` returns the instance, so it chains with keyword sets: `lamp.set("user:id", 7).set(intensity=2.0)`.

## Declaring a new prim type

A prim type is declared in C++, and everything else follows from that declaration. Say you want a fiducial marker for a pose-estimation pipeline.

**1. List its attributes** in a header, as an X-macro: one `X(C++ type, Name, USD name, default, (AttributeMeta{...}))` entry per attribute. The C++ types are `bool`, `int`, `double`, `glm::dvec2`, `glm::dvec3`, `glm::dvec4`, `std::string` (a token or file path), `ernie::Path` (a relationship), `std::vector<ernie::Path>` (a list of targets), `std::vector<float>` (a small array such as (nm, value) pairs) and `ernie::Tensor` (an array, default `nullptr`). Wrap a type as `ernie::OrArray<T>` for an attribute that also takes an array written in place of its value, as `display_color` does with a colour per vertex: its view then has `Read<Name>()` (the array, or null) and `Write<Name>(tensor)` beside `Get<Name>()`, loaders read nested lists for it as such an array, and Python's `AttributeDef.takes_array` is true. Then a second macro with the base's attributes first, for the accessors:

```cpp
// ernie/scene/fiducial_schemas.h
#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"

// clang-format off
#define ERNIE_FIDUCIAL_ATTRIBUTES(X)                                                              \
    X(std::string, Family, "ernie:fiducial:family", "tag36h11",                                    \
      (::ernie::AttributeMeta{.doc = "The tag family.", .group = "Fiducial",                       \
                              .options = {"tag36h11", "tag25h9", "tag16h5"}}))                     \
    X(int, TagId, "ernie:fiducial:id", 0,                                                          \
      (::ernie::AttributeMeta{.label = "ID", .doc = "The tag's code in its family.",              \
                              .group = "Fiducial", .min = 0.0, .max = 586.0}))                     \
    X(double, Size, "ernie:fiducial:size", 0.16,                                                   \
      (::ernie::AttributeMeta{.doc = "Edge of the black square, metres.", .group = "Fiducial",    \
                              .min = 0.0, .max = 10.0}))

#define ERNIE_FIDUCIAL_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_FIDUCIAL_ATTRIBUTES(X)
// clang-format on
```

`AttributeMeta`'s fields, in the order designated initializers need: `label` (for panels; derived from the name if empty, "Tag id"), `doc`, `group` (the panel section), `min` and `max` (edit limits), `options` (a token's choices, or a relationship's target type), `color` (a vector that is linear RGB), `asset` (a string that is a file path), `lenient` (accept values of other types, for legacy forms) and `dirty` (what a change stamps, below).

**2. Declare the view class** with `ERNIE_SCHEMA_BODY(Class, Base, ALL_ATTRIBUTES)`, which gives the constructors, `Definition()` and every accessor, and `ERNIE_SCHEMA_DEFINE(Class)` for `Define` (leave it out for an abstract type):

```cpp
namespace ernie {

/// A square fiducial marker (AprilTag) in its xy plane, facing +z.
class Fiducial : public Imageable {
    ERNIE_SCHEMA_BODY(Fiducial, Imageable, ERNIE_FIDUCIAL_ALL)
    ERNIE_SCHEMA_DEFINE(Fiducial)
};

std::vector<const PrimTypeDef*> FiducialSchemaTypes();

}  // namespace ernie
```

**3. Define its registry entry** in the source with `ERNIE_DEFINE_SCHEMA(Class, type name, base definition, abstract, doc, OWN_ATTRIBUTES)`. It takes the type's own attributes only; the bases' come from the base:

```cpp
// ernie/scene/fiducial_schemas.cpp
#include "ernie/scene/fiducial_schemas.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(Fiducial, "Fiducial", &Imageable::Definition(), false,
                    "A square fiducial marker (AprilTag) in its xy plane, facing +z.",
                    ERNIE_FIDUCIAL_ATTRIBUTES)

std::vector<const PrimTypeDef*> FiducialSchemaTypes() {
    return {&Fiducial::Definition()};
}

}  // namespace ernie
```

For a type whose attributes in some namespaces depend on data (a Shader's `inputs:*` depend on its `info:id`), use `ERNIE_DEFINE_OPEN_SCHEMA` and list those namespaces after the attributes: `ERNIE_DEFINE_OPEN_SCHEMA(Shader, ..., ERNIE_SHADER_ATTRIBUTES, "inputs", "outputs")`. `"*"` opens every name, as `Xform` and `Scope` do.

**4. Register it.** Each family's `*SchemaTypes()` returns its types' definitions, bases before the types derived from them; add the type to an existing family's list, or add a new family's function to the list `PrimTypes()` builds in `ernie/scene/prim_type.cpp`, and the source to `ernie/scene/CMakeLists.txt`.

That is all. With it registered you get:

- the C++ view, `Fiducial::Define(scene, path).SetTagId(12).SetSize(0.2)`, with `Get`, `Has` and `Clear` for each attribute;
- the Python class `ernie.Fiducial`, built from the registry at import, with properties `family`, `tag_id` and `size` (the snake_case of the C++ names), keyword attributes on `define`, `.at` and `typed`;
- the Python stubs (`schema.pyi`, written at build by `ernie/python/stub_schema.py`), so IDEs complete and type-check the new attributes, and the API docs list them;
- property panels in the viewer: its attributes in their groups, labelled, with sliders between `min` and `max`, menus of `options` and colour pickers;
- validation of every attribute set on a `Fiducial`, with suggestions for misspellings;
- `load_scene` and `LoadScene` reading `{"type": "Fiducial", "tag_id": 12}`;
- change tracking: a change to an attribute stamps the dirty bit its declaration's `dirty` names (`Dirty::Points` for bulk positions, `Dirty::Topology` for connectivity, `Dirty::Material`, `Dirty::Visibility`), else `Dirty::Params`, so render delegates know what to rebuild.

What the declaration does not give is drawing: a render delegate reads the new type's attributes to draw it (see [Custom render delegates](../extending/custom-render-delegates.md)). A Python name may not hide a `Prim` member (`path`, `type`, `get`, `set`, `define`, ...); importing `ernie` raises if one does. And a USD name means the same to every type: `DirtyOf` keys its bits on the name alone.
