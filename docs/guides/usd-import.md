# Importing USD

Ernie imports OpenUSD files (`.usd`, `.usda`, `.usdc`, `.usdz`) into a scene with `Scene.import_usd` in Python and `ImportUsd` in C++. OpenUSD composes the stage first, with its sublayers, references, payloads, inherits, specializes and variants. Ernie then reads it through UsdImaging's Hydra scene indices, as Hydra renderers do: they resolve native and point instancing into instancers of shared prototypes, material bindings (collections, purposes, inheritance), inherited primvars and purposes. Each prim becomes an Ernie prim of the same name and type. What you get is an ordinary subtree of Ernie prims: typed views, the viewer's panels, batched writes and `SaveScene` all work on it as on prims you define yourself.

## Building with USD

OpenUSD lives only in a plugin, `libernie_usd`, which Ernie loads the first time you import. Ernie's library and Python module never link OpenUSD, so a build without the plugin runs unchanged and only the import raises.

```sh
cmake -S . -B build -DERNIE_USD=ON
cmake --build build --target ernie_usd
```

By default the plugin's first build builds OpenUSD 26.08 from source with OpenUSD's own `build_usd.py`, which also builds the oneTBB, OpenSubdiv and MaterialX it needs. This happens once, takes about half an hour, and works on Linux and macOS. The result is one library with UsdImaging's scene indices but without OpenGL, Metal, Storm, Python or usdview, installed beside the fetched dependencies (`FETCHCONTENT_BASE_DIR`, `build/_deps` by default) in a directory named for the version, the options and the compiler. Configuring and building other targets never wait for it. CI builds it in a job of its own and caches it. The plugin finds it through its rpath.

A process must load only one OpenUSD. When Ernie runs inside an application that already has one, such as Isaac Sim, build the plugin against that application's OpenUSD instead, with `-DERNIE_USD_ROOT=<its OpenUSD>` and `-DERNIE_USD_TBB_ROOT=<its oneTBB>`. If the plugin finds another OpenUSD already loaded, the import fails with a message saying so rather than mixing the two.

`ernie.usd_import_available()` returns whether the plugin loads, and why not when it does not. The environment variable `ERNIE_USD_PLUGIN` names a plugin to load in place of the one beside Ernie's library or Python module.

## Importing a file

```python
import ernie

scene = ernie.Scene()
scene.define("/World", ernie.Xform)
rover = scene.import_usd("/World/lrv", "lrv/lrv.usda", variants={"/lrv": {"Mission": "a15"}})
print(rover)  # UsdImport(/World/lrv, 4316 prims, 0 warnings)
```

The same in C++:

```cpp
#include <ernie/scene/usd_import.h>

ernie::UsdImportOptions options;
options.variants.push_back({"/lrv", "Mission", "a15"});
const ernie::UsdImportResult rover =
    ernie::ImportUsd(scene, ernie::Path("/World/lrv"), "lrv/lrv.usda", options);
```

The file's default prim becomes the prim you name, as a USD reference to the file would place it, with its type and its own transform. A file without a default prim gets an `Xform` at that path, with the file's root prims under it. USD prims outside the default prim that the file uses, such as materials bound from elsewhere in the file, go under `_external` beneath the root.

The root's transform also converts the file's units into the scene's. A file's `upAxis` and `metersPerUnit` (0.01, centimetres, when it authors none, as OpenUSD assumes) become the scene's `upAxis` and `metersPerUnit` (Z and metres by default). Pixar's Kitchen Set, in centimetres, comes in in metres. Place an asset with a parent `Xform` of your own rather than by overwriting the root's transform.

## Options

| Python | C++ | What it does |
|---|---|---|
| `variants={prim: {set: selection}}` | `options.variants` | Variant selections made on the session layer before composing, in the given order, so a selection may pick a nested variant set that an earlier one brought in. |
| `time=t` | `options.time` | The time code to read. Without one, each attribute's earliest value is read, which is its default value when it has no time samples. |
| `proxy=True`, `guide=True` | `options.proxy`, `options.guide` | Also translate prims of purpose `proxy` and `guide`. Prims of purpose `default` and `render` are always translated. |

To pick other variants, remove the subtree and import again. Prim paths stay the same, so the renderer keeps the textures that did not change.

```python
scene.remove("/World/lrv")
scene.import_usd("/World/lrv", "lrv/lrv.usda", variants={"/lrv": {"Mission": "a17"}})
```

## What comes back

`import_usd` returns a `UsdImport`: `root`, the prim the file became; `prims`, the Ernie path of each translated USD prim by its USD path; and `warnings`, what the importer skipped or approximated. Use `prims` to find parts by the names you know from the file:

```python
steer = "/lrv/Geometry/chassis/chassis_fwd/rf_lower_arm/rf_upright/rf_steer"
scene.get_prim(rover.prims[steer]).set_xform(rotate=(0.0, 0.0, 15.0))
```

## What is translated

- **Transforms and hierarchy.** `Xform` and `Scope` keep their names and nesting. Each prim's local transform becomes its `xformOp:transform`. A prim that resets the transform stack keeps its world transform, made relative to its parent.
- **Meshes.** Points, faces and texture coordinates (any float2 primvar with the texture coordinate role, or named `st`, `uv` and the like), indexed or not, at any interpolation. Normals that differ per face corner (faceVarying or uniform) split the points they share, so hard edges stay hard. Left-handed faces are reversed. Faces in `GeomSubset`s bound to materials of their own become child meshes named after the subsets.
- **Subdivision surfaces.** A mesh's `subdivisionScheme` comes through as composed, so a mesh that authors none gets USD's fallback, Catmull-Clark, as most assets expect: Pixar's Kitchen Set authors none. So do its `interpolateBoundary`, `faceVaryingLinearInterpolation`, `triangleSubdivisionRule`, creases, corners and holes. Authored normals are dropped, as USD ignores them on subdivision surfaces. Ernie draws the limit surface, refined by OpenSubdiv (see below). Material subsets of a subdivision surface are refined as separate meshes, so their borders become boundaries.
- **Shapes.** `Cube`, `Sphere`, `Cylinder`, `Cone`, `Capsule` and `Plane`, with their sizes and axes and a constant `displayColor`.
- **Points and instancers.** `Points` with their widths. `PointInstancer` with positions, orientations, scales, prototype indices and its prototypes, which Ernie draws only through the instancer.
- **Instancing.** Native instances (instanceable prims) and point instancers become `PointInstancer`s of shared prototypes, which Ernie draws as instances. The prototypes are translated once, under the root's `_external`. The instanced Kitchen Set imports far fewer meshes than its plain version.
- **Cameras and lights.** `Camera` with its projection, focal length, apertures, clipping range, f-stop and focus distance. The UsdLux lights `DistantLight`, `DomeLight`, `SphereLight`, `RectLight`, `DiskLight` and `CylinderLight` with their `inputs:*`.
- **Materials.** The material each gprim resolves to (direct and collection bindings, inherited from ancestors, the full purpose first) is bound as `material:binding`. A UsdPreviewSurface network in the universal render context is copied from Hydra's material network as it is, with its `UsdUVTexture`, `UsdPrimvarReader` and `UsdTransform2d` shaders, their inputs and their connections. Texture paths are resolved.
- **MDL, approximated.** Ernie does not compile MDL yet. A material with only an MDL surface becomes a UsdPreviewSurface from the MDL shader's main inputs: OmniPBR's base colour and texture, roughness, metallic, normal map, emission and opacity, and OmniSurface's base colour, roughness, metalness and emission. Other MDL materials keep their base colour, with a warning.

Prims of other types become `Xform`s, with one warning per type, so what is below them still comes in. Physics prims, render settings and other prims without a transform are skipped.

## Subdivision surfaces

Any Ernie `Mesh` with a `subdivisionScheme` other than `none` (Ernie's own default) draws its limit surface rather than its faces. The scheme can be `catmullClark`, `loop` (triangles only) or `bilinear`. OpenSubdiv refines it into `2^level` segments along each control edge. The surface's own normals shade it, edges and vertices shared between faces stay shared, and face-varying texture coordinates follow `faceVaryingLinearInterpolation`.

The level is the mesh's `ernie:refineLevel`. At -1, its default, the stage's `ernie:subdivision:refineLevel` applies (1 by default; usdview's complexity Low to Very High is 0 to 3):

```python
ernie.Stage.at(scene, "/").subdivision_refine_level = 2
ernie.Mesh.at(scene, mesh_path).refine_level = 0  # this mesh only
```

Each level roughly quadruples a mesh's triangles. The Kitchen Set's 276,000 control faces become about 2.2 million triangles at level 1 and 8.8 million at level 2. A refined mesh's coarser levels come with it as its levels of detail, so distant ones draw fewer triangles without simplifying the refined mesh again.

## Skeletal animation

UsdSkel characters come in as Ernie's own skeletal prims, read through OpenUSD's UsdSkel API, which resolves each mesh's binding, joint order and blend shapes:

- a `Skeleton` with its joints, bind transforms and rest transforms;
- a `SkelAnimation` with its joints' translations, rotations and scales and its blend shape weights, sampled at each of the file's time codes (`ernie:times`);
- a `BlendShape` under each mesh for each of its shapes, inbetweens included;
- on each skinned `Mesh`, its rest points and `skel:skeleton`, `skel:joints`, `primvars:skel:jointIndices` and `jointWeights` (normalized), `primvars:skel:geomBindTransform`, `skel:skinningMethod`, `skel:blendShapes` and `skel:blendShapeTargets`.

The file's `startTimeCode`, `endTimeCode`, `timeCodesPerSecond` and `framesPerSecond` become the stage's when it has no range of its own, so the viewer's timeline plays the animation (`UsdImport.time_code_range` has them either way).

Ernie poses skinned meshes on the GPU at the stage's `ernie:timeCode`: the CPU samples each skeleton's animation once a frame, and a compute pass adds the weighted blend shapes and blends the joints' transforms, linearly (`classicLinear`) or as dual quaternions (`dualQuaternion`), as UsdSkel defines them. A skinned subdivision surface is posed at its control points, then refined by weights OpenSubdiv gives once, so it stays the limit surface of its posed cage. Normals follow from the posed triangles. `ernie::SkinnedPoints` (`ernie/session/skinning.h`) does the same on the CPU.

## Time samples

Animated transforms (a character's root motion, a moving part), visibility and point caches (a mesh's sampled `points`) come in as the attributes' time samples in Ernie's Scene, which holds each attribute's value at the stage's time code:

```python
scene.set("/World/Arm", ernie.tokens.TRANSFORM, matrix, time=24.0)  # a sample
scene.sample_times("/World/Arm", ernie.tokens.TRANSFORM)             # [..., 24.0]
ernie.Stage.at(scene, "/").time_code = 12.0  # every sampled attribute at 12
```

Between samples, numbers, vectors and matrices interpolate linearly (matrices by element, as USD does), quaternions spherically, and tokens, ints and paths hold; beyond the first and last sample the end value holds. Setting an attribute without a time clears its samples. Arrays blend lazily: a CPU array is blended only when read, and the renderer blends a mesh's point cache on the GPU from samples it uploads once, refining a subdivision surface's blended control points there too, so playing it back costs the CPU nothing per frame. Samples save with the scene (`$timeSamples` in scene JSON).

## Not yet

- Time samples of attributes other than transforms, visibility, points and skeletal animation (a light's intensity, a camera's focal length) are read at one time.
- Blend shapes' normal offsets: a skinned mesh's normals are recomputed from its posed triangles.
- Subdivision surfaces refine to one level everywhere: no refinement by distance or screen size yet. Their points written on the GPU draw the control mesh.
- `BasisCurves`, `NurbsPatch` and volumes.
- Textures inside `.usdz` archives: the stage loads, but Ernie cannot read the images yet (a warning says so).
- `invisibleIds` on point instancers.
- Windows: the plugin builds on Linux for now.

## Demos

`demo_usd_kitchen` imports Pixar's Kitchen Set, plain or with native instancing, and shows it from inside and from overhead with the ceiling cut away. The set (2.7 MB) is fetched once from openusd.org into Ernie's cache (`$ERNIE_CACHE`, else `~/.cache/ernie`).

`demo_usd_med_city` imports Pixar's Point Instanced Med City: one `PointInstancer` placing 40,000 buildings of eight prototypes over hills about 2 m across (the asset is in centimetres), under a sun you can move. The asset (0.5 MB) is fetched and cached the same way.

`demo_usd_human_female` imports Pixar's HumanFemale from the UsdSkelExamples and plays it on the timeline: walking or idling, at the asset's three levels of detail, skinned linearly or with dual quaternions. The examples (38 MB) are fetched and cached the same way.
