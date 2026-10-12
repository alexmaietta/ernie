# Volumes

Volumes are participating media such as engine plumes, smoke and dust, after OpenUSD's UsdVol schemas. A `Volume` prim names its fields with relationships `field:<name>`, each to a field prim. An `OpenVDBAsset` field reads one grid of a NanoVDB file. The Volume's bound Material says how the fields shade.

```python
import ernie

scene = ernie.Scene()
plume = scene.define("/World/Plume", ernie.Volume, material_binding="/Looks/Exhaust")
plume.set_field("density", scene.define("/World/Plume/Density", ernie.OpenVDBAsset, file_path="plume.nvdb", field_name="density"))
plume.set_field("temperature", scene.define("/World/Plume/Temperature", ernie.OpenVDBAsset, file_path="plume.nvdb", field_name="temperature"))
scene.define("/Looks/Exhaust", ernie.Material, density=20.0)
```

## Grids

Ernie reads NanoVDB files (`.nvdb`), uncompressed. The GPU walks each grid's tree in place, so sparse grids stay sparse in memory. Convert OpenVDB files (`.vdb`) first, with OpenVDB's `nanovdb_convert` (without `-z` or `-b`) or with Warp (`wp.Volume.load_from_nvdb`, `save_to_nvdb`).

`fieldName` picks a grid of the file by name, and `fieldIndex` picks among grids of the same name. Without a name, `fieldIndex` counts every grid in the file. Every grid type NanoVDB stores as floats is read: float, half, Fp4, Fp8, Fp16 and FpN. A vec3f grid is read as the lengths of its vectors.

A field prim's transform places its grid in the Volume, and the Volume's transform places the Volume in the world. Values sit at the grid's integer index coordinates, and Ernie interpolates trilinearly between them.

## Shading

The Material's volume inputs, in its "Volume" group:

| Input | Meaning |
| --- | --- |
| `inputs:density` | Scales the density field into the extinction coefficient, in 1/m. |
| `inputs:densityField` | The field holding density, `density` by default. |
| `inputs:volumeAlbedo` | The share of extinction that scatters; the rest absorbs. |
| `inputs:phaseFunction` | How what scatters spreads its light: `henyeyGreenstein`, `doubleHenyeyGreenstein`, `rayleigh` or `isotropic`. |
| `inputs:anisotropy` | The Henyey-Greenstein g: above 0 forwards, below 0 backwards. |
| `inputs:b`, `inputs:c` | The double Henyey-Greenstein lobe width and backscatter weight, the same inputs the `hapke` BRDF reads. |
| `inputs:emissionColor`, `inputs:emissionStrength` | Radiance added per metre, in cd/m² per m for white, times the emission field. |
| `inputs:emissionField` | The field emission follows; the density field when unset. |
| `inputs:temperature`, `inputs:temperatureField`, `inputs:temperatureScale` | Kelvin: of the whole volume, or a field of it times its scale. |
| `inputs:blackbody` | Scales the glow of what absorbs at its temperature: Planck's law per band times the absorption coefficient (Kirchhoff's law). 1 is physical. |

Extinction is the same in every band. Albedo, emission and black-body glow are spectral, binned to the camera's bands like everything else Ernie renders.

Volumes have no BRDF. The medium's counterpart is its single-scattering albedo and phase function, so Hapke's model carries over in its particle parameters: the albedo is Hapke's w, and the double Henyey-Greenstein phase reads the same `inputs:b` and `inputs:c` as a Hapke surface. One Material can then shade both the regolith (`ernie:brdf` `hapke`) and the dust lofted from it, the dust lacking only the packed surface's opposition effect and roughness.

```python
regolith = scene.define("/Looks/Regolith", ernie.Material, brdf="hapke", albedo=(0.25, 0.25, 0.25),
                        volume_albedo=(0.25, 0.25, 0.25), phase_function="doubleHenyeyGreenstein", density=5.0)
regolith.set("inputs:b", 0.25).set("inputs:c", 0.5)
```

## Rendering

After shading, each pixel's ray is marched through the volumes it crosses, nearest first, up to the surface behind or on to infinity. The radiance behind is dimmed by the volumes' transmittance, and what they emit along the way is added. Steps are half a voxel of the density grid, at most 256 per volume, with each step's emission integrated exactly for its coefficients.

Each grid is loaded once, however many Volumes name it, and loaded again when its file changes. In a batch of worlds, a Volume's `ernie:world` puts it in one world, and each view marches only the volumes of its own world and the shared ones that its frustum reaches.

Distant and sphere lights (spots included) light the medium: each step adds their light scattered once towards the eye, dimmed by the volume between the step and the light, a short march towards it. Other volumes, geometry and the atmosphere do not dim it yet.

Not yet: rect, disk, cylinder and laser lights in the medium, multiple scattering, shadows cast by volumes, and fields written from the GPU.
