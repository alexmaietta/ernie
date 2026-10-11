# Terrain and planets

## Colouring

A Terrain's `ernie:colorMode` (Python `color_mode`) sets how it is coloured:

| Mode | What it draws |
| --- | --- |
| `shaded` (default) | The bound material, lit by the scene's lights: the physical image, for sensors. |
| `colormap` | A height ramp from blue lows to white highs, unlit. |
| `shadedColormap` | The height ramp times the material, lit. |
| `hillshade` | A GIS hillshade: grey and unlit, the cosine of the angle between the terrain's normal and a fixed light at `ernie:hillshade:azimuth` (degrees clockwise from north, default 315) and `ernie:hillshade:altitude` (degrees above the horizon, default 45), whatever the scene's lights. |

The ramps and the hillshade are for showing heights; only `shaded` gives radiances a camera would see. On a planet (a Terrain with `ernie:planetRadius`), north is towards +z and the hillshade's light is in each point's own east, north and up.

```python
scene.define("/World/Terrain", ernie.Terrain, heights=heights, color_mode=tokens.HILLSHADE, hillshade_azimuth=315.0)
```
