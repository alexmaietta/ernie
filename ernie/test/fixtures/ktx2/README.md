KTX2 textures in Basis Universal's formats, for the transcoding tests (`util_texture_file_test.cpp`, `raster_textures_gpu_test.cpp`). Ernie links only Basis's transcoder, so these were encoded once with the `basisu` tool of Basis Universal `v2_50` (the version `cmake/third_party/basisu.cmake` pins) from two 16 x 16 PNGs made in code:

- `quadrants.png`: red and green on top, blue and white below, 8 x 8 each (the texture tests' 2 x 2 image, enlarged).
- `tilted_normal.png`: one normal tilted 30 degrees towards +x stored as n / 2 + 1/2: (191, 128, 238).

| File | Command | Holds |
| --- | --- | --- |
| `quadrants_etc1s.ktx2` | `basisu -ktx2 -etc1s -mipmap -linear quadrants.png` | ETC1S, BasisLZ supercompression, 5 levels |
| `quadrants_uastc_zstd.ktx2` | `basisu -ktx2 -uastc -mipmap -linear quadrants.png` | UASTC, zstd supercompression, 5 levels |
| `tilted_normal_etc1s.ktx2` | `basisu -ktx2 -etc1s -normal_map -separate_rg_to_color_alpha tilted_normal.png` | ETC1S, x in the colour slice and y in the alpha slice |
| `tilted_normal_uastc.ktx2` | `basisu -ktx2 -uastc -normal_map -separate_rg_to_color_alpha tilted_normal.png` | UASTC, x in rgb and y in alpha, zstd |

`basisu` labels the normal maps' channels as colour and alpha (ETC1S RGB and AAA, UASTC RGBA), so a reader cannot tell them from colour with alpha. Their data format descriptors' channel ids were then set to what KTX-Software's `toktx --normal_mode` writes for the same layout, ETC1S RRR and GGG and UASTC RRRG (the low four bits of each sample's channel type), by which Ernie transcodes them to BC5.
