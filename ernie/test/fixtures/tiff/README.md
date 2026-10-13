TIFFs in the encodings planetary and gigapixel data ship in, for `util_tiff_test.cpp`. Each is 37 x 23, made once by Pillow (uncompressed) and re-encoded by GDAL 3's `gdal_translate`, from values the test computes again at pixel (x, y):

- `rgb_lzw_predictor2.tif`: RGB8 (7x mod 256, 11y mod 256, 3(x + y) mod 256), LZW with horizontal prediction, strips of 8 rows (`-co COMPRESS=LZW -co PREDICTOR=2 -co BLOCKYSIZE=8`), as the LROC colour mosaics.
- `rgb_tiled_deflate.tif`: the same in 16 x 16 tiles, Deflate (`-co COMPRESS=DEFLATE -co TILED=YES -co BLOCKXSIZE=16 -co BLOCKYSIZE=16`).
- `float_raw.tif`: float32 0.5x - 1.25y + 0.125, uncompressed strips of one row (`-co BLOCKYSIZE=1`), as the LOLA elevation models.
- `float_deflate_predictor3.tif`: the same, Deflate with floating-point prediction, strips of 5 rows (`-co COMPRESS=DEFLATE -co PREDICTOR=3 -co BLOCKYSIZE=5`).
- `int16_bigtiff_bigendian.tif`: int16 100x - 300y, BigTIFF, big-endian, LZW with horizontal prediction, GeoTIFF's grid of the whole Earth and no-data -32768 (`-ot Int16 -co BIGTIFF=YES -co ENDIANNESS=BIG -co COMPRESS=LZW -co PREDICTOR=2 -a_nodata -32768 -a_srs EPSG:4326 -a_ullr -180 90 180 -90`).
