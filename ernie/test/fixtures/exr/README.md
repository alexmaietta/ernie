EXRs for `image_test.cpp`, written by `make_exr.py` from the file layout alone (no OpenEXR library), with values the test computes again at window-relative pixel (x, y):

- `rgba_half_window.exr`: RGBA half, uncompressed, 5 x 3 data window at (2, 1) in an 8 x 5 display window; R 0.25x, G 2^y, B -0.5, A 0.5.
- `rgb_float_zip.exr`: RGB float, ZIP (16 lines a block), 16 x 16, no alpha; R x/16, G y/16, B 100 + x.
- `y_half_zips.exr`: luminance half, ZIPS (one line a block, stored raw where zlib does not shrink it), 4 x 2; Y 0.5x + y.
