# SPDX-License-Identifier: BSD-3-Clause
"""Writes ernie/util/reflectance_table.h, the measured reflectance
spectra Ernie embeds (`ReflectanceLibrary`), from the ECOSTRESS
Spectral Library (NASA JPL; Meerdink et al. 2019, Baldridge et al.
2009), whose files give directional hemispherical reflectance in
percent against wavelength in micrometres.

Download each spectrum below into one directory, then run:

    for f in $(python tools/reflectance_library.py --files); do
        curl -o "spectra/$f" \\
            "https://speclib.jpl.nasa.gov/ecospeclibdata/$f"
    done
    python tools/reflectance_library.py spectra \\
        ernie/util/reflectance_table.h

Each spectrum keeps 10 nm bins from 350 to 2500 nm, then bins 1%
wide in wavelength to its end (12.5 to 15 um for most, whose thermal
infrared the emissivity of the long-wave infrared will need), each
its samples' mean at the bin's centre, as a fraction.

References:
- S. K. Meerdink, S. J. Hook, D. A. Roberts and E. A. Abbott, "The
  ECOSTRESS spectral library version 1.0", Remote Sensing of
  Environment 230:111196, 2019. https://doi.org/10.1016/j.rse.2019.05.015
- A. M. Baldridge, S. J. Hook, C. I. Grove and G. Rivera, "The ASTER
  spectral library version 2.0", Remote Sensing of Environment
  113(4):711-715, 2009. https://doi.org/10.1016/j.rse.2008.11.007
"""

import os
import sys

# (id, name, ECOSTRESS file).
SPECTRA = [
    (
        "grass",
        "Green rye grass",
        "vegetation.grass.unknown.unknown.all.grass.jhu.becknic.spectrum.txt",
    ),
    (
        "chaparral",
        "Chamise (Adenostoma fasciculatum) leaves",
        "vegetation.shrub.adenostoma.fasciculatum.vswir.vh033.ucsb.asd.spectrum.txt",
    ),
    (
        "dry_grass",
        "Wild oat (Avena fatua) litter",
        "nonphotosyntheticvegetation.leaves.avena.fatua.vswir.vh354.ucsb.asd.spectrum.txt",
    ),
    (
        "loam",
        "Brown to dark brown sandy loam",
        "soil.mollisol.hapludoll.none.all.87p764.jhu.becknic.spectrum.txt",
    ),
    (
        "sand",
        "Brown to dark brown sand",
        "soil.entisol.quartzipsamment.none.all.87p706.jhu.becknic.spectrum.txt",
    ),
    (
        "concrete",
        "Weathered construction concrete",
        "manmade.concrete.constructionconcrete.solid.all.0598uuucnc.jhu.becknic.spectrum.txt",
    ),
    (
        "asphalt",
        "Weathered construction asphalt",
        "manmade.road.pavingasphalt.solid.all.0674uuuasp.jhu.becknic.spectrum.txt",
    ),
    (
        "brick",
        "Weathered red brick",
        "manmade.generalconstructionmaterial.brick.solid.all.0412uuubrk.jhu.becknic.spectrum.txt",
    ),
    (
        "pine_wood",
        "Weathered pine lumber",
        "manmade.generalconstructionmaterial.wood.solid.all.0404uuuwod.jhu.becknic.spectrum.txt",
    ),
    (
        "black_paint",
        "Black alkyd gloss paint, two coats on aluminium",
        "manmade.generalconstructionmaterial.paint.solid.all.0403uuupnt.jhu.becknic.spectrum.txt",
    ),
    (
        "olive_paint",
        "Olive green alkyd gloss paint, two coats on aluminium",
        "manmade.generalconstructionmaterial.paint.solid.all.0386uuupnt.jhu.becknic.spectrum.txt",
    ),
    (
        "white_coating",
        "Weathered white rubberized roof coating",
        "manmade.roofingmaterial.rubber.solid.all.0795uuurbr.jhu.becknic.spectrum.txt",
    ),
    (
        "aluminium",
        "Weathered, scratched bare aluminium",
        "manmade.roofingmaterial.metal.solid.all.0384uuualm.jhu.becknic.spectrum.txt",
    ),
    (
        "galvanized_steel",
        "Weathered galvanized steel",
        "manmade.roofingmaterial.metal.solid.all.0525uuustla.jhu.becknic.spectrum.txt",
    ),
    (
        "copper",
        "Slightly weathered bare copper",
        "manmade.roofingmaterial.metal.solid.all.0681uuucop.jhu.becknic.spectrum.txt",
    ),
    (
        "fine_snow",
        "Fine snow (modelled to 2.08 um, measured beyond)",
        "water.snow.finegranular.fine.all.fine_snw_.jhu.becknic.spectrum.txt",
    ),
    (
        "granular_snow",
        "Medium granular snow (modelled to 2.08 um, measured beyond)",
        "water.snow.mediumgranular.medium.all.medgran_snw_.jhu.becknic.spectrum.txt",
    ),
    (
        "ice",
        "Ice (modelled to 2.08 um, measured beyond)",
        "water.ice.none.solid.all.ice_dat_.jhu.becknic.spectrum.txt",
    ),
    (
        "water",
        "Tap water",
        "water.tapwater.none.liquid.all.tapwater.jhu.becknic.spectrum.txt",
    ),
]

FINE_START_NM = 350.0
FINE_END_NM = 2500.0
FINE_NM = 10.0
COARSE_RATIO = 1.01


def read(path):
    """(nm, fraction) samples in increasing wavelength."""
    samples = []
    with open(path, encoding="latin-1") as file:
        for line in file:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                um, percent = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            samples.append((um * 1000.0, percent / 100.0))
    return sorted(samples)


def bins(end_nm):
    """[low, high) bins: 10 nm to 2500 nm, then 1% wide to `end_nm`."""
    out = []
    low = FINE_START_NM
    while low < FINE_END_NM:
        out.append((low, low + FINE_NM))
        low += FINE_NM
    while low < end_nm:
        out.append((low, low * COARSE_RATIO))
        low *= COARSE_RATIO
    return out


def reduce(samples):
    pairs = []
    for low, high in bins(samples[-1][0]):
        values = [r for w, r in samples if low <= w < high]
        if values:
            mean = sum(values) / len(values)
            pairs.append(
                (0.5 * (low + high), min(max(mean, 0.0), 1.0))
            )
    return pairs


def write(spectra, output):
    lines = [
        "// SPDX-License-Identifier: BSD-3-Clause",
        "// Generated by tools/reflectance_library.py; do not edit.",
        "//",
        "// Measured reflectance spectra from the ECOSTRESS Spectral",
        "// Library (NASA JPL; Meerdink et al. 2019, Baldridge et al.",
        "// 2009): directional hemispherical reflectance as a fraction,",
        "// in 10 nm bins from 350 to 2500 nm, then bins 1% wide to each",
        "// spectrum's end, each its samples' mean at the bin's centre.",
        "#pragma once",
        "",
        "namespace ernie::reflectance_table {",
        "",
        "// clang-format off",
        "struct Entry {",
        "    const char* id;",
        "    const char* name;",
        "    const char* source;  ///< Its ECOSTRESS file.",
        "    int first;           ///< Its first pair in `kPairs`.",
        "    int count;",
        "};",
        "",
    ]
    entries = []
    pairs = []
    for (key, name, source), values in spectra:
        entries.append(
            f'    {{"{key}", "{name}", "{source}", {len(pairs)}, '
            f"{len(values)}}},"
        )
        pairs.extend(values)
    lines.append(
        f"inline constexpr Entry kEntries[{len(entries)}] = {{"
    )
    lines += entries
    lines += ["};", "", "/// (nm, reflectance) pairs."]
    lines.append(
        f"inline constexpr double kPairs[{2 * len(pairs)}] = {{"
    )
    for nm, value in pairs:
        lines.append(f"    {nm:.6g}, {value:.5g},")
    lines += [
        "};",
        "// clang-format on",
        "",
        "}  // namespace ernie::reflectance_table",
        "",
    ]
    with open(output, "w") as file:
        file.write("\n".join(lines))


def main(argv):
    if argv[1:] == ["--files"]:
        print("\n".join(source for _, _, source in SPECTRA))
        return
    directory, output = argv[1], argv[2]
    spectra = [
        (
            (key, name, source),
            reduce(read(os.path.join(directory, source))),
        )
        for key, name, source in SPECTRA
    ]
    write(spectra, output)


if __name__ == "__main__":
    main(sys.argv)
