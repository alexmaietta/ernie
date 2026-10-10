# SPDX-License-Identifier: BSD-3-Clause
"""A star field over a house at night: real bright stars placed for a site and time, their
brightness from their magnitudes and their colour from their B-V indices, behind a dim airglow
dome, through a wide lens and a 35 mm one on Orion, both at a long night exposure. Stars are points
for now: a sensor filter for the lens's point-spread would grow the bright ones into discs.

Each star is a black body (temperature from B-V, Ballesteros 2012) whose illuminance is 2.54e-6 lux
times 10^(-0.4 V) (Allen's Astrophysical Quantities): Sirius gives about 1e-5 lux, a magnitude 6
star 1e-8. The renderer spreads each star's irradiance over the pixels around it, so stars keep
their brightness at any resolution, and draws them only where no geometry is in front.

    PYTHONPATH=build/python python demos/python/demo_lighting_star_field.py
    PYTHONPATH=build/python python demos/python/demo_lighting_star_field.py --output stars.png
    PYTHONPATH=build/python python demos/python/demo_lighting_star_field.py --catalogue bsc.csv

The built-in stars are about forty of the brightest, with J2000 positions, V magnitudes and B-V
indices rounded from the Yale Bright Star Catalogue, filled out with made-up faint stars
(`ernie.synthetic_stars`) for the look of a dark sky. A --catalogue CSV with columns
ra_deg,dec_deg,vmag,bv (J2000, a header line) replaces them all, such as one exported from
VizieR's copy of the Yale Bright Star Catalogue (V/50) or Hipparcos (I/239).

The site is 40 N 105 W (near Boulder, Colorado) at 21:00 local time on 14 January 2025
(04:00 UT on the 15th), when Orion stands high in the south. Precession since 2000 (about a
third of a degree) and refraction are left out.

References:
- H. W. Jensen, F. Durand, J. Dorsey, M. M. Stark, P. Shirley and S. Premoze, "A Physically-Based
  Night Sky Model", SIGGRAPH 2001, 399-408. https://doi.org/10.1145/383259.383306
- D. Hoffleit and W. H. Warren Jr., "The Bright Star Catalogue", 5th revised edition, Yale
  University Observatory, 1991 (VizieR V/50).
- F. J. Ballesteros, "New insights into black bodies", EPL 97(3):34008, 2012.
- A. N. Cox (ed.), "Allen's Astrophysical Quantities", 4th edition, Springer, 2000.
- J. Meeus, "Astronomical Algorithms", 2nd edition, Willmann-Bell, 1998 (sidereal time).
"""

import argparse
import csv
import math

import numpy as np

import ernie
from ernie import tokens

from demo_support import side_by_side, write_png

SIZE = (900, 600)
LATITUDE, LONGITUDE = 40.0, -105.0
WHEN = ernie.julian_date(2025, 1, 15, 4.0)

# name, right ascension (deg), declination (deg), V, B-V: J2000, rounded.
BRIGHT_STARS = [
    ("Sirius", 101.29, -16.72, -1.46, 0.00),
    ("Canopus", 95.99, -52.70, -0.74, 0.15),
    ("Arcturus", 213.92, 19.18, -0.05, 1.23),
    ("Vega", 279.23, 38.78, 0.03, 0.00),
    ("Capella", 79.17, 46.00, 0.08, 0.80),
    ("Rigel", 78.63, -8.20, 0.13, -0.03),
    ("Procyon", 114.83, 5.22, 0.34, 0.42),
    ("Betelgeuse", 88.79, 7.41, 0.50, 1.85),
    ("Achernar", 24.43, -57.24, 0.46, -0.16),
    ("Altair", 297.70, 8.87, 0.76, 0.22),
    ("Aldebaran", 68.98, 16.51, 0.86, 1.54),
    ("Antares", 247.35, -26.43, 0.96, 1.83),
    ("Spica", 201.30, -11.16, 0.97, -0.23),
    ("Pollux", 116.33, 28.03, 1.14, 1.00),
    ("Fomalhaut", 344.41, -29.62, 1.16, 0.09),
    ("Deneb", 310.36, 45.28, 1.25, 0.09),
    ("Regulus", 152.09, 11.97, 1.35, -0.11),
    ("Castor", 113.65, 31.89, 1.58, 0.03),
    ("Bellatrix", 81.28, 6.35, 1.64, -0.22),
    ("Alnilam", 84.05, -1.20, 1.69, -0.18),
    ("Alnitak", 85.19, -1.94, 1.77, -0.21),
    ("Mintaka", 83.00, -0.30, 2.23, -0.22),
    ("Saiph", 86.94, -9.67, 2.09, -0.17),
    ("Meissa", 83.78, 9.93, 3.39, -0.16),
    ("Elnath", 81.57, 28.61, 1.65, -0.13),
    ("Polaris", 37.95, 89.26, 1.98, 0.60),
    ("Dubhe", 165.93, 61.75, 1.79, 1.07),
    ("Merak", 165.46, 56.38, 2.37, -0.02),
    ("Phecda", 178.46, 53.69, 2.44, 0.04),
    ("Megrez", 183.86, 57.03, 3.31, 0.08),
    ("Alioth", 193.51, 55.96, 1.77, -0.02),
    ("Mizar", 200.98, 54.93, 2.27, 0.02),
    ("Alkaid", 206.89, 49.31, 1.86, -0.19),
    ("Schedar", 10.13, 56.54, 2.24, 1.17),
    ("Caph", 2.29, 59.15, 2.28, 0.34),
    ("Gamma Cassiopeiae", 14.18, 60.72, 2.47, -0.15),
    ("Ruchbah", 21.45, 60.24, 2.68, 0.13),
    ("Mirfak", 51.08, 49.86, 1.79, 0.48),
    ("Hamal", 31.79, 23.46, 2.00, 1.15),
    ("Alhena", 99.43, 16.40, 1.93, 0.00),
    ("Adhara", 104.66, -28.97, 1.50, -0.21),
    ("Wezen", 107.10, -26.39, 1.83, 0.68),
]


def catalogue(path):
    """Right ascension, declination, magnitude and colour index arrays: from a CSV, or the bright
    stars above with made-up faint ones."""
    if path:
        with open(path, newline="") as file:
            rows = [r for r in csv.DictReader(file)]
        return (np.array([float(r["ra_deg"]) for r in rows], np.float32),
                np.array([float(r["dec_deg"]) for r in rows], np.float32),
                np.array([float(r["vmag"]) for r in rows], np.float32),
                np.array([float(r["bv"]) for r in rows], np.float32))
    faint = ernie.synthetic_stars(9000, seed=7, faintest=6.5)
    keep = np.array(faint["magnitude"]) > 3.0  # the bright ones are real
    ra = [s[1] for s in BRIGHT_STARS] + list(np.array(faint["right_ascension"])[keep])
    dec = [s[2] for s in BRIGHT_STARS] + list(np.array(faint["declination"])[keep])
    mag = [s[3] for s in BRIGHT_STARS] + list(np.array(faint["magnitude"])[keep])
    bv = [s[4] for s in BRIGHT_STARS] + list(np.array(faint["color_index"])[keep])
    return tuple(np.array(v, np.float32) for v in (ra, dec, mag, bv))


def sky_map(rows=64, cols=128):
    """Airglow and a little light pollution: brighter towards the horizon, faintly warm low in
    the north-west (towards a town); the ground below is dark. Its photometric illuminance scales
    it, so only its shape matters."""
    theta = (np.arange(rows) + 0.5) / rows * np.pi
    phi = (np.arange(cols) + 0.5) / cols * 2.0 * np.pi
    t, p = np.meshgrid(theta, phi, indexing="ij")
    up = np.cos(t)[..., None]
    horizon = np.clip(1.0 - up, 0.0, 1.0) ** 4
    town = np.exp(-((p - 2.4) ** 2) / 0.3)[..., None] * horizon
    sky = np.array([0.55, 0.7, 1.0]) * (0.5 + 1.5 * horizon) + 2.0 * town * np.array([1.0, 0.7, 0.4])
    return np.where(up > 0.0, sky, 0.05).astype(np.float32)


def placed(x, y, z, sx=1.0, sy=1.0, sz=1.0):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    return m


def direction_of(ra, dec, transform):
    """World direction of a star at `ra`, `dec` (degrees) under the StarField's `transform`."""
    r, d = math.radians(ra), math.radians(dec)
    local = np.array([math.cos(d) * math.cos(r), math.cos(d) * math.sin(r), math.sin(d)])
    return np.asarray(transform)[:3, :3] @ local


def build_scene(catalogue_path):
    scene = ernie.Scene()
    ground = ernie.Path("/Looks/Ground")
    scene.define(ground, tokens.MATERIAL).set(tokens.ALBEDO, (0.18, 0.16, 0.14))
    scene.define("/World/Ground", tokens.GROUND_PLANE).set(tokens.MATERIAL_BINDING, ground)
    # A house and a tree in silhouette to the south-east.
    for path, transform in [("/World/House", placed(9.0, -26.0, 2.5, 9.0, 6.0, 5.0)),
                            ("/World/Roof", placed(9.0, -26.0, 5.6, 9.5, 6.5, 1.2))]:
        scene.define(path, tokens.CUBE).set(tokens.SIZE, 1.0).set(
            tokens.TRANSFORM, transform).set(tokens.MATERIAL_BINDING, ground)
    scene.define("/World/Tree", tokens.SPHERE).set(tokens.GEOM_RADIUS, 3.0).set(
        tokens.TRANSFORM, placed(-12.0, -22.0, 5.5)).set(tokens.MATERIAL_BINDING, ground)

    # A moonless night: about 6e-4 lux from the sky's glow.
    scene.define("/World/Night", tokens.DOME_LIGHT).set(
        tokens.PHOTOMETRIC_ILLUMINANCE, 6e-4).write(tokens.LIGHT_TEXTURE, sky_map())
    sky = ernie.celestial_to_local(LATITUDE, LONGITUDE, WHEN)
    ra, dec, mag, bv = catalogue(catalogue_path)
    scene.define("/World/Stars", tokens.STAR_FIELD).set(tokens.TRANSFORM, sky).write(
        tokens.STARS_RIGHT_ASCENSION, ra).write(tokens.STARS_DECLINATION, dec).write(
        tokens.STARS_MAGNITUDE, mag).write(tokens.STARS_COLOR_INDEX, bv)

    # A night exposure, ISO 1600, f/2.8, 15 s: the brightest stars clip to white, magnitude 6
    # ones are just visible, and the sky's glow is dark grey.
    scene.define("/World/Camera", tokens.CAMERA).set(tokens.EXPOSURE_ISO, 1600.0).set(
        tokens.EXPOSURE_TIME, 15.0).set(tokens.EXPOSURE_FSTOP, 2.8).set(
        tokens.SENSOR_NOISE, False)
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")).set(tokens.RESOLUTION, SIZE)
    return scene, sky


def aim(scene, target, focal_length):
    eye = np.array([0.0, 0.0, 1.6])
    scene.set("/World/Camera", tokens.TRANSFORM,
              ernie.look_at_transform(tuple(eye), tuple(eye + 100.0 * target)))
    scene.set("/World/Camera", tokens.FOCAL_LENGTH, focal_length)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", help="write the image here instead of opening a window")
    parser.add_argument("--catalogue", help="CSV of ra_deg,dec_deg,vmag,bv in place of the "
                                            "built-in stars")
    args = parser.parse_args()
    scene, sky = build_scene(args.catalogue)
    alnilam = direction_of(84.05, -1.20, sky)
    print(f"Orion's belt: {math.degrees(math.asin(alnilam[2])):.0f} degrees up, azimuth "
          f"{math.degrees(math.atan2(alnilam[0], alnilam[1])) % 360:.0f}")
    renderer = ernie.Renderer()
    # Wide, south and up; then 35 mm on Orion.
    south = np.array([0.0, -math.cos(math.radians(30.0)), math.sin(math.radians(30.0))])
    aim(scene, south, 14.0)
    wide = np.array(renderer.render(scene))[..., :3]
    aim(scene, alnilam, 35.0)
    orion = np.array(renderer.render(scene))[..., :3]
    image = side_by_side([wide, orion])
    if args.output:
        write_png(args.output, image)
    else:
        from ernie.app import Viewer
        aim(scene, south, 14.0)
        Viewer(scene).run()


if __name__ == "__main__":
    main()
