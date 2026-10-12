Image support data (ISD) fixtures from usgscsm's test data (https://github.com/DOI-USGS/usgscsm, `tests/data`), which is in the public domain (CC0 1.0):

- `constVelocityLineScan.json`: a synthetic 16 x 16 line scanner moving at constant velocity over a 10 m body.
- `orbitalLineScan.json`: a synthetic 16 x 16 line scanner in a circular orbit.
- `ctxLineScan.json`: Mars Reconnaissance Orbiter's Context Camera (CTX).
- `toughLroNacLineScan.json`: Lunar Reconnaissance Orbiter's Narrow Angle Camera (NAC, left), 2 x 2 summed.
- `orbitalPushFrame.json`: a synthetic push-frame camera of 30 framelets of 12 x 16 pixels in a circular orbit.
- `lroWacPushFrame.json`: Lunar Reconnaissance Orbiter's Wide Angle Camera (WAC), one visible band (604 nm) of 364 framelets of 14 x 1024 pixels.
- `simpleFramerISD.json`: a synthetic 15 x 15 frame camera 1 km from a 10 m body, with transverse distortion.
- `orbitalFramer.json`: a synthetic 16 x 16 frame camera 50 km above a 1000 km body.
- `jitterFramer.json`: `orbitalFramer.json` with rolling-shutter jitter, a quadratic in each line's exposure time.
- `dawnfc_radtan.json`: Dawn's Framing Camera 2 over Vesta, 1024 x 1024, with radial-tangential (radtan) distortion.

Generated here:

- `lroNacM1282557272RE.json`: Lunar Reconnaissance Orbiter's Narrow Angle Camera (NAC, right), image M1282557272RE (2018-06-02, unsummed, 5064 x 25600) over Tycho's central peak. Made with ISIS (`lronac2isis`, `spiceinit web=true shape=ellipsoid`, the pointing from the USGS SPICE web service) and ALE 1.2.0's `isd_generate`, from the EDR in the PDS LROC archive (https://pds.lroc.im-ldi.com/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0035/DATA/ESM3/2018153/NAC/M1282557272RE.IMG).
