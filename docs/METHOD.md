# TIANGOU Eclipse Model method

## End-to-end stages

The production workflow has ten computational stages after input setup.

1. Download NAIF SPICE kernels (`DE440`, lunar orientation, Earth orientation,
   leap seconds, and planetary constants).
2. Download SLDEM2015 at 128 pixels per degree for 60 S to 60 N and polar LOLA
   LDEM at 240 m per pixel for the 60-90 degree caps.
3. Download 5-minute AIA images for eight channels and the nearest 5-minute
   SUVI 284 A images.
4. Download the event-day FISM2 `flare_hr` and `flare_bands` NetCDF products.
5. Build the inclusive 5-minute time grid and the descending WACCM6 70-level
   height grid. The local runner skips existing snapshots and continues with
   each missing time-height calculation.
6. For each snapshot, use vectorized SPICE geometry to identify grid points
   where the Moon intersects the angular extent of the solar image.
7. Construct the lunar limb from every applicable native SLDEM/LDEM pixel and
   integrate the visible intensity over every native solar-image pixel at each
   selected grid point.
8. Concatenate the two-dimensional snapshots into one
   `source_mask(time, height, lat, lon)` file for each of the nine source
   channels.
9. Use the FISM2 0.1 nm spectrum and official Stan-band membership to calculate
   `source_weight(time, band, source)`, then combine the nine source masks into
   23 bands.
10. Validate dimensions, source ordering, TIANGOU configuration attributes,
    selected-slab value ranges, and NaN counts.

## True-limb source mask

Let `I_p` be the intensity of solar-image pixel `p`, and let `V_p` be one when
that ray is visible past the terrain-resolved lunar limb and zero otherwise.
The transmission for source channel `s` is

```text
M_s = sum_p(I_p V_p) / sum_p(I_p).
```

The lunar limb is observer- and time-dependent. SPICE supplies the Earth,
Moon, and Sun geometry and lunar body-fixed orientation. SLDEM2015 supplies
the equatorward terrain, while the polar LDEM products replace the regions
poleward of 60 degrees. Every applicable pixel from both terrain products is
used. The arrays are processed in memory-bounded blocks without changing
their resolution. The resulting limb is accumulated into 18,000 angular bins.

## Nine-source spectral mapping

The source order is fixed:

```text
aia_94, aia_131, aia_171, aia_193, aia_211,
suvi_284, aia_304, aia_335, aia_1600
```

For wavelengths below 9.4 nm, the mapping is an equal AIA 94/AIA 131 blend.
Between 9.4 and 33.5 nm, weights are interpolated in log wavelength between
the channel anchors. Above 33.5 nm, TIANGOU uses AIA 1600 as the
long-wave source. These wavelength-level weights are integrated within each
Stan band using the event-time FISM2 photon spectrum.

For output band `b`, time `t`, source `s`, height `z`, and grid point `x`,

```text
M(t,b,z,x) = sum_s W(t,b,s) M_s(t,z,x),
```

where the source weights sum to one. Repeated Stan bands are assigned with the
N2 cross-section thresholds in `pipeline/stan_bands/cross_sections.dat` rather
than by wavelength intervals alone.

The official FISM2 Stan-band irradiance is stored separately as
`fism2_flux(time, band)`. The output `mask_euv` remains a transmission factor.

## Production entry points

- `run_from_zero.sh`: Python setup, input staging, preflight, and local execution.
- `scripts/run_event.py`: restartable nine-source calculation and 23-band synthesis.
- `scripts/compute_tiangou_snapshot.py`: one time-height source snapshot.
- `pipeline/assemble_source_cube.py`: source-cube assembly.
- `pipeline/make_tiangou_23band_mask.py`: final 23-band synthesis.
- `scripts/validate_eclipsemask.py`: final contract validation.
