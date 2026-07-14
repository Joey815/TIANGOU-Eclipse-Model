# TIANGOU Eclipse Model data sources

No large scientific input is stored in this repository. The download scripts
preserve the original filenames and stage the products under `data/`.

## Lunar topography

The equatorward DEM is `SLDEM2015_128_60S_60N_000_360_FLOAT`, a 128
pixels-per-degree float IMG product covering 60 S to 60 N. It combines LRO
LOLA altimetry with SELENE Terrain Camera topography.

Source directory:

<https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/global/float_img/>

The polar replacement uses the LOLA `LDEM_60N/S_240M` float IMG products.
All DEM products are read at their archive-native pixel spacing.

Source directory:

<https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/float_img/>

For SLDEM2015, cite Barker et al. (2016), *A new lunar digital elevation model
from the Lunar Orbiter Laser Altimeter and SELENE Terrain Camera*, Icarus 273,
346-355, <https://doi.org/10.1016/j.icarus.2015.07.039>.

## SPICE kernels

The workflow downloads generic NAIF kernels from:

<https://naif.jpl.nasa.gov/pub/naif/generic_kernels/>

TIANGOU uses `de440.bsp`, `earth_latest_high_prec.bpc`,
`moon_pa_de440_200625.bpc`, `moon_de440_200625.tf`, `pck00011.tpc`, and
`naif0012.tls`.

## Solar images

SDO/AIA level-1 images are selected through JSOC DRMS from
`aia.lev1_euv_12s` and `aia.lev1_uv_24s`. The nearest quality-zero record to
each five-minute target is retained.

GOES-16/SUVI level-2 images are downloaded from NOAA NCEI:

<https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/>

The TIANGOU nine-source product uses SUVI 284 A only; the other eight source
images are from AIA.

## FISM2

The workflow downloads event-day NetCDF files from the CU/LASP FISM2 archive:

- `flare_hr_data`: 0.1 nm spectral bins at 60-second cadence.
- `flare_bands`: 23 Stan bands at five-minute cadence.

Archive root:

<https://lasp.colorado.edu/eve/data_access/evewebdata/fism/>

The model reference is Chamberlin et al. (2020), *The Flare Irradiance
Spectral Model-Version 2 (FISM2)*, Space Weather 18,
<https://doi.org/10.1029/2020SW002588>.

## Generated-data availability

The repository records code and source URLs, but not the multi-gigabyte inputs
or generated NetCDF products. For a published data release, archive final
products in a DOI-bearing repository and record their SHA-256 checksums and
the Git commit used for generation.
