"""
CAMS NRT — Quantitative Concentration Estimation
Wildfire Teano 16/08/2025

What this dataset is:
  CAMS (Copernicus Atmosphere Monitoring Service) is NOT a satellite.
  It is a numerical weather prediction model run by ECMWF (Reading, UK).
  Every 12 hours the model ingests observations from worldwide sources
  (ground stations, radiosondes, aircraft, multiple satellites) and
  produces a globally consistent atmospheric composition field.
  Each GEE image = one 6-hourly snapshot of that field.

  Key difference vs S5P:
    - S5P measures a column integral from space once per day over one spot.
    - CAMS estimates SURFACE concentrations in µg/m³ every 6h everywhere.
    - CAMS resolution: ~44 km/pixel (much coarser than S5P 1-3.5 km).

What each band measures:
  - PM2.5  [kg/m3 -> ug/m3]: fine particles <2.5um. Main smoke tracer.
            Wildfire values: 50-500 ug/m3 at 44km resolution.
  - PM10   [kg/m3 -> ug/m3]: coarser particles <10um. PM10 - PM2.5 = ash.
  - AOD    [dimensionless]:  total aerosol optical depth at 550nm.
            How much sunlight is blocked by the aerosol column (0 = clear).
  - OA AOD [dimensionless]:  organic aerosol fraction of AOD.
            High OA/AOD (>0.4) = smoke-dominated, not dust or traffic.
  - BC AOD [dimensionless]:  black carbon fraction of AOD. Soot from fire.
  - CO col [kg/m2 -> mg/m2]: total CO column. Incomplete combustion tracer.
            Elevated CO (>200 mg/m2) confirms active burning.

  Note: full band set (PM10, CO, BC, OA...) only available from 2021-07-01.
        Before that date only PM2.5 and total AOD exist on GEE.

Pipeline:
  1. Filter CAMS collection by date, ROI, and forecast step
  2. Download all 6 bands as individual GeoTIFFs at native resolution
  3. Convert units: kg/m3 -> ug/m3, kg/m2 -> mg/m2
  4. Compute source-zone enrichment vs background
  5. Compare PM2.5 against WHO 2021 air quality guidelines
  6. Multi-panel map plot (6 panels, one per band)
  7. Summary plot: PM2.5 map + enrichment bar chart
  8. Diagnostic hints if no signal is detected
"""

import ee
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rasterio.io import MemoryFile


# ── CONFIG ────────────────────────────────────────────────────────────────────

SOURCE_LON = 14.1130910
SOURCE_LAT = 41.2716850

# ROI wider than S5P scripts: CAMS pixels are ~44 km, so the Teano area
# would be only 1-2 pixels if we used the S5P bounding box.
# Widening the ROI lets us see regional spatial gradients.
ROI_BOUNDS = [13.0, 40.8, 15.0, 41.8]   # [lon_min, lat_min, lon_max, lat_max]

DATE_START = '2025-08-18'
DATE_END   = '2025-08-19'
# NOTE: CAMS runs twice daily (00 UTC and 12 UTC initialisation).
# model_forecast_hour = 0 is the analysis step (no lead time = best quality).
# Set either to None to retrieve all available steps in the date range.
MODEL_INIT_HOUR     = 0    # 0 = midnight UTC run  |  12 = noon UTC run
MODEL_FORECAST_HOUR = 0    # 0 = analysis field (best)  |  3, 6, 9... = forecast

# Source zone radius for enrichment statistics (degrees).
# ~0.10 degrees ~ 11 km at 41 N. Must be wider than S5P version
# because CAMS pixels are 44 km.
SOURCE_RADIUS_DEG = 0.10

# WHO 2021 24-hour mean PM guidelines (ug/m3) — reference lines in plots
WHO_PM25_24H = 15.0
WHO_PM10_24H = 45.0

# Unit conversion factors
KG_M3_TO_UG_M3 = 1e9   # kg/m3  ->  ug/m3  (for PM2.5, PM10)
KG_M2_TO_MG_M2 = 1e6   # kg/m2  ->  mg/m2  (for CO column)


ROOT_PATH = "/home/dcaramiello/dev/SmokeEye"
PATH_OUT = f"{ROOT_PATH}/out/Teano_StudyCase/"

# ── EARTH ENGINE SETUP ────────────────────────────────────────────────────────

ee.Authenticate()
ee.Initialize(project='earthenginesmokeeye-488609')

roi = ee.Geometry.Rectangle(ROI_BOUNDS)

# Build the collection and apply optional forecast-step filters.
# GEE image properties: model_initialization_hour (0 or 12)
#                       model_forecast_hour       (0, 3, 6, ... 120)
base = (
    ee.ImageCollection('ECMWF/CAMS/NRT')
    .filterDate(DATE_START, DATE_END)
    .filterBounds(roi)
)
if MODEL_INIT_HOUR is not None:
    base = base.filter(ee.Filter.eq('model_initialization_hour', MODEL_INIT_HOUR))
if MODEL_FORECAST_HOUR is not None:
    base = base.filter(ee.Filter.eq('model_forecast_hour', MODEL_FORECAST_HOUR))

BANDS = [
    'particulate_matter_d_less_than_25_um_surface',           # PM2.5
    'particulate_matter_d_less_than_10_um_surface',           # PM10
    'total_aerosol_optical_depth_at_550nm_surface',           # AOD total
    'organic_matter_aerosol_optical_depth_at_550nm_surface',  # OA AOD
    'black_carbon_aerosol_optical_depth_at_550nm_surface',    # BC AOD
    'total_column_carbon_monoxide_surface',                   # CO column
]

collection = base.select(BANDS)
n          = collection.size().getInfo()
print(f"CAMS images found: {n}")

if n == 0:
    print("No images found.")
    print("  Try: MODEL_INIT_HOUR = None  and  MODEL_FORECAST_HOUR = None")
    print("  Try widening DATE_END (e.g. '2025-08-20')")
    print("  Full band set only available from 2021-07-01 onward.")
    exit()

coll_list = collection.toList(n)


# ── HELPER: download one band as numpy array ──────────────────────────────────

def download_band(img_ee, band_name, roi, scale=44000):
    """
    Download a single band from GEE as a float32 numpy array.
    scale=44000 matches the CAMS native pixel size (~44 km).
    Requesting a finer scale would only bilinearly interpolate the data
    without adding real information, so we keep it at native resolution.
    Returns the array with NaN replacing nodata values.
    """
    url = img_ee.select(band_name).getDownloadURL({
        'scale': scale,
        'region': roi,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326'
    })
    r = requests.get(url)
    r.raise_for_status()

    with MemoryFile(r.content) as mf:
        with mf.open() as ds:
            arr    = ds.read(1).astype(float)
            nodata = ds.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < -9000.0] = np.nan
    return arr


# ── HELPER: source-zone enrichment statistics ─────────────────────────────────

def enrichment_stats(arr, bounds, src_lon, src_lat, radius_deg):
    """
    Compare the mean concentration inside a circle of radius_deg centred on
    the fire source against the mean outside (background).

    Returns (mean_inside, mean_outside, enrichment_ratio).
    enrichment > 1.3 indicates a meaningful elevation near the source.

    Uses Euclidean distance in degrees, which is a valid approximation
    for small areas (<50 km) at mid-latitudes.
    """
    nrows, ncols = arr.shape
    col_lons = (bounds[0]
                + (np.arange(ncols) + 0.5) * (bounds[2] - bounds[0]) / ncols)
    row_lats = (bounds[3]
                - (np.arange(nrows) + 0.5) * (bounds[3] - bounds[1]) / nrows)
    lon_g, lat_g = np.meshgrid(col_lons, row_lats)

    dist    = np.hypot(lon_g - src_lon, lat_g - src_lat)
    valid   = ~np.isnan(arr)
    inside  = (dist <= radius_deg) & valid
    outside = (dist >  radius_deg) & valid

    m_in  = float(np.mean(arr[inside]))  if inside.any()  else np.nan
    m_out = float(np.mean(arr[outside])) if outside.any() else np.nan
    enr   = (m_in / m_out
             if not (np.isnan(m_in) or np.isnan(m_out) or m_out == 0)
             else np.nan)
    return m_in, m_out, enr


# ── FORMATTING HELPERS ────────────────────────────────────────────────────────

def fmt(v, dec=1):
    return f"{v:.{dec}f}" if not np.isnan(v) else "n/a"

def fmtenr(v):
    return f"{v:.2f}x" if not np.isnan(v) else "n/a"


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

for i in range(n):
    img_ee    = ee.Image(coll_list.get(i))
    info      = img_ee.getInfo()
    ms        = info['properties']['system:time_start']
    init_h    = info['properties'].get('model_initialization_hour', '?')
    fcst_h    = info['properties'].get('model_forecast_hour', '?')
    dt_str    = datetime.datetime.fromtimestamp(ms / 1000).strftime('%Y-%m-%d %H:%M UTC')
    timestamp = datetime.datetime.fromtimestamp(ms / 1000).strftime('%Y-%m-%d_%H%M')

    print(f"\n{'='*60}")
    print(f"Image {i}  —  {dt_str}  (init +{init_h}h, forecast +{fcst_h}h)")

    # ── 1. DOWNLOAD ALL 6 BANDS ───────────────────────────────────────────────
    print("  Downloading bands...")
    pm25_raw = download_band(img_ee, BANDS[0], roi)
    pm10_raw = download_band(img_ee, BANDS[1], roi)
    aod      = download_band(img_ee, BANDS[2], roi)
    oa_aod   = download_band(img_ee, BANDS[3], roi)
    bc_aod   = download_band(img_ee, BANDS[4], roi)
    co_raw   = download_band(img_ee, BANDS[5], roi)

    nrows, ncols = pm25_raw.shape

    # ── 2. UNIT CONVERSION ────────────────────────────────────────────────────
    # PM2.5 / PM10: kg/m3 -> ug/m3
    # Example: 3e-9 kg/m3 (background clean air) -> 3 ug/m3
    #          3e-7 kg/m3 (wildfire peak)         -> 300 ug/m3
    pm25 = pm25_raw * KG_M3_TO_UG_M3
    pm10 = pm10_raw * KG_M3_TO_UG_M3

    # CO column: kg/m2 -> mg/m2
    co = co_raw * KG_M2_TO_MG_M2

    # Aerosol type fractions (OA and BC relative to total AOD).
    # These ratios identify the aerosol source:
    #   OA/AOD 0.4-0.7  ->  biomass burning smoke
    #   OA/AOD ~0.05    ->  traffic / industrial
    #   OA/AOD ~0.0     ->  mineral dust (Saharan)
    #   BC/AOD >0.05    ->  active combustion (soot)
    with np.errstate(invalid='ignore', divide='ignore'):
        oa_frac = np.where(aod > 0, oa_aod / aod, np.nan)
        bc_frac = np.where(aod > 0, bc_aod / aod, np.nan)

    print(f"  Grid: {nrows}x{ncols}")
    print(f"  PM2.5  {fmt(np.nanmin(pm25))} - {fmt(np.nanmax(pm25))} ug/m3")
    print(f"  PM10   {fmt(np.nanmin(pm10))} - {fmt(np.nanmax(pm10))} ug/m3")
    print(f"  AOD    {fmt(np.nanmin(aod),3)} - {fmt(np.nanmax(aod),3)}")
    print(f"  CO     {fmt(np.nanmin(co))} - {fmt(np.nanmax(co))} mg/m2")

    # ── 3. SOURCE ZONE STATISTICS ─────────────────────────────────────────────
    pm25_in, pm25_out, pm25_enr = enrichment_stats(
        pm25, ROI_BOUNDS, SOURCE_LON, SOURCE_LAT, SOURCE_RADIUS_DEG)
    pm10_in, pm10_out, pm10_enr = enrichment_stats(
        pm10, ROI_BOUNDS, SOURCE_LON, SOURCE_LAT, SOURCE_RADIUS_DEG)
    aod_in,  aod_out,  aod_enr  = enrichment_stats(
        aod,  ROI_BOUNDS, SOURCE_LON, SOURCE_LAT, SOURCE_RADIUS_DEG)
    co_in,   co_out,   co_enr   = enrichment_stats(
        co,   ROI_BOUNDS, SOURCE_LON, SOURCE_LAT, SOURCE_RADIUS_DEG)

    print(f"\n  Source zone statistics (r={SOURCE_RADIUS_DEG} deg):")
    print(f"    PM2.5  src={fmt(pm25_in)} ug/m3   bg={fmt(pm25_out)}   enrich={fmtenr(pm25_enr)}")
    print(f"    PM10   src={fmt(pm10_in)} ug/m3   bg={fmt(pm10_out)}   enrich={fmtenr(pm10_enr)}")
    print(f"    AOD    src={fmt(aod_in,3)}         bg={fmt(aod_out,3)}  enrich={fmtenr(aod_enr)}")
    print(f"    CO     src={fmt(co_in)} mg/m2      bg={fmt(co_out)}    enrich={fmtenr(co_enr)}")

    # WHO guideline check
    if not np.isnan(pm25_in):
        ratio = pm25_in / WHO_PM25_24H
        if ratio > 3:
            print(f"  ⚠  PM2.5 source zone = {ratio:.1f}x WHO 24h limit — very unhealthy")
        elif ratio > 1:
            print(f"  ⚠  PM2.5 source zone = {ratio:.1f}x WHO 24h limit — above guideline")
        else:
            print(f"  ✔  PM2.5 source zone below WHO 24h guideline ({WHO_PM25_24H} ug/m3)")

    # Diagnostic hints if signal is absent
    if np.nanmax(pm25) < 5 and np.nanmax(aod) < 0.05:
        print("  ⚠  Very low concentrations — fire signal may be absent or outside ROI.")
        print("     Try: wider DATE range, or set MODEL_FORECAST_HOUR=None")

    # ── 4. MULTI-PANEL MAP PLOT (6 panels) ────────────────────────────────────
    lon_min, lat_min, lon_max, lat_max = ROI_BOUNDS
    extent = [lon_min, lon_max, lat_min, lat_max]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)
    axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax in axs:
        ax.set_facecolor('#0d1117')

    def map_panel(ax, data, cmap, title, unit, vmin=None, vmax=None):
        _vmin = vmin if vmin is not None else np.nanpercentile(data, 2)
        _vmax = vmax if vmax is not None else np.nanpercentile(data, 98)
        im = ax.imshow(data, cmap=cmap, vmin=_vmin, vmax=_vmax,
                       extent=extent, origin='upper', alpha=0.92)
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label(unit, color='white', fontsize=8)
        cb.ax.yaxis.set_tick_params(color='white', labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
        cb.outline.set_edgecolor('#555566')
        ax.plot(SOURCE_LON, SOURCE_LAT, marker='*', markersize=10,
                color='#ff3333', markeredgecolor='white',
                markeredgewidth=0.6, zorder=5)
        ax.add_patch(plt.Circle((SOURCE_LON, SOURCE_LAT), SOURCE_RADIUS_DEG,
                                color='#ffdd57', fill=False, linewidth=1.0,
                                linestyle='--', alpha=0.7, zorder=4))
        ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=4)
        ax.set_xlabel('Lon (E)', color='white', fontsize=7)
        ax.set_ylabel('Lat (N)', color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333344')

    map_panel(axs[0], pm25, 'hot_r',
              f'PM2.5  [src {fmt(pm25_in)} ug/m3, {fmtenr(pm25_enr)} bgd]',
              'ug/m3')
    map_panel(axs[1], pm10, 'YlOrRd',
              f'PM10  [src {fmt(pm10_in)} ug/m3, {fmtenr(pm10_enr)} bgd]',
              'ug/m3')
    map_panel(axs[2], aod, 'plasma',
              f'AOD 550nm  [src {fmt(aod_in,3)}, {fmtenr(aod_enr)} bgd]',
              'dimensionless')
    map_panel(axs[3], oa_frac, 'RdPu',
              'Organic fraction  OA/AOD\n>0.4 = smoke signal',
              'ratio', vmin=0, vmax=1)
    map_panel(axs[4], bc_frac, 'Reds',
              'Black carbon fraction  BC/AOD',
              'ratio', vmin=0, vmax=0.3)
    map_panel(axs[5], co, 'cividis',
              f'CO column  [src {fmt(co_in)} mg/m2, {fmtenr(co_enr)} bgd]',
              'mg/m2')

    fig.suptitle(
        f'CAMS NRT — Wildfire concentration estimate\n'
        f'{dt_str}  (init +{init_h}h, forecast +{fcst_h}h)',
        color='white', fontsize=12, fontweight='bold', y=0.98
    )

    summary = (
        f"Source zone  r={SOURCE_RADIUS_DEG} deg\n"
        f"PM2.5  {fmt(pm25_in)} ug/m3  ({fmtenr(pm25_enr)} bgd)\n"
        f"PM10   {fmt(pm10_in)} ug/m3\n"
        f"AOD    {fmt(aod_in,3)}   CO {fmt(co_in)} mg/m2\n"
        f"WHO PM2.5 24h limit: {WHO_PM25_24H} ug/m3"
    )
    fig.text(0.98, 0.95, summary, ha='right', va='top', fontsize=8,
             color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                       edgecolor='#00e5cc', alpha=0.88))

    out_maps = f'CAMS_maps_{i}_{timestamp}.png'
    out = f"{PATH_OUT}{out_maps}"
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Saved: {out}")

    # ── 5. SUMMARY PLOT: PM2.5 map + enrichment bar chart ────────────────────
    fig2, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(13, 5))
    fig2.patch.set_facecolor('#0d1117')
    for ax in (ax_map, ax_bar):
        ax.set_facecolor('#0d1117')

    # Left: PM2.5 concentration map
    vmin_p = np.nanpercentile(pm25, 2)
    vmax_p = np.nanpercentile(pm25, 98)
    im2 = ax_map.imshow(pm25, cmap='hot_r', vmin=vmin_p, vmax=vmax_p,
                        extent=extent, origin='upper', alpha=0.92)
    cb2 = plt.colorbar(im2, ax=ax_map, fraction=0.046, pad=0.04)
    cb2.set_label('PM2.5 (ug/m3)', color='white', fontsize=9)
    cb2.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')
    ax_map.plot(SOURCE_LON, SOURCE_LAT, marker='*', markersize=12,
                color='#ff3333', markeredgecolor='white',
                markeredgewidth=0.7, zorder=5, label='Fire source')
    ax_map.add_patch(plt.Circle((SOURCE_LON, SOURCE_LAT), SOURCE_RADIUS_DEG,
                                color='#ffdd57', fill=False, linewidth=1.2,
                                linestyle='--', alpha=0.7, zorder=4))
    ax_map.set_title(f'PM2.5 — {dt_str}', color='white', fontsize=10)
    ax_map.set_xlabel('Lon (E)', color='white')
    ax_map.set_ylabel('Lat (N)', color='white')
    ax_map.tick_params(colors='white')
    ax_map.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    for sp in ax_map.spines.values():
        sp.set_edgecolor('#333344')

    # Right: bar chart comparing source zone vs background for PM2.5, PM10, AOD
    labels   = ['PM2.5\n(ug/m3)', 'PM10\n(ug/m3)', 'AOD x100']
    src_vals = [pm25_in,
                pm10_in,
                aod_in * 100 if not np.isnan(aod_in) else np.nan]
    bgd_vals = [pm25_out,
                pm10_out,
                aod_out * 100 if not np.isnan(aod_out) else np.nan]

    x     = np.arange(len(labels))
    width = 0.35
    b_src = ax_bar.bar(x - width/2, src_vals, width, label='Source zone',
                       color='#ff8c00', alpha=0.85)
    b_bgd = ax_bar.bar(x + width/2, bgd_vals, width, label='Background',
                       color='#378ADD', alpha=0.85)

    # WHO PM2.5 reference line
    ax_bar.axhline(y=WHO_PM25_24H, color='#ff4444',
                   linewidth=1.2, linestyle='--', alpha=0.8)
    ax_bar.text(len(labels) - 0.55, WHO_PM25_24H + 0.4,
                f'WHO 24h = {WHO_PM25_24H}', color='#ff8888', fontsize=8)

    # Value labels on bars
    for bar in list(b_src) + list(b_bgd):
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                        f'{h:.1f}', ha='center', va='bottom',
                        color='white', fontsize=8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, color='white', fontsize=9)
    ax_bar.set_title('Source zone vs background', color='white', fontsize=10)
    ax_bar.set_ylabel('Concentration / AOD value', color='white')
    ax_bar.tick_params(colors='white')
    ax_bar.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    for sp in ax_bar.spines.values():
        sp.set_edgecolor('#333344')

    plt.suptitle(f'CAMS NRT summary  |  {dt_str}',
                 color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()

    out_summary = f'CAMS_summary_{i}_{timestamp}.png'
    out = f"{PATH_OUT}{out_summary}"
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig2.get_facecolor())
    plt.show()
    print(f"  Saved: {out}")