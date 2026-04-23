"""
S5P Sentinel-5P — NO2 Source Zone Validation
Wildfire Teano 16/08/2025

Goal: validate the area near the fire SOURCE using tropospheric NO2.
Unlike AAI (which traces the distant plume), NO2 is a direct combustion
tracer — elevated peaks near the source confirm active fire in the area.

Key differences vs the AAI pipeline:
  - Dataset  : COPERNICUS/S5P/NRTI/L3_NO2
  - Band     : tropospheric_NO2_column_number_density (mol/m²)
  - Scale    : 1113 m (native NO2 resolution, ~3x finer than AAI)
  - Mask     : cloud_fraction < 0.3 (cloud pre-filter applied server-side)
  - Focus    : source zone (tight ROI), not plume trajectory
  - Units    : mol/m² → converted to µmol/m² for readability
  - Threshold: local percentile (not a fixed absolute value), because
               the NO2 background varies with season, time, and area

Pipeline:
  1. Download NO2 GeoTIFF with cloud mask applied on GEE (server-side)
  2. Convert mol/m² → µmol/m²
  3. Segment the NO2 anomaly (local p75 as adaptive threshold)
  4. Morphological cleaning (open → close → remove small objects)
  5. Compute weighted centroid + distance from known fire source
  6. Compute source-zone enrichment ratio vs background
  7. Annotated map with source marker and stats info box
  8. Diagnostic plot if no anomaly is detected
"""

import ee
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from rasterio.io import MemoryFile
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, remove_small_objects, disk
from skimage.measure import label, find_contours
from scipy.ndimage import binary_fill_holes


# ── CONFIG ────────────────────────────────────────────────────────────────────

# Known fire source coordinates (Teano area, August 2025)
SOURCE_LON = 14.1130910
SOURCE_LAT = 41.2716850

# Region of interest: [lon_min, lat_min, lon_max, lat_max]
ROI_BOUNDS = [13.85, 41.15, 14.20, 41.37]

DATE_START = '2025-08-18'
DATE_END   = '2025-08-19'
# NOTE: S5P passes over a given area roughly once per day.
# If no signal is found, try widening the window: DATE_END = '2025-08-21'

# Maximum cloud cover fraction accepted per pixel (0–1)
# Pixels with cloud_fraction >= this value are masked out before download.
# The mask is applied server-side on GEE, so àùcloudy pixels arrive as NaN.
CLOUD_THRESHOLD = 0.3

# NO2 anomaly threshold: percentile of local pixel values above which
# we consider a pixel part of the anomaly (i.e. potential combustion signal).
# p75 = top quartile — conservative, captures only real peaks.
# Unlike AAI, there is no physically meaningful absolute threshold for NO2
# because the background level varies with season, time of day, and location.
NO2_PERCENTILE_THRESHOLD = 75

# Minimum cluster size (pixels) to keep after morphological cleaning.
# Smaller than in the AAI pipeline because the source ROI is tighter.
MIN_CLUSTER_PIXELS = 10

# Radius (in degrees) around the fire source for the enrichment analysis.
# ~0.05° ≈ 5.5 km at this latitude. Used to compare NO2 inside vs outside
# the source zone and compute the enrichment ratio.
SOURCE_ANALYSIS_RADIUS_DEG = 0.05

ROOT_PATH = "/home/dcaramiello/dev/SmokeEye"
PATH_OUT = f"{ROOT_PATH}/out/Teano_StudyCase/"


# ── EARTH ENGINE SETUP ────────────────────────────────────────────────────────

ee.Authenticate()
ee.Initialize(project='earthenginesmokeeye-488609')

roi = ee.Geometry.Rectangle(ROI_BOUNDS)

# Select only the two bands we need: NO2 column density + cloud fraction.
# cloud_fraction is used for masking only and is not downloaded.
collection = (
    ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
    .filterDate(DATE_START, DATE_END)
    .filterBounds(roi)
    .select(['tropospheric_NO2_column_number_density', 'cloud_fraction'])
)

n = collection.size().getInfo()
print(f"Images found: {n}")

if n == 0:
    print("No images in date range. Try widening DATE_END.")
    exit()

coll_list = collection.toList(n)


# ── HELPER: geographic bearing (degrees clockwise from North) ─────────────────

def bearing_deg(lat1, lon1, lat2, lon2):
    """
    Compute the forward azimuth from point 1 → point 2 using
    the spherical-planar approximation. Returns degrees [0, 360).
    """
    dlon = np.radians(lon2 - lon1)
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2_r)
    y = (np.cos(lat1_r) * np.sin(lat2_r)
         - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon))
    return np.degrees(np.arctan2(x, y)) % 360


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

for i in range(n):
    img_ee    = ee.Image(coll_list.get(i))
    info      = img_ee.getInfo()
    ms        = info['properties']['system:time_start']
    dt_str    = datetime.datetime.fromtimestamp(ms / 1000).strftime('%Y-%m-%d %H:%M UTC')
    timestamp = img_ee.date().format('YYYY-MM-dd_HH-mm').getInfo()

    print(f"\n{'='*60}")
    print(f"Image {i}  —  {dt_str}")

    # ── 1. SERVER-SIDE CLOUD MASK + DOWNLOAD ─────────────────────────────────
    # updateMask() sets pixels where cloud_fraction >= CLOUD_THRESHOLD to
    # masked (no-data). When exported as GeoTIFF they become the nodata value.
    # Doing this server-side avoids downloading corrupt NO2 readings that
    # occur when the sensor is looking through cloud rather than at the ground.
    img_clean = img_ee.updateMask(
        img_ee.select('cloud_fraction').lt(CLOUD_THRESHOLD)
    )

    url = img_clean.select('tropospheric_NO2_column_number_density').getDownloadURL({
        'scale': 1113,       # native NO2 pixel size in metres
        'region': roi,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326'
    })

    response = requests.get(url)
    response.raise_for_status()

    with MemoryFile(response.content) as memfile:
        with memfile.open() as ds:
            no2_raw = ds.read(1).astype(float)
            nodata  = ds.nodata

    # Replace the raster nodata sentinel with NaN for numpy operations
    if nodata is not None:
        no2_raw[no2_raw == nodata] = np.nan
    no2_raw[no2_raw < -9000.0] = np.nan

    nrows, ncols = no2_raw.shape

    # ── 2. UNIT CONVERSION: mol/m² → µmol/m² ─────────────────────────────────
    # Raw values are on the order of 1e-4 mol/m², which is hard to read.
    # Multiplying by 1e6 gives µmol/m² (typical range: 50–300), much more
    # interpretable in plots and printed stats.
    no2 = no2_raw * 1e6

    valid_count = np.sum(~np.isnan(no2))
    print(f"  Grid: {nrows}×{ncols}  |  valid pixels: {valid_count}")

    if valid_count == 0:
        # Every pixel was cloudy or missing — nothing to analyse
        print("  ⚠  All pixels masked (cloud cover or nodata). Skipping.")
        continue

    print(f"  NO2  min={np.nanmin(no2):.3f}  max={np.nanmax(no2):.3f}  "
          f"mean={np.nanmean(no2):.3f}  µmol/m²")

    # ── 3. ANOMALY SEGMENTATION (adaptive percentile threshold) ───────────────
    # We cannot use a fixed absolute NO2 threshold because the background level
    # changes with time of day, season, and local sources (traffic, industry).
    # Instead we compute a LOCAL threshold as the p75 of positive-valued pixels
    # in the ROI: this captures the top quartile of the signal regardless of
    # the absolute magnitude, adapting to each individual image.
    no2_filled = no2.copy()
    no2_filled[np.isnan(no2_filled)] = 0.0  # replace NaN with 0 for masking ops

    if np.any(no2 > 0):
        # Compute threshold only on positive pixels to avoid bias from zeros
        thresh = np.nanpercentile(no2[no2 > 0], NO2_PERCENTILE_THRESHOLD)
    else:
        # Fall back to the full distribution if no positive values exist
        thresh = np.nanpercentile(no2, NO2_PERCENTILE_THRESHOLD)

    print(f"  Anomaly threshold: NO2 > {thresh:.3f} µmol/m²  (p{NO2_PERCENTILE_THRESHOLD})")
    anomaly_mask = no2_filled > thresh

    # ── 4. MORPHOLOGICAL CLEANING ─────────────────────────────────────────────
    # binary_fill_holes: fills internal NaN holes that break the mask continuity
    anomaly_mask = binary_fill_holes(anomaly_mask)

    # opening (erosion then dilation): removes isolated single-pixel noise
    # disk(1) = 3×3 structuring element — minimal, preserves small clusters
    anomaly_mask = opening(anomaly_mask, footprint=disk(1))

    # closing (dilation then erosion): bridges small gaps inside the anomaly
    # disk(1) here because the source ROI is dense — avoid over-merging clusters
    anomaly_mask = closing(anomaly_mask, footprint=disk(1))

    # Remove clusters smaller than MIN_CLUSTER_PIXELS.
    # min_size + 1 because skimage >= 0.26 removes objects <= min_size (inclusive).
    labeled      = label(anomaly_mask)
    anomaly_mask = remove_small_objects(labeled > 0, min_size=MIN_CLUSTER_PIXELS + 1)

    n_pixels = anomaly_mask.sum()
    print(f"  Anomaly pixels after cleaning: {n_pixels}")

    # ── DIAGNOSTIC PLOT: no anomaly detected ──────────────────────────────────
    if n_pixels < MIN_CLUSTER_PIXELS:
        print(f"  ⚠  No NO2 anomaly detected.")
        print(f"     Possible causes:")
        print(f"       • Cloud cover > {CLOUD_THRESHOLD*100:.0f}%  (try raising CLOUD_THRESHOLD)")
        print(f"       • Fire not yet active or NO2 signal too weak")
        print(f"       • Try a lower NO2_PERCENTILE_THRESHOLD (current: {NO2_PERCENTILE_THRESHOLD})")

        # Two-panel diagnostic: raw NO2 map + value histogram with threshold line
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor('#0d1117')
        for ax in axes:
            ax.set_facecolor('#0d1117')

        lon_min, lat_min, lon_max, lat_max = ROI_BOUNDS
        extent = [lon_min, lon_max, lat_min, lat_max]

        # Left panel: raw NO2 heatmap
        im = axes[0].imshow(
            no2, cmap='YlOrRd',
            vmin=np.nanpercentile(no2, 2),
            vmax=np.nanpercentile(no2, 98),
            extent=extent, origin='upper'
        )
        cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label('NO2 (µmol/m²)', color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        axes[0].plot(SOURCE_LON, SOURCE_LAT, marker='*', color='red',
                     markersize=12, zorder=5, label='Fire source')
        axes[0].set_title(f'Raw NO2 — {dt_str}\n(no anomaly detected)',
                          color='white', fontsize=10)
        axes[0].set_xlabel('Lon', color='white')
        axes[0].set_ylabel('Lat', color='white')
        axes[0].tick_params(colors='white')
        axes[0].legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

        # Right panel: NO2 value histogram with threshold marker
        # Shows visually how far the data is from the anomaly threshold
        valid_vals = no2[~np.isnan(no2)].flatten()
        axes[1].hist(valid_vals, bins=30, color='#EF9F27',
                     edgecolor='#0d1117', linewidth=0.5)
        axes[1].axvline(x=thresh, color='#ff4444', linewidth=1.5,
                        linestyle='--',
                        label=f'Threshold p{NO2_PERCENTILE_THRESHOLD} = {thresh:.3f}')
        axes[1].set_title('NO2 value distribution', color='white', fontsize=10)
        axes[1].set_xlabel('NO2 (µmol/m²)', color='white')
        axes[1].set_ylabel('Pixel count', color='white')
        axes[1].tick_params(colors='white')
        axes[1].legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        for spine in axes[1].spines.values():
            spine.set_edgecolor('#333344')

        plt.suptitle(f'NO2 Diagnostic — {dt_str}',
                     color='white', fontsize=11, fontweight='bold')
        plt.tight_layout()
        diag_name = f'S5P_NO2_diagnostic_{i}_{timestamp}.png'
        plt.savefig(diag_name, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.show()
        print(f"  Diagnostic plot saved: {diag_name}")
        continue

    # ── 5. SPATIAL ANOMALY ANALYSIS ───────────────────────────────────────────

    # Convert anomaly pixel indices to geographic coordinates
    rows_a, cols_a = np.where(anomaly_mask)
    lons_a = (ROI_BOUNDS[0]
              + (cols_a + 0.5) * (ROI_BOUNDS[2] - ROI_BOUNDS[0]) / ncols)
    lats_a = (ROI_BOUNDS[3]
              - (rows_a + 0.5) * (ROI_BOUNDS[3] - ROI_BOUNDS[1]) / nrows)

    # Weighted centroid: each pixel contributes proportionally to its NO2 value.
    # This pulls the centroid toward the hottest part of the anomaly, giving a
    # better estimate of the actual combustion core than a geometric mean would.
    weights = no2_filled[rows_a, cols_a]
    cen_lon = np.average(lons_a, weights=weights)
    cen_lat = np.average(lats_a, weights=weights)

    # Distance and bearing from centroid to known fire source
    dist_km = np.hypot(cen_lon - SOURCE_LON, cen_lat - SOURCE_LAT) * 111.0
    bear    = bearing_deg(SOURCE_LAT, SOURCE_LON, cen_lat, cen_lon)
    compass_dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    compass = compass_dirs[int((bear + 11.25) / 22.5) % 16]

    # Count anomaly pixels within SOURCE_ANALYSIS_RADIUS_DEG of the source.
    # Note: Euclidean distance in degrees is used (not Haversine) because the
    # radius is small (~5.5 km) and the lat/lon degree sizes are similar here.
    dist_from_source = np.hypot(lons_a - SOURCE_LON, lats_a - SOURCE_LAT)
    n_near_source    = np.sum(dist_from_source <= SOURCE_ANALYSIS_RADIUS_DEG)
    pct_near         = 100.0 * n_near_source / n_pixels

    # Build a boolean mask for all pixels (not just anomaly pixels) that fall
    # inside the source circle, using broadcasting over the full grid.
    # This is used to compare mean NO2 inside vs outside the circle.
    col_coords = (np.arange(ncols)[None, :] + 0.5) * (ROI_BOUNDS[2] - ROI_BOUNDS[0]) / ncols + ROI_BOUNDS[0]
    row_coords = ROI_BOUNDS[3] - (np.arange(nrows)[:, None] + 0.5) * (ROI_BOUNDS[3] - ROI_BOUNDS[1]) / nrows
    source_mask = np.hypot(col_coords - SOURCE_LON, row_coords - SOURCE_LAT) <= SOURCE_ANALYSIS_RADIUS_DEG

    # Mean NO2 inside the source zone and outside (background)
    no2_source_zone = no2[source_mask & ~np.isnan(no2)]
    no2_outside     = no2[~source_mask & ~np.isnan(no2)]
    no2_mean_source = np.mean(no2_source_zone) if no2_source_zone.size > 0 else np.nan
    no2_mean_bg     = np.mean(no2_outside)     if no2_outside.size > 0     else np.nan

    # Enrichment ratio: how much higher is NO2 at the source vs the background?
    # > 1.0 = some excess; > 1.3 = meaningful signal consistent with active fire.
    # Threshold of 1.3 is conservative: high enough to exclude natural variation,
    # low enough not to miss moderate-intensity fires.
    enrichment = (no2_mean_source / no2_mean_bg
                  if (no2_mean_bg and no2_mean_bg > 0) else np.nan)

    print(f"  Anomaly centroid    : ({cen_lat:.4f}°N, {cen_lon:.4f}°E)")
    print(f"  Distance to source  : {dist_km:.1f} km ({bear:.0f}° {compass})")
    print(f"  Pixels near source  : {n_near_source}/{n_pixels} ({pct_near:.0f}%)"
          f"  [r = {SOURCE_ANALYSIS_RADIUS_DEG}°]")
    print(f"  NO2 source zone mean: {no2_mean_source:.3f} µmol/m²")
    print(f"  NO2 background mean : {no2_mean_bg:.3f} µmol/m²")
    if not np.isnan(enrichment):
        print(f"  NO2 enrichment      : {enrichment:.2f}x")
        if enrichment > 1.3:
            print("  ✔  Elevated NO2 near source — consistent with active fire")
        else:
            print("  x  Weak enrichment — low-intensity fire or diluted signal")

    # ── 6. CONTOUR EXTRACTION ─────────────────────────────────────────────────
    # find_contours returns pixel-space (row, col) arrays; converted to lon/lat below
    contours = find_contours(anomaly_mask.astype(float), level=0.5)

    # ── 7. MAIN PLOT ──────────────────────────────────────────────────────────
    lon_min, lat_min, lon_max, lat_max = ROI_BOUNDS
    extent = [lon_min, lon_max, lat_min, lat_max]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # NO2 heatmap: YlOrRd colormap (yellow = low, red = high)
    vmin = np.nanpercentile(no2, 2)
    vmax = np.nanpercentile(no2, 98)
    im = ax.imshow(
        no2,
        cmap='YlOrRd',
        vmin=vmin, vmax=vmax,
        extent=extent,
        origin='upper',
        alpha=0.92,
        zorder=1
    )

    # Semi-transparent orange overlay for anomaly pixels
    anomaly_rgba = np.zeros((*anomaly_mask.shape, 4))
    anomaly_rgba[anomaly_mask, 0] = 1.0   # R
    anomaly_rgba[anomaly_mask, 1] = 0.55  # G
    anomaly_rgba[anomaly_mask, 2] = 0.0   # B
    anomaly_rgba[anomaly_mask, 3] = 0.28  # alpha
    ax.imshow(anomaly_rgba, extent=extent, origin='upper', zorder=2)

    # Dashed circle showing the source analysis zone
    circle = plt.Circle(
        (SOURCE_LON, SOURCE_LAT),
        SOURCE_ANALYSIS_RADIUS_DEG,
        color='#ffdd57', fill=False,
        linewidth=1.2, linestyle='--', alpha=0.7, zorder=4,
        label=f'Analysis zone  (r ≈ {SOURCE_ANALYSIS_RADIUS_DEG * 111:.0f} km)'
    )
    ax.add_patch(circle)

    # Anomaly contours (pixel-space → geographic coordinates)
    for contour in contours:
        c_lons = lon_min + (contour[:, 1] + 0.5) * (lon_max - lon_min) / ncols
        c_lats = lat_max - (contour[:, 0] + 0.5) * (lat_max - lat_min) / nrows
        ax.plot(c_lons, c_lats, color='#ff8c00', linewidth=1.4,
                alpha=0.85, zorder=3)

    # Weighted centroid of the anomaly
    ax.plot(cen_lon, cen_lat, marker='+', markersize=13, color='#ff8c00',
            markeredgewidth=2.2, zorder=6, label='NO2 anomaly centroid')

    # Known fire source
    ax.plot(SOURCE_LON, SOURCE_LAT, marker='*', markersize=15,
            color='#ff3333', markeredgecolor='white', markeredgewidth=0.8,
            zorder=7, label='Fire source')
    ax.annotate(
        'Source',
        xy=(SOURCE_LON, SOURCE_LAT),
        xytext=(SOURCE_LON + 0.01, SOURCE_LAT + 0.013),
        color='#ff8888', fontsize=8.5, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2, foreground='black')],
        zorder=8
    )

    # Dotted line connecting centroid to fire source
    ax.plot([SOURCE_LON, cen_lon], [SOURCE_LAT, cen_lat],
            color='white', linewidth=0.8, linestyle=':', alpha=0.5, zorder=5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label('Tropospheric NO₂ (µmol/m²)', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')

    # Stats info box (bottom-left corner)
    enrich_str = f"{enrichment:.2f}x" if not np.isnan(enrichment) else "n/a"
    info_text = (
        f"Anomaly centroid\n"
        f"  {cen_lat:.4f}°N  {cen_lon:.4f}°E\n"
        f"Dist. to source: {dist_km:.1f} km ({bear:.0f}° {compass})\n"
        f"Anomaly pixels: {n_pixels}  |  near src: {pct_near:.0f}%\n"
        f"NO2 source zone: {no2_mean_source:.3f} µmol/m²\n"
        f"NO2 background:  {no2_mean_bg:.3f} µmol/m²\n"
        f"Enrichment:      {enrich_str}"
    )
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=8, color='white',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                  edgecolor='#ff8c00', alpha=0.88),
        zorder=9
    )

    # Title, axis labels, tick colours
    ax.set_title(
        f'Sentinel-5P NO₂ — Source zone validation\n{dt_str}',
        fontsize=12, fontweight='bold', color='white', pad=10
    )
    ax.set_xlabel('Longitude (°E)', color='white', fontsize=10)
    ax.set_ylabel('Latitude (°N)', color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

    ax.legend(loc='upper right', fontsize=8.5,
              facecolor='#1a1a2e', edgecolor='#ff8c00',
              labelcolor='white', framealpha=0.9)

    plt.tight_layout()
    out_name = f'S5P_NO2_source_validation_{i}_{timestamp}.png'
    out = f"{PATH_OUT}{out_name}"
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Saved: {out}")