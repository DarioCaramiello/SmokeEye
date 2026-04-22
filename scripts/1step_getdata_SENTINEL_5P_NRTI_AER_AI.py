"""
S5P Sentinel-5P — Plume Shape & Trajectory Analysis

Goal: extract the smoke plume shape and dispersion direction from
Sentinel-5P Absorbing Aerosol Index (AAI) data. AAI is sensitive to
UV-absorbing particles (smoke, dust) and remains detectable far from
the source — making it ideal for tracing plume trajectory rather than
pinpointing the combustion zone.

Key differences vs the NO2 pipeline:
  - Dataset   : COPERNICUS/S5P/NRTI/L3_AER_AI
  - Band      : absorbing_aerosol_index (dimensionless)
  - Scale     : 3500 m (native AAI resolution)
  - No cloud  : AAI is UV-based and partially cloud-penetrating;
                no explicit cloud mask is applied
  - Focus     : plume trajectory (shape + direction), not source zone
  - Threshold : fixed physical floor (AAI > 0) + Otsu adaptive refinement
                because AAI = 0 is a physically meaningful boundary
                (positive = absorbing aerosols, negative = scattering)

Pipeline:
  1. Download AAI GeoTIFF from Google Earth Engine
  2. Segment the plume: fixed floor threshold + Otsu adaptive refinement
  3. Morphological cleaning (fill holes, remove noise, bridge gaps)
  4. Extract plume contour (pixel-space → geographic coordinates)
  5. PCA on plume pixels → principal axis = maximum elongation = direction
  6. Resolve 180° ambiguity using known fire source coordinates
  7. Compute centroid, bearing, estimated plume length
  8. Annotated map overlay with principal axis arrow
  9. Diagnostic plot if AAI is all-negative or plume is absent
"""

import ee
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from rasterio.io import MemoryFile
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, remove_small_objects, disk
from skimage.measure import label, regionprops, find_contours
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
# If no plume is detected, widen the window: DATE_END = '2025-08-19'
# or shift DATE_START back by one day to catch earlier orbits.

# Minimum absolute AAI value to consider a pixel as potential plume.
# AAI > 0 already indicates UV-absorbing aerosols (smoke, dust).
# This floor prevents including scattering aerosols (marine sulfates, etc.)
# that produce negative AAI. Increase if you get too much background noise;
# decrease if the plume is faint.
AAI_MIN_THRESHOLD = 0.5

# Minimum cluster size (pixels) to retain after morphological cleaning.
# Clusters smaller than this are discarded as noise.
MIN_CLUSTER_PIXELS = 20


ROOT_PATH = "/home/dcaramiello/dev/SmokeEye"
PATH_OUT = f"{ROOT_PATH}/out/Teano_StudyCase/"


# ── EARTH ENGINE SETUP ────────────────────────────────────────────────────────

ee.Authenticate()
ee.Initialize(project='earthenginesmokeeye-488609')

roi = ee.Geometry.Rectangle(ROI_BOUNDS)

collection = (
    ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI')
    .filterDate(DATE_START, DATE_END)
    .filterBounds(roi)
    .select('absorbing_aerosol_index')
)

n = collection.size().getInfo()
print(f"Images found: {n}")

coll_list = collection.toList(n)


# ── HELPER: pixel index → geographic coordinate ───────────────────────────────

def pixel_to_geo(row, col, shape, bounds):
    """
    Convert a (row, col) pixel index to (lon, lat) geographic coordinates
    given the raster bounding box [lon_min, lat_min, lon_max, lat_max].
    Assumes pixels are evenly spaced and origin is upper-left.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    nrows, ncols = shape
    lon = lon_min + (col + 0.5) * (lon_max - lon_min) / ncols
    lat = lat_max - (row + 0.5) * (lat_max - lat_min) / nrows
    return lon, lat


# ── HELPER: geographic bearing (degrees clockwise from North) ─────────────────

def bearing_deg(lat1, lon1, lat2, lon2):
    """
    Compute the forward azimuth from point 1 → point 2 using the
    spherical-planar approximation. Returns degrees in [0, 360).
    Accurate enough for distances < ~500 km at mid-latitudes.
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
    timestamp = img_ee.date().format('YYYY-MM-dd').getInfo()

    print(f"\n{'='*60}")
    print(f"Image {i}  —  {dt_str}")

    # ── 1. DOWNLOAD ───────────────────────────────────────────────────────────
    # No server-side cloud mask here: AAI is derived from UV reflectance and
    # retains a partial signal even under thin cloud cover. Masking aggressively
    # would remove valid plume pixels. Cloud contamination is handled implicitly
    # by the thresholding step (spurious cloud AAI tends to be near-zero).
    url = img_ee.getDownloadURL({
        'scale': 3500,        # native AAI pixel size in metres
        'region': roi,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326'
    })

    response = requests.get(url)
    response.raise_for_status()

    with MemoryFile(response.content) as memfile:
        with memfile.open() as ds:
            aai    = ds.read(1).astype(float)
            nodata = ds.nodata

    # Replace raster nodata sentinel with NaN for numpy operations
    if nodata is not None:
        aai[aai == nodata] = np.nan
    aai[aai < -9000.0] = np.nan

    nrows, ncols = aai.shape
    print(f"  Grid: {nrows}×{ncols}  |  AAI min={np.nanmin(aai):.2f}  max={np.nanmax(aai):.2f}")

    # ── 2. PLUME SEGMENTATION (two-stage adaptive threshold) ─────────────────
    # Stage A — fixed physical floor:
    #   AAI > 0 means UV-absorbing aerosols are present. We use 0.5 as a
    #   conservative floor to exclude near-zero noise near the AAI = 0 boundary.
    #   Unlike NO2, this threshold has a physical meaning independent of the
    #   image statistics, so a fixed value is appropriate here.
    #
    # Stage B — Otsu refinement on positive pixels:
    #   Among the pixels that passed stage A, Otsu's method finds the threshold
    #   that minimises within-class variance between "background" and "plume".
    #   It splits the histogram at its natural valley, adapting to each image
    #   without any manual tuning. We take max(fixed, Otsu) to never go below
    #   the physical floor.
    aai_valid = aai.copy()
    aai_valid[np.isnan(aai_valid)] = 0.0   # NaN → 0 for mask operations

    # Stage A: primary mask
    mask_fixed    = aai_valid > AAI_MIN_THRESHOLD
    positive_vals = aai_valid[mask_fixed]

    # Stage B: Otsu on positive pixels (only if enough samples exist)
    if positive_vals.size > 50:
        otsu_thresh  = threshold_otsu(positive_vals)
        final_thresh = max(AAI_MIN_THRESHOLD, otsu_thresh)
    else:
        # Not enough positive pixels to fit a bimodal distribution → keep floor
        final_thresh = AAI_MIN_THRESHOLD

    print(f"  Plume threshold used: AAI > {final_thresh:.3f}")
    plume_mask = aai_valid > final_thresh

    # ── 3. MORPHOLOGICAL CLEANING ─────────────────────────────────────────────
    # binary_fill_holes: fills internal NaN-caused holes in the plume mask
    plume_mask = binary_fill_holes(plume_mask)

    # opening (erosion → dilation): removes isolated pixel-scale noise.
    # disk(1) = 3×3 footprint — minimal erosion, preserves the plume body.
    plume_mask = opening(plume_mask, footprint=disk(1))

    # closing (dilation → erosion): bridges small internal gaps.
    # disk(2) is slightly larger than opening to reconnect fragmented plume
    # segments caused by missing satellite pixels or thin cloud patches.
    plume_mask = closing(plume_mask, footprint=disk(2))

    # Remove clusters smaller than MIN_CLUSTER_PIXELS.
    # min_size + 1 because skimage >= 0.26 uses inclusive semantics (removes
    # objects whose size <= min_size, not strictly less than).
    labeled    = label(plume_mask)
    plume_mask = remove_small_objects(labeled > 0, min_size=MIN_CLUSTER_PIXELS + 1)

    n_pixels = plume_mask.sum()
    print(f"  Plume pixels after cleaning: {n_pixels}")

    # ── DIAGNOSTIC PLOT: all-negative AAI or plume too small ─────────────────
    aai_max = np.nanmax(aai)
    if n_pixels < MIN_CLUSTER_PIXELS:
        if aai_max < 0:
            print(f"  ⚠  All-negative AAI (max={aai_max:.2f}) — no absorbing aerosols detected.")
            print(f"     Possible causes:")
            print(f"       • Fire not yet active, or plume outside the ROI")
            print(f"       • Cloud cover blocking the UV sensor")
            print(f"       • Try widening the date range to include more orbits")
        else:
            print(f"  ⚠  Plume too small (max AAI={aai_max:.2f}, threshold={final_thresh:.3f}).")
            print(f"     Try lowering AAI_MIN_THRESHOLD (currently: {AAI_MIN_THRESHOLD})")

        # Two-panel diagnostic: raw AAI map + value histogram with threshold markers.
        # This lets you visually assess how far the data is from producing a valid plume.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor('#0d1117')
        for ax in axes:
            ax.set_facecolor('#0d1117')

        lon_min, lat_min, lon_max, lat_max = ROI_BOUNDS
        extent = [lon_min, lon_max, lat_min, lat_max]

        # Left panel: raw AAI with diverging colormap centred on zero
        # (red = absorbing / positive, blue = scattering / negative)
        im = axes[0].imshow(
            aai,
            cmap='RdBu_r',
            vmin=np.nanpercentile(aai, 2),
            vmax=np.nanpercentile(aai, 98),
            extent=extent, origin='upper'
        )
        cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label('AAI', color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        axes[0].plot(SOURCE_LON, SOURCE_LAT, marker='*', color='red',
                     markersize=12, zorder=5, label='Fire source')
        axes[0].set_title(f'Raw AAI — {dt_str}\n(no plume detected)',
                          color='white', fontsize=10)
        axes[0].set_xlabel('Lon', color='white')
        axes[0].set_ylabel('Lat', color='white')
        axes[0].tick_params(colors='white')
        axes[0].legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

        # Right panel: histogram with the two threshold lines
        # Orange line = physical zero boundary; red line = configured floor
        valid_vals = aai[~np.isnan(aai)].flatten()
        axes[1].hist(valid_vals, bins=30, color='#378ADD',
                     edgecolor='#0d1117', linewidth=0.5)
        axes[1].axvline(x=0, color='#EF9F27', linewidth=1.5,
                        linestyle='--', label='AAI = 0  (physical boundary)')
        axes[1].axvline(x=AAI_MIN_THRESHOLD, color='#ff4444', linewidth=1.5,
                        linestyle='--', label=f'Threshold = {AAI_MIN_THRESHOLD}')
        axes[1].set_title('AAI value distribution', color='white', fontsize=10)
        axes[1].set_xlabel('AAI', color='white')
        axes[1].set_ylabel('Pixel count', color='white')
        axes[1].tick_params(colors='white')
        axes[1].legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        for spine in axes[1].spines.values():
            spine.set_edgecolor('#333344')

        plt.tight_layout()
        diag_name = f'S5P_AAI_diagnostic_{i}_{timestamp}.png'
        out = f"{PATH_OUT}{diag_name}"
        plt.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.show()
        print(f"  Diagnostic plot saved: {out}")
        continue

    # ── 4. CONTOUR EXTRACTION ─────────────────────────────────────────────────
    # Returns contours in pixel-space (row, col); converted to lon/lat in the
    # plotting section below.
    contours = find_contours(plume_mask.astype(float), level=0.5)

    # ── 5. PCA → PRINCIPAL AXIS OF THE PLUME ─────────────────────────────────
    # Convert plume pixel indices to geographic coordinates first, so that PCA
    # operates in degree-space rather than pixel-space. This avoids distortion
    # if the pixel is not square (different lon/lat degree sizes).
    rows, cols = np.where(plume_mask)
    lons = ROI_BOUNDS[0] + (cols + 0.5) * (ROI_BOUNDS[2] - ROI_BOUNDS[0]) / ncols
    lats = ROI_BOUNDS[3] - (rows + 0.5) * (ROI_BOUNDS[3] - ROI_BOUNDS[1]) / nrows

    # Geometric centroid of the plume (unweighted — unlike NO2, AAI pixels are
    # already thresholded so all included pixels represent the plume equally)
    cen_lon = lons.mean()
    cen_lat = lats.mean()

    # Build the 2×n centred coordinate matrix and compute the 2×2 covariance matrix.
    # Cov[0,0] = variance in longitude, Cov[1,1] = variance in latitude,
    # Cov[0,1] = Cov[1,0] = covariance (how lon and lat vary together).
    coords_centered = np.stack([lons - cen_lon, lats - cen_lat], axis=1)
    cov_matrix      = np.cov(coords_centered.T)

    # Eigendecomposition: eigenvectors are the natural axes of the point cloud.
    # The eigenvector corresponding to the largest eigenvalue points along the
    # direction of maximum variance — i.e. the elongation axis of the plume.
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]   # (dlon, dlat) unit vector

    # Extend the axis by ±2σ from the centroid to visualise it on the map.
    # 2σ covers ~95% of the plume mass along the principal direction.
    spread   = 2.0 * np.sqrt(eigenvalues.max())
    tip_lon  = cen_lon + principal_axis[0] * spread
    tip_lat  = cen_lat + principal_axis[1] * spread
    tail_lon = cen_lon - principal_axis[0] * spread
    tail_lat = cen_lat - principal_axis[1] * spread

    # ── 6. RESOLVE 180° AMBIGUITY ─────────────────────────────────────────────
    # PCA returns an axis, not a directed vector: the tip and tail are
    # mathematically equivalent. We resolve this using the known fire source:
    # the plume travels AWAY from the source, so the correct tip is the
    # endpoint that is farther from the source coordinates.
    dist_tip  = np.hypot(tip_lon  - SOURCE_LON, tip_lat  - SOURCE_LAT)
    dist_tail = np.hypot(tail_lon - SOURCE_LON, tail_lat - SOURCE_LAT)
    if dist_tail > dist_tip:
        # Swap: tail was farther from source — it should be the tip
        tip_lon, tip_lat, tail_lon, tail_lat = tail_lon, tail_lat, tip_lon, tip_lat

    # Compute bearing (direction of plume dispersion from fire source)
    bearing = bearing_deg(SOURCE_LAT, SOURCE_LON, tip_lat, tip_lon)
    compass_dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    compass = compass_dirs[int((bearing + 11.25) / 22.5) % 16]

    # Plume length estimate along the principal axis (Euclidean in degrees → km)
    # Note: this is an approximation; 1° ≈ 111 km holds reasonably at 41°N.
    plume_length_deg = np.hypot(tip_lon - tail_lon, tip_lat - tail_lat)
    plume_length_km  = plume_length_deg * 111.0

    print(f"  Plume centroid  : ({cen_lat:.4f}°N, {cen_lon:.4f}°E)")
    print(f"  Plume direction : {bearing:.1f}° ({compass})")
    print(f"  Estimated length: {plume_length_km:.1f} km")

    # ── 7. MAIN PLOT ──────────────────────────────────────────────────────────
    lon_min, lat_min, lon_max, lat_max = ROI_BOUNDS
    extent = [lon_min, lon_max, lat_min, lat_max]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # AAI heatmap: inferno colormap (dark = low AAI, bright = high AAI)
    vmin = np.nanpercentile(aai, 2)
    vmax = np.nanpercentile(aai, 98)
    im = ax.imshow(
        aai,
        cmap='inferno',
        vmin=vmin, vmax=vmax,
        extent=extent,
        origin='upper',
        alpha=0.9,
        zorder=1
    )

    # Semi-transparent cyan overlay for plume pixels
    plume_rgba = np.zeros((*plume_mask.shape, 4))
    plume_rgba[plume_mask, 0] = 0.0   # R
    plume_rgba[plume_mask, 1] = 0.9   # G
    plume_rgba[plume_mask, 2] = 0.8   # B
    plume_rgba[plume_mask, 3] = 0.30  # alpha
    ax.imshow(plume_rgba, extent=extent, origin='upper', zorder=2)

    # Plume contours (pixel-space row/col → geographic lon/lat)
    for contour in contours:
        c_lons = lon_min + (contour[:, 1] + 0.5) * (lon_max - lon_min) / ncols
        c_lats = lat_max - (contour[:, 0] + 0.5) * (lat_max - lat_min) / nrows
        ax.plot(c_lons, c_lats, color='#00e5cc',
                linewidth=1.4, alpha=0.85, zorder=3)

    # Dashed principal axis line (tail → tip)
    ax.plot([tail_lon, tip_lon], [tail_lat, tip_lat],
            color='white', linewidth=1.2, linestyle='--', alpha=0.6, zorder=4,
            label=f'Principal axis ({bearing:.0f}° {compass})')

    # Direction arrow from centroid toward tip
    ax.annotate(
        '',
        xy=(tip_lon, tip_lat),
        xytext=(cen_lon, cen_lat),
        arrowprops=dict(
            arrowstyle='->',
            color='#ffdd57',
            lw=2.2,
            mutation_scale=18
        ),
        zorder=5
    )

    # Known fire source marker
    ax.plot(SOURCE_LON, SOURCE_LAT,
            marker='*', markersize=14, color='#ff4444',
            markeredgecolor='white', markeredgewidth=0.8,
            zorder=6, label='Fire source')
    ax.annotate(
        'Source',
        xy=(SOURCE_LON, SOURCE_LAT),
        xytext=(SOURCE_LON + 0.01, SOURCE_LAT + 0.012),
        color='#ff8888', fontsize=8.5, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2, foreground='black')],
        zorder=7
    )

    # Plume centroid marker
    ax.plot(cen_lon, cen_lat,
            marker='+', markersize=12, color='#00e5cc',
            markeredgewidth=2, zorder=6, label='Plume centroid')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label('Absorbing Aerosol Index (AAI)', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')

    # Stats info box (bottom-left corner)
    info_text = (
        f"Bearing: {bearing:.1f}° ({compass})\n"
        f"Axis length: {plume_length_km:.1f} km\n"
        f"Centroid: {cen_lat:.3f}°N  {cen_lon:.3f}°E\n"
        f"Plume pixels: {n_pixels}"
    )
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=8.5, color='white',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                  edgecolor='#00e5cc', alpha=0.85),
        zorder=8
    )

    # Title, axis labels, tick colours
    ax.set_title(f'Sentinel-5P AAI — Plume Analysis\n{dt_str}',
                 fontsize=13, fontweight='bold', color='white', pad=10)
    ax.set_xlabel('Longitude (°E)', color='white', fontsize=10)
    ax.set_ylabel('Latitude (°N)', color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

    ax.legend(loc='upper right', fontsize=8.5,
              facecolor='#1a1a2e', edgecolor='#00e5cc',
              labelcolor='white', framealpha=0.9)

    plt.tight_layout()
    out_name = f'S5P_plume_analysis_{i}_{timestamp}.png'
    out = f"{PATH_OUT}{out_name}"
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Saved: {out}")