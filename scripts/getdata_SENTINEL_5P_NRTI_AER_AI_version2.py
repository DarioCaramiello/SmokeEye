"""
S5P Sentinel-5P — Plume Shape & Trajectory Analysis
Rogo Teano 16/08/2025

Pipeline:
  1. Download AAI GeoTIFF from Google Earth Engine
  2. Segment the plume via adaptive thresholding (Otsu on positive AAI values)
  3. Morphological cleaning (remove noise, fill holes)
  4. Extract plume contour
  5. PCA on plume pixels → principal axis = plume direction
  6. Compute centroid, tip, and angular bearing from source
  7. Produce annotated map overlay
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
from skimage.morphology import (opening, closing,
                                remove_small_objects, disk)
from skimage.measure import label, regionprops, find_contours
from scipy.ndimage import binary_fill_holes


# ── CONFIG ────────────────────────────────────────────────────────────────────

SOURCE_LON = 14.1130910   # rogo noto (Teano area)
SOURCE_LAT = 41.2716850

ROI_BOUNDS = [13.85, 41.15, 14.20, 41.37]   # [lon_min, lat_min, lon_max, lat_max]
DATE_START = '2025-08-17'
DATE_END   = '2025-08-18'

# Soglia minima assoluta AAI per considerare un pixel come "plume"
# (valori > 0 già indicano aerosol assorbenti; regola in base ai tuoi dati)
AAI_MIN_THRESHOLD = 0.5

# Dimensione minima del cluster (pixel) da mantenere
MIN_CLUSTER_PIXELS = 20


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
print(f"Immagini trovate: {n}")

coll_list = collection.toList(n)


# ── HELPER: pixel → geographic coordinate ────────────────────────────────────
def pixel_to_geo(row, col, shape, bounds):
    """Converte indice pixel (row, col) in (lon, lat) dato bounds [lon_min, lat_min, lon_max, lat_max]."""
    lon_min, lat_min, lon_max, lat_max = bounds
    nrows, ncols = shape
    lon = lon_min + (col + 0.5) * (lon_max - lon_min) / ncols
    lat = lat_max - (row + 0.5) * (lat_max - lat_min) / nrows
    return lon, lat


# ── HELPER: bearing (gradi da Nord, senso orario) ─────────────────────────────
def bearing_deg(lat1, lon1, lat2, lon2):
    """Calcola il bearing geografico da punto 1 → punto 2."""
    dlon = np.radians(lon2 - lon1)
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y)) % 360
    return bearing


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
for i in range(n):
    img_ee = ee.Image(coll_list.get(i))
    info   = img_ee.getInfo()
    ms     = info['properties']['system:time_start']
    dt_str = datetime.datetime.fromtimestamp(ms / 1000).strftime('%Y-%m-%d %H:%M UTC')
    timestamp = img_ee.date().format('YYYY-MM-dd').getInfo()

    print(f"\n{'='*60}")
    print(f"Immagine {i}  —  {dt_str}")

    # ── 1. DOWNLOAD ──────────────────────────────────────────────────────────
    url = img_ee.getDownloadURL({
        'scale': 3500,
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

    # Pulizia nodata
    if nodata is not None:
        aai[aai == nodata] = np.nan
    aai[aai < -9000.0] = np.nan

    nrows, ncols = aai.shape
    print(f"  Grid: {nrows}x{ncols}  |  AAI min={np.nanmin(aai):.2f}  max={np.nanmax(aai):.2f}")

    # ── 2. SEGMENTAZIONE PLUME ───────────────────────────────────────────────
    aai_valid = aai.copy()
    aai_valid[np.isnan(aai_valid)] = 0.0

    # Maschera primaria: solo valori > soglia fissa
    mask_fixed = aai_valid > AAI_MIN_THRESHOLD

    # Raffina con Otsu sui soli pixel positivi (se ce ne sono abbastanza)
    positive_vals = aai_valid[mask_fixed]
    if positive_vals.size > 50:
        otsu_thresh = threshold_otsu(positive_vals)
        final_thresh = max(AAI_MIN_THRESHOLD, otsu_thresh)
    else:
        final_thresh = AAI_MIN_THRESHOLD

    print(f"  Soglia plume usata: AAI > {final_thresh:.3f}")
    plume_mask = aai_valid > final_thresh

    # ── 3. PULIZIA MORFOLOGICA ───────────────────────────────────────────────
    # Riempimento buchi interni
    plume_mask = binary_fill_holes(plume_mask)

    # Apertura: rimuove pixel isolati
    plume_mask = opening(plume_mask, footprint=disk(1))

    # Chiusura: colma piccole lacune
    plume_mask = closing(plume_mask, footprint=disk(2))

    # Rimozione cluster piccoli
    labeled      = label(plume_mask)
    plume_mask   = remove_small_objects(labeled > 0, min_size=MIN_CLUSTER_PIXELS + 1)

    n_pixels = plume_mask.sum()
    print(f"  Pixel plume dopo pulizia: {n_pixels}")

    # ── DIAGNOSTICA: se AAI tutto negativo o plume assente, plot comunque ────
    aai_max = np.nanmax(aai)
    if n_pixels < MIN_CLUSTER_PIXELS:
        if aai_max < 0:
            print(f"  ⚠  AAI tutto negativo (max={aai_max:.2f}) — nessun aerosol assorbente rilevato.")
            print(f"     Possibili cause:")
            print(f"       • il rogo non era ancora attivo o la plume era fuori ROI")
            print(f"       • copertura nuvolosa che blocca il sensore")
            print(f"       • prova DATE_END = '{DATE_START[:8]}{int(DATE_START[8:])+2:02d}' per includere più ore")
        else:
            print(f"  ⚠  Plume troppo piccola (max AAI={aai_max:.2f}, soglia={final_thresh:.3f}).")
            print(f"     Prova ad abbassare AAI_MIN_THRESHOLD (attuale: {AAI_MIN_THRESHOLD})")

        # Plot diagnostico: mostra comunque l'AAI grezzo con istogramma
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor('#0d1117')
        for ax in axes:
            ax.set_facecolor('#0d1117')

        lon_min, lat_min, lon_max, lat_max = ROI_BOUNDS
        extent = [lon_min, lon_max, lat_min, lat_max]

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
                     markersize=12, zorder=5, label='Sorgente')
        axes[0].set_title(f'AAI grezzo — {dt_str}\n(nessuna plume rilevata)',
                          color='white', fontsize=10)
        axes[0].set_xlabel('Lon', color='white'); axes[0].set_ylabel('Lat', color='white')
        axes[0].tick_params(colors='white')
        axes[0].legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

        # Istogramma dei valori AAI
        valid_vals = aai[~np.isnan(aai)].flatten()
        axes[1].hist(valid_vals, bins=30, color='#378ADD', edgecolor='#0d1117', linewidth=0.5)
        axes[1].axvline(x=0, color='#EF9F27', linewidth=1.5, linestyle='--', label='AAI = 0')
        axes[1].axvline(x=AAI_MIN_THRESHOLD, color='#ff4444', linewidth=1.5,
                        linestyle='--', label=f'Soglia = {AAI_MIN_THRESHOLD}')
        axes[1].set_title('Distribuzione AAI', color='white', fontsize=10)
        axes[1].set_xlabel('AAI', color='white'); axes[1].set_ylabel('Conteggio pixel', color='white')
        axes[1].tick_params(colors='white')
        axes[1].legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        for spine in axes[1].spines.values():
            spine.set_edgecolor('#333344')

        plt.tight_layout()
        diag_name = f'S5P_diagnostic_{i}_{timestamp}.png'
        plt.savefig(diag_name, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.show()
        print(f"  📊 Plot diagnostico salvato: {diag_name}")
        continue

    # ── 4. CONTORNO ─────────────────────────────────────────────────────────
    contours = find_contours(plume_mask.astype(float), level=0.5)

    # ── 5. PCA → ASSE PRINCIPALE DELLA PLUME ────────────────────────────────
    rows, cols = np.where(plume_mask)

    # Converti pixel → gradi geografici
    lons = ROI_BOUNDS[0] + (cols + 0.5) * (ROI_BOUNDS[2] - ROI_BOUNDS[0]) / ncols
    lats = ROI_BOUNDS[3] - (rows + 0.5) * (ROI_BOUNDS[3] - ROI_BOUNDS[1]) / nrows

    # Centroide geografico della plume
    cen_lon = lons.mean()
    cen_lat = lats.mean()

    # Matrice di covarianza → autovettori
    coords_centered = np.stack([lons - cen_lon, lats - cen_lat], axis=1)
    cov_matrix = np.cov(coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Asse principale = autovettore con autovalore maggiore
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]   # (dlon, dlat) normalizzato

    # Estensione dell'asse (2σ in ogni direzione)
    spread = 2.0 * np.sqrt(eigenvalues.max())
    tip_lon = cen_lon + principal_axis[0] * spread
    tip_lat = cen_lat + principal_axis[1] * spread
    tail_lon = cen_lon - principal_axis[0] * spread
    tail_lat = cen_lat - principal_axis[1] * spread

    # ── 6. SCELTA ORIENTAMENTO (dal source verso il tip, non il contrario) ──
    # Se il tip è più lontano dalla sorgente del tail, ok; altrimenti inverti
    dist_tip  = np.hypot(tip_lon  - SOURCE_LON, tip_lat  - SOURCE_LAT)
    dist_tail = np.hypot(tail_lon - SOURCE_LON, tail_lat - SOURCE_LAT)
    if dist_tail > dist_tip:
        tip_lon, tip_lat, tail_lon, tail_lat = tail_lon, tail_lat, tip_lon, tip_lat

    bearing = bearing_deg(SOURCE_LAT, SOURCE_LON, tip_lat, tip_lon)
    compass_dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                    'S','SSW','SW','WSW','W','WNW','NW','NNW']
    compass = compass_dirs[int((bearing + 11.25) / 22.5) % 16]

    plume_length_deg = np.hypot(tip_lon - tail_lon, tip_lat - tail_lat)
    plume_length_km  = plume_length_deg * 111.0   # approssimazione

    print(f"  Centroide plume : ({cen_lat:.4f}°N, {cen_lon:.4f}°E)")
    print(f"  Direzione plume : {bearing:.1f}° ({compass})")
    print(f"  Lunghezza stim. : {plume_length_km:.1f} km")

    # ── 7. PLOT ──────────────────────────────────────────────────────────────
    lon_min, lat_min, lon_max, lat_max = ROI_BOUNDS
    extent = [lon_min, lon_max, lat_min, lat_max]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # -- AAI heatmap
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

    # -- Plume mask (overlay semitrasparente ciano)
    plume_rgba = np.zeros((*plume_mask.shape, 4))
    plume_rgba[plume_mask, 0] = 0.0
    plume_rgba[plume_mask, 1] = 0.9
    plume_rgba[plume_mask, 2] = 0.8
    plume_rgba[plume_mask, 3] = 0.30
    ax.imshow(plume_rgba, extent=extent, origin='upper', zorder=2)

    # -- Contorni plume
    for contour in contours:
        # contour[:,0]=row, contour[:,1]=col → converti in lon/lat
        c_lons = lon_min + (contour[:, 1] + 0.5) * (lon_max - lon_min) / ncols
        c_lats = lat_max - (contour[:, 0] + 0.5) * (lat_max - lat_min) / nrows
        ax.plot(c_lons, c_lats,
                color='#00e5cc', linewidth=1.4, alpha=0.85, zorder=3)

    # -- Asse principale (tratteggiato)
    ax.plot([tail_lon, tip_lon], [tail_lat, tip_lat],
            color='white', linewidth=1.2, linestyle='--', alpha=0.6, zorder=4,
            label=f'Asse principale ({bearing:.0f}° {compass})')

    # -- Freccia direzionale (centroide → tip)
    ax.annotate(
        '',
        xy=(tip_lon, tip_lat),
        xytext=(cen_lon, cen_lat),
        arrowprops=dict(
            arrowstyle='->', color='#ffdd57',
            lw=2.2,
            mutation_scale=18
        ),
        zorder=5
    )

    # -- Sorgente del rogo
    ax.plot(SOURCE_LON, SOURCE_LAT,
            marker='*', markersize=14, color='#ff4444',
            markeredgecolor='white', markeredgewidth=0.8,
            zorder=6, label='Sorgente rogo')

    ax.annotate(
        'Sorgente',
        xy=(SOURCE_LON, SOURCE_LAT),
        xytext=(SOURCE_LON + 0.01, SOURCE_LAT + 0.012),
        color='#ff8888', fontsize=8.5, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2, foreground='black')],
        zorder=7
    )

    # -- Centroide plume
    ax.plot(cen_lon, cen_lat,
            marker='+', markersize=12, color='#00e5cc',
            markeredgewidth=2, zorder=6, label='Centroide plume')

    # -- Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label('Absorbing Aerosol Index (AAI)', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')

    # -- Info box
    info_text = (
        f"Bearing: {bearing:.1f}° ({compass})\n"
        f"Lung. asse: {plume_length_km:.1f} km\n"
        f"Centroide: {cen_lat:.3f}°N  {cen_lon:.3f}°E\n"
        f"Pixel plume: {n_pixels}"
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

    # -- Titolo e assi
    ax.set_title(f'Sentinel-5P AAI — Plume Analysis\n{dt_str}',
                 fontsize=13, fontweight='bold', color='white', pad=10)
    ax.set_xlabel('Longitudine (°E)', color='white', fontsize=10)
    ax.set_ylabel('Latitudine (°N)', color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

    legend = ax.legend(
        loc='upper right', fontsize=8.5,
        facecolor='#1a1a2e', edgecolor='#00e5cc',
        labelcolor='white', framealpha=0.9
    )

    plt.tight_layout()
    out_name = f'S5P_plume_analysis_{i}_{timestamp}.png'
    plt.savefig(out_name, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  ✔  Salvato: {out_name}")