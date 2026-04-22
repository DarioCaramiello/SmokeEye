"""
S5P Sentinel-5P — NO2 Source Zone Validation
Rogo Teano 16/08/2025

Obiettivo: validare la zona vicino alla SORGENTE del rogo usando NO2
troposferico. A differenza dell'AAI (che traccia la plume lontana),
l'NO2 è un tracciante della combustione — picchi elevati vicino alla
sorgente confermano l'attività del rogo nell'area.

Pipeline:
  1. Download NO2 GeoTIFF con maschera cloud
  2. Conversione mol/m² → µmol/m²
  3. Segmentazione anomalia NO2 (percentile 75 locale come soglia)
  4. Pulizia morfologica
  5. Calcolo centroide dell'anomalia + distanza dalla sorgente nota
  6. Plot annotato con marker sorgente e info box
  7. Plot diagnostico se no anomalia rilevata
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
SOURCE_LON = 14.1130910
SOURCE_LAT = 41.2716850

ROI_BOUNDS = [13.85, 41.15, 14.20, 41.37]   # [lon_min, lat_min, lon_max, lat_max]
DATE_START = '2025-08-17'
DATE_END   = '2025-08-18'

# Frazione massima di copertura nuvolosa accettata (0–1)
CLOUD_THRESHOLD = 0.3

# Soglia NO2: percentile locale sopra il quale consideriamo "anomalia"
# 75 = top quartile dei pixel → conservativo, cattura solo i picchi reali
NO2_PERCENTILE_THRESHOLD = 75

# Dimensione minima cluster anomalia (pixel)
MIN_CLUSTER_PIXELS = 10   # più basso di AAI perché la ROI è più piccola

# Raggio di analisi intorno alla sorgente (gradi) per il report
SOURCE_ANALYSIS_RADIUS_DEG = 0.05   # ~5.5 km


# ── EARTH ENGINE SETUP ────────────────────────────────────────────────────────
ee.Authenticate()
ee.Initialize(project='earthenginesmokeeye-488609')

roi = ee.Geometry.Rectangle(ROI_BOUNDS)

collection = (
    ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
    .filterDate(DATE_START, DATE_END)
    .filterBounds(roi)
    .select(['tropospheric_NO2_column_number_density', 'cloud_fraction'])
)

n = collection.size().getInfo()
print(f"Immagini trovate: {n}")

if n == 0:
    print("Nessuna immagine nel periodo. Prova ad allargare DATE_END.")
    exit()

coll_list = collection.toList(n)


# ── HELPER: bearing ───────────────────────────────────────────────────────────
def bearing_deg(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2 - lon1)
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
for i in range(n):
    img_ee = ee.Image(coll_list.get(i))
    info   = img_ee.getInfo()
    ms     = info['properties']['system:time_start']
    dt_str = datetime.datetime.fromtimestamp(ms / 1000).strftime('%Y-%m-%d %H:%M UTC')
    timestamp = img_ee.date().format('YYYY-MM-dd_HH-mm').getInfo()

    print(f"\n{'='*60}")
    print(f"Immagine {i}  —  {dt_str}")

    # ── 1. DOWNLOAD con maschera cloud ───────────────────────────────────────
    img_clean = img_ee.updateMask(
        img_ee.select('cloud_fraction').lt(CLOUD_THRESHOLD)
    )

    url = img_clean.select('tropospheric_NO2_column_number_density').getDownloadURL({
        'scale': 1113,
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

    # Pulizia nodata
    if nodata is not None:
        no2_raw[no2_raw == nodata] = np.nan
    no2_raw[no2_raw < -9000.0] = np.nan

    nrows, ncols = no2_raw.shape

    # ── 2. CONVERSIONE mol/m² → µmol/m² (più leggibile) ─────────────────────
    no2 = no2_raw * 1e6   # µmol/m²

    valid_count = np.sum(~np.isnan(no2))
    print(f"  Grid: {nrows}×{ncols}  |  pixel validi: {valid_count}")
    if valid_count == 0:
        print("  ⚠  Tutti i pixel mascherati (nuvole o nodata). Salto.")
        continue

    print(f"  NO2 min={np.nanmin(no2):.3f}  max={np.nanmax(no2):.3f}  "
          f"media={np.nanmean(no2):.3f}  µmol/m²")

    # ── 3. SEGMENTAZIONE ANOMALIA NO2 ────────────────────────────────────────
    # Per NO2 usiamo il percentile locale come soglia invece di un valore fisso:
    # il background NO2 varia con la stagione, l'ora e l'area geografica.
    # Il percentile 75 cattura solo il top quartile del segnale nella ROI.
    no2_filled = no2.copy()
    no2_filled[np.isnan(no2_filled)] = 0.0

    thresh = np.nanpercentile(no2[no2 > 0], NO2_PERCENTILE_THRESHOLD) \
             if np.any(no2 > 0) else np.nanpercentile(no2, NO2_PERCENTILE_THRESHOLD)

    print(f"  Soglia anomalia NO2: {thresh:.3f} µmol/m² (p{NO2_PERCENTILE_THRESHOLD})")
    anomaly_mask = no2_filled > thresh

    # ── 4. PULIZIA MORFOLOGICA ───────────────────────────────────────────────
    anomaly_mask = binary_fill_holes(anomaly_mask)
    anomaly_mask = opening(anomaly_mask, footprint=disk(1))
    anomaly_mask = closing(anomaly_mask, footprint=disk(1))   # disk più piccolo: ROI più densa

    labeled       = label(anomaly_mask)
    anomaly_mask  = remove_small_objects(labeled > 0, min_size=MIN_CLUSTER_PIXELS + 1)

    n_pixels = anomaly_mask.sum()
    print(f"  Pixel anomalia dopo pulizia: {n_pixels}")

    # ── DIAGNOSTICA se anomalia assente ──────────────────────────────────────
    if n_pixels < MIN_CLUSTER_PIXELS:
        cloud_pct_info = info['properties'].get('cloud_fraction', 'n/d')
        print(f"  ⚠  Nessuna anomalia NO2 rilevata.")
        print(f"     Possibili cause:")
        print(f"       • copertura nuvolosa > {CLOUD_THRESHOLD*100:.0f}% (prova ad alzare CLOUD_THRESHOLD)")
        print(f"       • rogo non ancora attivo o segnale NO2 debole")
        print(f"       • prova NO2_PERCENTILE_THRESHOLD più basso (attuale: {NO2_PERCENTILE_THRESHOLD})")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor('#0d1117')
        for ax in axes:
            ax.set_facecolor('#0d1117')

        lon_min, lat_min, lon_max, lat_max = ROI_BOUNDS
        extent = [lon_min, lon_max, lat_min, lat_max]

        im = axes[0].imshow(no2, cmap='YlOrRd',
                            vmin=np.nanpercentile(no2, 2),
                            vmax=np.nanpercentile(no2, 98),
                            extent=extent, origin='upper')
        cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label('NO2 (µmol/m²)', color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        axes[0].plot(SOURCE_LON, SOURCE_LAT, marker='*', color='red',
                     markersize=12, zorder=5, label='Sorgente')
        axes[0].set_title(f'NO2 grezzo — {dt_str}\n(nessuna anomalia rilevata)',
                          color='white', fontsize=10)
        axes[0].set_xlabel('Lon', color='white')
        axes[0].set_ylabel('Lat', color='white')
        axes[0].tick_params(colors='white')
        axes[0].legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

        valid_vals = no2[~np.isnan(no2)].flatten()
        axes[1].hist(valid_vals, bins=30, color='#EF9F27',
                     edgecolor='#0d1117', linewidth=0.5)
        axes[1].axvline(x=thresh, color='#ff4444', linewidth=1.5,
                        linestyle='--', label=f'Soglia p{NO2_PERCENTILE_THRESHOLD} = {thresh:.3f}')
        axes[1].set_title('Distribuzione NO2', color='white', fontsize=10)
        axes[1].set_xlabel('NO2 (µmol/m²)', color='white')
        axes[1].set_ylabel('Conteggio pixel', color='white')
        axes[1].tick_params(colors='white')
        axes[1].legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        for spine in axes[1].spines.values():
            spine.set_edgecolor('#333344')

        plt.suptitle(f'Diagnostica NO2 — {dt_str}', color='white',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        diag_name = f'S5P_NO2_diagnostic_{i}_{timestamp}.png'
        plt.savefig(diag_name, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.show()
        print(f"  📊 Plot diagnostico salvato: {diag_name}")
        continue

    # ── 5. ANALISI SPAZIALE ANOMALIA ─────────────────────────────────────────
    rows_a, cols_a = np.where(anomaly_mask)

    lons_a = ROI_BOUNDS[0] + (cols_a + 0.5) * (ROI_BOUNDS[2] - ROI_BOUNDS[0]) / ncols
    lats_a = ROI_BOUNDS[3] - (rows_a + 0.5) * (ROI_BOUNDS[3] - ROI_BOUNDS[1]) / nrows

    # Centroide dell'anomalia (pesato per valore NO2 = centroide "caldo")
    weights   = no2_filled[rows_a, cols_a]
    cen_lon   = np.average(lons_a, weights=weights)
    cen_lat   = np.average(lats_a, weights=weights)

    # Distanza centroide ↔ sorgente
    dist_km = np.hypot(cen_lon - SOURCE_LON, cen_lat - SOURCE_LAT) * 111.0
    bear     = bearing_deg(SOURCE_LAT, SOURCE_LON, cen_lat, cen_lon)
    compass_dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                    'S','SSW','SW','WSW','W','WNW','NW','NNW']
    compass  = compass_dirs[int((bear + 11.25) / 22.5) % 16]

    # Pixel anomalia entro SOURCE_ANALYSIS_RADIUS_DEG dalla sorgente
    dist_from_source = np.hypot(lons_a - SOURCE_LON, lats_a - SOURCE_LAT)
    n_near_source    = np.sum(dist_from_source <= SOURCE_ANALYSIS_RADIUS_DEG)
    pct_near         = 100.0 * n_near_source / n_pixels

    # NO2 medio nella zona sorgente vs resto della ROI
    source_mask = np.hypot(
        (np.arange(ncols)[None, :] + 0.5) * (ROI_BOUNDS[2]-ROI_BOUNDS[0])/ncols + ROI_BOUNDS[0] - SOURCE_LON,
        ROI_BOUNDS[3] - (np.arange(nrows)[:, None] + 0.5) * (ROI_BOUNDS[3]-ROI_BOUNDS[1])/nrows - SOURCE_LAT
    ) <= SOURCE_ANALYSIS_RADIUS_DEG

    no2_source_zone = no2[source_mask & ~np.isnan(no2)]
    no2_outside     = no2[~source_mask & ~np.isnan(no2)]
    no2_mean_source = np.mean(no2_source_zone) if no2_source_zone.size > 0 else np.nan
    no2_mean_bg     = np.mean(no2_outside)     if no2_outside.size > 0     else np.nan
    enrichment      = (no2_mean_source / no2_mean_bg) if (no2_mean_bg and no2_mean_bg > 0) else np.nan

    print(f"  Centroide anomalia  : ({cen_lat:.4f}°N, {cen_lon:.4f}°E)")
    print(f"  Distanza da sorgente: {dist_km:.1f} km ({bear:.0f}° {compass})")
    print(f"  Pixel vicino sorgente (r={SOURCE_ANALYSIS_RADIUS_DEG}°): "
          f"{n_near_source}/{n_pixels} ({pct_near:.0f}%)")
    print(f"  NO2 medio zona sorgente : {no2_mean_source:.3f} µmol/m²")
    print(f"  NO2 medio background    : {no2_mean_bg:.3f} µmol/m²")
    if not np.isnan(enrichment):
        print(f"  Arricchimento NO2       : {enrichment:.2f}x")
        if enrichment > 1.3:
            print(f"  ✔  Segnale NO2 elevato vicino alla sorgente — coerente con rogo attivo")
        else:
            print(f"  x  Arricchimento debole — rogo poco intenso o segnale diluito")

    # ── 6. CONTORNI ──────────────────────────────────────────────────────────
    contours = find_contours(anomaly_mask.astype(float), level=0.5)

    # ── 7. PLOT PRINCIPALE ───────────────────────────────────────────────────
    lon_min, lat_min, lon_max, lat_max = ROI_BOUNDS
    extent = [lon_min, lon_max, lat_min, lat_max]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # -- NO2 heatmap (YlOrRd: giallo=basso, rosso=alto)
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

    # -- Anomaly mask overlay (arancio semitrasparente)
    anomaly_rgba = np.zeros((*anomaly_mask.shape, 4))
    anomaly_rgba[anomaly_mask, 0] = 1.0
    anomaly_rgba[anomaly_mask, 1] = 0.55
    anomaly_rgba[anomaly_mask, 2] = 0.0
    anomaly_rgba[anomaly_mask, 3] = 0.28
    ax.imshow(anomaly_rgba, extent=extent, origin='upper', zorder=2)

    # -- Cerchio zona analisi sorgente
    circle = plt.Circle(
        (SOURCE_LON, SOURCE_LAT),
        SOURCE_ANALYSIS_RADIUS_DEG,
        color='#ffdd57', fill=False,
        linewidth=1.2, linestyle='--', alpha=0.7, zorder=4,
        label=f'Zona analisi (r≈{SOURCE_ANALYSIS_RADIUS_DEG*111:.0f} km)'
    )
    ax.add_patch(circle)

    # -- Contorni anomalia
    for contour in contours:
        c_lons = lon_min + (contour[:, 1] + 0.5) * (lon_max - lon_min) / ncols
        c_lats = lat_max - (contour[:, 0] + 0.5) * (lat_max - lat_min) / nrows
        ax.plot(c_lons, c_lats, color='#ff8c00', linewidth=1.4,
                alpha=0.85, zorder=3)

    # -- Centroide anomalia
    ax.plot(cen_lon, cen_lat, marker='+', markersize=13, color='#ff8c00',
            markeredgewidth=2.2, zorder=6, label='Centroide anomalia NO2')

    # -- Sorgente rogo
    ax.plot(SOURCE_LON, SOURCE_LAT, marker='*', markersize=15,
            color='#ff3333', markeredgecolor='white', markeredgewidth=0.8,
            zorder=7, label='Sorgente rogo')
    ax.annotate(
        'Sorgente',
        xy=(SOURCE_LON, SOURCE_LAT),
        xytext=(SOURCE_LON + 0.01, SOURCE_LAT + 0.013),
        color='#ff8888', fontsize=8.5, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2, foreground='black')],
        zorder=8
    )

    # -- Linea centroide → sorgente
    ax.plot([SOURCE_LON, cen_lon], [SOURCE_LAT, cen_lat],
            color='white', linewidth=0.8, linestyle=':', alpha=0.5, zorder=5)

    # -- Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label('NO₂ troposferico (µmol/m²)', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')

    # -- Info box
    enrich_str = f"{enrichment:.2f}x" if not np.isnan(enrichment) else "n/d"
    info_text = (
        f"Centroide anomalia\n"
        f"  {cen_lat:.4f}°N  {cen_lon:.4f}°E\n"
        f"Dist. sorgente: {dist_km:.1f} km ({bear:.0f}° {compass})\n"
        f"Pixel anomalia: {n_pixels}  |  vicino src: {pct_near:.0f}%\n"
        f"NO2 sorgente:  {no2_mean_source:.3f} µmol/m²\n"
        f"NO2 background: {no2_mean_bg:.3f} µmol/m²\n"
        f"Arricchimento:  {enrich_str}"
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

    # -- Titolo e assi
    ax.set_title(
        f'Sentinel-5P NO₂ — Validazione zona sorgente\n{dt_str}',
        fontsize=12, fontweight='bold', color='white', pad=10
    )
    ax.set_xlabel('Longitudine (°E)', color='white', fontsize=10)
    ax.set_ylabel('Latitudine (°N)', color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

    ax.legend(loc='upper right', fontsize=8.5,
              facecolor='#1a1a2e', edgecolor='#ff8c00',
              labelcolor='white', framealpha=0.9)

    plt.tight_layout()
    out_name = f'S5P_NO2_source_validation_{i}_{timestamp}.png'
    plt.savefig(out_name, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  ✔  Salvato: {out_name}")