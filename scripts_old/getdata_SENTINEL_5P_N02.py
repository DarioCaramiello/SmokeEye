import ee
import time
import requests, io
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Auth
ee.Authenticate()
ee.Initialize(project='earthenginesmokeeye-488609')

# Area of interest 
# Teano - 16/08/2025  2025-08-16 / 2025-08-18
roi = ee.Geometry.Rectangle([
    13.85,   # lon_min
    41.15,   # lat_min
    14.20,   # lon_max
    41.37    # lat_max
])

img = (
    ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
    .filterDate('2025-08-18', '2025-08-19')
    .filterBounds(roi)
    .select(['absorbing_aerosol_index', 'tropospheric_NO2_column_number_density', 'cloud_fraction'])
)

# Tot of response image 
n = img.size().getInfo()
print(f"Tot images : {n}")

# Transform the response into a collection of images 
coll_img = img.toList(n)

# Itaration on all response images 
for i in range(n):
    img = ee.Image(coll_img.get(i))
    info = img.getInfo()
    ms = info['properties']['system:time_start']
    dt = datetime.datetime.fromtimestamp(ms / 1000).strftime('%Y-%m-%d %H:%M:%S')

    print(f"Considering image index : {i}")
    print(f"------ info : {info}")
    print(f"------ dt : {dt}")

    timestamp = img.date().format('YYYY-MM-dd').getInfo()
    filename  = f'S5P_NO2_{i}_{timestamp}'

    # Mask Clouds
    img_clean = img.updateMask(img.select('cloud_fraction').lt(0.3))

    url = img_clean.select('tropospheric_NO2_column_number_density').getDownloadURL({
	'scale': 1113,
	'region': roi,
	'format': 'GEO_TIFF',
	'crs': 'EPSG:4326'
    })

    print("------ download image")
    response = requests.get(url)
    response.raise_for_status() 

    with MemoryFile(response.content) as memfile:
        with memfile.open() as dataset:
            aai =  dataset.read(1).astype(float)
            transform = dataset.transform,
            crs = dataset.crs,
            nodata = dataset.nodata

    # Mask NoData
    if nodata is not None:
        aai[aai == nodata] = np.nan
    aai[aai < -9000.0] = np.nan


    print(f"Shape dati: {aai.shape}")
    print(f"Min AAI: {np.nanmin(aai):.3f}")
    print(f"Max AAI: {np.nanmax(aai):.3f}")
    print(f"Media AAI: {np.nanmean(aai):.3f}")

    # View 
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        aai,
        cmap='RdYlBu_r',          # rosso = AAI alto (più aerosol assorbenti)
        vmin=np.nanpercentile(aai, 2),
        vmax=np.nanpercentile(aai, 98),
        extent=[13.85, 14.20, 41.15, 41.37],
        origin='upper'
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Tropospheric NO2 column (mol/m²)', fontsize=11)

    ax.set_title(f'Sentinel-5P NO2 – \n{dt}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitudine')
    ax.set_ylabel('Latitudine')
    ax.legend()

    #TODO: aggiungere la sorgente del rogo
    ax.plot(14.1130910, 41.2716850, marker='*', color='red', zorder=10, label='Source')

    plt.tight_layout()
    plt.savefig(f'S5P_N02_{i}_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot salvato come: S5P_N02_{i}_{timestamp}.png")


