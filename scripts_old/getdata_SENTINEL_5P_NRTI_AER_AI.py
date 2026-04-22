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
# Torre del Greco 
'''
roi = ee.Geometry.Rectangle([
    14.35,  # lon_min (ovest)
    40.77,  # lat_min (sud)
    14.48,  # lon_max (est)
    40.87   # lat_max (nord)
])
'''
# Vico Equense Tordigliano -  2024-08-18
#roi = ee.Geometry.Rectangle([
#    14.40,  # lon_min (ovest)
#    40.63,  # lat_min (sud)
#    14.50,  # lon_max (est)
#    40.70   # lat_max (nord)
#])

# Teano - 16/08/2025  2025-08-16 / 2025-08-18
roi = ee.Geometry.Rectangle([
    13.85,   # lon_min
    41.15,   # lat_min
    14.20,   # lon_max
    41.37    # lat_max
])


img = (
    ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI')
    .filterDate('2025-08-16', '2025-08-17')
    .filterBounds(roi)
    .select(['absorbing_aerosol_index', 'sensor_altitude'])
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

    print("Considering image index : {i}")
    print(f"------ info : {info}")
    print(f"------ dt : {dt}")

    timestamp = img.date().format('YYYY-MM-dd').getInfo()
    filename  = f'S5P_AER_AI_{i}_{timestamp}'

    url = img.select('absorbing_aerosol_index').getDownloadURL({
	'scale': 3500,
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
    cbar.set_label('Absorbing Aerosol Index (AAI)', fontsize=11)

    ax.set_title(f'Sentinel-5P AAI - \n{dt}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitudine')
    ax.set_ylabel('Latitudine')
    ax.legend()

    ax.plot(14.1130910, 41.2716850, marker='*', color='red', zorder=10, label='Source')

    plt.tight_layout()
    plt.savefig(f'S5P_AAI_{i}_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot salvato come: S5P_AAI_{i}_{timestamp}.png")


    # To directly export the response image on Google Drive
    '''
    # Export on Google Drive
    task = ee.batch.Export.image.toDrive(
        image          = img,
        description    = filename,
        folder         = 'GEE_Vesuvio_AER_AI',
        fileNamePrefix = filename,
        scale          = 3500,
        region         = roi,
    a    fileFormat     = 'GeoTIFF',
        maxPixels      = 1e9,
        crs            = 'EPSG:4326'
    )

    task.start()
    print(f"Task avviato: {task.id}")

    while True:
        status = task.status()['state']
        print(f"Stato: {status}")
        if status in ('COMPLETED', 'FAILED', 'CANCELLED'):
            break
        time.sleep(15)

    #Risultato finale
    if status == 'COMPLETED':
        print(f"\nFile salvato su Google Drive nella cartella: GEE_Vesuvio_AER_AI")
        print(f"Nome file: {filename}.tif")
    else:
        err = task.status().get('error_message', 'sconosciuto')
        print(f"\nTask fallito. Errore: {err}")
    '''

