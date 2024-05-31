# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:10:29 2023

@author: lindgrv1
"""

import rasterio
import rasterio.mask
from shapely.geometry import box
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os

##############################################################################
# Create bounding box

radar_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\ARTIKKELI_2\data_a2\kiira_radar"
file_list1 = os.listdir(radar_dir)
file_list1 = [x for x in file_list1 if ".tif" in x]
file_list1 = [f for f in file_list1 if f.endswith(".tif")]

temp_raster = rasterio.open(os.path.join(radar_dir, file_list1[0])) #import one rain field
print(temp_raster.crs) #coordinate reference system of the raster
temp_bounds = temp_raster.bounds #raster corner coordinates
bbox = box(*temp_bounds) #raster to GeoDataFrame
# print(bbox.wkt)
bbox_df = gpd.GeoDataFrame({"geometry":[bbox]}, crs=temp_raster.crs)

##############################################################################
# Mask the input data 

indata_dir = r"W:\lindgrv1\Kiira_whole_day\12\dbz"
file_list2 = os.listdir(indata_dir)
file_list2 = [x for x in file_list2 if ".tif" in x]
file_list2 = [f for f in file_list2 if f.endswith(".tif")]

# (14:05) 15:00 - 18:00 (22:00)
file_list3 = file_list2[169:265]

out_dir = r"W:\lindgrv1\Kiira_whole_day\kiira_14-22_dbz"

for i in range(len(file_list3)):
    with rasterio.open(os.path.join(indata_dir, file_list3[i])) as src:
        out_image, out_transform = rasterio.mask.mask(src, [bbox], crop=True)
        out_meta = src.meta
    
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(os.path.join(out_dir, f"kiira_{file_list3[i]}"), "w", **out_meta) as dest:
        dest.write(out_image)
        
##############################################################################
# Create tif for fft-filter

bbox_big = box(minx=131412.26858513, miny=6429589.44575806, maxx=643420.498961918, maxy=6941618.08472581)
bbox_big_df = gpd.GeoDataFrame({"geometry":[bbox_big]}, crs=temp_raster.crs)

out_dir_big = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"

with rasterio.open(os.path.join(indata_dir, "Radar-suomi_dbz_eureffin_1208171635.tif")) as src:
    out_image, out_transform = rasterio.mask.mask(src, [bbox_big], crop=True)
    out_meta = src.meta

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open(os.path.join(out_dir_big, "fft_filter_tif_kiira.tif"), "w", **out_meta) as dest:
    dest.write(out_image)

##############################################################################
# KANSIKUVA

POLYGON ((515418.4413677212 6557596.605500001, 515418.4413677212 6813610.924983871, 259414.3261793271 6813610.924983871, 259414.3261793271 6557596.605500001, 515418.4413677212 6557596.605500001))
POLYGON ((643420.4989619181 6429589.44575806, 643420.4989619181 6941618.08472581, 131412.26858513 6941618.08472581, 131412.26858513 6429589.44575806, 643420.4989619181 6429589.44575806))

maxx-miny, maxx-maxy, minx-maxy, minx-miny, maxx-miny

#isompi neliö
643420.498961918-131412.26858513 #maxx-minx
6941618.08472581-6429589.44575806 #maxy-miny

#pienempi
515418.4413677212-259414.3261793271 #maxx-minx
6813610.924983871-6557596.605500001 #maxy-miny

#kantta varten uusi isoin neliö
maxx_iso = 643420.498961918 + 256000
minx_iso = 131412.26858513 - 256000
maxy_iso = 6941618.08472581 + 256000
miny_iso = 6429589.44575806 - 256000

# maxx_iso-minx_iso
# maxy_iso-miny_iso

bbox_kansi = box(minx=minx_iso, miny=miny_iso, maxx=maxx_iso, maxy=maxy_iso)
bbox_kansi_df = gpd.GeoDataFrame({"geometry":[bbox_kansi]}, crs=temp_raster.crs)

out_dir_kansi = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"

with rasterio.open(os.path.join(indata_dir, "Radar-suomi_dbz_eureffin_1208171635.tif")) as src:
    out_image, out_transform = rasterio.mask.mask(src, [bbox_kansi], crop=True)
    out_meta = src.meta

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open(os.path.join(out_dir_kansi, "kansi.tif"), "w", **out_meta) as dest:
    dest.write(out_image)

##############################################################################
# Calculate areal mean rainfall time series and find starting and ending timesteps for the event

input_dir = r"W:\lindgrv1\Kiira_whole_day\kiira_14-21_dbz"
files_input_dir = os.listdir(input_dir)
files_input_dir = [x for x in files_input_dir if ".tif" in x]
files_input_dir = [f for f in files_input_dir if f.endswith(".tif")]

radar_kiira = []
for i in range(len(files_input_dir)):
    src = rasterio.open(os.path.join(input_dir, files_input_dir[i]))
    array = src.read(1)
    radar_kiira.append(array)
    
radar_kiira = np.concatenate([radar_kiira_[None, :, :] for radar_kiira_ in radar_kiira])
    
#This have to be added for new event data: Remove last column from each layer
radar_kiira = radar_kiira[:,:,:-1]

#The following data is available for Finnish radar composite: radar reflectivity (dbz), conversion: Z[dBZ] = 0.5 * pixel value - 32
radar_kiira = (radar_kiira * 0.5) - 32

#Values less than threshold to wanted value, which represents no-rain
radar_kiira[radar_kiira < 10] = 3.1830486304816077

#Areal mean rainfall in dbz
R = radar_kiira.copy()
areal_rainfall_ts = np.zeros(len(files_input_dir))
for i in range (len(files_input_dir)):
    areal_rainfall_ts[i] = np.nanmean(R[i])
#Plot the time series
plt.figure()
plt.plot(areal_rainfall_ts)
plt.axhline(y = 5, color = "black", linestyle = "-", linewidth=0.5)
plt.title("Areal mean rainfall (dBZ)")

# -> Start of the event: timestep when areal mean rainfall > 5 dBZ is 9 -> first image is 14:45
# -> End of the event: timestep when areal mean rainfall < 5 dBZ is 97 -> last image is 22:00
