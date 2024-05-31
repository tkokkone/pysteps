# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:03:10 2023

@author: lindgrv1
"""

import os
import wradlib as wrl
import wradlib.adjust as adjust
import wradlib.verify as verify
import wradlib.util as util
# https://docs.wradlib.org/en/stable/adjust.html
# https://debug-docs.readthedocs.io/en/test/notebooks/multisensor/wradlib_adjust_example.html
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from rasterstats import point_query
import rasterstats

##############################################################################

# import random

# random_field = np.random.randint(low=0, high=10, size=(10,10), dtype='l')
# random_field2 = np.hstack(random_field)

# random_field3 = tuple([dim.ravel() for dim in reversed(np.meshgrid(random_field2, indexing="ij"))])
# random_field4 = np.vstack(random_field3).transpose()

# random_field5 = random_field.reshape([(random_field.shape[0]*random_field.shape[1]), ])

# random_field_arr = util.gridaspoints(random_field)

##############################################################################

radar_dir = r"W:\lindgrv1\Kiira_whole_day\kiira_14-22_dbz"

file_list = os.listdir(radar_dir)
file_list = [x for x in file_list if ".tif" in x]
file_list = [f for f in file_list if f.endswith(".tif")]

radar_kiira = []
for i in range(len(file_list)):
    src = rasterio.open(os.path.join(radar_dir, file_list[i]))
    array = src.read(1)
    radar_kiira.append(array)
    
radar_kiira = np.concatenate([radar_kiira_[None, :, :] for radar_kiira_ in radar_kiira])

#This have to be added for new event data: Remove last column from each layer
radar_kiira = radar_kiira[:,:,:-1]
#The following data is available for Finnish radar composite: radar reflectivity (dbz), conversion: Z[dBZ] = 0.5 * pixel value - 32
radar_kiira = (radar_kiira * 0.5) - 32
#Values less than threshold to wanted value, which represents no-rain
radar_kiira[radar_kiira < 10] = 3.1830486304816077

# radar_field = radar_kiira[25]
radar_field = radar_kiira[32]
radar_field_dbz = radar_kiira[32]

a_R=223
b_R=1.53
radar_field = 10**((radar_field-10*np.log10(a_R))/(10*b_R))
#Values less than threshold to zero
radar_field[radar_field < 0.1] = 0

plt.figure()
plt.imshow(radar_field, cmap="nipy_spectral")
plt.figure()
plt.imshow(radar_field_dbz, cmap="nipy_spectral")

radar_field2 = radar_field.reshape([(radar_field.shape[0]*radar_field.shape[1]), ])

##############################################################################

# #extra files for gauge adjustments
# gauge_adj_dir = r"W:\lindgrv1\Kiira_whole_day\kiira_for_gauge_adj"
# gauge_adj_files = os.listdir(gauge_adj_dir)
# gauge_adj_files = [x for x in gauge_adj_files if ".tif" in x]
# gauge_adj_files = [f for f in gauge_adj_files if f.endswith(".tif")]

temp_raster = rasterio.open(os.path.join(radar_dir, file_list[0])) #import one rain field
print(temp_raster.crs) #coordinate reference system of the raster
temp_bounds = temp_raster.bounds #raster corner coordinates
bbox = box(*temp_bounds) #raster to GeoDataFrame
bbox_df = gpd.GeoDataFrame({"geometry":[bbox]}, crs=temp_raster.crs)

# # for i in range(len(gauge_adj_files)):
# #     with rasterio.open(os.path.join(gauge_adj_dir, gauge_adj_files[i])) as src:
# #         out_image, out_transform = rasterio.mask.mask(src, [bbox], crop=True)
# #         out_meta = src.meta
# #     out_meta.update({"driver": "GTiff",
# #                      "height": out_image.shape[1],
# #                      "width": out_image.shape[2],
# #                      "transform": out_transform})
# #     with rasterio.open(os.path.join(gauge_adj_dir, f"crop_{gauge_adj_files[i]}"), "w", **out_meta) as dest:
# #         dest.write(out_image)

# gauge_adj_files_crop = os.listdir(gauge_adj_dir)
# gauge_adj_files_crop = [x for x in gauge_adj_files_crop if ".tif" in x]
# gauge_adj_files_crop = [f for f in gauge_adj_files_crop if f.endswith(".tif")]
# gauge_adj_files_crop = [e for e in gauge_adj_files_crop if "crop" in e]

# gauge_adj_fields = []
# for i in range(len(gauge_adj_files_crop)):
#     src_gauge_adj = rasterio.open(os.path.join(gauge_adj_dir, gauge_adj_files_crop[i]))
#     array_gauge_adj = src_gauge_adj.read(1)
#     gauge_adj_fields.append(array_gauge_adj)
    
# gauge_adj_fields = np.concatenate([gauge_adj_fields_[None, :, :] for gauge_adj_fields_ in gauge_adj_fields])
# gauge_adj_fields = gauge_adj_fields[:,:,1:-1]

# gauge_adj_fields = (gauge_adj_fields * 0.5) - 32
# gauge_adj_fields[gauge_adj_fields < 10] = 3.1830486304816077
# gauge_adj_fields = 10**((gauge_adj_fields-10*np.log10(a_R))/(10*b_R))
# gauge_adj_fields[gauge_adj_fields < 0.1] = 0

##############################################################################
#Hourly accumulation maps

#files to calculate hourly accumulation fields
file_list_14_15 = file_list[:12]
file_list_15_16 = file_list[12:24]
file_list_16_17 = file_list[24:36]
file_list_17_18 = file_list[36:48]
file_list_18_19 = file_list[48:60]
file_list_19_20 = file_list[60:72]
file_list_20_21 = file_list[72:84]
file_list_21_22 = file_list[84:]

#fields for every hour
radar_kiira_14_15 = radar_kiira[:12]
radar_kiira_15_16 = radar_kiira[12:24]
radar_kiira_16_17 = radar_kiira[24:36]
radar_kiira_17_18 = radar_kiira[36:48]
radar_kiira_18_19 = radar_kiira[48:60]
radar_kiira_19_20 = radar_kiira[60:72]
radar_kiira_20_21 = radar_kiira[72:84]
radar_kiira_21_22 = radar_kiira[84:]

#dbz to mm/h
radar_kiira_mm_14_15 = 10**((radar_kiira_14_15-10*np.log10(a_R))/(10*b_R))
radar_kiira_mm_15_16 = 10**((radar_kiira_15_16-10*np.log10(a_R))/(10*b_R))
radar_kiira_mm_16_17 = 10**((radar_kiira_16_17-10*np.log10(a_R))/(10*b_R))
radar_kiira_mm_17_18 = 10**((radar_kiira_17_18-10*np.log10(a_R))/(10*b_R))
radar_kiira_mm_18_19 = 10**((radar_kiira_18_19-10*np.log10(a_R))/(10*b_R))
radar_kiira_mm_19_20 = 10**((radar_kiira_19_20-10*np.log10(a_R))/(10*b_R))
radar_kiira_mm_20_21 = 10**((radar_kiira_20_21-10*np.log10(a_R))/(10*b_R))
radar_kiira_mm_21_22 = 10**((radar_kiira_21_22-10*np.log10(a_R))/(10*b_R))

#threshold 0.1
radar_kiira_mm_14_15[radar_kiira_mm_14_15 < 0.1] = 0
radar_kiira_mm_15_16[radar_kiira_mm_15_16 < 0.1] = 0
radar_kiira_mm_16_17[radar_kiira_mm_16_17 < 0.1] = 0
radar_kiira_mm_17_18[radar_kiira_mm_17_18 < 0.1] = 0
radar_kiira_mm_18_19[radar_kiira_mm_18_19 < 0.1] = 0
radar_kiira_mm_19_20[radar_kiira_mm_19_20 < 0.1] = 0
radar_kiira_mm_20_21[radar_kiira_mm_20_21 < 0.1] = 0
radar_kiira_mm_21_22[radar_kiira_mm_21_22 < 0.1] = 0

# radar_kiira_mm_14_15_extras = np.concatenate((gauge_adj_fields,radar_kiira_mm_14_15))

#accumulation maps
radar_kiira_mm_14_15_accu = np.sum(radar_kiira_mm_14_15, axis=0) * (5/60)
radar_kiira_mm_15_16_accu = np.sum(radar_kiira_mm_15_16, axis=0) * (5/60)
radar_kiira_mm_16_17_accu = np.sum(radar_kiira_mm_16_17, axis=0) * (5/60)
radar_kiira_mm_17_18_accu = np.sum(radar_kiira_mm_17_18, axis=0) * (5/60)
radar_kiira_mm_18_19_accu = np.sum(radar_kiira_mm_18_19, axis=0) * (5/60)
radar_kiira_mm_19_20_accu = np.sum(radar_kiira_mm_19_20, axis=0) * (5/60)
radar_kiira_mm_20_21_accu = np.sum(radar_kiira_mm_20_21, axis=0) * (5/60)
radar_kiira_mm_21_22_accu = np.sum(radar_kiira_mm_21_22, axis=0) * (5/60)

# plt.figure()
# plt.imshow(radar_kiira_mm_14_15_accu, cmap="nipy_spectral")

#max grid accumulation
maxaccu = np.max(np.concatenate((radar_kiira_mm_14_15_accu, radar_kiira_mm_15_16_accu, radar_kiira_mm_16_17_accu, radar_kiira_mm_17_18_accu, 
                                 radar_kiira_mm_18_19_accu, radar_kiira_mm_19_20_accu, radar_kiira_mm_20_21_accu, radar_kiira_mm_21_22_accu)).ravel())
#open figure
fig = plt.figure(figsize=(10, 6))
#14-15
ax = fig.add_subplot(241, aspect="equal")
plt.imshow(radar_kiira_mm_14_15_accu, cmap="nipy_spectral", vmin=0, vmax=maxaccu)
plt.title("14-15")
plt.colorbar(shrink=0.75)
#15-16
ax = fig.add_subplot(242, aspect="equal")
plt.imshow(radar_kiira_mm_15_16_accu, cmap="nipy_spectral", vmin=0, vmax=maxaccu)
plt.title("15-16")
plt.colorbar(shrink=0.75)
#16-17
ax = fig.add_subplot(243, aspect="equal")
plt.imshow(radar_kiira_mm_16_17_accu, cmap="nipy_spectral", vmin=0, vmax=maxaccu)
plt.title("16-17")
plt.colorbar(shrink=0.75)
#17_18
ax = fig.add_subplot(244, aspect="equal")
plt.imshow(radar_kiira_mm_17_18_accu, cmap="nipy_spectral", vmin=0, vmax=maxaccu)
plt.title("17_18")
plt.colorbar(shrink=0.75)
#18-19
ax = fig.add_subplot(245, aspect="equal")
plt.imshow(radar_kiira_mm_18_19_accu, cmap="nipy_spectral", vmin=0, vmax=maxaccu)
plt.title("18-19")
plt.colorbar(shrink=0.75)
#19-20
ax = fig.add_subplot(246, aspect="equal")
plt.imshow(radar_kiira_mm_19_20_accu, cmap="nipy_spectral", vmin=0, vmax=maxaccu)
plt.title("19-20")
plt.colorbar(shrink=0.75)
#20-21
ax = fig.add_subplot(247, aspect="equal")
plt.imshow(radar_kiira_mm_20_21_accu, cmap="nipy_spectral", vmin=0, vmax=maxaccu)
plt.title("20-21")
plt.colorbar(shrink=0.75)
#21-22
ax = fig.add_subplot(248, aspect="equal")
plt.imshow(radar_kiira_mm_21_22_accu, cmap="nipy_spectral", vmin=0, vmax=maxaccu)
plt.title("21-22")
plt.colorbar(shrink=0.75)

##############################################################################
#Gauge data
gauge_data_dir = r"W:\lindgrv1\Kiira_whole_day"
gauge_data = pd.read_csv(os.path.join(gauge_data_dir, "kiira_gauge_mm.csv"), delimiter=(";"))

data_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/GIS-aineistot" 
mittarit = gpd.read_file(os.path.join(data_dir, "FMI_stations_2022-03.gpkg"))
# mittarit.crs
mittarit_tutkakuva = gpd.clip(mittarit, bbox_df)
dir_gauges_kiira = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\ARTIKKELI_2\data_a2\mittarit"
fp_gauges_kiira = os.path.join(dir_gauges_kiira, "gauges_kiira.csv")
mittarit_tutkakuva_data = pd.read_csv(fp_gauges_kiira, delimiter=(";"))
mittarit_tutkakuva_kiira = mittarit_tutkakuva[mittarit_tutkakuva["field_1"].isin(mittarit_tutkakuva_data["Station"])]

test_matrix = np.arange(0, radar_kiira.shape[1]*radar_kiira.shape[2]).reshape(radar_kiira.shape[1], radar_kiira.shape[2])
test_affine = temp_raster.transform
mittarit_tutkakuva_kiira = mittarit_tutkakuva_kiira.to_crs(temp_raster.crs)
temp_point = point_query(mittarit_tutkakuva_kiira, test_matrix, affine=test_affine, nodata=-999)
mittarit_tutkakuva_kiira["point"] = temp_point

outrows = np.zeros(len(mittarit_tutkakuva_kiira))
outcols = np.zeros(len(mittarit_tutkakuva_kiira))
for i in range(len(mittarit_tutkakuva_kiira)):
    outrow = np.where(np.any(test_matrix == int(mittarit_tutkakuva_kiira.iloc[i]["point"]), axis = 1))
    outrows[i] = int(outrow[0])
    outcol = np.where(np.any(test_matrix == int(mittarit_tutkakuva_kiira.iloc[i]["point"]), axis = 0))
    outcols[i] = int(outcol[0])
mittarit_tutkakuva_kiira["row"] = outrows
mittarit_tutkakuva_kiira["col"] = outcols

point_laune = Point(25.6309, 60.96211)
point_laune_df = gpd.GeoDataFrame({"geometry":[point_laune]}, crs="epsg:4326")
point_laune_df.crs
point_laune_df = point_laune_df.to_crs(temp_raster.crs)
temp_point = point_query(point_laune_df, test_matrix, affine=test_affine, nodata=-999)
point_laune_df["point"] = temp_point
outrow_laune = np.where(np.any(test_matrix == int(point_laune_df.iloc[0]["point"]), axis = 1))
outcol_laune = np.where(np.any(test_matrix == int(point_laune_df.iloc[0]["point"]), axis = 0))
point_laune_df["row"] = int(outrow_laune[0])
point_laune_df["col"] = int(outcol_laune[0])
point_laune_df["id"] = int(101152)

##############################################################################
# Hourly radar accumulations from gauge locations
hourly_accus = np.zeros((22, 8))

temp_accus_14_15 = point_query(mittarit_tutkakuva_kiira, radar_kiira_mm_14_15_accu, affine=test_affine)
temp_accus_15_16 = point_query(mittarit_tutkakuva_kiira, radar_kiira_mm_15_16_accu, affine=test_affine)
temp_accus_16_17 = point_query(mittarit_tutkakuva_kiira, radar_kiira_mm_16_17_accu, affine=test_affine)
temp_accus_17_18 = point_query(mittarit_tutkakuva_kiira, radar_kiira_mm_17_18_accu, affine=test_affine)
temp_accus_18_19 = point_query(mittarit_tutkakuva_kiira, radar_kiira_mm_18_19_accu, affine=test_affine)
temp_accus_19_20 = point_query(mittarit_tutkakuva_kiira, radar_kiira_mm_19_20_accu, affine=test_affine)
temp_accus_20_21 = point_query(mittarit_tutkakuva_kiira, radar_kiira_mm_20_21_accu, affine=test_affine)
temp_accus_21_22 = point_query(mittarit_tutkakuva_kiira, radar_kiira_mm_21_22_accu, affine=test_affine)
hourly_accus[:-1,0] = temp_accus_14_15
hourly_accus[:-1,1] = temp_accus_15_16
hourly_accus[:-1,2] = temp_accus_16_17
hourly_accus[:-1,3] = temp_accus_17_18
hourly_accus[:-1,4] = temp_accus_18_19
hourly_accus[:-1,5] = temp_accus_19_20
hourly_accus[:-1,6] = temp_accus_20_21
hourly_accus[:-1,7] = temp_accus_21_22

temp_laune_accus_14_15 = point_query(point_laune_df, radar_kiira_mm_14_15_accu, affine=test_affine)
temp_laune_accus_15_16 = point_query(point_laune_df, radar_kiira_mm_15_16_accu, affine=test_affine)
temp_laune_accus_16_17 = point_query(point_laune_df, radar_kiira_mm_16_17_accu, affine=test_affine)
temp_laune_accus_17_18 = point_query(point_laune_df, radar_kiira_mm_17_18_accu, affine=test_affine)
temp_laune_accus_18_19 = point_query(point_laune_df, radar_kiira_mm_18_19_accu, affine=test_affine)
temp_laune_accus_19_20 = point_query(point_laune_df, radar_kiira_mm_19_20_accu, affine=test_affine)
temp_laune_accus_20_21 = point_query(point_laune_df, radar_kiira_mm_20_21_accu, affine=test_affine)
temp_laune_accus_21_22 = point_query(point_laune_df, radar_kiira_mm_21_22_accu, affine=test_affine)
hourly_accus[-1,0] = temp_laune_accus_14_15[0]
hourly_accus[-1,1] = temp_laune_accus_15_16[0]
hourly_accus[-1,2] = temp_laune_accus_16_17[0]
hourly_accus[-1,3] = temp_laune_accus_17_18[0]
hourly_accus[-1,4] = temp_laune_accus_18_19[0]
hourly_accus[-1,5] = temp_laune_accus_19_20[0]
hourly_accus[-1,6] = temp_laune_accus_20_21[0]
hourly_accus[-1,7] = temp_laune_accus_21_22[0]

hourly_accus_gauge = gauge_data.copy()
hourly_accus_gauge = hourly_accus_gauge.iloc[: , 3:]
hourly_accus_gauge = hourly_accus_gauge.to_numpy()

#Plot hourly accus
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(5, 5, 1, aspect="equal")
plt.plot(hourly_accus[0], color="red")
plt.plot(hourly_accus_gauge[0], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[0]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 2, aspect="equal")
plt.plot(hourly_accus[1], color="red")
plt.plot(hourly_accus_gauge[1], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[1]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 3, aspect="equal")
plt.plot(hourly_accus[2], color="red")
plt.plot(hourly_accus_gauge[2], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[2]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 4, aspect="equal")
plt.plot(hourly_accus[3], color="red")
plt.plot(hourly_accus_gauge[3], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[3]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 5, aspect="equal")
plt.plot(hourly_accus[4], color="red")
plt.plot(hourly_accus_gauge[4], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[4]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 6, aspect="equal")
plt.plot(hourly_accus[5], color="red")
plt.plot(hourly_accus_gauge[5], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[5]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 7, aspect="equal")
plt.plot(hourly_accus[6], color="red")
plt.plot(hourly_accus_gauge[6], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[6]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 8, aspect="equal")
plt.plot(hourly_accus[7], color="red")
plt.plot(hourly_accus_gauge[7], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[7]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 9, aspect="equal")
plt.plot(hourly_accus[8], color="red")
plt.plot(hourly_accus_gauge[8], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[8]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 10, aspect="equal")
plt.plot(hourly_accus[9], color="red")
plt.plot(hourly_accus_gauge[9], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[9]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 11, aspect="equal")
plt.plot(hourly_accus[10], color="red")
plt.plot(hourly_accus_gauge[10], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[10]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 12, aspect="equal")
plt.plot(hourly_accus[11], color="red")
plt.plot(hourly_accus_gauge[11], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[11]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 13, aspect="equal")
plt.plot(hourly_accus[12], color="red")
plt.plot(hourly_accus_gauge[12], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[12]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 14, aspect="equal")
plt.plot(hourly_accus[13], color="red")
plt.plot(hourly_accus_gauge[13], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[13]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 15, aspect="equal")
plt.plot(hourly_accus[14], color="red")
plt.plot(hourly_accus_gauge[14], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[14]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 16, aspect="equal")
plt.plot(hourly_accus[15], color="red")
plt.plot(hourly_accus_gauge[15], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[15]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 17, aspect="equal")
plt.plot(hourly_accus[16], color="red")
plt.plot(hourly_accus_gauge[16], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[16]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 18, aspect="equal")
plt.plot(hourly_accus[17], color="red")
plt.plot(hourly_accus_gauge[17], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[17]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 19, aspect="equal")
plt.plot(hourly_accus[18], color="red")
plt.plot(hourly_accus_gauge[18], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[18]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 20, aspect="equal")
plt.plot(hourly_accus[19], color="red")
plt.plot(hourly_accus_gauge[19], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[19]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 21, aspect="equal")
plt.plot(hourly_accus[20], color="red")
plt.plot(hourly_accus_gauge[20], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[20]["field_1"])
plt.xlim(0,8)
plt.ylim(0,16)
ax = fig.add_subplot(5, 5, 22, aspect="equal")
plt.plot(hourly_accus[21], color="red")
plt.plot(hourly_accus_gauge[21], color="blue")
plt.title(gauge_data.iloc[21]["station"])
plt.xlim(0,8)
plt.ylim(0,16)
plt.tight_layout()

##############################################################################
#10min accumulation maps
temp_radar_kiira_10min = []
for i in range (0, len(radar_kiira)-1, 2):
    print(i)
    temp_radar_kiira_dbz = radar_kiira[i:i+2]
    temp_radar_kiira_mmh = 10**((temp_radar_kiira_dbz-10*np.log10(a_R))/(10*b_R))
    temp_radar_kiira_mmh[temp_radar_kiira_mmh < 0.1] = 0
    temp_radar_kiira_mm = np.sum(temp_radar_kiira_mmh, axis=0) * (5/60)
    temp_radar_kiira_10min.append(temp_radar_kiira_mm)
temp_radar_kiira_10min = np.concatenate([temp_radar_kiira_10min_[None, :, :] for temp_radar_kiira_10min_ in temp_radar_kiira_10min])

#10min accumulation time series in gauge locations
accus_10min = np.zeros((22, len(temp_radar_kiira_10min)))
for i in range(0, len(temp_radar_kiira_10min)):
    temp_accus_10min = point_query(mittarit_tutkakuva_kiira, temp_radar_kiira_10min[i], affine=test_affine)
    temp_accus_10min_laune = point_query(point_laune_df, temp_radar_kiira_10min[i], affine=test_affine)
    accus_10min[:-1,i] = temp_accus_10min
    accus_10min[-1,i] = temp_accus_10min_laune[0]

#10min gauge accumulation
gauge_data_mmh = pd.read_csv(os.path.join(gauge_data_dir, "kiira_gauge_mmh.csv"), delimiter=(";"))
accus_10min_gauge = gauge_data_mmh.copy()
accus_10min_gauge = accus_10min_gauge.iloc[: , 2:]
accus_10min_gauge = accus_10min_gauge.to_numpy()
accus_10min_gauge = accus_10min_gauge * (10/60)

#Plot 10min accus
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(5, 5, 1, aspect="equal")
plt.plot(accus_10min[0], color="red")
plt.plot(accus_10min_gauge[0], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[0]["field_1"])

ax = fig.add_subplot(5, 5, 2, aspect="equal")
plt.plot(accus_10min[1], color="red")
plt.plot(accus_10min_gauge[1], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[1]["field_1"])

ax = fig.add_subplot(5, 5, 3, aspect="equal")
plt.plot(accus_10min[2], color="red")
plt.plot(accus_10min_gauge[2], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[2]["field_1"])

ax = fig.add_subplot(5, 5, 4, aspect="equal")
plt.plot(accus_10min[3], color="red")
plt.plot(accus_10min_gauge[3], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[3]["field_1"])

ax = fig.add_subplot(5, 5, 5, aspect="equal")
plt.plot(accus_10min[4], color="red")
plt.plot(accus_10min_gauge[4], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[4]["field_1"])

ax = fig.add_subplot(5, 5, 6, aspect="equal")
plt.plot(accus_10min[5], color="red")
plt.plot(accus_10min_gauge[5], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[5]["field_1"])

ax = fig.add_subplot(5, 5, 7, aspect="equal")
plt.plot(accus_10min[6], color="red")
plt.plot(accus_10min_gauge[6], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[6]["field_1"])

ax = fig.add_subplot(5, 5, 8, aspect="equal")
plt.plot(accus_10min[7], color="red")
plt.plot(accus_10min_gauge[7], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[7]["field_1"])

ax = fig.add_subplot(5, 5, 9, aspect="equal")
plt.plot(accus_10min[8], color="red")
plt.plot(accus_10min_gauge[8], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[8]["field_1"])

ax = fig.add_subplot(5, 5, 10, aspect="equal")
plt.plot(accus_10min[9], color="red")
plt.plot(accus_10min_gauge[9], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[9]["field_1"])

ax = fig.add_subplot(5, 5, 11, aspect="equal")
plt.plot(accus_10min[10], color="red")
plt.plot(accus_10min_gauge[10], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[10]["field_1"])

ax = fig.add_subplot(5, 5, 12, aspect="equal")
plt.plot(accus_10min[11], color="red")
plt.plot(accus_10min_gauge[11], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[11]["field_1"])

ax = fig.add_subplot(5, 5, 13, aspect="equal")
plt.plot(accus_10min[12], color="red")
plt.plot(accus_10min_gauge[12], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[12]["field_1"])

ax = fig.add_subplot(5, 5, 14, aspect="equal")
plt.plot(accus_10min[13], color="red")
plt.plot(accus_10min_gauge[13], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[13]["field_1"])

ax = fig.add_subplot(5, 5, 15, aspect="equal")
plt.plot(accus_10min[14], color="red")
plt.plot(accus_10min_gauge[14], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[14]["field_1"])

ax = fig.add_subplot(5, 5, 16, aspect="equal")
plt.plot(accus_10min[15], color="red")
plt.plot(accus_10min_gauge[15], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[15]["field_1"])

ax = fig.add_subplot(5, 5, 17, aspect="equal")
plt.plot(accus_10min[16], color="red")
plt.plot(accus_10min_gauge[16], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[16]["field_1"])

ax = fig.add_subplot(5, 5, 18, aspect="equal")
plt.plot(accus_10min[17], color="red")
plt.plot(accus_10min_gauge[17], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[17]["field_1"])

ax = fig.add_subplot(5, 5, 19, aspect="equal")
plt.plot(accus_10min[18], color="red")
plt.plot(accus_10min_gauge[18], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[18]["field_1"])

ax = fig.add_subplot(5, 5, 20, aspect="equal")
plt.plot(accus_10min[19], color="red")
plt.plot(accus_10min_gauge[19], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[19]["field_1"])

ax = fig.add_subplot(5, 5, 21, aspect="equal")
plt.plot(accus_10min[20], color="red")
plt.plot(accus_10min_gauge[20], color="blue")
plt.title(mittarit_tutkakuva_kiira.iloc[20]["field_1"])

ax = fig.add_subplot(5, 5, 22, aspect="equal")
plt.plot(accus_10min[21], color="red")
plt.plot(accus_10min_gauge[21], color="blue")
plt.title(gauge_data.iloc[21]["station"])

plt.tight_layout()

for im in range(0,len(accus_10min)):
    # im=1
    plt.figure()
    plt.plot(accus_10min[im,0:], color="red", label="rad")
    plt.plot(accus_10min_gauge[im,1:], color="blue", label="gauge")
    plt.title(gauge_data.iloc[im]["station"])
    # plt.figure()
    # plt.plot(accus_10min[im], color="red", label="rad")
    # plt.plot(accus_10min_gauge[im], color="blue", label="gauge")
    # plt.title(gauge_data.iloc[im]["station"])
    plt.legend()

##############################################################################
#Scatterplotit

    
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
### Create 2-D synthetic data

# # # grid axes
# # xgrid = np.arange(0, 10)
# # ygrid = np.arange(20, 30)

# # # number of observations
# # num_obs = 10

# # # create grid
# # gridshape = len(xgrid), len(ygrid)
# # grid_coords = wrl.util.gridaspoints(ygrid, xgrid)

# ###########
# xgrid = np.arange(0, radar_field.shape[0])
# ygrid = np.arange(0, radar_field.shape[1])
# ygrid = np.flip(ygrid)

# # number of observations
# num_obs = 22
# # 
# # create grid
# gridshape = len(xgrid), len(ygrid)
# grid_coords = wrl.util.gridaspoints(ygrid, xgrid)

# ###############
# # Synthetic true rainfall
# truth = np.abs(10.0 * np.sin(0.1 * grid_coords).sum(axis=1))
# truth = np.abs(100 * (0.1 * grid_coords).sum(axis=1))
# truth = 0.008 * truth
# # Creating radar data by perturbing truth with multiplicative and
# # additive error
# np.random.seed(1319622840)
# # radar = 0.6 * truth + 1.0 * np.random.uniform(low=-1.0, high=1, size=len(truth))
# radar = radar_field2
# radar[radar < 0.0] = 0.0

# # indices for creating obs from raw (random placement of gauges)
# obs_ix = np.random.uniform(low=0, high=len(grid_coords), size=num_obs).astype("i4")

# # creating obs_coordinates
# obs_coords = grid_coords[obs_ix]

# # creating gauge observations from truth
# # obs = truth[obs_ix]
# obs = radar[obs_ix]

# # gauge1_location = np.reshape(np.array([51, 48]), (1,2))
# # gauge2_location = np.reshape(np.array([690, 500]), (1,2))
# # gauge3_location = np.reshape(np.array([616, 780]), (1,2))
# # gauge4_location = np.reshape(np.array([671, 599]), (1,2))
# # gauge5_location = np.reshape(np.array([378, 743]), (1,2))
# # gauge6_location = np.reshape(np.array([268, 43]), (1,2))
# # gauge7_location = np.reshape(np.array([635, 488]), (1,2))
# # gauge8_location = np.reshape(np.array([126, 296]), (1,2))
# # gauge9_location = np.reshape(np.array([482, 320]), (1,2))
# # gauge10_location = np.reshape(np.array([237, 257]), (1,2))

# # gauge1_location = np.reshape(np.array([142, 327]), (1,2))
# # gauge2_location = np.reshape(np.array([112, 957]), (1,2))
# # gauge3_location = np.reshape(np.array([695, 881]), (1,2))
# # gauge4_location = np.reshape(np.array([561, 455]), (1,2))
# # gauge5_location = np.reshape(np.array([560, 344]), (1,2))
# # gauge6_location = np.reshape(np.array([552, 697]), (1,2))
# # gauge7_location = np.reshape(np.array([542, 350]), (1,2))
# # gauge8_location = np.reshape(np.array([527, 53]), (1,2))
# # gauge9_location = np.reshape(np.array([509, 611]), (1,2))
# # gauge10_location = np.reshape(np.array([494, 810]), (1,2))
# # gauge11_location = np.reshape(np.array([479, 938]), (1,2))
# # gauge12_location = np.reshape(np.array([471, 964]), (1,2))
# # gauge13_location = np.reshape(np.array([458, 223]), (1,2))
# # gauge14_location = np.reshape(np.array([424, 842]), (1,2))
# # gauge15_location = np.reshape(np.array([412, 303]), (1,2))
# # gauge16_location = np.reshape(np.array([373, 400]), (1,2))
# # gauge17_location = np.reshape(np.array([362, 735]), (1,2))
# # gauge18_location = np.reshape(np.array([342, 36]), (1,2))
# # gauge19_location = np.reshape(np.array([337, 279]), (1,2))
# # gauge20_location = np.reshape(np.array([267, 319]), (1,2))
# # gauge21_location = np.reshape(np.array([192, 807]), (1,2))

# gauge1_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[0]["row"]), int(mittarit_tutkakuva_kiira.iloc[0]["col"])]), (1,2))
# gauge2_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[1]["row"]), int(mittarit_tutkakuva_kiira.iloc[1]["col"])]), (1,2))
# gauge3_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[2]["row"]), int(mittarit_tutkakuva_kiira.iloc[2]["col"])]), (1,2))
# gauge4_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[3]["row"]), int(mittarit_tutkakuva_kiira.iloc[3]["col"])]), (1,2))
# gauge5_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[4]["row"]), int(mittarit_tutkakuva_kiira.iloc[4]["col"])]), (1,2))
# gauge6_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[5]["row"]), int(mittarit_tutkakuva_kiira.iloc[5]["col"])]), (1,2))
# gauge7_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[6]["row"]), int(mittarit_tutkakuva_kiira.iloc[6]["col"])]), (1,2))
# gauge8_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[7]["row"]), int(mittarit_tutkakuva_kiira.iloc[7]["col"])]), (1,2))
# gauge9_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[8]["row"]), int(mittarit_tutkakuva_kiira.iloc[8]["col"])]), (1,2))
# gauge10_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[9]["row"]), int(mittarit_tutkakuva_kiira.iloc[9]["col"])]), (1,2))
# gauge11_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[10]["row"]), int(mittarit_tutkakuva_kiira.iloc[10]["col"])]), (1,2))
# gauge12_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[11]["row"]), int(mittarit_tutkakuva_kiira.iloc[11]["col"])]), (1,2))
# gauge13_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[12]["row"]), int(mittarit_tutkakuva_kiira.iloc[12]["col"])]), (1,2))
# gauge14_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[13]["row"]), int(mittarit_tutkakuva_kiira.iloc[13]["col"])]), (1,2))
# gauge15_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[14]["row"]), int(mittarit_tutkakuva_kiira.iloc[14]["col"])]), (1,2))
# gauge16_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[15]["row"]), int(mittarit_tutkakuva_kiira.iloc[15]["col"])]), (1,2))
# gauge17_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[16]["row"]), int(mittarit_tutkakuva_kiira.iloc[16]["col"])]), (1,2))
# gauge18_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[17]["row"]), int(mittarit_tutkakuva_kiira.iloc[17]["col"])]), (1,2))
# gauge19_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[18]["row"]), int(mittarit_tutkakuva_kiira.iloc[18]["col"])]), (1,2))
# gauge20_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[19]["row"]), int(mittarit_tutkakuva_kiira.iloc[19]["col"])]), (1,2))
# gauge21_location = np.reshape(np.array([int(mittarit_tutkakuva_kiira.iloc[20]["row"]), int(mittarit_tutkakuva_kiira.iloc[20]["col"])]), (1,2))
# gauge22_location = np.reshape(np.array([int(point_laune_df.iloc[0]["row"]), int(point_laune_df.iloc[0]["col"])]), (1,2))

# # gauge_coords = np.zeros([10,2])
# # gauge_coords[0] = gauge1_location
# # gauge_coords[1] = gauge2_location
# # gauge_coords[2] = gauge3_location
# # gauge_coords[3] = gauge4_location
# # gauge_coords[4] = gauge5_location
# # gauge_coords[5] = gauge6_location
# # gauge_coords[6] = gauge7_location
# # gauge_coords[7] = gauge8_location
# # gauge_coords[8] = gauge9_location
# # gauge_coords[9] = gauge10_location

# # gauge_coords = np.zeros([21,2])
# # gauge_coords[0] = gauge1_location
# # gauge_coords[1] = gauge2_location
# # gauge_coords[2] = gauge3_location
# # gauge_coords[3] = gauge4_location
# # gauge_coords[4] = gauge5_location
# # gauge_coords[5] = gauge6_location
# # gauge_coords[6] = gauge7_location
# # gauge_coords[7] = gauge8_location
# # gauge_coords[8] = gauge9_location
# # gauge_coords[9] = gauge10_location
# # gauge_coords[10] = gauge11_location
# # gauge_coords[11] = gauge12_location
# # gauge_coords[12] = gauge13_location
# # gauge_coords[13] = gauge14_location
# # gauge_coords[14] = gauge15_location
# # gauge_coords[15] = gauge16_location
# # gauge_coords[16] = gauge17_location
# # gauge_coords[17] = gauge18_location
# # gauge_coords[18] = gauge19_location
# # gauge_coords[19] = gauge20_location
# # gauge_coords[20] = gauge21_location

# # gauge_coords_test = np.zeros([21,2])
# # gauge_coords_test[:,0] = gauge_coords[:,1]
# # gauge_coords_test[:,1] = gauge_coords[:,0]
# # gauge_coords = gauge_coords_test.copy()

# gauge_coords = np.zeros([22,2])
# gauge_coords[0] = gauge1_location
# gauge_coords[1] = gauge2_location
# gauge_coords[2] = gauge3_location
# gauge_coords[3] = gauge4_location
# gauge_coords[4] = gauge5_location
# gauge_coords[5] = gauge6_location
# gauge_coords[6] = gauge7_location
# gauge_coords[7] = gauge8_location
# gauge_coords[8] = gauge9_location
# gauge_coords[9] = gauge10_location
# gauge_coords[10] = gauge11_location
# gauge_coords[11] = gauge12_location
# gauge_coords[12] = gauge13_location
# gauge_coords[13] = gauge14_location
# gauge_coords[14] = gauge15_location
# gauge_coords[15] = gauge16_location
# gauge_coords[16] = gauge17_location
# gauge_coords[17] = gauge18_location
# gauge_coords[18] = gauge19_location
# gauge_coords[19] = gauge20_location
# gauge_coords[20] = gauge21_location
# gauge_coords[21] = gauge22_location

# gauge_coords_test = np.zeros([22,2])
# gauge_coords_test[:,0] = gauge_coords[:,1]
# gauge_coords_test[:,1] = gauge_coords[:,0]
# gauge_coords = gauge_coords_test.copy()

# # gauge_obs = np.zeros([10,])
# # gauge_obs[0] = radar_field[int(gauge_coords[0,1]), int(gauge_coords[0,0])]
# # gauge_obs[1] = radar_field[int(gauge_coords[1,1]), int(gauge_coords[1,0])]
# # gauge_obs[2] = radar_field[int(gauge_coords[2,1]), int(gauge_coords[2,0])]
# # gauge_obs[3] = radar_field[int(gauge_coords[3,1]), int(gauge_coords[3,0])]
# # gauge_obs[4] = radar_field[int(gauge_coords[4,1]), int(gauge_coords[4,0])]
# # gauge_obs[5] = radar_field[int(gauge_coords[5,1]), int(gauge_coords[5,0])]
# # gauge_obs[6] = radar_field[int(gauge_coords[6,1]), int(gauge_coords[6,0])]
# # gauge_obs[7] = radar_field[int(gauge_coords[7,1]), int(gauge_coords[7,0])]
# # gauge_obs[8] = radar_field[int(gauge_coords[8,1]), int(gauge_coords[8,0])]
# # gauge_obs[9] = radar_field[int(gauge_coords[9,1]), int(gauge_coords[9,0])]

# # gauge_obs = np.zeros([21,])
# # gauge_obs[0] = radar_field[int(gauge_coords[0,1]), int(gauge_coords[0,0])]
# # gauge_obs[1] = radar_field[int(gauge_coords[1,1]), int(gauge_coords[1,0])]
# # gauge_obs[2] = radar_field[int(gauge_coords[2,1]), int(gauge_coords[2,0])]
# # gauge_obs[3] = radar_field[int(gauge_coords[3,1]), int(gauge_coords[3,0])]
# # gauge_obs[4] = radar_field[int(gauge_coords[4,1]), int(gauge_coords[4,0])]
# # gauge_obs[5] = radar_field[int(gauge_coords[5,1]), int(gauge_coords[5,0])]
# # gauge_obs[6] = radar_field[int(gauge_coords[6,1]), int(gauge_coords[6,0])]
# # gauge_obs[7] = radar_field[int(gauge_coords[7,1]), int(gauge_coords[7,0])]
# # gauge_obs[8] = radar_field[int(gauge_coords[8,1]), int(gauge_coords[8,0])]
# # gauge_obs[9] = radar_field[int(gauge_coords[9,1]), int(gauge_coords[9,0])]
# # gauge_obs[10] = radar_field[int(gauge_coords[10,1]), int(gauge_coords[10,0])]
# # gauge_obs[11] = radar_field[int(gauge_coords[11,1]), int(gauge_coords[11,0])]
# # gauge_obs[12] = radar_field[int(gauge_coords[12,1]), int(gauge_coords[12,0])]
# # gauge_obs[13] = radar_field[int(gauge_coords[13,1]), int(gauge_coords[13,0])]
# # gauge_obs[14] = radar_field[int(gauge_coords[14,1]), int(gauge_coords[14,0])]
# # gauge_obs[15] = radar_field[int(gauge_coords[15,1]), int(gauge_coords[15,0])]
# # gauge_obs[16] = radar_field[int(gauge_coords[16,1]), int(gauge_coords[16,0])]
# # gauge_obs[17] = radar_field[int(gauge_coords[17,1]), int(gauge_coords[17,0])]
# # gauge_obs[18] = radar_field[int(gauge_coords[18,1]), int(gauge_coords[18,0])]
# # gauge_obs[19] = radar_field[int(gauge_coords[19,1]), int(gauge_coords[19,0])]
# # gauge_obs[20] = radar_field[int(gauge_coords[20,1]), int(gauge_coords[20,0])]

# gauge_obs = np.zeros([22,8])
# gauge_obs[:,0] = gauge_data["15"]
# gauge_obs[:,1] = gauge_data["16"]
# gauge_obs[:,2] = gauge_data["17"]
# gauge_obs[:,3] = gauge_data["18"]
# gauge_obs[:,4] = gauge_data["19"]
# gauge_obs[:,5] = gauge_data["20"]
# gauge_obs[:,6] = gauge_data["21"]
# gauge_obs[:,7] = gauge_data["22"]
# gauge_obs[9,:] = np.nan

# # gauge_obs2 = gauge_obs.copy()
# # gauge_obs2[0] = gauge_obs2[0] * (random.randint(100,200)/100)
# # gauge_obs2[1] = gauge_obs2[1] * (random.randint(100,200)/100)
# # gauge_obs2[2] = gauge_obs2[2] * (random.randint(100,200)/100)
# # gauge_obs2[3] = gauge_obs2[3] * (random.randint(100,200)/100)
# # gauge_obs2[4] = gauge_obs2[4] * (random.randint(100,200)/100)
# # gauge_obs2[5] = gauge_obs2[5] * (random.randint(100,200)/100)
# # gauge_obs2[6] = gauge_obs2[6] * (random.randint(100,200)/100)
# # gauge_obs2[7] = gauge_obs2[7] * (random.randint(100,200)/100)
# # gauge_obs2[8] = gauge_obs2[8] * (random.randint(100,200)/100)
# # gauge_obs2[9] = gauge_obs2[9] * (random.randint(100,200)/100)

# # gauge_obs2 = gauge_obs.copy()
# # gauge_obs2[0] = gauge_obs2[0] * (random.randint(100,200)/100)
# # gauge_obs2[1] = gauge_obs2[1] * (random.randint(100,200)/100)
# # gauge_obs2[2] = gauge_obs2[2] * (random.randint(100,200)/100)
# # gauge_obs2[3] = gauge_obs2[3] * (random.randint(100,200)/100)
# # gauge_obs2[4] = gauge_obs2[4] * (random.randint(100,200)/100)
# # gauge_obs2[5] = gauge_obs2[5] * (random.randint(100,200)/100)
# # gauge_obs2[6] = gauge_obs2[6] * (random.randint(100,200)/100)
# # gauge_obs2[7] = gauge_obs2[7] * (random.randint(100,200)/100)
# # gauge_obs2[8] = gauge_obs2[8] * (random.randint(100,200)/100)
# # gauge_obs2[9] = gauge_obs2[9] * (random.randint(100,200)/100)
# # gauge_obs2[10] = gauge_obs2[10] * (random.randint(100,200)/100)
# # gauge_obs2[11] = gauge_obs2[11] * (random.randint(100,200)/100)
# # gauge_obs2[12] = gauge_obs2[12] * (random.randint(100,200)/100)
# # gauge_obs2[13] = gauge_obs2[13] * (random.randint(100,200)/100)
# # gauge_obs2[14] = gauge_obs2[14] * (random.randint(100,200)/100)
# # gauge_obs2[15] = gauge_obs2[15] * (random.randint(100,200)/100)
# # gauge_obs2[16] = gauge_obs2[16] * (random.randint(100,200)/100)
# # gauge_obs2[17] = gauge_obs2[17] * (random.randint(100,200)/100)
# # gauge_obs2[18] = gauge_obs2[18] * (random.randint(100,200)/100)
# # gauge_obs2[19] = gauge_obs2[19] * (random.randint(100,200)/100)
# # gauge_obs2[20] = gauge_obs2[20] * (random.randint(100,200)/100)

# obs_coords = gauge_coords
# # obs = gauge_obs2
# # obs = gauge_obs[:,4]

# ##############################################################################
# ### Apply different adjustment methods
# #https://docs.wradlib.org/en/latest/generated/wradlib.adjust.AdjustBase.html#wradlib.adjust.AdjustBase

# # mingages (int) – Defaults to 5. Minimum number of valid gages required for 
# # an adjustment. If less valid gauges are available, the adjustment procedure 
# # will return unadjusted raw values. If you do not want to use this feature, 
# # you need to set mingages=0.

# # minval (float) – If the gage or radar observation is below this threshold, 
# # the location will not be used for adjustment. For additive adjustment, this 
# # value should be set to zero (default value). For multiplicative adjustment, 
# # values larger than zero might be chosen in order to minimize artifacts.

# # radar_field_temp = radar_kiira_mm_18_19_accu.copy()
# # radar_field_temp2 = radar_field_temp.reshape([(radar_field_temp.shape[0]*radar_field_temp.shape[1]), ])
# # radar = radar_field_temp2

# # Mean Field Bias Adjustment
# mfbadjuster = wrl.adjust.AdjustMFB(obs_coords, grid_coords, minval=0.11)
# mfbadjusted = mfbadjuster(obs, radar)

# # Additive Error Model
# addadjuster = wrl.adjust.AdjustAdd(obs_coords, grid_coords, minval=0.11)
# addadjusted = addadjuster(obs, radar)

# # Multiplicative Error Model
# multadjuster = wrl.adjust.AdjustMultiply(obs_coords, grid_coords, minval=0.11)
# multadjusted = multadjuster(obs, radar)

# # adjust the radar observation by AdjustMixed
# mixed_adjuster = wrl.adjust.AdjustMixed(obs_coords, grid_coords, minval=0.11)
# mixed_adjusted = mixed_adjuster(obs, radar)

# # check if arrays are equal to each other
# np.array_equal(radar, mfbadjusted)
# np.array_equal(radar, addadjusted)
# np.array_equal(radar, multadjusted)
# np.array_equal(radar, mixed_adjusted)

# ##############################################################################
# ### Plot 2-D adjustment results

# # Helper functions for grid plots
# def gridplot(data, title):
#     """Quick and dirty helper function to produce a grid plot"""
#     xplot = np.append(xgrid, xgrid[-1] + 1.0) - 0.5
#     yplot = np.append(ygrid, ygrid[-1] + 1.0) - 0.5
#     grd = ax.pcolormesh(xplot, yplot, data.reshape(gridshape), vmin=0, vmax=maxval, cmap="nipy_spectral")
#     ax.scatter(
#         obs_coords[:, 0],
#         obs_coords[:, 1],
#         c=obs.ravel(),
#         marker="o",
#         edgecolors="red",
#         s=50,
#         vmin=0,
#         vmax=maxval,
#         cmap="nipy_spectral",
#     )
#     plt.colorbar(grd, shrink=0.5)
#     plt.title(title)
    
# # Maximum value (used for normalisation of colorscales)
# # maxval = np.max(np.concatenate((truth, radar, obs, addadjusted)).ravel())
# maxval = np.max(np.concatenate((radar, obs, addadjusted, mfbadjusted, multadjusted, mixed_adjusted)).ravel())

# truth = truth - truth

# plt.figure(figsize=(10, 10))
# for i in range(len(obs_coords)):
#     plt.scatter(obs_coords[i, 0], obs_coords[i, 1], label=i)
#     plt.text(obs_coords[i, 0], obs_coords[i, 1], str(i))
# plt.xlim(0,1024)
# plt.ylim(1024,0)
# plt.legend(bbox_to_anchor=(1, 1))
           
# # open figure
# fig = plt.figure(figsize=(10, 6))
# # True rainfall
# ax = fig.add_subplot(231, aspect="equal")
# # gridplot(truth, "True rainfall")
# ax.scatter(obs_coords[:, 0], obs_coords[:, 1], c=obs.ravel(), marker="o", s=50, vmin=0, vmax=maxval, cmap="nipy_spectral")
# ax.set_xlim([0, 1024])
# ax.set_ylim([0, 1024])
# ax.set_title("Gauges")
# # Unadjusted radar rainfall
# ax = fig.add_subplot(232, aspect="equal")
# gridplot(radar, "Radar rainfall")
# # Adjusted radar rainfall (MFB)
# ax = fig.add_subplot(234, aspect="equal")
# gridplot(mfbadjusted, "Adjusted (MFB)")
# # Adjusted radar rainfall (additive)
# ax = fig.add_subplot(235, aspect="equal")
# gridplot(addadjusted, "Adjusted (Add.)")
# # Adjusted radar rainfall (multiplicative)
# ax = fig.add_subplot(236, aspect="equal")
# gridplot(multadjusted, "Adjusted (Mult.)")
# # Adjusted radar rainfall (Mixed)
# ax = fig.add_subplot(233, aspect="equal")
# gridplot(mixed_adjusted, "Adjusted (Mixed)")
# plt.tight_layout()

# test_radar = radar.reshape(gridshape)
# test_mfbadjusted = mfbadjusted.reshape(gridshape)
# test_addadjusted = addadjusted.reshape(gridshape)
# test_multadjusted = multadjusted.reshape(gridshape)
# test_mixed_adjusted = mixed_adjusted.reshape(gridshape)

# # gauge_loc_values = np.zeros([10,6])
# gauge_loc_values = np.zeros([21,6])
# for i in range(len(obs_coords)):
#     gauge_loc_values[i,0] = gauge_obs2[i]
#     gauge_loc_values[i,1] = test_radar[int(obs_coords[i,1]),int(obs_coords[i,0])]
#     gauge_loc_values[i,2] = test_mfbadjusted[int(obs_coords[i,1]),int(obs_coords[i,0])]
#     gauge_loc_values[i,3] = test_addadjusted[int(obs_coords[i,1]),int(obs_coords[i,0])]
#     gauge_loc_values[i,4] = test_multadjusted[int(obs_coords[i,1]),int(obs_coords[i,0])]
#     gauge_loc_values[i,5] = test_mixed_adjusted[int(obs_coords[i,1]),int(obs_coords[i,0])]

# #stats
# field_stats = np.zeros([5,2])
# field_stats[0,0] = np.mean(radar)
# field_stats[0,1] = np.max(radar)
# field_stats[1,0] = np.mean(mfbadjusted)
# field_stats[1,1] = np.max(mfbadjusted)
# field_stats[2,0] = np.mean(addadjusted)
# field_stats[2,1] = np.max(addadjusted)
# field_stats[3,0] = np.mean(multadjusted)
# field_stats[3,1] = np.max(multadjusted)
# field_stats[4,0] = np.mean(mixed_adjusted)
# field_stats[4,1] = np.max(mixed_adjusted)

# ##############################################################################
# ### Verification

# # Verification for this example
# rawerror = wrl.verify.ErrorMetrics(truth, radar)
# mfberror = wrl.verify.ErrorMetrics(truth, mfbadjusted)
# adderror = wrl.verify.ErrorMetrics(truth, addadjusted)
# multerror = wrl.verify.ErrorMetrics(truth, multadjusted)
# mixerror = wrl.verify.ErrorMetrics(truth, mixed_adjusted)

# # Helper function for scatter plot
# def scatterplot(x, y, title=""):
#     """Quick and dirty helper function to produce scatter plots"""
#     plt.scatter(x, y)
#     plt.plot([0, 1.2 * maxval], [0, 1.2 * maxval], "-", color="grey")
#     plt.xlabel("True rainfall (mm)")
#     plt.ylabel("Estimated rainfall (mm)")
#     plt.xlim(0, maxval + 0.1 * maxval)
#     plt.ylim(0, maxval + 0.1 * maxval)
#     plt.title(title)

# # Enlarge all label fonts
# font = {"size": 10}
# plt.rc("font", **font)
# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(231, aspect=1.0)
# scatterplot(rawerror.obs, rawerror.est, title="Unadjusted radar")
# ax.text(0.2, maxval, "Nash=%.1f" % rawerror.nash(), fontsize=12)
# ax = fig.add_subplot(232, aspect=1.0)
# scatterplot(adderror.obs, adderror.est, title="Additive adjustment")
# ax.text(0.2, maxval, "Nash=%.1f" % adderror.nash(), fontsize=12)
# ax = fig.add_subplot(233, aspect=1.0)
# scatterplot(multerror.obs, multerror.est, title="Multiplicative adjustment")
# ax.text(0.2, maxval, "Nash=%.1f" % multerror.nash(), fontsize=12)
# ax = fig.add_subplot(234, aspect=1.0)
# scatterplot(mixerror.obs, mixerror.est, title="Mixed (mult./add.) adjustment")
# ax.text(0.2, maxval, "Nash=%.1f" % mixerror.nash(), fontsize=12)
# ax = fig.add_subplot(235, aspect=1.0)
# scatterplot(mfberror.obs, mfberror.est, title="Mean Field Bias adjustment")
# ax.text(0.2, maxval, "Nash=%.1f" % mfberror.nash(), fontsize=12)
# plt.tight_layout()

# # Open figure
# fig = plt.figure(figsize=(14, 8))
# # Scatter plot radar vs. observations
# ax = fig.add_subplot(231, aspect="equal")
# scatterplot(truth, radar, "Radar vs. Truth (red: Gauges)")
# plt.plot(obs, radar[obs_ix], linestyle="None", marker="o", color="red")
# # Adjusted (MFB) vs. radar (for control purposes)
# ax = fig.add_subplot(235, aspect="equal")
# scatterplot(truth, mfbadjusted, "Adjusted (MFB) vs. Truth")
# # Adjusted (Add) vs. radar (for control purposes)
# ax = fig.add_subplot(232, aspect="equal")
# scatterplot(truth, addadjusted, "Adjusted (Add.) vs. Truth")
# # Adjusted (Mult.) vs. radar (for control purposes)
# ax = fig.add_subplot(233, aspect="equal")
# scatterplot(truth, multadjusted, "Adjusted (Mult.) vs. Truth")
# # Adjusted (Mixed) vs. radar (for control purposes)
# ax = fig.add_subplot(234, aspect="equal")
# scatterplot(truth, mixed_adjusted, "Adjusted (Mixed) vs. Truth")
# plt.tight_layout()

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
### Test field

# radar_dir = r"W:\lindgrv1\Kiira_whole_day\kiira_14-21_dbz"

# file_list = os.listdir(radar_dir)
# file_list = [x for x in file_list if ".tif" in x]
# file_list = [f for f in file_list if f.endswith(".tif")]

# radar_kiira = []
# for i in range(len(file_list)):
#     src = rasterio.open(os.path.join(radar_dir, file_list[i]))
#     array = src.read(1)
#     radar_kiira.append(array)
    
# radar_kiira = np.concatenate([radar_kiira_[None, :, :] for radar_kiira_ in radar_kiira])

# #This have to be added for new event data: Remove last column from each layer
# radar_kiira = radar_kiira[:,:,:-1]
# #The following data is available for Finnish radar composite: radar reflectivity (dbz), conversion: Z[dBZ] = 0.5 * pixel value - 32
# radar_kiira = (radar_kiira * 0.5) - 32
# #Values less than threshold to wanted value, which represents no-rain
# radar_kiira[radar_kiira < 10] = 3.1830486304816077

# radar_field = radar_kiira[25]

# a_R=223
# b_R=1.53
# radar_field = 10**((radar_field-10*np.log10(a_R))/(10*b_R))
# #Values less than threshold to zero
# radar_field[radar_field < 0.1] = 0

# plt.figure()
# plt.imshow(radar_field, cmap="nipy_spectral")

# # #x- and y-coords in rainfield
# # gauge1_loc = np.reshape(np.array([51, 48]), (1,2)) #37.1309 
# # gauge2_loc = np.reshape(np.array([690, 500]), (1,2)) #58.3195
# # gauge3_loc = np.reshape(np.array([616, 780]), (1,2)) #114.797

# # gauge1_loc_row_flipped = (radar_field.shape[0]-1)-gauge1_loc[0,0]
# # gauge2_loc_row_flipped = (radar_field.shape[0]-1)-gauge2_loc[0,0]
# # gauge3_loc_row_flipped = (radar_field.shape[0]-1)-gauge3_loc[0,0]

# # gauge1_loc_flipped = gauge1_loc.copy()
# # gauge1_loc_flipped[0,0] = gauge1_loc_row_flipped
# # gauge2_loc_flipped = gauge2_loc.copy()
# # gauge2_loc_flipped[0,0] = gauge2_loc_row_flipped
# # gauge3_loc_flipped = gauge3_loc.copy()
# # gauge3_loc_flipped[0,0] = gauge3_loc_row_flipped

# # gauge_coords = np.zeros([3,2])
# # gauge_coords[0] = gauge1_loc
# # gauge_coords[1] = gauge2_loc
# # gauge_coords[2] = gauge3_loc
# # gauge_coords_f = np.zeros([3,2])
# # gauge_coords_f[0] = gauge1_loc_flipped 
# # gauge_coords_f[1] = gauge2_loc_flipped
# # gauge_coords_f[2] = gauge3_loc_flipped

# # # plt.figure()
# # # plt.imshow(radar_field)
# # # plt.scatter(gauge_coords[0,0], gauge_coords[0,1], s=30, c='red', marker='o')
# # # plt.scatter(gauge_coords[1,0], gauge_coords[1,1], s=30, c='red', marker='o')
# # # plt.scatter(gauge_coords[2,0], gauge_coords[2,1], s=30, c='red', marker='o')

# # radar_field_edit = radar_field.reshape([(radar_field.shape[0]*radar_field.shape[1]), ])

# # gauge_obs = np.zeros([3,])
# # gauge_obs[0] = radar_field[int(gauge_coords[0,1]), int(gauge_coords[0,0])] #37.1309 
# # gauge_obs[1] = radar_field[int(gauge_coords[1,1]), int(gauge_coords[1,0])] #58.3195
# # gauge_obs[2] = radar_field[int(gauge_coords[2,1]), int(gauge_coords[2,0])] #114.797

# # radar_loc_x = np.array(np.arange(radar_field.shape[0]))
# # # radar_loc_x = np.flipud(radar_loc_x)
# # radar_loc_y = np.array(np.arange(radar_field.shape[1]))
# # # radar_loc_y = np.flipud(radar_loc_y)

# # radar_coords = util.gridaspoints(radar_loc_y, radar_loc_x)

# # #Mielivaltainen modifionti testi mielessä
# # gauge_obs = gauge_obs * 1.5

# # # Mean Field Bias Adjustment
# # adjuster_mfb_rad = adjust.AdjustMFB(gauge_coords, radar_coords)
# # adjusted_mfb_rad = adjuster_mfb_rad(gauge_obs, radar_field_edit)
# # # Additive Error Model
# # adjuster_add_rad = adjust.AdjustAdd(gauge_coords, radar_coords)
# # adjusted_add_rad = adjuster_add_rad(gauge_obs, radar_field_edit)
# # # Multiplicative Error Model
# # adjuster_mult_rad = adjust.AdjustMultiply(gauge_coords, radar_coords)
# # adjusted_mult_rad = adjuster_mult_rad(gauge_obs, radar_field_edit)
# # # Mixed Error
# # adjuster_mixed_rad = adjust.AdjustMixed(gauge_coords, radar_coords)
# # adjusted_mixed_rad = adjuster_mixed_rad(gauge_obs, radar_field_edit)

# # #Added here
# # ((radar_field_edit == adjusted_mfb_rad) | (np.isnan(radar_field_edit) & np.isnan(adjusted_mfb_rad))).all() #Ville added
# # ((radar_field_edit== adjusted_add_rad) | (np.isnan(radar_field_edit) & np.isnan(adjusted_add_rad))).all() #Ville added
# # ((radar_field_edit== adjusted_mult_rad) | (np.isnan(radar_field_edit) & np.isnan(adjusted_mult_rad))).all() #Ville added
# # ((radar_field_edit == adjusted_mixed_rad) | (np.isnan(radar_field_edit) & np.isnan(adjusted_mixed_rad))).all() #Ville added

# # radar_shape = len(radar_loc_x), len(radar_loc_y)
# # maxval_radar = np.nanmax(np.concatenate((gauge_obs, radar_field_edit, gauge_obs, adjusted_add_rad)).ravel())

# # gauge_coords2 = gauge_coords.copy()
# # gauge_coords2[0,1] = (radar_field.shape[1]-1)-gauge1_loc[0,1]
# # gauge_coords2[1,1] = (radar_field.shape[1]-1)-gauge2_loc[0,1]
# # gauge_coords2[2,1] = (radar_field.shape[1]-1)-gauge3_loc[0,1]

# # # open figure
# # fig = plt.figure(figsize=(10, 6))
# # # Unadjusted radar rainfall
# # ax = fig.add_subplot(232, aspect="equal")
# # gridplot(radar_field_edit, "Radar rainfall")
# # # Adjusted radar rainfall (MFB)
# # ax = fig.add_subplot(234, aspect="equal")
# # gridplot(adjusted_mfb_rad, "Adjusted (MFB)")
# # # Adjusted radar rainfall (additive)
# # ax = fig.add_subplot(235, aspect="equal")
# # gridplot(adjusted_add_rad, "Adjusted (Add.)")
# # # Adjusted radar rainfall (multiplicative)
# # ax = fig.add_subplot(236, aspect="equal")
# # gridplot(adjusted_mult_rad, "Adjusted (Mult.)")
# # ax = fig.add_subplot(233, aspect="equal") #Ville added
# # gridplot(adjusted_mixed_rad, "Adjusted (Mixed)") #Ville added
# # plt.tight_layout()

# # grid axes
# xgrid = np.arange(0, radar_field.shape[0])
# ygrid = np.arange(0, radar_field.shape[1])

# # number of observations
# num_obs = 10

# # create grid
# gridshape = len(xgrid), len(ygrid)
# grid_coords = wrl.util.gridaspoints(ygrid, xgrid)

# radar_field2 = radar_field.reshape([(radar_field.shape[0]*radar_field.shape[1]), ])

# gauge1_location = np.reshape(np.array([51, 48]), (1,2)) #37.1309 
# gauge2_location = np.reshape(np.array([690, 500]), (1,2)) #58.3195
# gauge3_location = np.reshape(np.array([616, 780]), (1,2)) #114.797
# gauge4_location = np.reshape(np.array([200, 700]), (1,2))
# gauge5_location = np.reshape(np.array([300, 200]), (1,2))
# gauge6_location = np.reshape(np.array([400, 600]), (1,2))
# gauge7_location = np.reshape(np.array([500, 100]), (1,2))
# gauge8_location = np.reshape(np.array([750, 900]), (1,2))
# gauge9_location = np.reshape(np.array([850, 200]), (1,2))
# gauge10_location = np.reshape(np.array([950, 500]), (1,2))

# gauge_coords = np.zeros([10,2])
# gauge_coords[0] = gauge1_location
# gauge_coords[1] = gauge2_location
# gauge_coords[2] = gauge3_location
# gauge_coords[3] = gauge4_location
# gauge_coords[4] = gauge5_location
# gauge_coords[5] = gauge6_location
# gauge_coords[6] = gauge7_location
# gauge_coords[7] = gauge8_location
# gauge_coords[8] = gauge9_location
# gauge_coords[9] = gauge10_location

# gauge_obs = np.zeros([10,])
# gauge_obs[0] = radar_field[int(gauge_coords[0,1]), int(gauge_coords[0,0])] #37.1309 
# gauge_obs[1] = radar_field[int(gauge_coords[1,1]), int(gauge_coords[1,0])] #58.3195
# gauge_obs[2] = radar_field[int(gauge_coords[2,1]), int(gauge_coords[2,0])] #114.797
# gauge_obs[3] = radar_field[int(gauge_coords[3,1]), int(gauge_coords[3,0])] #114.797
# gauge_obs[4] = radar_field[int(gauge_coords[4,1]), int(gauge_coords[4,0])] #114.797
# gauge_obs[5] = radar_field[int(gauge_coords[5,1]), int(gauge_coords[5,0])] #114.797
# gauge_obs[6] = radar_field[int(gauge_coords[6,1]), int(gauge_coords[6,0])] #114.797
# gauge_obs[7] = radar_field[int(gauge_coords[7,1]), int(gauge_coords[7,0])] #114.797
# gauge_obs[8] = radar_field[int(gauge_coords[8,1]), int(gauge_coords[8,0])] #114.797
# gauge_obs[9] = radar_field[int(gauge_coords[9,1]), int(gauge_coords[9,0])] #114.797

# # gauge_obs = gauge_obs * 1.5 #mielivaltainen muutos arvoihin

# # # Mixed Error
# # adjuster_mixed_rad = adjust.AdjustMixed(gauge_coords, grid_coords)
# # adjusted_mixed_rad = adjuster_mixed_rad(gauge_obs, radar_field2)

# # adjuster_mfb_rad = adjust.AdjustMFB(gauge_coords, grid_coords)
# # adjusted_mfb_rad = mfbadjuster(gauge_obs, radar_field2)

# # # Maximum value (used for normalisation of colorscales)
# # # maxval = np.max(np.concatenate((radar_field2, gauge_obs, adjusted_mixed_rad)).ravel())
# # maxval = max(np.max(radar_field2),np.max(gauge_obs),np.max(adjusted_mixed_rad))

# # # Helper functions for grid plots
# # def gridplot_radar(data, title):
# #     """Quick and dirty helper function to produce a grid plot"""
# #     xplot = np.append(xgrid, xgrid[-1] + 1.0) - 0.5
# #     yplot = np.append(ygrid, ygrid[-1] + 1.0) - 0.5
# #     grd = ax.pcolormesh(xplot, yplot, data.reshape(gridshape), vmin=0, vmax=maxval, cmap="nipy_spectral")
# #     ax.scatter(
# #         gauge_coords[:, 0],
# #         gauge_coords[:, 1],
# #         c=gauge_obs.ravel(),
# #         marker="s",
# #         s=50,
# #         vmin=0,
# #         vmax=maxval,
# #         # cmap="nipy_spectral",
# #     )
# #     # plt.colorbar(grd, shrink=0.5)
# #     plt.title(title)
    
# # # open figure
# # fig = plt.figure(figsize=(10, 6))
# # # Unadjusted radar rainfall
# # ax = fig.add_subplot(131, aspect="equal")
# # gridplot_radar(radar_field2, "Radar rainfall")
# # # Adjusted radar rainfall (Mixed)
# # ax = fig.add_subplot(132, aspect="equal")
# # gridplot_radar(adjusted_mixed_rad, "Adjusted (Mixed)")
# # # Adjusted radar rainfall (Mixed)
# # ax = fig.add_subplot(133, aspect="equal")
# # gridplot_radar(adjusted_mfb_rad, "Adjusted (MFB)")
# # plt.tight_layout()
    
