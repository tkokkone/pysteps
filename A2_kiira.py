# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:30:51 2023

@author: lindgrv1
"""

import os
import rasterio
import numpy as np
from matplotlib import cm, pyplot as plt
import pysteps
from shapely.geometry import box
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd

from rasterstats import point_query

##############################################################################

# radar_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\ARTIKKELI_2\data_a2\kiira_radar"
radar_dir = r"W:\lindgrv1\Kiira_whole_day\kiira_14-21_dbz"
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

# Extent	259414.3261793270648923,6557596.6055000005289912 : 515418.4413677211850882,6813610.924983871169388
# Width	1024
# Height	1024
# Dimensions	X: 1024 Y: 1024 Bands: 1
# Origin	259414.3261793270648923,6813610.9249838711693883
# Pixel Size	250.004018738666133,-250.0139838709677349
# CRS EPSG:3067 - ETRS89 / TM35FIN(E,N)
# Units	meters
# 5 min time resolution
# 250x250 m2 spatial resolution

np.max(radar_kiira)
np.min(radar_kiira)
np.min(radar_kiira[np.nonzero(radar_kiira)])

#The following data is available for Finnish radar composite: radar reflectivity (dbz), conversion: Z[dBZ] = 0.5 * pixel value - 32
radar_kiira = (radar_kiira * 0.5) - 32

radar_kiira[radar_kiira < 10] = 3.1830486304816077  #Values less than threshold to zero
# radar_kiira[radar_kiira > 45] = 0.5*(45 + radar_kiira[radar_kiira > 45]) #Clear values over threshold of 45 dBZ

plt.figure()
plt.imshow(radar_kiira[25])

# ani = pysteps.visualization.animations.animate_interactive(radar_kiira,False,True,False,"Blues")
# ani2 = pysteps.visualization.animations.animate_interactive(radar_kiira,False,True,False,None)

##############################################################################

# Create bounding box
temp_raster = rasterio.open(os.path.join(radar_dir, file_list[0])) #import one rain field
print(temp_raster.crs) #coordinate reference system of the raster
temp_bounds = temp_raster.bounds #raster corner coordinates
bbox = box(*temp_bounds) #raster to GeoDataFrame
# print(bbox.wkt)
bbox_df = gpd.GeoDataFrame({"geometry":[bbox]}, crs=temp_raster.crs)

data_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/GIS-aineistot" 
mittarit = gpd.read_file(os.path.join(data_dir, "FMI_stations_2022-03.gpkg"))
# mittarit.crs
mittarit_tutkakuva = gpd.clip(mittarit, bbox_df)

dir_gauges_kiira = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\ARTIKKELI_2\data_a2\mittarit"
fp_gauges_kiira = os.path.join(dir_gauges_kiira, "gauges_kiira.csv")
mittarit_tutkakuva_data = pd.read_csv(fp_gauges_kiira, delimiter=(";"))
mittarit_tutkakuva_kiira = mittarit_tutkakuva[mittarit_tutkakuva["field_1"].isin(mittarit_tutkakuva_data["Station"])]

point_laune = Point(25.6309, 60.96211)
point_laune_df = gpd.GeoDataFrame({"geometry":[point_laune]}, crs="epsg:4326")
point_laune_df.crs
# point_laune_df = point_laune_df.to_crs(temp_raster.crs)

# point_pakila = Point(, )
# point_pakila_df = gpd.GeoDataFrame({"geometry":[point_pakila]}, crs="epsg:4326")
# point_pakila_df.crs
# # point_pakila_df = point_pakila_df.to_crs(temp_raster.crs)

suomi = gpd.read_file(os.path.join(data_dir, "mml-hallintorajat_10k", "2021", "SuomenValtakunta_2021_10k.shp"))
suomi = suomi.to_crs(temp_raster.crs)

countries = gpd.read_file(os.path.join(data_dir, "ne_10m_admin_0_countries", "ne_10m_admin_0_countries.shp"))
# countries.crs
countries_finland = countries[countries["SOVEREIGNT"]=="Finland"]
countries_sweden = countries[countries["SOVEREIGNT"]=="Sweden"]
countries_norway = countries[countries["SOVEREIGNT"]=="Norway"]
countries_estonia = countries[countries["SOVEREIGNT"]=="Estonia"]
countries_russia = countries[countries["SOVEREIGNT"]=="Russia"]

suomi = suomi.to_crs(countries.crs)
bbox_df = bbox_df.to_crs(countries.crs)

fp_paajako = os.path.join(data_dir, "syke-valuma", "Paajako.shp") #Syken valuma-alue rasteri
alue_paajako = gpd.read_file(fp_paajako)
alue_paajako = alue_paajako.to_crs(countries.crs)
mittarit_tutkakuva = mittarit_tutkakuva.to_crs(countries.crs)
mittarit_tutkakuva_kiira = mittarit_tutkakuva_kiira.to_crs(countries.crs)

countries_sweden = countries_sweden.to_crs(temp_raster.crs)
countries_norway = countries_norway.to_crs(temp_raster.crs)
countries_estonia = countries_estonia.to_crs(temp_raster.crs)
countries_russia = countries_russia.to_crs(temp_raster.crs)
alue_paajako = alue_paajako.to_crs(temp_raster.crs)
suomi = suomi.to_crs(temp_raster.crs)
bbox_df = bbox_df.to_crs(temp_raster.crs)
mittarit_tutkakuva_kiira = mittarit_tutkakuva_kiira.to_crs(temp_raster.crs)
point_laune_df = point_laune_df.to_crs(temp_raster.crs)

# temp_df = mittarit_tutkakuva_kiira.copy()
# temp_df = temp_df.drop(temp_df.index[0:2])
# temp_df = temp_df.drop(temp_df.index[1:])

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_xlim(0, 800000)
ax.set_ylim(6500000, 7850000)
# ax.set_xlim(16, 33)
# ax.set_ylim(58.5, 71)
countries_sweden.plot(ax = ax, color="gainsboro")
countries_norway.plot(ax = ax, color="gainsboro")
countries_estonia.plot(ax = ax, color="gainsboro")
countries_russia.plot(ax = ax, color="gainsboro")
alue_paajako.plot(ax = ax, color="darkgray", ec="black", linewidth=0.5)
suomi.plot(ax = ax, fc="none", ec="black", linewidth=2)
bbox_df.plot(ax = ax, fc="none", ec="black", linewidth=2)
mittarit_tutkakuva_kiira.plot(ax = ax, color="red", marker="o")#, edgecolor="black", linewidth=1)
# temp_df.plot(ax = ax, color="red", marker="o")#, edgecolor="black", linewidth=1)
point_laune_df.plot(ax = ax, color="blue", marker="o")

##############################################################################
#LOCATIONS OF GAUGES

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

# fp_gauges_kiira_mm = os.path.join(dir_gauges_kiira, "gauges_kiira_mm.csv")
# gauge_data_mm = pd.read_csv(fp_gauges_kiira_mm, delimiter=(";"))
# gauge_data_mm_array = np.array(gauge_data_mm)
# gauge_data_mm_array = gauge_data_mm_array[:,5:-1]

# dir_pakila_mm = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\ARTIKKELI_2\data_a2\pakila_gauges"
# fp_pakila_mm = os.path.join(dir_pakila_mm, "kiira_gauges.csv")
# pakila_data_mm = pd.read_csv(fp_pakila_mm, delimiter=(";"))
# pakila_data_mm_array = np.array(pakila_data_mm)

# pakila_green = pakila_data_mm_array[:,1]
# pakila_green_accus = np.zeros([5,1])

# pakila_orange = pakila_data_mm_array[:,2]
# pakila_blue = pakila_data_mm_array[:,3]


##############################################################################

# Areal mean rainfall in dbz
areal_mean_rainfall_ts = np.zeros(len(file_list))
for i in range (len(file_list)):
    areal_mean_rainfall_ts[i] = np.nanmean(radar_kiira[i])
plt.figure()
plt.plot(areal_mean_rainfall_ts)

# Time step of maximum areal mean rainfall
max_idx = np.where(areal_mean_rainfall_ts == areal_mean_rainfall_ts.max())
max_idx = int(max_idx[0])


# Moving averages of areal mean rainfall and timestep of maximum moving average
window_size = 10
moving_averages = []
i = 0
while i < len(areal_mean_rainfall_ts) - window_size + 1:
    window = areal_mean_rainfall_ts[i : i + window_size]
    window_average = round(sum(window) / window_size, 2)
    moving_averages.append(window_average)
    i += 1
moving_averages = np.array(moving_averages)

max_idx2 = np.where(moving_averages == moving_averages.max())
max_idx2 = int(max_idx2[0])

##############################################################################

# Power law filters using only one field (max areal mean rainfall) and average of 10 fields (max moving average)

R2 = radar_kiira.copy() 
R3 = R2[max_idx2:max_idx2+window_size]
for i in range(R2.shape[0]):
    R2[i, ~np.isfinite(R2[i, :])] = np.nanmin(R2[i, :])

step=0
Fp = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(R2[max_idx])
Fnp = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2[max_idx])
Fp_ave = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(R3)
Fnp_ave = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R3)

seed = 1234
num_realizations = 3

# Generate noise
Np = []
Nnp = []
Np_ave = []
Nnp_ave = []
for k in range(num_realizations):
    Np.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fp, seed=seed + k))
    Nnp.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp, seed=seed + k))
    Np_ave.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fp_ave, seed=seed + k))
    Nnp_ave.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_ave, seed=seed + k))


# Plot the generated noise fields
fig, ax = plt.subplots(nrows=2, ncols=3)
# parametric noise
ax[0, 0].imshow(Np[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 1].imshow(Np[1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 2].imshow(Np[2], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# nonparametric noise
ax[1, 0].imshow(Nnp[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 1].imshow(Nnp[1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 2].imshow(Nnp[2], cmap=cm.RdBu_r, vmin=-3, vmax=3)

for i in range(2):
    for j in range(3):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=2, ncols=3)
# parametric noise
ax[0, 0].imshow(Np_ave[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 1].imshow(Np_ave[1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 2].imshow(Np_ave[2], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# nonparametric noise
ax[1, 0].imshow(Nnp_ave[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 1].imshow(Nnp_ave[1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 2].imshow(Nnp_ave[2], cmap=cm.RdBu_r, vmin=-3, vmax=3)

for i in range(2):
    for j in range(3):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.tight_layout()
plt.show()

##############################################################################



