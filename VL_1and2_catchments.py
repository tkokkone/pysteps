# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 23:39:27 2022

@author: lindgrv1
"""

##############################################################################
# IMPORT PACKAGES

import geopandas as gpd #https://geopandas.org/en/stable/getting_started/introduction.html
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import rasterio.mask
from shapely.geometry import box
from rasterstats import zonal_stats, point_query
import numpy as np
import pandas as pd
from datetime import datetime
import pysteps
import time
from matplotlib_scalebar.scalebar import ScaleBar
import random
from shapely.ops import unary_union
from numpy import genfromtxt

##############################################################################
# DATA DIRECTORIES

data_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/GIS-aineistot" #letter "r" needs to be added in front of the path in Windows
fp1 = os.path.join(data_dir, "syke-valuma", "Paajako.shp") #Syken valuma-alue rasteri
fp2 = os.path.join(data_dir, "syke-valuma", "Jako3.shp") #Syken kolmannen jakovaiheen valuma-alue rasteri
fp3 = os.path.join(data_dir, "FMI_stations_2022-03.gpkg") #Ilmatieteenlaitoksen sää- ja sadehavaintoasemat
fp4 = os.path.join(data_dir, "FMI_stations_2022-05_rain.csv") #Ilmatieteenlaitoksen sää- ja sadehavaintoasemat, jotka mittaa sadetta (toukokuu 2022)

data_dir_location = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Event3_new/Simulations"
dir_list = os.listdir(data_dir_location)
chosen_realization = dir_list[0]
data_dir2 = os.path.join(data_dir_location, chosen_realization, "Event_tiffs") #simulated event
# data_dir2 = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simulations_pysteps/Event1_new/Simulations/Simulation_2087_4095_3282_3135/Event_tiffs"

fp5 = os.path.join(data_dir2, "test_0.tif") #for purpose of creating a bounding box

files = os.listdir(data_dir2)

#Output directory
# out_dir = os.path.join(data_dir_location, chosen_realization, "Calculations_thresholded")
out_dir = os.path.join(data_dir_location, chosen_realization, "Calculations_500")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
out_figs = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Figures"
# plt.savefig(os.path.join(out_figs,"figure_1.pdf"), format="pdf", bbox_inches="tight")

##############################################################################
# VARIABLES AND PARAMETERS

timestep = 5 #mins
timeseries_size = len(files) #steps
timeseries_length = timeseries_size * timestep #mins

window_length = 60 #mins
window_size = window_length / timestep #steps

##############################################################################
# READ SHAPEFILES AND CSV-FILE

alue_paajako = gpd.read_file(fp1)
alue_jako3 = gpd.read_file(fp2)
mittarit = gpd.read_file(fp3)
mittarit.crs

#Info which gauges measured rain during May 1.-12. (retrieved 13.5.2022)
#Either daily measurements or hourly + intensity/10min
mittarit_sade = pd.read_csv(fp4, delimiter=(";"))

#Import one simulated field for purpose of creating bounding box
raster0 = rasterio.open(fp5)

if raster0.crs == mittarit.crs:
    print("Projection is", raster0.crs, "on both. So all good!")

#Create bounding box
bounds = raster0.bounds #Raster corner coordinates
bbox = box(*bounds) #Raster to GeoDataFrame
#print(bbox.wkt)

##############################################################################
# SELECT KOKEMAENJOKI BASIN, RELATED SUBBASINS, AND GAUGES

kokemaki_index = ["35"]
kokemaki_alue = alue_paajako[alue_paajako["PaajakoTun"].isin(kokemaki_index)]
# kokemaki_alue.plot()

kokemaki_jako3 = gpd.clip(alue_jako3, kokemaki_alue, keep_geom_type=True)

#Tutka-alueen mittarit, jotka mittaa tunnistaista sadetta
bbox_df = gpd.GeoDataFrame({"geometry":[bbox]}, crs=kokemaki_alue.crs)
mittarit_tutkakuva = gpd.clip(mittarit, bbox_df)
mittarit_sade_h = mittarit_tutkakuva[mittarit_tutkakuva["field_1"].isin(mittarit_sade["field_1_may_13052022"][mittarit_sade["sademaara/h"]==1])]

##############################################################################
# CENTROIDS OF SUBBASINS (center of mass of a geometry)

kokemaki_jako3["centroid"] = kokemaki_jako3["geometry"].centroid

##############################################################################
# SOME MORE STATISTICS IN SUBBASINS SCALE 

#Area
kokemaki_jako3["area/m2"] = kokemaki_jako3.apply(lambda row: row["geometry"].area, axis=1)
kokemaki_jako3["area/km2"] = kokemaki_jako3["area/m2"]/1000000
area_min = kokemaki_jako3["area/km2"].min()
area_max = kokemaki_jako3["area/km2"].max()
area_mean = kokemaki_jako3["area/km2"].mean()
area_std = kokemaki_jako3["area/km2"].std()
area_median = kokemaki_jako3["area/km2"].median()
kokemaki_area_km2 = np.sum(kokemaki_jako3["area/km2"])

##############################################################################
# PÄÄVALUMA-ALUE (tunnus: 35)
#Kussakin osajaossa aina kolmanteen jakovaiheeseen asti on käsiteltävä alue 
#jaettu enintään 9 osa-alueeseen, jolloin yhdellä vesistöalueella on enintään 
#729 osa-aluetta. Näin on tehty päävesistöalueen koon ollessa yli 10 000 km2.

# 1. JAKOVAIHEEN OSAVALUMA-ALUEET (tunnus: 35.1 - 35.9)
# -> 9 osavaluma-aluetta
kokemaki_jako1 = []
for i in range(1,10):
    kokemaki_jako1.append(kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"].str.contains(f"35.{i}")])
len(kokemaki_jako1)

#Areas
areas_jako1 = []
for subbasin in range(len(kokemaki_jako1)):
    areas_jako1.append(np.sum(kokemaki_jako1[subbasin]["area/km2"]))
areas_jako1_arr = np.vstack(areas_jako1)
#Polygons
kokemaki_jako1_polys = []
for subbasin in range(len(kokemaki_jako1)):
    kokemaki_jako1_polys.append(gpd.GeoSeries(unary_union(kokemaki_jako1[subbasin]["geometry"])))
#Centroids
kokemaki_jako1_centroids = []
for subbasin in range(len(kokemaki_jako1_polys)):
    kokemaki_jako1_centroids.append(kokemaki_jako1_polys[subbasin].centroid)

# 2. JAKOVAIHEEN OSAVALUMA-ALUEET (tunnus: 35.11 - 35.99)
# -> 81 osavaluma-aluetta
kokemaki_jako2 = []
for i in range(1,10):
    for j in range(1,10):
        kokemaki_jako2.append(kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"].str.contains(f"35.{i}{j}")])
len(kokemaki_jako2)

#Areas
areas_jako2 = []
for subbasin in range(len(kokemaki_jako2)):
    areas_jako2.append(np.sum(kokemaki_jako2[subbasin]["area/km2"]))
areas_jako2_arr = np.vstack(areas_jako2)
#Polygons
kokemaki_jako2_polys = []
for subbasin in range(len(kokemaki_jako2)):
    kokemaki_jako2_polys.append(gpd.GeoSeries(unary_union(kokemaki_jako2[subbasin]["geometry"])))
#Centroids
kokemaki_jako2_centroids = []
for subbasin in range(len(kokemaki_jako2_polys)):
    kokemaki_jako2_centroids.append(kokemaki_jako2_polys[subbasin].centroid)

# 3. JAKOVAIHEEN OSAVALUMA-ALUEET (tunnus: 35.111 - 35.999)
# -> 494 osavaluma-aluetta
len(kokemaki_jako3)

#Check if sum of areas are equal
print(np.sum(kokemaki_jako3["area/km2"]))
print(np.sum(areas_jako2_arr))
print(np.sum(areas_jako1_arr))

##############################################################################
#Z-R relationship: Z = a*R^b (Reflectivity)

a_R=223
b_R=1.53

##############################################################################
# OSAPOL

# Events used in the study:
# 1. last radar image: 201306271955 -> number of previous files: 141
# 3. last radar image: 201310290345 -> number of previous files: 115
# 6. last radar image: 201408071800 -> number of previous files: 97

#Read in the event with pySTEPS
date = datetime.strptime("201310290345", "%Y%m%d%H%M") #last radar image of the event
data_source = pysteps.rcparams.data_sources["osapol"]

#Load the data from the archive
root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]
timestep = data_source["timestep"]

#Find the input files from the archive
fns = pysteps.io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=115)

#Select the importer
importer = pysteps.io.get_method(importer_name, "importer")

#Read the radar composites
event_osapol_array_mmh, quality, event_osapol_metadata_mmh = pysteps.io.read_timeseries(fns, importer, **importer_kwargs)
del quality  #delete quality variable because it is not used

event_osapol_array_mmh_raw = event_osapol_array_mmh.copy()

#Add unit to metadata
event_osapol_metadata_mmh["unit"] = "mm/h"

#Values less than threshold to zero
event_osapol_array_mmh[event_osapol_array_mmh<0.1] = 0

#Add some metadata
event_osapol_metadata_mmh["zerovalue"] = 0
event_osapol_metadata_mmh["threshold"] = 0.1

#dBZ transformation for mm/h-data
event_osapol_array, event_osapol_metadata = pysteps.utils.conversion.to_reflectivity(event_osapol_array_mmh, event_osapol_metadata_mmh, zr_a=a_R, zr_b=b_R)

##############################################################################
# LOOP TO GO THROUGH ALL ENSEMBLE MEMBERS

zonal_1_accus = np.zeros((len(kokemaki_jako1_polys), len(dir_list)))
zonal_2_accus = np.zeros((len(kokemaki_jako2_polys), len(dir_list)))
zonal_3_accus = np.zeros((len(kokemaki_jako3), len(dir_list)))

for member in range(len(dir_list)):
    chosen_realization = dir_list[member]
    data_dir2 = os.path.join(data_dir_location, chosen_realization, "Event_tiffs") #simulated event
    
    fp5 = os.path.join(data_dir2, "test_0.tif") #for purpose of creating a bounding box
    
    files = os.listdir(data_dir2)

    #Output directory
    out_dir2 = os.path.join(data_dir_location, chosen_realization, "Calculations_500")
    if not os.path.exists(out_dir2):
        os.makedirs(out_dir2)

##############################################################################
# IMPORT SIMULATED EVENT
    
    #Import simulated event in raster-format and convert it into array
    event_sim_array  = []
    for i in range(len(files)):
        temp_raster = rasterio.open(os.path.join(data_dir2, f"test_{i}.tif"))
        temp_array = temp_raster.read(1)
        event_sim_array.append(temp_array)
        if i == 0:
            event_affine = temp_raster.transform  
    event_sim_array = np.concatenate([event_sim_array_[None, :, :] for event_sim_array_ in event_sim_array])
    
    #Clear values over threshold of 45 dBZ
    event_sim_array[event_sim_array > 45] = 0.5*(45 + event_sim_array[event_sim_array > 45])
    
    # #Test plot before thresholding
    # plt.figure()
    # plt.imshow(event_sim_array[78])
    
    #Mean and std timeseries before thresholding
    mean_ts_before = np.zeros(len(event_sim_array))
    std_ts_before = np.zeros(len(event_sim_array))
    for i in range (len(event_sim_array)):
        mean_ts_before[i] = np.nanmean(event_sim_array[i])
        std_ts_before[i] = np.nanstd(event_sim_array[i])
    
    # plt.figure()
    # plt.plot(mean_ts_before)
    # plt.figure()
    # plt.plot(std_ts_before)

    # #Clear values over threshold of 45 dBZ
    # event_sim_array[event_sim_array > 45] = 45
    
    # #Test plot after thresholding
    # plt.figure()
    # plt.imshow(event_sim_array[78])
    
    #Mean and std timeseries after thresholding
    mean_ts_after = np.zeros(len(event_sim_array))
    std_ts_after = np.zeros(len(event_sim_array))
    for i in range (len(event_sim_array)):
        mean_ts_after[i] = np.nanmean(event_sim_array[i])
        std_ts_after[i] = np.nanstd(event_sim_array[i])
    
    # plt.figure()
    # plt.plot(mean_ts_after)
    # plt.figure()
    # plt.plot(std_ts_after)
    
    #Before-After plot
    plt.figure()
    plt.plot(mean_ts_before)
    plt.plot(mean_ts_after)
    plt.savefig(os.path.join(out_dir2, "means.png"))
    plt.figure()
    plt.plot(std_ts_before)
    plt.plot(std_ts_after)
    plt.savefig(os.path.join(out_dir2, "stds.png"))
    
    #Save timeseries
    data_temp = [mean_ts_before, mean_ts_after, std_ts_before, std_ts_after]
    stats_tss = pd.DataFrame(data_temp, index=['mean_before', 'mean_after', 'std_before', 'std_after'])
    # fp_thresholding = os.path.join(r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simulations_pysteps/Event1_new/Subbasins/Thresholding_with_45dbz")
    pd.DataFrame(stats_tss).to_csv(os.path.join(out_dir2, "mean_and_std_tss.csv"))
    
    #Event from dbz into mm/h
    event_sim_array_mmh = event_sim_array.copy()
    event_sim_array_mmh = 10**((event_sim_array_mmh-10*np.log10(a_R))/(10*b_R))
    
    #Values less than threshold to zero
    event_sim_array_mmh[event_sim_array_mmh < 0.1] = 0
    
##############################################################################
# ZONAL MEANS

    #ZONAL: 1.JAKOVAIHE
    zonal_1_means_all = np.zeros((len(kokemaki_jako1_polys), len(event_sim_array_mmh)))
    for j in range(len(event_sim_array_mmh)): #timesteps
        zonal_1_test = []    
        for i in range(len(kokemaki_jako1_polys)): #subbasins
            zonal_1_temp = zonal_stats(kokemaki_jako1_polys[i], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
            zonal_1_test.append(float(zonal_1_temp[0]["mean"]))
        zonal_1_test_arr = np.array(zonal_1_test)
        zonal_1_means_all[:,j] = zonal_1_test_arr
    
    #ZONAL: 2.JAKOVAIHE
    zonal_2_means_all = np.zeros((len(kokemaki_jako2_polys), len(event_sim_array_mmh)))
    for j in range(len(event_sim_array_mmh)): #timesteps
        zonal_2_test = []    
        for i in range(len(kokemaki_jako2_polys)): #subbasins
            zonal_2_temp = zonal_stats(kokemaki_jako2_polys[i], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
            zonal_2_test.append(float(zonal_2_temp[0]["mean"]))
        zonal_2_test_arr = np.array(zonal_2_test)
        zonal_2_means_all[:,j] = zonal_2_test_arr
        
    #ZONAL: 3.JAKOVAIHE
    fp7 = os.path.join(data_dir_location, chosen_realization, "Calculations_500", "subbasin_zonal_means.csv")
    temp_zonal_means_all_pd = pd.read_csv(fp7, delimiter=(","))
    temp_zonal_means_all = temp_zonal_means_all_pd.to_numpy()
    zonal_3_means_all = temp_zonal_means_all[:, 1:]
    
##############################################################################
# ZONAL ACCUMULATIONS

    zonal_1_means_all_accu = np.sum(zonal_1_means_all, axis=1) * (timestep/60)
    zonal_2_means_all_accu = np.sum(zonal_2_means_all, axis=1) * (timestep/60)
    zonal_3_means_all_accu = np.sum(zonal_3_means_all, axis=1) * (timestep/60)
    
    zonal_1_accus[:, member] = zonal_1_means_all_accu
    zonal_2_accus[:, member] = zonal_2_means_all_accu
    zonal_3_accus[:, member] = zonal_3_means_all_accu

#Save csv
fp_zonal = os.path.join(r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Event3_new/Subbasins_500")
zonal_1_accus_pd = pd.DataFrame(zonal_1_accus)
pd.DataFrame(zonal_1_accus_pd).to_csv(os.path.join(fp_zonal, "zonal_1_accus.csv"))
zonal_2_accus_pd = pd.DataFrame(zonal_2_accus)
pd.DataFrame(zonal_2_accus_pd).to_csv(os.path.join(fp_zonal, "zonal_2_accus.csv"))
zonal_3_accus_pd = pd.DataFrame(zonal_3_accus)
pd.DataFrame(zonal_3_accus_pd).to_csv(os.path.join(fp_zonal, "zonal_3_accus.csv"))

##############################################################################
# READ ACCUMULATION CSVS

zonal_1_accus_pd = pd.read_csv(os.path.join(fp_zonal, "zonal_1_accus.csv"), delimiter=(","))
zonal_2_accus_pd = pd.read_csv(os.path.join(fp_zonal, "zonal_2_accus.csv"), delimiter=(","))
zonal_3_accus_pd = pd.read_csv(os.path.join(fp_zonal, "zonal_3_accus.csv"), delimiter=(","))

zonal_1_accus = zonal_1_accus_pd.to_numpy()
zonal_2_accus = zonal_2_accus_pd.to_numpy()
zonal_3_accus = zonal_3_accus_pd.to_numpy()

zonal_1_accus = zonal_1_accus[:, 1:]
zonal_2_accus = zonal_2_accus[:, 1:]
zonal_3_accus = zonal_3_accus[:, 1:]

##############################################################################
# STATISTICS

zonal_1_accus_min = zonal_1_accus.min(axis=1)
zonal_1_accus_max = zonal_1_accus.max(axis=1)
zonal_1_accus_mean = zonal_1_accus.mean(axis=1)
zonal_1_accus_var = zonal_1_accus.var(axis=1)
zonal_1_accus_std = zonal_1_accus.std(axis=1)

zonal_2_accus_min = zonal_2_accus.min(axis=1)
zonal_2_accus_max = zonal_2_accus.max(axis=1)
zonal_2_accus_mean = zonal_2_accus.mean(axis=1)
zonal_2_accus_var = zonal_2_accus.var(axis=1)
zonal_2_accus_std = zonal_2_accus.std(axis=1)

zonal_3_accus_min = zonal_3_accus.min(axis=1)
zonal_3_accus_max = zonal_3_accus.max(axis=1)
zonal_3_accus_mean = zonal_3_accus.mean(axis=1)
zonal_3_accus_var = zonal_3_accus.var(axis=1)
zonal_3_accus_std = zonal_3_accus.std(axis=1)

##############################################################################
# GEODATAFRAMES

# 1.JAKOVAIHE
#Copy geodataframe
zonal_1_geo = kokemaki_jako3.copy()
#Drop some rows to have only 9 subbasins
zonal_1_geo = zonal_1_geo.head(-(len(kokemaki_jako3)-len(kokemaki_jako1_polys)))
#Edit geometry-column
kokemaki_jako1_polys_arr = np.array(kokemaki_jako1_polys)
zonal_1_geo["geometry"] = kokemaki_jako1_polys_arr
#Edit centroids-column
kokemaki_jako1_centroids_arr = np.array(kokemaki_jako1_centroids)
zonal_1_geo["centroid"] = kokemaki_jako1_centroids_arr
#Edit area-columns
zonal_1_geo["area/km2"] = areas_jako1_arr
zonal_1_geo["area/m2"] = zonal_1_geo["area/km2"] * 1000000
#Add new columns
zonal_1_geo["zonal_min"] = zonal_1_accus_min
zonal_1_geo["zonal_max"] = zonal_1_accus_max
zonal_1_geo["zonal_mean"] = zonal_1_accus_mean
zonal_1_geo["zonal_var"] = zonal_1_accus_var
zonal_1_geo["zonal_std"] = zonal_1_accus_std

# 2.JAKOVAIHE
#Copy geodataframe
zonal_2_geo = kokemaki_jako3.copy()
#Drop some rows to have only 9 subbasins
zonal_2_geo = zonal_2_geo.head(-(len(kokemaki_jako3)-len(kokemaki_jako2_polys)))
#Edit geometry-column
kokemaki_jako2_polys_arr = np.array(kokemaki_jako2_polys)
zonal_2_geo["geometry"] = kokemaki_jako2_polys_arr
#Edit centroids-column
kokemaki_jako2_centroids_arr = np.array(kokemaki_jako2_centroids)
zonal_2_geo["centroid"] = kokemaki_jako2_centroids_arr
#Edit area-columns
zonal_2_geo["area/km2"] = areas_jako2_arr
zonal_2_geo["area/m2"] = zonal_2_geo["area/km2"] * 1000000
#Add new columns
zonal_2_geo["zonal_min"] = zonal_2_accus_min
zonal_2_geo["zonal_max"] = zonal_2_accus_max
zonal_2_geo["zonal_mean"] = zonal_2_accus_mean
zonal_2_geo["zonal_var"] = zonal_2_accus_var
zonal_2_geo["zonal_std"] = zonal_2_accus_std

# 3.JAKOVAIHE
zonal_3_geo = kokemaki_jako3.copy()
zonal_3_geo["zonal_min"] = zonal_3_accus_min
zonal_3_geo["zonal_max"] = zonal_3_accus_max
zonal_3_geo["zonal_mean"] = zonal_3_accus_mean
zonal_3_geo["zonal_var"] = zonal_3_accus_var
zonal_3_geo["zonal_std"] = zonal_3_accus_std

##############################################################################
# PLOTS

##########
#Subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1.set_xlim(185000, 475000)
ax1.set_ylim(6700000, 6990000)
ax2.set_xlim(185000, 475000)
ax2.set_ylim(6700000, 6990000)
ax3.set_xlim(185000, 475000)
ax3.set_ylim(6700000, 6990000)
alue_paajako.plot(ax = ax1, color="darkgray", ec="black", linewidth=0.5)
kokemaki_alue.plot(ax = ax1, fc="none", ec="black", linewidth=2)
bbox_df.plot(ax = ax1, fc="none", ec="black", linewidth=2)
alue_paajako.plot(ax = ax2, color="darkgray", ec="black", linewidth=0.5)
kokemaki_alue.plot(ax = ax2, fc="none", ec="black", linewidth=2)
bbox_df.plot(ax = ax2, fc="none", ec="black", linewidth=2)
alue_paajako.plot(ax = ax3, color="darkgray", ec="black", linewidth=0.5)
kokemaki_alue.plot(ax = ax3, fc="none", ec="black", linewidth=2)
bbox_df.plot(ax = ax3, fc="none", ec="black", linewidth=2)

#JAKO_1
for i in range(len(kokemaki_jako1)):
    kokemaki_jako1_polys[i].plot(ax = ax1, color="lightblue", ec="black", linewidth=0.5)
#JAKO_2
for j in range(len(kokemaki_jako2)):
    kokemaki_jako2_polys[j].plot(ax = ax2, color="lightblue", ec="black", linewidth=0.5)
#JAKO_3
kokemaki_jako3.plot(ax = ax3, color="lightblue", ec="black", linewidth=0.5)
mittarit_sade_h.plot(ax = ax3, color="red", marker="^", markersize=200, edgecolor="black", linewidth=1)

#Titles
ax1.title.set_text("1st level")
ax2.title.set_text("2nd level")
ax3.title.set_text("3rd level")

#Arrow
x, y, arrow_length = 0.9, 0.90, 0.1
ax1.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax1.transAxes)
ax2.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax2.transAxes)
ax3.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax3.transAxes)

#Scalebar
scalebar1 = ScaleBar(1, "m", length_fraction=0.4, location="lower right", scale_loc="top")
scalebar2 = ScaleBar(1, "m", length_fraction=0.4, location="lower right", scale_loc="top")
scalebar3 = ScaleBar(1, "m", length_fraction=0.4, location="lower right", scale_loc="top")
ax1.add_artist(scalebar1)
ax2.add_artist(scalebar2)
ax3.add_artist(scalebar3)

fig.tight_layout()

# plt.savefig(os.path.join(fp_zonal, "subbasin_levels.png"))

############################
#Ensemble MEAN for subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#scale
minmin_mean = np.min([np.min(zonal_1_geo["zonal_mean"]), np.min(zonal_2_geo["zonal_mean"]), np.min(zonal_3_geo["zonal_mean"])])
minmin_mean = 0.0
maxmax_mean = np.max([np.max(zonal_1_geo["zonal_mean"]), np.max(zonal_2_geo["zonal_mean"]), np.max(zonal_3_geo["zonal_mean"])])
maxmax_mean = round(maxmax_mean + 0.5)
#maps
im1 = zonal_1_geo.plot(column = "zonal_mean", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_mean, vmax=maxmax_mean)
im2 = zonal_2_geo.plot(column = "zonal_mean", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_mean, vmax=maxmax_mean)
im3 = zonal_3_geo.plot(column = "zonal_mean", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_mean, vmax=maxmax_mean)
#titles
ax1.title.set_text("1.jako: Ensemble MEAN")
ax2.title.set_text("2.jako: Ensemble MEAN")
ax3.title.set_text("3.jako: Ensemble MEAN")
#edges
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
zonal_1_geo.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_2_geo.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_3_geo.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_1_geo.plot(ax=ax2, facecolor="None", edgecolor="red", linewidth=0.5)
zonal_1_geo.plot(ax=ax3, facecolor="None", edgecolor="red", linewidth=0.5)

# plt.savefig(os.path.join(fp_zonal, "zonal_accu_ensemble_mean.png"))

###########################
#Ensemble STD for subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#scale
minmin_std = np.min([np.min(zonal_1_geo["zonal_std"]), np.min(zonal_2_geo["zonal_std"]), np.min(zonal_3_geo["zonal_std"])])
minmin_std = 0.0
maxmax_std = np.max([np.max(zonal_1_geo["zonal_std"]), np.max(zonal_2_geo["zonal_std"]), np.max(zonal_3_geo["zonal_std"])])
maxmax_std = round(maxmax_std + 0.5)
#maps
im1 = zonal_1_geo.plot(column = "zonal_std", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_std, vmax=maxmax_std)
im2 = zonal_2_geo.plot(column = "zonal_std", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_std, vmax=maxmax_std)
im3 = zonal_3_geo.plot(column = "zonal_std", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_std, vmax=maxmax_std)
#titles
ax1.title.set_text("1.jako: Ensemble STD")
ax2.title.set_text("2.jako: Ensemble STD")
ax3.title.set_text("3.jako: Ensemble STD")
#edges
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
zonal_1_geo.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_2_geo.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_3_geo.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_1_geo.plot(ax=ax2, facecolor="None", edgecolor="red", linewidth=0.5)
zonal_1_geo.plot(ax=ax3, facecolor="None", edgecolor="red", linewidth=0.5)

# plt.savefig(os.path.join(fp_zonal, "zonal_accu_ensemble_std.png"))

###########################
#Ensemble VAR for subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#scale
minmin_var = np.min([np.min(zonal_1_geo["zonal_var"]), np.min(zonal_2_geo["zonal_var"]), np.min(zonal_3_geo["zonal_var"])])
minmin_var = 0.0
maxmax_var = np.max([np.max(zonal_1_geo["zonal_var"]), np.max(zonal_2_geo["zonal_var"]), np.max(zonal_3_geo["zonal_var"])])
maxmax_var = round(maxmax_var + 0.5)
#maps
im1 = zonal_1_geo.plot(column = "zonal_var", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_var, vmax=maxmax_var)
im2 = zonal_2_geo.plot(column = "zonal_var", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_var, vmax=maxmax_var)
im3 = zonal_3_geo.plot(column = "zonal_var", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_var, vmax=maxmax_var)
#titles
ax1.title.set_text("1.jako: Ensemble VAR")
ax2.title.set_text("2.jako: Ensemble VAR")
ax3.title.set_text("3.jako: Ensemble VAR")
#edges
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
zonal_1_geo.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_2_geo.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_3_geo.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_1_geo.plot(ax=ax2, facecolor="None", edgecolor="red", linewidth=0.5)
zonal_1_geo.plot(ax=ax3, facecolor="None", edgecolor="red", linewidth=0.5)

# plt.savefig(os.path.join(fp_zonal, "zonal_accu_ensemble_var.png"))

###########################
#Ensemble MAX for subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#scale
minmin_max = np.min([np.min(zonal_1_geo["zonal_max"]), np.min(zonal_2_geo["zonal_max"]), np.min(zonal_3_geo["zonal_max"])])
minmin_max = 0.0
maxmax_max = np.max([np.max(zonal_1_geo["zonal_max"]), np.max(zonal_2_geo["zonal_max"]), np.max(zonal_3_geo["zonal_max"])])
maxmax_max = round(maxmax_max + 0.5)
#maps
im1 = zonal_1_geo.plot(column = "zonal_max", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_max, vmax=maxmax_max)
im2 = zonal_2_geo.plot(column = "zonal_max", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_max, vmax=maxmax_max)
im3 = zonal_3_geo.plot(column = "zonal_max", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_max, vmax=maxmax_max)
#titles
ax1.title.set_text("1.jako: Ensemble MAX")
ax2.title.set_text("2.jako: Ensemble MAX")
ax3.title.set_text("3.jako: Ensemble MAX")
#edges
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
zonal_1_geo.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_2_geo.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_3_geo.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_1_geo.plot(ax=ax2, facecolor="None", edgecolor="red", linewidth=0.5)
zonal_1_geo.plot(ax=ax3, facecolor="None", edgecolor="red", linewidth=0.5)

# plt.savefig(os.path.join(fp_zonal, "zonal_accu_ensemble_max.png"))

###########################
#Ensemble MIN for subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#scale
minmin_min = np.min([np.min(zonal_1_geo["zonal_min"]), np.min(zonal_2_geo["zonal_min"]), np.min(zonal_3_geo["zonal_min"])])
minmin_min = 0.0
maxmax_min = np.max([np.max(zonal_1_geo["zonal_min"]), np.max(zonal_2_geo["zonal_min"]), np.max(zonal_3_geo["zonal_min"])])
maxmax_min = round(maxmax_min + 0.5)
#maps
im1 = zonal_1_geo.plot(column = "zonal_min", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_min, vmax=maxmax_min)
im2 = zonal_2_geo.plot(column = "zonal_min", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_min, vmax=maxmax_min)
im3 = zonal_3_geo.plot(column = "zonal_min", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=minmin_min, vmax=maxmax_min)
#titles
ax1.title.set_text("1.jako: Ensemble MIN")
ax2.title.set_text("2.jako: Ensemble MIN")
ax3.title.set_text("3.jako: Ensemble MIN")
#edges
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
zonal_1_geo.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_2_geo.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_3_geo.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=0.2)
zonal_1_geo.plot(ax=ax2, facecolor="None", edgecolor="red", linewidth=0.5)
zonal_1_geo.plot(ax=ax3, facecolor="None", edgecolor="red", linewidth=0.5)

# plt.savefig(os.path.join(fp_zonal, "zonal_accu_ensemble_min.png"))

##############################################################################
##############################################################################
##############################################################################
# POINT MEANS

point_1_accus = np.zeros((len(kokemaki_jako1_polys), len(dir_list)))
point_2_accus = np.zeros((len(kokemaki_jako2_polys), len(dir_list)))
point_3_accus = np.zeros((len(kokemaki_jako3), len(dir_list)))

for member in range(len(dir_list)):
    print(member)
    chosen_realization = dir_list[member]

##############################################################################
    #POINT: 3.JAKOVAIHE
    fp6 = os.path.join(data_dir_location, chosen_realization, "Calculations_500", "subbasin_point_means.csv")
    temp_point_means_all_pd = pd.read_csv(fp6, delimiter=(","))
    temp_point_means_all = temp_point_means_all_pd.to_numpy()
    point_3_means_all = temp_point_means_all[:, 1:]
    
    #lisätään jakotunnukset
    subbasins_all = np.array(kokemaki_jako3["Jako3Tunnu"])
    subbasins_all = np.expand_dims(subbasins_all, axis=1)
    subbasins_all = np.hstack((subbasins_all,point_3_means_all))
    
    #POINT: 2.JAKOVAIHE
    point_2_means_all = []
    point_2_areas = []
    for i in range(len(kokemaki_jako2)):
        temp_list = []
        temp_list_areas = []
        temp_arr = np.array(kokemaki_jako2[i]["Jako3Tunnu"])
        for element in range(len(temp_arr)):
            temp_list.append(subbasins_all[subbasins_all[:,0] == temp_arr[element]])
            temp_list_areas.append(kokemaki_jako2[i]["area/km2"][kokemaki_jako2[i]["Jako3Tunnu"] == temp_arr[element]])
        temp_list = np.vstack(temp_list)
        temp_list_areas = np.vstack(temp_list_areas)
        point_2_means_all.append(temp_list)
        point_2_areas.append(temp_list_areas)
    
    #POINT: 1.JAKOVAIHE
    point_1_means_all = []
    point_1_areas = []
    for i in range(len(kokemaki_jako1)):
        temp_list = []
        temp_list_areas = []
        temp_arr = np.array(kokemaki_jako1[i]["Jako3Tunnu"])
        for element in range(len(temp_arr)):
            temp_list.append(subbasins_all[subbasins_all[:,0] == temp_arr[element]])
            temp_list_areas.append(kokemaki_jako1[i]["area/km2"][kokemaki_jako1[i]["Jako3Tunnu"] == temp_arr[element]])
        temp_list = np.vstack(temp_list)
        temp_list_areas = np.vstack(temp_list_areas)
        point_1_means_all.append(temp_list)
        point_1_areas.append(temp_list_areas)
    
##############################################################################
    # POINT ACCUMULATIONS
    point_3_means_all_accu = np.sum(point_3_means_all, axis=1) * (timestep/60)
    
    point_2_means_all_accu = []
    for i in range(len(kokemaki_jako2)):
        temp_list = []
        temp_arr = np.sum(point_2_means_all[i][:,1:], axis=1) * (timestep/60)
        temp_accu = 0
        temp_kerroin = 0
        for j in range(len(temp_arr)):
            temp_accu = temp_accu + (temp_arr[j] * (float(point_2_areas[i][j])))
            temp_kerroin = temp_kerroin + (float(point_2_areas[i][j]))
        temp_accu = temp_accu / temp_kerroin
        point_2_means_all_accu.append(temp_accu)
    point_2_means_all_accu = np.vstack(point_2_means_all_accu)
    
    point_1_means_all_accu = []
    for i in range(len(kokemaki_jako1)):
        temp_list = []
        temp_arr = np.sum(point_1_means_all[i][:,1:], axis=1) * (timestep/60)
        temp_accu = 0
        temp_kerroin = 0
        for j in range(len(temp_arr)):
            temp_accu = temp_accu + (temp_arr[j] * (float(point_1_areas[i][j])))
            temp_kerroin = temp_kerroin + (float(point_1_areas[i][j]))
        temp_accu = temp_accu / temp_kerroin
        point_1_means_all_accu.append(temp_accu)
    point_1_means_all_accu = np.vstack(point_1_means_all_accu)
    
    point_1_accus[:, member] = point_1_means_all_accu[:,0]
    point_2_accus[:, member] = point_2_means_all_accu[:,0]
    point_3_accus[:, member] = point_3_means_all_accu

##############################################################################
#Save csv
fp_point = os.path.join(r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Event3_new/Subbasins_500")
point_1_accus_pd = pd.DataFrame(point_1_accus)
pd.DataFrame(point_1_accus_pd).to_csv(os.path.join(fp_point, "point_1_accus.csv"))
point_2_accus_pd = pd.DataFrame(point_2_accus)
pd.DataFrame(point_2_accus_pd).to_csv(os.path.join(fp_point, "point_2_accus.csv"))
point_3_accus_pd = pd.DataFrame(point_3_accus)
pd.DataFrame(point_3_accus_pd).to_csv(os.path.join(fp_point, "point_3_accus.csv"))

##############################################################################
# READ ACCUMULATION CSVS

point_1_accus_pd = pd.read_csv(os.path.join(fp_point, "point_1_accus.csv"), delimiter=(","))
point_2_accus_pd = pd.read_csv(os.path.join(fp_point, "point_2_accus.csv"), delimiter=(","))
point_3_accus_pd = pd.read_csv(os.path.join(fp_point, "point_3_accus.csv"), delimiter=(","))

point_1_accus = point_1_accus_pd.to_numpy()
point_2_accus = point_2_accus_pd.to_numpy()
point_3_accus = point_3_accus_pd.to_numpy()

point_1_accus = point_1_accus[:, 1:]
point_2_accus = point_2_accus[:, 1:]
point_3_accus = point_3_accus[:, 1:]

##############################################################################
# STATISTICS

point_1_accus_min = point_1_accus.min(axis=1)
point_1_accus_max = point_1_accus.max(axis=1)
point_1_accus_mean = point_1_accus.mean(axis=1)
point_1_accus_var = point_1_accus.var(axis=1)
point_1_accus_std = point_1_accus.std(axis=1)

point_2_accus_min = point_2_accus.min(axis=1)
point_2_accus_max = point_2_accus.max(axis=1)
point_2_accus_mean = point_2_accus.mean(axis=1)
point_2_accus_var = point_2_accus.var(axis=1)
point_2_accus_std = point_2_accus.std(axis=1)

point_3_accus_min = point_3_accus.min(axis=1)
point_3_accus_max = point_3_accus.max(axis=1)
point_3_accus_mean = point_3_accus.mean(axis=1)
point_3_accus_var = point_3_accus.var(axis=1)
point_3_accus_std = point_3_accus.std(axis=1)

##############################################################################
# GEODATAFRAMES

# 1.JAKOVAIHE
#Copy geodataframe
point_1_geo = kokemaki_jako3.copy()
#Drop some rows to have only 9 subbasins
point_1_geo = point_1_geo.head(-(len(kokemaki_jako3)-len(kokemaki_jako1_polys)))
#Edit geometry-column
kokemaki_jako1_polys_arr = np.array(kokemaki_jako1_polys)
point_1_geo["geometry"] = kokemaki_jako1_polys_arr
#Edit centroids-column
kokemaki_jako1_centroids_arr = np.array(kokemaki_jako1_centroids)
point_1_geo["centroid"] = kokemaki_jako1_centroids_arr
#Edit area-columns
point_1_geo["area/km2"] = areas_jako1_arr
point_1_geo["area/m2"] = point_1_geo["area/km2"] * 1000000
#Add new columns
point_1_geo["zonal_min"] = point_1_accus_min
point_1_geo["zonal_max"] = point_1_accus_max
point_1_geo["zonal_mean"] = point_1_accus_mean
point_1_geo["zonal_var"] = point_1_accus_var
point_1_geo["zonal_std"] = point_1_accus_std

# 2.JAKOVAIHE
#Copy geodataframe
point_2_geo = kokemaki_jako3.copy()
#Drop some rows to have only 9 subbasins
point_2_geo = point_2_geo.head(-(len(kokemaki_jako3)-len(kokemaki_jako2_polys)))
#Edit geometry-column
kokemaki_jako2_polys_arr = np.array(kokemaki_jako2_polys)
point_2_geo["geometry"] = kokemaki_jako2_polys_arr
#Edit centroids-column
kokemaki_jako2_centroids_arr = np.array(kokemaki_jako2_centroids)
point_2_geo["centroid"] = kokemaki_jako2_centroids_arr
#Edit area-columns
point_2_geo["area/km2"] = areas_jako2_arr
point_2_geo["area/m2"] = zonal_2_geo["area/km2"] * 1000000
#Add new columns
point_2_geo["zonal_min"] = point_2_accus_min
point_2_geo["zonal_max"] = point_2_accus_max
point_2_geo["zonal_mean"] = point_2_accus_mean
point_2_geo["zonal_var"] = point_2_accus_var
point_2_geo["zonal_std"] = point_2_accus_std

# 3.JAKOVAIHE
point_3_geo = kokemaki_jako3.copy()
point_3_geo["zonal_min"] = point_3_accus_min
point_3_geo["zonal_max"] = point_3_accus_max
point_3_geo["zonal_mean"] = point_3_accus_mean
point_3_geo["zonal_var"] = point_3_accus_var
point_3_geo["zonal_std"] = point_3_accus_std

##############################################################################
# PLOTS

############################
#Ensemble MEAN for subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#scale
point_minmin_mean = np.min([np.min(point_1_geo["zonal_mean"]), np.min(point_2_geo["zonal_mean"]), np.min(point_3_geo["zonal_mean"])])
point_minmin_mean = 0.0
point_maxmax_mean = np.max([np.max(point_1_geo["zonal_mean"]), np.max(point_2_geo["zonal_mean"]), np.max(point_3_geo["zonal_mean"])])
point_maxmax_mean = round(point_maxmax_mean + 0.5)
#maps
im1 = point_1_geo.plot(column = "zonal_mean", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_mean, vmax=point_maxmax_mean)
im2 = point_2_geo.plot(column = "zonal_mean", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_mean, vmax=point_maxmax_mean)
im3 = point_3_geo.plot(column = "zonal_mean", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_mean, vmax=point_maxmax_mean)
#titles
ax1.title.set_text("1.jako: Ensemble MEAN")
ax2.title.set_text("2.jako: Ensemble MEAN")
ax3.title.set_text("3.jako: Ensemble MEAN")
#edges
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
point_1_geo.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=0.2)
point_2_geo.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=0.2)
point_3_geo.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=0.2)
point_1_geo.plot(ax=ax2, facecolor="None", edgecolor="red", linewidth=0.5)
point_1_geo.plot(ax=ax3, facecolor="None", edgecolor="red", linewidth=0.5)

# plt.savefig(os.path.join(fp_point, "point_accu_ensemble_mean.png"))

###########################
#Ensemble STD for subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#scale
point_minmin_std = np.min([np.min(point_1_geo["zonal_std"]), np.min(point_2_geo["zonal_std"]), np.min(point_3_geo["zonal_std"])])
point_minmin_std = 0.0
point_maxmax_std = np.max([np.max(point_1_geo["zonal_std"]), np.max(point_2_geo["zonal_std"]), np.max(point_3_geo["zonal_std"])])
point_maxmax_std = round(point_maxmax_std + 0.5)
#maps
im1 = point_1_geo.plot(column = "zonal_std", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_std, vmax=point_maxmax_std)
im2 = point_2_geo.plot(column = "zonal_std", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_std, vmax=point_maxmax_std)
im3 = point_3_geo.plot(column = "zonal_std", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_std, vmax=point_maxmax_std)
#titles
ax1.title.set_text("1.jako: Ensemble STD")
ax2.title.set_text("2.jako: Ensemble STD")
ax3.title.set_text("3.jako: Ensemble STD")
#edges
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
point_1_geo.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=0.2)
point_2_geo.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=0.2)
point_3_geo.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=0.2)
point_1_geo.plot(ax=ax2, facecolor="None", edgecolor="red", linewidth=0.5)
point_1_geo.plot(ax=ax3, facecolor="None", edgecolor="red", linewidth=0.5)

# plt.savefig(os.path.join(fp_point, "point_accu_ensemble_std.png"))

###########################
#Ensemble VAR for subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#scale
point_minmin_var = np.min([np.min(point_1_geo["zonal_var"]), np.min(point_2_geo["zonal_var"]), np.min(point_3_geo["zonal_var"])])
point_minmin_var = 0.0
point_maxmax_var = np.max([np.max(point_1_geo["zonal_var"]), np.max(point_2_geo["zonal_var"]), np.max(point_3_geo["zonal_var"])])
point_maxmax_var = round(point_maxmax_var + 0.5)
#maps
im1 = point_1_geo.plot(column = "zonal_var", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_var, vmax=point_maxmax_var)
im2 = point_2_geo.plot(column = "zonal_var", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_var, vmax=point_maxmax_var)
im3 = point_3_geo.plot(column = "zonal_var", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_var, vmax=point_maxmax_var)
#titles
ax1.title.set_text("1.jako: Ensemble VAR")
ax2.title.set_text("2.jako: Ensemble VAR")
ax3.title.set_text("3.jako: Ensemble VAR")
#edges
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
point_1_geo.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=0.2)
point_2_geo.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=0.2)
point_3_geo.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=0.2)
point_1_geo.plot(ax=ax2, facecolor="None", edgecolor="red", linewidth=0.5)
point_1_geo.plot(ax=ax3, facecolor="None", edgecolor="red", linewidth=0.5)

# plt.savefig(os.path.join(fp_point, "point_accu_ensemble_var.png"))

###########################
#Ensemble MAX for subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#scale
point_minmin_max = np.min([np.min(point_1_geo["zonal_max"]), np.min(point_2_geo["zonal_max"]), np.min(point_3_geo["zonal_max"])])
point_minmin_max = 0.0
point_maxmax_max = np.max([np.max(point_1_geo["zonal_max"]), np.max(point_2_geo["zonal_max"]), np.max(point_3_geo["zonal_max"])])
point_maxmax_max = round(point_maxmax_max + 0.5)
#maps
im1 = point_1_geo.plot(column = "zonal_max", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_max, vmax=point_maxmax_max)
im2 = point_2_geo.plot(column = "zonal_max", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_max, vmax=point_maxmax_max)
im3 = point_3_geo.plot(column = "zonal_max", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_max, vmax=point_maxmax_max)
#titles
ax1.title.set_text("1.jako: Ensemble MAX")
ax2.title.set_text("2.jako: Ensemble MAX")
ax3.title.set_text("3.jako: Ensemble MAX")
#edges
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
point_1_geo.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=0.2)
point_2_geo.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=0.2)
point_3_geo.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=0.2)
point_1_geo.plot(ax=ax2, facecolor="None", edgecolor="red", linewidth=0.5)
point_1_geo.plot(ax=ax3, facecolor="None", edgecolor="red", linewidth=0.5)

# plt.savefig(os.path.join(fp_point, "point_accu_ensemble_max.png"))

###########################
#Ensemble MIN for subbasins
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
#scale
point_minmin_min = np.min([np.min(point_1_geo["zonal_min"]), np.min(point_2_geo["zonal_min"]), np.min(point_3_geo["zonal_min"])])
point_minmin_min = 0.0
point_maxmax_min = np.max([np.max(point_1_geo["zonal_min"]), np.max(point_2_geo["zonal_min"]), np.max(point_3_geo["zonal_min"])])
point_maxmax_min = round(point_maxmax_min + 0.5)
#maps
im1 = point_1_geo.plot(column = "zonal_min", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_min, vmax=point_maxmax_min)
im2 = point_2_geo.plot(column = "zonal_min", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_min, vmax=point_maxmax_min)
im3 = point_3_geo.plot(column = "zonal_min", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=point_minmin_min, vmax=point_maxmax_min)
#titles
ax1.title.set_text("1.jako: Ensemble MIN")
ax2.title.set_text("2.jako: Ensemble MIN")
ax3.title.set_text("3.jako: Ensemble MIN")
#edges
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
point_1_geo.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=0.2)
point_2_geo.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=0.2)
point_3_geo.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=0.2)
point_1_geo.plot(ax=ax2, facecolor="None", edgecolor="red", linewidth=0.5)
point_1_geo.plot(ax=ax3, facecolor="None", edgecolor="red", linewidth=0.5)

# plt.savefig(os.path.join(fp_point, "point_accu_ensemble_min.png"))

##############################################################################
##############################################################################
##############################################################################
# DIFFERENCE BETWEEN ZONAL AND POINT

# 1.JAKOVAIHE
comp_1_geo = zonal_1_geo.copy()
comp_1_geo = comp_1_geo.drop(["area/m2","area/km2","zonal_min","zonal_var"], axis=1)
comp_1_geo["point_max"] = point_1_geo["zonal_max"]
comp_1_geo["point_mean"] = point_1_geo["zonal_mean"]
comp_1_geo["point_std"] = point_1_geo["zonal_std"]
comp_1_geo["dif_max"] = comp_1_geo["zonal_max"] - comp_1_geo["point_max"]
comp_1_geo["dif_mean"] = comp_1_geo["zonal_mean"] - comp_1_geo["point_mean"]
comp_1_geo["dif_std"] = comp_1_geo["zonal_std"] - comp_1_geo["point_std"]

# 2.JAKOVAIHE
comp_2_geo = zonal_2_geo.copy()
comp_2_geo = comp_2_geo.drop(["area/m2","area/km2","zonal_min","zonal_var"], axis=1)
comp_2_geo["point_max"] = point_2_geo["zonal_max"]
comp_2_geo["point_mean"] = point_2_geo["zonal_mean"]
comp_2_geo["point_std"] = point_2_geo["zonal_std"]
comp_2_geo["dif_max"] = comp_2_geo["zonal_max"] - comp_2_geo["point_max"]
comp_2_geo["dif_mean"] = comp_2_geo["zonal_mean"] - comp_2_geo["point_mean"]
comp_2_geo["dif_std"] = comp_2_geo["zonal_std"] - comp_2_geo["point_std"]

# 3.JAKOVAIHE
comp_3_geo = zonal_3_geo.copy()
comp_3_geo = comp_3_geo.drop(["area/m2","area/km2","zonal_min","zonal_var"], axis=1)
comp_3_geo["point_max"] = point_3_geo["zonal_max"]
comp_3_geo["point_mean"] = point_3_geo["zonal_mean"]
comp_3_geo["point_std"] = point_3_geo["zonal_std"]
comp_3_geo["dif_max"] = comp_3_geo["zonal_max"] - comp_3_geo["point_max"]
comp_3_geo["dif_mean"] = comp_3_geo["zonal_mean"] - comp_3_geo["point_mean"]
comp_3_geo["dif_std"] = comp_3_geo["zonal_std"] - comp_3_geo["point_std"]

##############################################################################
# COMPARISON PLOTS

############################
#Ensemble MEAN for subbasins
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
#scale
comp_minmin_mean = 0.0
comp_maxmax_mean = np.maximum(maxmax_mean, point_maxmax_mean)
#zonal maps
im1 = zonal_1_geo.plot(column = "zonal_mean", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
im2 = zonal_2_geo.plot(column = "zonal_mean", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
im3 = zonal_3_geo.plot(column = "zonal_mean", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
#titles
ax1.title.set_text("1.jako: Zonal MEAN")
ax2.title.set_text("2.jako: Zonal MEAN")
ax3.title.set_text("3.jako: Zonal MEAN")
#point maps
im4 = point_1_geo.plot(column = "zonal_mean", ax=ax4, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
im5 = point_2_geo.plot(column = "zonal_mean", ax=ax5, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
im6 = point_3_geo.plot(column = "zonal_mean", ax=ax6, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
#titles
ax4.title.set_text("1.jako: Point MEAN")
ax5.title.set_text("2.jako: Point MEAN")
ax6.title.set_text("3.jako: Point MEAN")

# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_mean.png"))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(9,12), constrained_layout = True)
#scale
comp_minmin_mean = 0.0
comp_maxmax_mean = np.maximum(maxmax_mean, point_maxmax_mean)
#zonal maps
im1 = zonal_1_geo.plot(column = "zonal_mean", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
im2 = zonal_2_geo.plot(column = "zonal_mean", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
im3 = zonal_3_geo.plot(column = "zonal_mean", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
#titles
ax1.title.set_text("1.jako: Zonal MEAN")
ax2.title.set_text("2.jako: Zonal MEAN")
ax3.title.set_text("3.jako: Zonal MEAN")
#point maps
im4 = point_1_geo.plot(column = "zonal_mean", ax=ax4, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
im5 = point_2_geo.plot(column = "zonal_mean", ax=ax5, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
im6 = point_3_geo.plot(column = "zonal_mean", ax=ax6, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_mean, vmax=comp_maxmax_mean)
#titles
ax4.title.set_text("1.jako: Point MEAN")
ax5.title.set_text("2.jako: Point MEAN")
ax6.title.set_text("3.jako: Point MEAN")
#scale difference maps
dif_1_mean_scale = np.maximum(abs(np.max(comp_1_geo["dif_mean"])), abs(np.min(comp_1_geo["dif_mean"])))
dif_2_mean_scale = np.maximum(abs(np.max(comp_2_geo["dif_mean"])), abs(np.min(comp_2_geo["dif_mean"])))
dif_3_mean_scale = np.maximum(abs(np.max(comp_3_geo["dif_mean"])), abs(np.min(comp_3_geo["dif_mean"])))
dif_mean_scale = np.maximum(dif_1_mean_scale, dif_2_mean_scale)
dif_mean_scale = np.maximum(dif_mean_scale, dif_3_mean_scale)
dif_mean_scale = round(dif_mean_scale + 0.5)
#difference maps
# im7 = comp_1_geo.plot(column = "dif_mean", ax=ax7, edgecolor=None, linewidth=1, cmap="bwr")
# im8 = comp_2_geo.plot(column = "dif_mean", ax=ax8, edgecolor=None, linewidth=1, cmap="bwr")
# im9 = comp_3_geo.plot(column = "dif_mean", ax=ax9, edgecolor=None, linewidth=1, cmap="bwr")
im7 = comp_1_geo.plot(column = "dif_mean", ax=ax7, edgecolor=None, linewidth=1, cmap="bwr", vmin=-dif_mean_scale, vmax=dif_mean_scale)
im8 = comp_2_geo.plot(column = "dif_mean", ax=ax8, edgecolor=None, linewidth=1, cmap="bwr", vmin=-dif_mean_scale, vmax=dif_mean_scale)
im9 = comp_3_geo.plot(column = "dif_mean", ax=ax9, edgecolor=None, linewidth=1, cmap="bwr", vmin=-dif_mean_scale, vmax=dif_mean_scale)
#titles
ax7.title.set_text("1.jako: Difference (Zonal - Point)")
ax8.title.set_text("2.jako: Difference (Zonal - Point)")
ax9.title.set_text("3.jako: Difference (Zonal - Point)")

#axis labels off
ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)
ax4.get_xaxis().set_visible(False)
ax5.get_xaxis().set_visible(False)
ax6.get_xaxis().set_visible(False)
ax7.get_xaxis().set_visible(False)
ax8.get_xaxis().set_visible(False)
ax9.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)
ax6.get_yaxis().set_visible(False)
ax7.get_yaxis().set_visible(False)
ax8.get_yaxis().set_visible(False)
ax9.get_yaxis().set_visible(False)

# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_mean_dif.png"))
# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_mean_dif_scaled.png"))

###########################
#Ensemble STD for subbasins
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
#scale
comp_minmin_std = 0.0
comp_maxmax_std = np.maximum(maxmax_std, point_maxmax_std)
#zonal maps
im1 = zonal_1_geo.plot(column = "zonal_std", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
im2 = zonal_2_geo.plot(column = "zonal_std", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
im3 = zonal_3_geo.plot(column = "zonal_std", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
#titles
ax1.title.set_text("1.jako: Zonal STD")
ax2.title.set_text("2.jako: Zonal STD")
ax3.title.set_text("3.jako: Zonal STD")
#point maps
im4 = point_1_geo.plot(column = "zonal_std", ax=ax4, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
im5 = point_2_geo.plot(column = "zonal_std", ax=ax5, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
im6 = point_3_geo.plot(column = "zonal_std", ax=ax6, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
#titles
ax4.title.set_text("1.jako: Point STD")
ax5.title.set_text("2.jako: Point STD")
ax6.title.set_text("3.jako: Point STD")

# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_std.png"))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(9,9), constrained_layout = True)
#scale
comp_minmin_std = 0.0
comp_maxmax_std = np.maximum(maxmax_std, point_maxmax_std)
comp_maxmax_std = 19.0 #add this for event3
#zonal maps
# im1 = zonal_1_geo.plot(column = "zonal_std", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
# im2 = zonal_2_geo.plot(column = "zonal_std", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
# im3 = zonal_3_geo.plot(column = "zonal_std", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
im1 = zonal_1_geo.plot(column = "zonal_std", ax=ax1, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_std, vmax=comp_maxmax_std)
im2 = zonal_2_geo.plot(column = "zonal_std", ax=ax2, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_std, vmax=comp_maxmax_std)
im3 = zonal_3_geo.plot(column = "zonal_std", ax=ax3, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_std, vmax=comp_maxmax_std)
#titles
ax1.title.set_text("a) 1.jako: Areal STD")
ax2.title.set_text("b) 2.jako: Areal STD")
ax3.title.set_text("c) 3.jako: Areal STD")
#point maps
# im4 = point_1_geo.plot(column = "zonal_std", ax=ax4, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
# im5 = point_2_geo.plot(column = "zonal_std", ax=ax5, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
# im6 = point_3_geo.plot(column = "zonal_std", ax=ax6, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_std, vmax=comp_maxmax_std)
im4 = point_1_geo.plot(column = "zonal_std", ax=ax4, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_std, vmax=comp_maxmax_std)
im5 = point_2_geo.plot(column = "zonal_std", ax=ax5, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_std, vmax=comp_maxmax_std)
im6 = point_3_geo.plot(column = "zonal_std", ax=ax6, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_std, vmax=comp_maxmax_std)
#titles
ax4.title.set_text("d) 1.jako: Point STD")
ax5.title.set_text("e) 2.jako: Point STD")
ax6.title.set_text("f) 3.jako: Point STD")
#scale difference maps
dif_1_std_scale = np.maximum(abs(np.max(comp_1_geo["dif_std"])), abs(np.min(comp_1_geo["dif_std"])))
dif_2_std_scale = np.maximum(abs(np.max(comp_2_geo["dif_std"])), abs(np.min(comp_2_geo["dif_std"])))
dif_3_std_scale = np.maximum(abs(np.max(comp_3_geo["dif_std"])), abs(np.min(comp_3_geo["dif_std"])))
dif_std_scale = np.maximum(dif_1_std_scale, dif_2_std_scale)
dif_std_scale = np.maximum(dif_std_scale, dif_3_std_scale)
dif_std_scale = round(dif_std_scale + 0.5)
dif_std_scale = 10.0 #add this for event3
#difference maps
# im7 = comp_1_geo.plot(column = "dif_std", ax=ax7, edgecolor=None, linewidth=1, cmap="bwr")
# im8 = comp_2_geo.plot(column = "dif_std", ax=ax8, edgecolor=None, linewidth=1, cmap="bwr")
# im9 = comp_3_geo.plot(column = "dif_std", ax=ax9, edgecolor=None, linewidth=1, cmap="bwr")
im7 = comp_1_geo.plot(column = "dif_std", ax=ax7, edgecolor=None, linewidth=1, cmap="bwr", vmin=-dif_std_scale, vmax=dif_std_scale)
im8 = comp_2_geo.plot(column = "dif_std", ax=ax8, edgecolor=None, linewidth=1, cmap="bwr", vmin=-dif_std_scale, vmax=dif_std_scale)
im9 = comp_3_geo.plot(column = "dif_std", ax=ax9, edgecolor=None, linewidth=1, cmap="bwr", vmin=-dif_std_scale, vmax=dif_std_scale)
#titles
ax7.title.set_text("g) 1.jako: Difference")
ax8.title.set_text("h) 2.jako: Difference")
ax9.title.set_text("i) 3.jako: Difference")

#axis labels off
ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)
ax4.get_xaxis().set_visible(False)
ax5.get_xaxis().set_visible(False)
ax6.get_xaxis().set_visible(False)
ax7.get_xaxis().set_visible(False)
ax8.get_xaxis().set_visible(False)
ax9.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)
ax6.get_yaxis().set_visible(False)
ax7.get_yaxis().set_visible(False)
ax8.get_yaxis().set_visible(False)
ax9.get_yaxis().set_visible(False)

norm_std = mpl.colors.Normalize(vmin=comp_minmin_std, vmax=comp_maxmax_std)
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_std), ax=ax1, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_std, comp_maxmax_std, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_std), ax=ax2, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_std, comp_maxmax_std, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_std), ax=ax3, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_std, comp_maxmax_std, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_std), ax=ax4, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_std, comp_maxmax_std, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_std), ax=ax5, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_std, comp_maxmax_std, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_std), ax=ax6, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_std, comp_maxmax_std, 11))

norm_std_dif = mpl.colors.Normalize(vmin=-dif_std_scale, vmax=dif_std_scale)
fig.colorbar(plt.cm.ScalarMappable(cmap="seismic", norm=norm_std_dif), ax=ax7, location="right", orientation="vertical", ticks=np.linspace(-dif_std_scale, dif_std_scale, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="seismic", norm=norm_std_dif), ax=ax8, location="right", orientation="vertical", ticks=np.linspace(-dif_std_scale, dif_std_scale, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="seismic", norm=norm_std_dif), ax=ax9, location="right", orientation="vertical", ticks=np.linspace(-dif_std_scale, dif_std_scale, 11))

#Add gauge stations
mittarit_sade_h.plot(ax = ax1, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax2, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax3, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax4, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax5, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax6, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax7, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax8, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax9, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)

# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_std_dif.png"))
# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_std_dif_scaled.png"))
plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_std_dif_colors.png"))

###########################
#Ensemble MAX for subbasins
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
#scale
comp_minmin_max = 0.0
comp_maxmax_max = np.maximum(maxmax_max, point_maxmax_max)
#zonal maps
im1 = zonal_1_geo.plot(column = "zonal_max", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_max, vmax=comp_maxmax_max)
im2 = zonal_2_geo.plot(column = "zonal_max", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_max, vmax=comp_maxmax_max)
im3 = zonal_3_geo.plot(column = "zonal_max", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_max, vmax=comp_maxmax_max)
#titles
ax1.title.set_text("1.jako: Zonal MAX")
ax2.title.set_text("2.jako: Zonal MAX")
ax3.title.set_text("3.jako: Zonal MAX")
#point maps
im4 = point_1_geo.plot(column = "zonal_max", ax=ax4, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_max, vmax=comp_maxmax_max)
im5 = point_2_geo.plot(column = "zonal_max", ax=ax5, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_max, vmax=comp_maxmax_max)
im6 = point_3_geo.plot(column = "zonal_max", ax=ax6, edgecolor=None, linewidth=1, cmap="Blues", vmin=comp_minmin_max, vmax=comp_maxmax_max)
#titles
ax4.title.set_text("1.jako: Point MAX")
ax5.title.set_text("2.jako: Point MAX")
ax6.title.set_text("3.jako: Point MAX")

# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_max.png"))

###########################
#MAX WITH COMPARISON
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(9,9), constrained_layout = True) #9,12
#scale
comp_minmin_max = 0.0
comp_maxmax_max = np.maximum(maxmax_max, point_maxmax_max)
# comp_maxmax_max = 158.0 #add this for event3
#zonal maps
im1 = zonal_1_geo.plot(column = "zonal_max", ax=ax1, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_max, vmax=comp_maxmax_max) #cmap="Blues"
im2 = zonal_2_geo.plot(column = "zonal_max", ax=ax2, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_max, vmax=comp_maxmax_max) #nipy_spectral
im3 = zonal_3_geo.plot(column = "zonal_max", ax=ax3, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_max, vmax=comp_maxmax_max)
#titles
ax1.title.set_text("a) 1st level: Areal MAX")
ax2.title.set_text("b) 2nd level: Areal MAX")
ax3.title.set_text("c) 3.jako: Areal MAX")
#point maps
im4 = point_1_geo.plot(column = "zonal_max", ax=ax4, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_max, vmax=comp_maxmax_max) #cmap="Blues"
im5 = point_2_geo.plot(column = "zonal_max", ax=ax5, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_max, vmax=comp_maxmax_max)
im6 = point_3_geo.plot(column = "zonal_max", ax=ax6, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=comp_minmin_max, vmax=comp_maxmax_max)
#titles
ax4.title.set_text("d) 1st level: Point MAX")
ax5.title.set_text("e) 2nd level: Point MAX")
ax6.title.set_text("f) 3rd level: Point MAX")
#scale difference maps
dif_1_max_scale = np.maximum(abs(np.max(comp_1_geo["dif_max"])), abs(np.min(comp_1_geo["dif_max"])))
dif_2_max_scale = np.maximum(abs(np.max(comp_2_geo["dif_max"])), abs(np.min(comp_2_geo["dif_max"])))
dif_3_max_scale = np.maximum(abs(np.max(comp_3_geo["dif_max"])), abs(np.min(comp_3_geo["dif_max"])))
dif_max_scale = np.maximum(dif_1_max_scale, dif_2_max_scale)
dif_max_scale = np.maximum(dif_max_scale, dif_3_max_scale)
dif_max_scale = round(dif_max_scale + 0.5)
# dif_max_scale = 85.0 #add this for event3
#difference maps
# im7 = comp_1_geo.plot(column = "dif_max", ax=ax7, edgecolor=None, linewidth=1, cmap="bwr")
# im8 = comp_2_geo.plot(column = "dif_max", ax=ax8, edgecolor=None, linewidth=1, cmap="bwr")
# im9 = comp_3_geo.plot(column = "dif_max", ax=ax9, edgecolor=None, linewidth=1, cmap="bwr")
im7 = comp_1_geo.plot(column = "dif_max", ax=ax7, edgecolor=None, linewidth=1, cmap="seismic", vmin=-dif_max_scale, vmax=dif_max_scale) #cmap="bwr"
im8 = comp_2_geo.plot(column = "dif_max", ax=ax8, edgecolor=None, linewidth=1, cmap="seismic", vmin=-dif_max_scale, vmax=dif_max_scale) #cmap="coolwarm"
im9 = comp_3_geo.plot(column = "dif_max", ax=ax9, edgecolor=None, linewidth=1, cmap="seismic", vmin=-dif_max_scale, vmax=dif_max_scale) #seismic
#titles
ax7.title.set_text("g) 1st level: Difference")
ax8.title.set_text("h) 2nd level: Difference")
ax9.title.set_text("i) 3rd level: Difference")
#axis labels off
ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)
ax4.get_xaxis().set_visible(False)
ax5.get_xaxis().set_visible(False)
ax6.get_xaxis().set_visible(False)
ax7.get_xaxis().set_visible(False)
ax8.get_xaxis().set_visible(False)
ax9.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)
ax6.get_yaxis().set_visible(False)
ax7.get_yaxis().set_visible(False)
ax8.get_yaxis().set_visible(False)
ax9.get_yaxis().set_visible(False)

norm_max = mpl.colors.Normalize(vmin=comp_minmin_max, vmax=comp_maxmax_max)
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_max), ax=ax1, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_max, comp_maxmax_max, 11)) #cmap="Blues"
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_max), ax=ax2, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_max, comp_maxmax_max, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_max), ax=ax3, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_max, comp_maxmax_max, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_max), ax=ax4, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_max, comp_maxmax_max, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_max), ax=ax5, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_max, comp_maxmax_max, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_max), ax=ax6, location="right", orientation="vertical", ticks=np.linspace(comp_minmin_max, comp_maxmax_max, 11))

norm_dif = mpl.colors.Normalize(vmin=-dif_max_scale, vmax=dif_max_scale)
fig.colorbar(plt.cm.ScalarMappable(cmap="seismic", norm=norm_dif), ax=ax7, location="right", orientation="vertical", ticks=np.linspace(-dif_max_scale, dif_max_scale, 11)) #cmap="bwr"
fig.colorbar(plt.cm.ScalarMappable(cmap="seismic", norm=norm_dif), ax=ax8, location="right", orientation="vertical", ticks=np.linspace(-dif_max_scale, dif_max_scale, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="seismic", norm=norm_dif), ax=ax9, location="right", orientation="vertical", ticks=np.linspace(-dif_max_scale, dif_max_scale, 11))

#Add gauge stations
mittarit_sade_h.plot(ax = ax1, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax2, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax3, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax4, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax5, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax6, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax7, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax8, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax9, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)

# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_max_dif.png"))
# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_max_dif_scaled.png"))
# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_max_dif_colors.png"))
# plt.savefig(os.path.join(fp_point, "comparison_accu_ensemble_max_dif_stations3.png"))

plt.savefig(os.path.join(out_figs,"figure_5_scaled.pdf"), format="pdf", bbox_inches="tight")

##############################################################################
##############################################################################
##############################################################################

mae_1_maxs = np.mean(abs(zonal_1_accus_max - point_1_accus_max))
mae_2_maxs = np.mean(abs(zonal_2_accus_max - point_2_accus_max))
mae_3_maxs = np.mean(abs(zonal_3_accus_max - point_3_accus_max))

##############################################################################
##############################################################################
##############################################################################
# Mean absolute error

mae_1 = np.mean(abs(zonal_1_accus - point_1_accus), axis=1)
comp_1_geo["mae"] = mae_1
mae_2 = np.mean(abs(zonal_2_accus - point_2_accus), axis=1)
comp_2_geo["mae"] = mae_2
mae_3 = np.mean(abs(zonal_3_accus - point_3_accus), axis=1)
comp_3_geo["mae"] = mae_3

#scale
mae_min = np.minimum(np.minimum(np.min(comp_1_geo["mae"]), np.min(comp_2_geo["mae"])), np.min(comp_3_geo["mae"]))
mae_min = 0.0
mae_max = np.maximum(np.maximum(np.max(comp_1_geo["mae"]), np.max(comp_2_geo["mae"])), np.max(comp_3_geo["mae"]))
mae_max = round(mae_max + 0.5)
mae_max = 8.0 #add this for event3

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9,3), constrained_layout = True) #20,6.5
#maps
# im1 = comp_1_geo.plot(column = "mae", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=mae_min, vmax=mae_max)
# im2 = comp_2_geo.plot(column = "mae", ax=ax2, edgecolor=None, linewidth=1, cmap="Blues", vmin=mae_min, vmax=mae_max)
# im3 = comp_3_geo.plot(column = "mae", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=mae_min, vmax=mae_max)
im1 = comp_1_geo.plot(column = "mae", ax=ax1, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=mae_min, vmax=mae_max)
im2 = comp_2_geo.plot(column = "mae", ax=ax2, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=mae_min, vmax=mae_max)
im3 = comp_3_geo.plot(column = "mae", ax=ax3, edgecolor=None, linewidth=1, cmap="nipy_spectral", vmin=mae_min, vmax=mae_max)
#titles
# ax1.title.set_text("a) 1st level: MAD") #event1
# ax2.title.set_text("b) 2nd level: MAD") #event1
# ax3.title.set_text("c) 3rd level: MAD") #event1
ax1.title.set_text("d) 1st level: MAD") #event3
ax2.title.set_text("e) 2nd level: MAD") #event3
ax3.title.set_text("f) 3rd level: MAD") #event3
#colorbar
norm_mae = mpl.colors.Normalize(vmin=mae_min, vmax=mae_max)
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_mae), ax=ax1, location="right", orientation="vertical", ticks=np.linspace(mae_min, mae_max, 11)) #cmap="bwr"
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_mae), ax=ax2, location="right", orientation="vertical", ticks=np.linspace(mae_min, mae_max, 11))
fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_mae), ax=ax3, location="right", orientation="vertical", ticks=np.linspace(mae_min, mae_max, 11)) 
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_mae), ax=ax1, location="bottom", orientation="horizontal", ticks=np.linspace(mae_min, mae_max, 5)) #cmap="bwr"
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_mae), ax=ax2, location="bottom", orientation="horizontal", ticks=np.linspace(mae_min, mae_max, 5))
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_mae), ax=ax3, location="bottom", orientation="horizontal", ticks=np.linspace(mae_min, mae_max, 5)) 

#Add gauge stations
mittarit_sade_h.plot(ax = ax1, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax2, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax3, color="black", marker="o", markersize=50, edgecolor="black", linewidth=1)

ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

# plt.savefig(os.path.join(fp_point, "comparison_mean_absolute_error.png"))
plt.savefig(os.path.join(fp_point, "comparison_mean_absolute_error_colors3.png"))

##############################################################################
##############################################################################
##############################################################################
# # INVESTICATING STRANGELY HIGH ACCUMULATIONS

# # zonal_3_accus
# # zonal_3_accus_max
# np.max(zonal_3_geo["zonal_max"]) #35.874, rivi 5043

# zonal_3_geo["row"] = 0
# for i in range(len(zonal_3_geo)):
#     zonal_3_geo.iloc[i]["row"] = i


# zonal_3_geo[zonal_3_geo["zonal_max"] == np.max(zonal_3_geo["zonal_max"])].index[0] # index value of subbasin, 5043

# zonal_3_geo.index.get_loc(zonal_3_geo[zonal_3_geo["zonal_max"] == np.max(zonal_3_geo["zonal_max"])].index[0]) #row number, 322

# zonal_3_geo.iloc[322]

# zonal_3_accus[322]

# plt.figure()
# plt.plot(zonal_3_accus[322])

# # np.max(zonal_3_accus[322])

# np.argmax(zonal_3_accus[322], axis=0) # realization with max value, 17

# dir_list[17] #Simulation_2087_4095_3282_3135

##############################################################################
##############################################################################
##############################################################################

#Painotettu etäisyys purkupisteeseen
distmean_dir = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Event3_new/Subbasins_500"
distmean_raster = rasterio.open(os.path.join(distmean_dir, "distmean.tif"))
distmean_array = distmean_raster.read(1)
# plt.imshow(distmean_array)

dist_arr_basin = np.zeros(len(dir_list))
dist_arr_radar_dbz_before = np.zeros(len(dir_list))
dist_arr_radar_dbz_after = np.zeros(len(dir_list))
dist_arr_radar_mm_before = np.zeros(len(dir_list))
dist_arr_radar_mm_after = np.zeros(len(dir_list))

for member in range(len(dir_list)):
    print(member)
    #Choose ensemble member
    chosen_realization = dir_list[member]
    data_dir2 = os.path.join(data_dir_location, chosen_realization, "Event_tiffs") #simulated event
    files = os.listdir(data_dir2)

    #Import simulated event in dbz
    event_sim_array  = []
    for i in range(len(files)):
        temp_raster = rasterio.open(os.path.join(data_dir2, f"test_{i}.tif"))
        temp_array = temp_raster.read(1)
        event_sim_array.append(temp_array)
        if i == 0:
            event_affine = temp_raster.transform  
    event_sim_array = np.concatenate([event_sim_array_[None, :, :] for event_sim_array_ in event_sim_array])
    # event_sim_array[event_sim_array == 3.1830486304816077] = np.nan #0
    #Event from dbz into mm/h without thresholding
    event_sim_array_mmh_before = event_sim_array.copy()
    event_sim_array_mmh_before = 10**((event_sim_array_mmh_before-10*np.log10(a_R))/(10*b_R))
    
    #Accumulation in dbz before thresholding /(256*256)
    event_sim_array_dbz_accu_before = np.nansum(event_sim_array, axis=0)
    # plt.imshow(event_sim_array_dbz_accu_before)
    
    #Accumulation in mm before thresholding /(256*256)
    event_sim_array_mm_accu_before = np.nansum(event_sim_array_mmh_before, axis=0) * (timestep/60)
    
    #Clear values over threshold of 45 dBZ
    event_sim_array[event_sim_array > 45] = 0.5*(45 + event_sim_array[event_sim_array > 45])
    
    #Event from dbz into mm/h
    event_sim_array_mmh = event_sim_array.copy()
    event_sim_array_mmh = 10**((event_sim_array_mmh-10*np.log10(a_R))/(10*b_R))
    #Values less than threshold to zero
    event_sim_array_mmh[event_sim_array_mmh < 0.1] = 0
    
    #Accumulation in dbz after thresholding /(256*256)
    event_sim_array_dbz_accu_after = np.nansum(event_sim_array, axis=0)
    
    #Accumulation raster in mm after thresholding /(256*256)
    event_sim_array_mm_accu_after = np.nansum(event_sim_array_mmh, axis=0) * (timestep/60)
    # plt.imshow(event_sim_array_mmh_accu)
    
    #Inverse distance weighting
    distmean_array_inv = distmean_array.copy()
    distmean_array_inv[distmean_array_inv == 0] = np.nan
    distmean_array_inv = 1 / distmean_array_inv
    # plt.imshow(distmean_array_inv)
    
    dist_basin_temp = np.nansum(event_sim_array_mm_accu_after * distmean_array_inv) / (np.nansum(distmean_array_inv))
    dist_arr_basin[member] = dist_basin_temp
    
    dist_arr_radar_dbz_before[member] = np.nansum(event_sim_array_dbz_accu_before)
    dist_arr_radar_dbz_after[member] = np.nansum(event_sim_array_dbz_accu_after)
    dist_arr_radar_mm_before[member] = np.nansum(event_sim_array_mm_accu_before)
    dist_arr_radar_mm_after[member] = np.nansum(event_sim_array_mm_accu_after)

plt.figure()
plt.hist(dist_arr_radar_dbz_before/(256*256)/(len(event_sim_array)))
plt.title("dbz before")
plt.figure()
plt.hist(dist_arr_radar_dbz_after/(256*256)/(len(event_sim_array)))
plt.title("dbz after")
plt.figure()
plt.hist(dist_arr_radar_mm_before/(256*256))
plt.title("mm before")
plt.figure()
plt.hist(dist_arr_radar_mm_after/(256*256))
plt.title("mm after")

# plt.figure()
# plt.plot(dist_arr_radar_dbz_after/(256*256)/(len(event_sim_array)))
# plt.axhline(y = (osapol_accu_dbz_after / (256*256)/(len(event_sim_array))), color = 'r', linestyle = '-')

# plt.figure()
# plt.plot(dist_arr_basin)
# plt.figure()
# plt.hist(dist_arr_basin)

# plt.figure()
# plt.plot(dist_arr_radar_mm_after)
# plt.figure()
# plt.hist(dist_arr_radar_mm_after)
# plt.figure()
# plt.hist(dist_arr_radar_mm_after/(256*256))

##########################
##########################
#OSAPOL accumulation with and without thresholding, and in mm/h and dbz
# osapol_accu = (np.nansum(event_osapol_array_mmh)) * (timestep/60) 
# osapol_accu / (256*256)

osapol_accu_mm_raw = np.nansum(event_osapol_array_mmh_raw) * (timestep/60)
osapol_accu_mm_01 = np.nansum(event_osapol_array_mmh) * (timestep/60) 

osapol_accu_dbz_before = np.nansum(event_osapol_array)
event_osapol_array_after = event_osapol_array.copy()
#Clear values over threshold of 45 dBZ
event_osapol_array_after[event_osapol_array_after > 45] = 0.5*(45 + event_osapol_array_after[event_osapol_array_after > 45])
osapol_accu_dbz_after = np.nansum(event_osapol_array_after)

event_osapol_array_mmh_after = event_osapol_array_after.copy()
event_osapol_array_mmh_after = 10**((event_osapol_array_mmh_after-10*np.log10(a_R))/(10*b_R))
osapol_accu_mm_after = np.nansum(event_osapol_array_mmh_after) * (timestep/60)

event_osapol_array_mmh_after[event_osapol_array_mmh_after < 0.1] = 0
osapol_accu_mm_after_01 = np.nansum(event_osapol_array_mmh_after) * (timestep/60)

print(osapol_accu_mm_raw / (256*256))
print(osapol_accu_mm_01 / (256*256))
print(osapol_accu_dbz_before / (256*256)/(len(event_sim_array)))
print(osapol_accu_dbz_after / (256*256)/(len(event_sim_array)))
print(osapol_accu_mm_after / (256*256))
print(osapol_accu_mm_after_01 / (256*256))


##############################################################################
#Sadanta-aikasrajat
mean_tss = []
for member in range(len(dir_list)):
    chosen_realization = dir_list[member]
    mean_ts = genfromtxt(fname=os.path.join(data_dir_location, chosen_realization, "sim_brokenlines.csv"), delimiter=',', skip_header=1)
    mean_ts = mean_ts[0,1:]
    mean_tss.append(mean_ts)

mean_tss = np.vstack(mean_tss)

##############################################################################

member #
np.nanmean(event_sim_array)

np.nanmean(event_osapol_array)

test_ts = []
for i in range(len(event_sim_array)):
    test_ts.append(np.mean(event_sim_array[i]))
plt.figure()
plt.plot(test_ts)

np.mean(test_ts)
