# -*- coding: utf-8 -*-
"""
pysteps_event_simulation.py

Scrip to do GIS and statistical analyses on different catchment scales.

Outputs:
    - 

Requirements:
    - geopandas
    - os
    - matplotlib
    - matplotlib.pyplot
    - rasterio
    - shapely
    - rasterstats
    - numpy
    - pandas
    - datetime
    - pysteps

References:
    - 

Created on Tue May 10 08:29:26 2022

@author: Ville Lindgren
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

##############################################################################
# TIME USED FOR SETTING UP EVERYTHING FOR CALCULATING ACCUMULATIONS: Start timer

run_start_0 = time.perf_counter()

##############################################################################
# DATA DIRECTORIES

data_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/GIS-aineistot" #letter "r" needs to be added in front of the path in Windows
fp1 = os.path.join(data_dir, "syke-valuma", "Paajako.shp") #Syken valuma-alue rasteri
fp2 = os.path.join(data_dir, "syke-valuma", "Jako3.shp") #Syken kolmannen jakovaiheen valuma-alue rasteri
fp3 = os.path.join(data_dir, "FMI_stations_2022-03.gpkg") #Ilmatieteenlaitoksen sää- ja sadehavaintoasemat
fp4 = os.path.join(data_dir, "FMI_stations_2022-05_rain.csv") #Ilmatieteenlaitoksen sää- ja sadehavaintoasemat, jotka mittaa sadetta (toukokuu 2022)

data_dir_location = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simuloinnit/Event_6/Simulations"
dir_list = os.listdir(data_dir_location)
chosen_realization = dir_list[0]
data_dir2 = os.path.join(data_dir_location, chosen_realization, "Event_tiffs") #simulated event
# data_dir2 = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simuloinnit/Event_6/Simulations/Simulation_2049_2098_1754_5517/Event_tiffs"

fp5 = os.path.join(data_dir2, "test_0.tif") #for purpose of creating a bounding box

files = os.listdir(data_dir2)

#Output directory
out_dir = os.path.join(data_dir_location, chosen_realization, "Calculations")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

##############################################################################
# OPTIONS TO SAVE PLOTS AND RESULTS

save_study_area = 0
save_centroids_rains = 0 #save maps of subbasin centreoids as well as gauges measuring daily and hourly rain
save_3closest = 0 #save example plot of 3 closest gauges as a png
save_accu_csv = 0 #save accumulation array as a csv
save_accu_raster = 1 #save accumulation array as a raster
save_accu_maps = 0 #save plot of accumulation maps and difference map as a png
save_maxmin_maps = 0
save_moving_accu_ts = 0
save_gdfs = 0 #save geodataframes as gpkg: subbasins, centroids, and gauges

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

#Plot geodataframes
alue_paajako.plot()
alue_jako3.plot()

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
# kokemaki_jako3.plot()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2) #, figsize=(20,6.5), constrained_layout = True
alue_paajako.plot(ax = ax1)
kokemaki_alue.plot(ax = ax1, color="red")
kokemaki_jako3.plot(ax = ax2)
fig.suptitle("Study area: Kokemäenjoki riverbasin", fontsize=16)
if save_study_area == 1:
    plt.savefig(os.path.join(out_dir, "map_study_area.png"))

# mittarit_kokemaki = gpd.clip(mittarit, kokemaki_alue)
# # mittarit_kokemaki.plot()

# mittari_index = ["sade", "s��, sade, ilmanlaatu (IL)"]
# mittarit_sade_kokemaki = mittarit_kokemaki[mittarit_kokemaki["field_8"].isin(mittari_index)]
# # mittarit_sade_kokemaki.plot()

bbox_df = gpd.GeoDataFrame({"geometry":[bbox]}, crs=kokemaki_alue.crs)
mittarit_tutkakuva = gpd.clip(mittarit, bbox_df)
# mittarit_tutkakuva.plot()

#Tutka-alueen mittarit, jotka mittaa päivittäistä sadetta
mittarit_sade_d = mittarit_tutkakuva[mittarit_tutkakuva["field_1"].isin(mittarit_sade["field_1_may_13052022"][mittarit_sade["sademaara/d"]==1])]
  
#Tutka-alueen mittarit, jotka mittaa tunnistaista sadetta
mittarit_sade_h = mittarit_tutkakuva[mittarit_tutkakuva["field_1"].isin(mittarit_sade["field_1_may_13052022"][mittarit_sade["sademaara/h"]==1])]

##############################################################################
# CENTROIDS OF SUBBASINS (center of mass of a geometry)

kokemaki_jako3["centroid"] = kokemaki_jako3["geometry"].centroid

##############################################################################
# PLOT SUBBASINS, CENTROIDS, AND GAUGES

fig, ax = plt.subplots()
kokemaki_jako3.plot(ax = ax)
kokemaki_jako3["centroid"].plot(ax = ax, color="yellow", label="centroids")
ax.set_title("Centroids")
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
if save_centroids_rains == 1:
    plt.savefig(os.path.join(out_dir, "map_centroids.png"))

fig, ax = plt.subplots()
kokemaki_jako3.plot(ax = ax)
mittarit_tutkakuva.plot(ax = ax, color="black", label="gauges in radar image area")
mittarit_sade_d.plot(ax = ax, color="red", label="rain/day")
ax.set_title("Rain/Day")
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
if save_centroids_rains == 1:
    plt.savefig(os.path.join(out_dir, "map_rain_day.png"))

fig, ax = plt.subplots()
kokemaki_jako3.plot(ax = ax)
mittarit_tutkakuva.plot(ax = ax, color="black", label="gauges in radar image area")
mittarit_sade_h.plot(ax = ax, color="red", label="rain/hour")
ax.set_title("Rain/Hour")
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
if save_centroids_rains == 1:
    plt.savefig(os.path.join(out_dir, "map_rain_hour.png"))

##############################################################################
# FIND 3 NEAREST GAUGES TO EACH CENTROID AND THEIR DISTANCES

#from shapely.ops import nearest_points
#https://autogis-site.readthedocs.io/en/latest/notebooks/L3/04_nearest-neighbour.html

#1st gauge, distance, and point
distances1 = kokemaki_jako3["centroid"].apply(lambda g: mittarit_sade_h["geometry"].distance(g))

kokemaki_distances = kokemaki_jako3.copy()
kokemaki_distances["min1_id"] = distances1.idxmin(axis=1, skipna=True) #https://thispointer.com/pandas-dataframe-get-minimum-values-in-rows-or-columns-their-index-position/
kokemaki_distances["min1_dist"] = distances1.min(axis=1, skipna=True)
distances_temp1 = mittarit_sade_h["geometry"][kokemaki_distances["min1_id"]]
distances_temp1.index = kokemaki_distances.index
kokemaki_distances["min1_point"] = distances_temp1

#2nd gauge, distance, and point
distances2 = distances1.copy()

for i in range(0, len(distances2)):
    distances2.iloc[i][kokemaki_distances.iloc[i]["min1_id"]] = None
    
kokemaki_distances["min2_id"] = distances2.idxmin(axis=1, skipna=True)
kokemaki_distances["min2_dist"] = distances2.min(axis=1, skipna=True)
distances_temp2 = mittarit_sade_h["geometry"][kokemaki_distances["min2_id"]]
distances_temp2.index = kokemaki_distances.index
kokemaki_distances["min2_point"] = distances_temp2

#3th gauge, distance, and point
distances3 = distances2.copy()

for i in range(0, len(distances3)):
    distances3.iloc[i][kokemaki_distances.iloc[i]["min2_id"]] = None
    
kokemaki_distances["min3_id"] = distances3.idxmin(axis=1, skipna=True)
kokemaki_distances["min3_dist"] = distances3.min(axis=1, skipna=True)
distances_temp3 = mittarit_sade_h["geometry"][kokemaki_distances["min3_id"]]
distances_temp3.index = kokemaki_distances.index
kokemaki_distances["min3_point"] = distances_temp3

#Example plot
fig, ax = plt.subplots()
kokemaki_distances.plot(ax = ax)
kokemaki_distances["centroid"].plot(ax = ax, color="yellow", label="centroids")
kokemaki_distances[100:101]["centroid"].plot(ax = ax, color="red", label="example centroid")
mittarit_sade_h.plot(ax = ax, color="orange", label="rain/hour")
kokemaki_distances[100:101]["min1_point"].plot(ax = ax, color="black", marker="o", label="#1 closest gauge")
kokemaki_distances[100:101]["min2_point"].plot(ax = ax, color="black", marker="^", label="#2 closest gauge")
kokemaki_distances[100:101]["min3_point"].plot(ax = ax, color="black", marker="s", label="#3 closest gauge")
ax.set_title("Example: 3 closest stations")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
if save_3closest == 1:
    plt.savefig(os.path.join(out_dir, "map_example_3closest.png"))

##############################################################################
# PARAMETERS FOR DBZ-MM/H TRANSFORMATION

#dBZ transformation for mm/h-data
#Z-R relationship: Z = a*R^b (Reflectivity)
#dBZ conversion: dBZ = 10 * log10(Z) (Decibels of Reflectivity)
#Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with size. J. Meteor., 5, 165–166.
a_R=223
b_R=1.53
#Leinonen et al. 2012 - A Climatology of Disdrometer Measurements of Rainfall in Finland over Five Years with Implications for Global Radar Observations
#https://doi.org/10.1175/JAMC-D-11-056.1

#-> R = 10^(log10(Z/a)/b)
#-> R = 10^((dBZ-10*log10(a))/10*b)

##############################################################################
# AVARAGE RAINFALL FOR EACH SUBBASIN USING

# #Read data into array
# test_array0 = raster0.read(1)

# #mm/h transformation for dBZ-data
# test_array0_mmh = test_array0.copy()
# test_array0_mmh = 10**((test_array0-10*np.log10(a_R))/(10*b_R))

# #Print field min and max in units of dbz and mmh
# print("min_dBZ:", np.min(test_array0), "\nmax_dBZ:", np.max(test_array0), "\nmin_mm/h:", np.min(test_array0_mmh), "\nmax_mm/h:", np.max(test_array0_mmh))

# #Plot field in units of dbz and mmh
# plt.figure()
# plt.imshow(test_array0) #, cmap="Blues"
# plt.figure()
# plt.imshow(test_array0_mmh) #, cmap="Blues"

# #Get the affine
# test_affine = raster0.transform

# #Plot subbasin polygons, gauge points, and raster below them
# fig, ax = plt.subplots(1,1)
# show((raster0, 1), ax=ax)
# kokemaki_jako3.plot(ax=ax, facecolor="None", edgecolor="red", linewidth=1)
# mittarit_sade_h.plot(ax=ax, color="orange")

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

#Event from dbz into mm/h
event_sim_array_mmh = event_sim_array.copy()
event_sim_array_mmh = 10**((event_sim_array_mmh-10*np.log10(a_R))/(10*b_R))

#Values less than threshold to zero
event_sim_array_mmh[event_sim_array_mmh<0.1] = 0

#Accumulation array of the event
event_sim_array_mmh_accu = sum(event_sim_array_mmh)
event_sim_array_mm_accu = event_sim_array_mmh_accu * (timestep/60)

if save_accu_csv == 1:
    
    #TODO: How to save 3D-array as csv? event_sim_array_mmh
    
    event_sim_array_mmh_accu_pd = pd.DataFrame(event_sim_array_mmh_accu)
    pd.DataFrame(event_sim_array_mmh_accu_pd).to_csv(os.path.join(out_dir, f"event_sim_{chosen_realization}_mmh_accu.csv"))
    
    event_sim_array_mm_accu_pd = pd.DataFrame(event_sim_array_mm_accu)
    pd.DataFrame(event_sim_array_mm_accu_pd).to_csv(os.path.join(out_dir, f"event_sim_{chosen_realization}_mm_accu.csv"))

##############################################################################
# OSAPOL TIMESERIES FOR EACH GAUGE

# Events used in the study:
# 1. last radar image: 201306271955 -> number of previous files: 141
# 3. last radar image: 201310290345 -> number of previous files: 115
# 6. last radar image: 201408071800 -> number of previous files: 97

#Read in the event with pySTEPS
date = datetime.strptime("201408071800", "%Y%m%d%H%M") #last radar image of the event
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
fns = pysteps.io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=97)

#Select the importer
importer = pysteps.io.get_method(importer_name, "importer")

#Read the radar composites
event_osapol_array_mmh, quality, event_osapol_metadata_mmh = pysteps.io.read_timeseries(fns, importer, **importer_kwargs)
del quality  #delete quality variable because it is not used

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

#AVARAGE RAINFALL FOR EACH SUBBASIN USING 3 POINT MEASUREMENTS FROM CLOSES GAUGE LOCATIONS (rasterstats.point_query)
#Painotetaan 1/etäisyydellä ja lasketaan keskiarvo kolmesta piste mittauksesta.
#Saadaan keskisadanta aikasarjat.

# #Get point values from raster
# test_point = point_query(mittarit_sade_h, test_array0, affine=test_affine, nodata=-999)
# #Add point values into GeoDataFrame
# mittarit_sade_h["rain"] = test_point

# #Calculate means using 3 closes point values
# test_point_means = np.zeros(len(kokemaki_jako3))
# for i in range(0, len(test_point_means)):
#     sade1 = float(mittarit_sade_h.iloc[[int(np.asarray(np.where(mittarit_sade_h.index.values==int(kokemaki_distances.iloc[[i]]["min1_id"]))))]]["rain"]) #1st gauge
#     sade2 = float(mittarit_sade_h.iloc[[int(np.asarray(np.where(mittarit_sade_h.index.values==int(kokemaki_distances.iloc[[i]]["min2_id"]))))]]["rain"]) #2st gauge
#     sade3 = float(mittarit_sade_h.iloc[[int(np.asarray(np.where(mittarit_sade_h.index.values==int(kokemaki_distances.iloc[[i]]["min3_id"]))))]]["rain"]) #3st gauge
#     kerroin1 = 1/float(kokemaki_distances.iloc[[i]]["min1_dist"])
#     kerroin2 = 1/float(kokemaki_distances.iloc[[i]]["min2_dist"])
#     kerroin3 = 1/float(kokemaki_distances.iloc[[i]]["min3_dist"])
    
#     #Painotettu aritmeettinen keskiarvo
#     keskisade = (kerroin1*sade1 + kerroin2*sade2 + kerroin3*sade3) / (kerroin1 + kerroin2 + kerroin3)
#     test_point_means[i] = keskisade
# kokemaki_jako3["mean_point"] = test_point_means

##############################################################################

# AVARAGE RAINFALL FOR EACH SUBBASIN USING AREAL MEANS (rasterstats.zonal_stats)
# Keskiarvo simuloidusta datasta.

# #Calculate zonal statistics
# test_zonal = zonal_stats(kokemaki_jako3, test_array0, affine=test_affine, nodata=-999, stats=["count", "min", "max", "mean", "median"])

# #Get mean values as a list
# test_zonal_means = np.zeros(len(kokemaki_jako3))
# for i in range(0, len(test_zonal_means)):
#     test_zonal_means[i] = float(test_zonal[i]["mean"])
# test_zonal_means = list(test_zonal_means)

# #Add mean values into GeoDataFrame
# kokemaki_jako3["mean_zonal"] = test_zonal_means

# #List to array
# test_zonal_means = np.asarray(test_zonal_means)

##############################################################################
# DIFFERENCE IN RESULTS OF TWO USED METHODS

# #Difference between two areal mean rainfalls calculated above
# kokemaki_jako3["difference"] = kokemaki_jako3["mean_zonal"] - kokemaki_jako3["mean_point"]

# #Visualize the difference of calculation methods of zonal mean
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)
# #Same min and max for all subplots
# minmin = np.min([np.min(raster0.read(1)), np.min(kokemaki_jako3["mean_zonal"]), np.min(kokemaki_jako3["mean_point"])])
# maxmax = np.max([np.max(raster0.read(1)), np.max(kokemaki_jako3["mean_zonal"]), np.max(kokemaki_jako3["mean_point"])])
# show((raster0, 1), ax=ax1, vmin=minmin, vmax=maxmax, cmap="Blues")
# kokemaki_jako3.plot(column = "mean_zonal", ax=ax2, edgecolor=None, linewidth=1, vmin=minmin, vmax=maxmax, cmap="Blues")
# kokemaki_jako3.plot(column = "mean_point", ax=ax3, edgecolor=None, linewidth=1, vmin=minmin, vmax=maxmax, cmap="Blues")
# kokemaki_jako3.plot(column = "difference", ax=ax4, edgecolor=None, linewidth=1, cmap="bwr")
# kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
# kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
# kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
# kokemaki_alue.plot(ax=ax4, facecolor="None", edgecolor="black", linewidth=1)
# ax1.title.set_text("Radar image")
# ax2.title.set_text("Areal mean for subbasins \nfrom radar image")
# ax3.title.set_text("Areal mean for subbasins \nusing 3 point measurements")
# ax4.title.set_text("Difference in means")

##############################################################################
# SAVE ACCUMULATION ARRAY AS A RASTER AND OPEN IT

if save_accu_raster == 1:
    event_sim_raster_mmh_accu = rasterio.open(
        os.path.join(out_dir, "event_sim_raster_mmh_accu.tif"),
        "w",
        driver="GTiff",
        height=event_sim_array_mmh_accu.shape[0],
        width=event_sim_array_mmh_accu.shape[1],
        count=1,
        dtype=event_sim_array_mmh_accu.dtype,
        crs=raster0.crs,
        transform=raster0.transform)
    event_sim_raster_mmh_accu.write(event_sim_array_mmh_accu, 1)
    event_sim_raster_mmh_accu.close()
    event_sim_raster_mmh_accu = rasterio.open(os.path.join(out_dir, "event_sim_raster_mmh_accu.tif"))
    
    event_sim_raster_mm_accu = rasterio.open(
        os.path.join(out_dir, "event_sim_raster_mm_accu.tif"),
        "w",
        driver="GTiff",
        height=event_sim_array_mm_accu.shape[0],
        width=event_sim_array_mm_accu.shape[1],
        count=1,
        dtype=event_sim_array_mm_accu.dtype,
        crs=raster0.crs,
        transform=raster0.transform)
    event_sim_raster_mm_accu.write(event_sim_array_mm_accu, 1)
    event_sim_raster_mm_accu.close()
    event_sim_raster_mm_accu = rasterio.open(os.path.join(out_dir, "event_sim_raster_mm_accu.tif"))

##############################################################################
# CALCULATING ACCUMULATIONS FOR WHOLE EVENT AND DIFFERENCE IN TWO METHODS

kokemaki_jako3["accu_point_mm/h"] = 0
kokemaki_jako3["accu_zonal_mm/h"] = 0
kokemaki_jako3["accu_dif_mm/h"] = 0

#Empty arrays to collect mean rainfalls for every timesteps
temp_point_means_all = np.zeros((len(kokemaki_jako3), len(event_sim_array_mmh)))
temp_zonal_means_all = np.zeros((len(kokemaki_jako3), len(event_sim_array_mmh)))

for j in range(0, len(event_sim_array_mmh)):
    ## POINT MEAN    
    #Get point values from raster
    temp_point = point_query(mittarit_sade_h, event_sim_array_mmh[j], affine=event_affine, nodata=-999)
    #Add point values into GeoDataFrame
    mittarit_sade_h["temp_point"] = temp_point
    #Calculate means using 3 closes point values
    temp_point_means = np.zeros(len(kokemaki_jako3))
    for i in range(0, len(temp_point_means)):
        sade1 = float(mittarit_sade_h.iloc[[int(np.asarray(np.where(mittarit_sade_h.index.values==int(kokemaki_distances.iloc[[i]]["min1_id"]))))]]["temp_point"]) #1st gauge
        sade2 = float(mittarit_sade_h.iloc[[int(np.asarray(np.where(mittarit_sade_h.index.values==int(kokemaki_distances.iloc[[i]]["min2_id"]))))]]["temp_point"]) #2st gauge
        sade3 = float(mittarit_sade_h.iloc[[int(np.asarray(np.where(mittarit_sade_h.index.values==int(kokemaki_distances.iloc[[i]]["min3_id"]))))]]["temp_point"]) #3st gauge
        kerroin1 = 1/float(kokemaki_distances.iloc[[i]]["min1_dist"])
        kerroin2 = 1/float(kokemaki_distances.iloc[[i]]["min2_dist"])
        kerroin3 = 1/float(kokemaki_distances.iloc[[i]]["min3_dist"])
        #Painotettu keskiarvo
        keskisade = (kerroin1*sade1 + kerroin2*sade2 + kerroin3*sade3) / (kerroin1 + kerroin2 + kerroin3)
        temp_point_means[i] = keskisade
    temp_point_means_all[:,j] = temp_point_means
    #Add mean values into GeoDataFrame
    kokemaki_jako3["accu_point_mm/h"] = kokemaki_jako3["accu_point_mm/h"] + temp_point_means

    ## ZONAL MEAN
    #Calculate zonal statistics
    temp_zonal = zonal_stats(kokemaki_jako3, event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
    #Get mean values as a list
    temp_zonal_means = np.zeros(len(kokemaki_jako3))
    for i in range(0, len(temp_zonal_means)):
        temp_zonal_means[i] = float(temp_zonal[i]["mean"])
    temp_zonal_means_all[:,j] = temp_zonal_means
    temp_zonal_means = list(temp_zonal_means)
    #Add mean values into GeoDataFrame
    kokemaki_jako3["accu_zonal_mm/h"] = kokemaki_jako3["accu_zonal_mm/h"] + temp_zonal_means
    
#Calculate difference between accumulations
kokemaki_jako3["accu_dif_mm/h"] = kokemaki_jako3["accu_zonal_mm/h"] - kokemaki_jako3["accu_point_mm/h"]

##############################################################################
# EVENT ACCUMULATIONS AND DIFFERENCE IN MM

#Calculate accumulations and difference as a depth [mm]
kokemaki_jako3["accu_point_mm"] = kokemaki_jako3["accu_point_mm/h"] * (timestep/60)
kokemaki_jako3["accu_zonal_mm"] = kokemaki_jako3["accu_zonal_mm/h"] * (timestep/60)
kokemaki_jako3["accu_dif_mm"] = kokemaki_jako3["accu_zonal_mm"] - kokemaki_jako3["accu_point_mm"]

##############################################################################
# VISUALIZING ACCUMULATIONS IN A MAP

## MM/H
#Visualize the difference of calculation methods of zonal mean
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(20,6.5), constrained_layout = True)
#Same min and max for all subplots
minmin = np.min([np.min(event_sim_raster_mmh_accu.read(1)), np.min(kokemaki_jako3["accu_zonal_mm/h"]), np.min(kokemaki_jako3["accu_point_mm/h"])])
maxmax = np.max([np.max(event_sim_raster_mmh_accu.read(1)), np.max(kokemaki_jako3["accu_zonal_mm/h"]), np.max(kokemaki_jako3["accu_point_mm/h"])])
im1 = show((event_sim_raster_mmh_accu,1), ax=ax1, vmin=minmin, vmax=maxmax, cmap="Blues")
# minmin = np.min([np.min(event_sim_array_mmh_accu), np.min(kokemaki_jako3["accu_zonal"]), np.min(kokemaki_jako3["accu_point"])])
# maxmax = np.max([np.max(event_sim_array_mmh_accu), np.max(kokemaki_jako3["accu_zonal"]), np.max(kokemaki_jako3["accu_point"])])
# im1 = show(event_sim_array_mmh_accu, ax=ax1, vmin=minmin, vmax=maxmax, cmap="Blues")
im2 = kokemaki_jako3.plot(column = "accu_zonal_mm/h", ax=ax2, edgecolor=None, linewidth=1, vmin=minmin, vmax=maxmax, cmap="Blues") #, legend=True
im3 = kokemaki_jako3.plot(column = "accu_point_mm/h", ax=ax3, edgecolor=None, linewidth=1, vmin=minmin, vmax=maxmax, cmap="Blues")
im4 = kokemaki_jako3.plot(column = "accu_dif_mm/h", ax=ax4, edgecolor=None, linewidth=1, cmap="bwr")
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax4, facecolor="None", edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax1, color="red")
ax2.set_xlim(bounds[0], bounds[2])
ax2.set_ylim(bounds[1], bounds[3])
ax3.set_xlim(bounds[0], bounds[2])
ax3.set_ylim(bounds[1], bounds[3])
ax4.set_xlim(bounds[0], bounds[2])
ax4.set_ylim(bounds[1], bounds[3])
fig.suptitle("Rainfall accumulations [mm/h] for the whole time period of the event", fontsize=16)
ax1.title.set_text("Simulated event")
ax2.title.set_text("Areal mean for subbasins \nusing zonalstats")
ax3.title.set_text("Areal mean for subbasins \nusing 3 point measurements")
ax4.title.set_text("Difference between two areal means")
norm_accu = mpl.colors.Normalize(vmin=minmin,vmax=maxmax)
norm_dif = mpl.colors.Normalize(vmin=np.min(kokemaki_jako3["accu_dif_mm/h"]),vmax=np.max(kokemaki_jako3["accu_dif_mm/h"]))
fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=norm_accu), ax=ax1, location="bottom", orientation="horizontal", ticks=np.linspace(minmin,maxmax,6))
fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=norm_accu), ax=ax2, location="bottom", orientation="horizontal", ticks=np.linspace(minmin,maxmax,6))
fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=norm_accu), ax=ax3, location="bottom", orientation="horizontal", ticks=np.linspace(minmin,maxmax,6))
fig.colorbar(plt.cm.ScalarMappable(cmap="bwr", norm=norm_dif), ax=ax4, location="bottom", orientation="horizontal", ticks=np.linspace(np.min(kokemaki_jako3["accu_dif_mm/h"]),np.max(kokemaki_jako3["accu_dif_mm/h"]),6))

if save_accu_maps == 1:
    plt.savefig(os.path.join(out_dir, "map_accumulations_mmh.png"))

## MM
#Visualize the difference of calculation methods of zonal mean
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(20,6.5), constrained_layout = True)
#Same min and max for all subplots
minmin = np.min([np.min(event_sim_raster_mm_accu.read(1)), np.min(kokemaki_jako3["accu_zonal_mm"]), np.min(kokemaki_jako3["accu_point_mm"])])
maxmax = np.max([np.max(event_sim_raster_mm_accu.read(1)), np.max(kokemaki_jako3["accu_zonal_mm"]), np.max(kokemaki_jako3["accu_point_mm"])])
im1 = show((event_sim_raster_mm_accu,1), ax=ax1, vmin=minmin, vmax=maxmax, cmap="Blues")
# minmin = np.min([np.min(event_sim_array_mmh_accu), np.min(kokemaki_jako3["accu_zonal"]), np.min(kokemaki_jako3["accu_point"])])
# maxmax = np.max([np.max(event_sim_array_mmh_accu), np.max(kokemaki_jako3["accu_zonal"]), np.max(kokemaki_jako3["accu_point"])])
# im1 = show(event_sim_array_mmh_accu, ax=ax1, vmin=minmin, vmax=maxmax, cmap="Blues")
im2 = kokemaki_jako3.plot(column = "accu_zonal_mm", ax=ax2, edgecolor=None, linewidth=1, vmin=minmin, vmax=maxmax, cmap="Blues") #, legend=True
im3 = kokemaki_jako3.plot(column = "accu_point_mm", ax=ax3, edgecolor=None, linewidth=1, vmin=minmin, vmax=maxmax, cmap="Blues")
im4 = kokemaki_jako3.plot(column = "accu_dif_mm", ax=ax4, edgecolor=None, linewidth=1, cmap="bwr")
kokemaki_alue.plot(ax=ax1, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax2, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax3, facecolor="None", edgecolor="black", linewidth=1)
kokemaki_alue.plot(ax=ax4, facecolor="None", edgecolor="black", linewidth=1)
mittarit_sade_h.plot(ax = ax1, color="red")
ax2.set_xlim(bounds[0], bounds[2])
ax2.set_ylim(bounds[1], bounds[3])
ax3.set_xlim(bounds[0], bounds[2])
ax3.set_ylim(bounds[1], bounds[3])
ax4.set_xlim(bounds[0], bounds[2])
ax4.set_ylim(bounds[1], bounds[3])
fig.suptitle("Rainfall accumulations [mm] for the whole time period of the event", fontsize=16)
ax1.title.set_text("Simulated event")
ax2.title.set_text("Areal mean for subbasins \nusing zonalstats")
ax3.title.set_text("Areal mean for subbasins \nusing 3 point measurements")
ax4.title.set_text("Difference between two areal means")
norm_accu = mpl.colors.Normalize(vmin=minmin,vmax=maxmax)
norm_dif = mpl.colors.Normalize(vmin=np.min(kokemaki_jako3["accu_dif_mm"]),vmax=np.max(kokemaki_jako3["accu_dif_mm"]))
fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=norm_accu), ax=ax1, location="bottom", orientation="horizontal", ticks=np.linspace(minmin,maxmax,6))
fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=norm_accu), ax=ax2, location="bottom", orientation="horizontal", ticks=np.linspace(minmin,maxmax,6))
fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=norm_accu), ax=ax3, location="bottom", orientation="horizontal", ticks=np.linspace(minmin,maxmax,6))
fig.colorbar(plt.cm.ScalarMappable(cmap="bwr", norm=norm_dif), ax=ax4, location="bottom", orientation="horizontal", ticks=np.linspace(np.min(kokemaki_jako3["accu_dif_mm"]),np.max(kokemaki_jako3["accu_dif_mm"]),6))

if save_accu_maps == 1:
    plt.savefig(os.path.join(out_dir, "map_accumulations_mm.png"))
    
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
# 1-HOUR MOVING WINDOW ACCUMULATIONS

#Moving window accumulations for 1-hour periods
#1-hour-window includes 12 timesteps of 5 min
moving_point_accu = np.zeros((len(kokemaki_jako3), int(timeseries_size - window_size)))
moving_zonal_accu = np.zeros((len(kokemaki_jako3), int(timeseries_size - window_size)))

for i in range(0, (int(timeseries_size - window_size))):
    moving_point_accu[:,i] = np.sum((temp_point_means_all[:, i:int(i+window_size-1)]), axis=1)
    moving_zonal_accu[:,i] = np.sum((temp_zonal_means_all[:, i:int(i+window_size-1)]), axis=1)

#1-hour period max and min accumulations
moving_point_subbasin_max = np.amax(moving_point_accu, axis=1)
moving_zonal_subbasin_max = np.amax(moving_zonal_accu, axis=1)
kokemaki_jako3["hourly_max_point"] = moving_point_subbasin_max
kokemaki_jako3["hourly_max_zonal"] = moving_zonal_subbasin_max
kokemaki_jako3["hourly_max_point_mm"] = kokemaki_jako3["hourly_max_point"] * (timestep/60)
kokemaki_jako3["hourly_max_zonal_mm"] = kokemaki_jako3["hourly_max_zonal"] * (timestep/60)

#Plot mins and maxs
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20,6.5), constrained_layout = True)
fig_zonalmax = np.max(kokemaki_jako3["hourly_max_zonal_mm"])
fig_zonalmin = np.min(kokemaki_jako3["hourly_max_zonal_mm"])
fig_pointmax = np.max(kokemaki_jako3["hourly_max_point_mm"])
fig_pointmin = np.min(kokemaki_jako3["hourly_max_point_mm"])
fig_zonalmean = np.mean(kokemaki_jako3["hourly_max_zonal_mm"])
fig_pointmean = np.mean(kokemaki_jako3["hourly_max_point_mm"])
#Which is greater, fig_zonalmax or fig_pointmax?
if fig_zonalmax > fig_pointmax:
    fig_max = fig_zonalmax.copy()
else:
    fig_max = fig_pointmax.copy()
# #Which is lesser, fig_zonalmin or fig_pointmin?
# if fig_zonalmin < fig_pointmin:
#     fig_min = fig_zonalmin.copy()
# else:
#     fig_min = fig_pointmin.copy()
im1 = kokemaki_jako3.plot(column = "hourly_max_zonal_mm", ax=ax1, edgecolor=None, linewidth=1, cmap="Blues", vmin=0, vmax=fig_max)
im3 = kokemaki_jako3.plot(column = "hourly_max_point_mm", ax=ax3, edgecolor=None, linewidth=1, cmap="Blues", vmin=0, vmax=fig_max)
im2 = ax2.bar(np.arange(len(kokemaki_jako3)), kokemaki_jako3["hourly_max_zonal_mm"])
im4 = ax4.bar(np.arange(len(kokemaki_jako3)), kokemaki_jako3["hourly_max_point_mm"])
ax2.set_ylim(0, fig_max+0.5)
ax4.set_ylim(0, fig_max+0.5)
ax2.axhline(y=fig_zonalmean, xmin=0, xmax=1, color="red", linestyle = "dashed")
ax4.axhline(y=fig_pointmean, xmin=0, xmax=1, color="red", linestyle = "dashed")
fig.suptitle("1-hour period max accumulations [mm] during the event in subbasin scale", fontsize=16)
ax1.title.set_text("Zonalstats")
ax2.title.set_text("Zonalstats")
ax3.title.set_text("3 point measurements")
ax4.title.set_text("3 point measurements")
norm_moving = mpl.colors.Normalize(vmin=0, vmax=fig_max)
fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=norm_moving), ax=ax1, location="bottom", orientation="horizontal", ticks=np.linspace(0, fig_max, 6))
fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=norm_moving), ax=ax3, location="bottom", orientation="horizontal", ticks=np.linspace(0, fig_max, 6))

if save_maxmin_maps == 1:
    plt.savefig(os.path.join(out_dir, "map_maxmin_accumulation.png"))

#Simulated event: 1-hour-period max accumulation
#Empty list to store all arrays
temp_masked_array = []
for radar_im in range(len(files)):
    temp_raster_accu = rasterio.open(os.path.join(data_dir2, f"test_{radar_im}.tif"))
    #Mask with single subbasin
    temp_masked_raster, temp_masked_affine = rasterio.mask.mask(temp_raster_accu, kokemaki_alue.geometry)
    #Reduce dimensions to two
    temp_masked_raster = temp_masked_raster[0]
    #Change zero values to nan, because there should not be data in those cells
    temp_masked_raster[temp_masked_raster==0] = ["nan"]
    #Transform unit from dBZ into mm/h
    temp_masked_raster = 10**((temp_masked_raster-10*np.log10(a_R))/(10*b_R))
    #Values less than threshold to zero
    temp_masked_raster[temp_masked_raster<0.1] = 0
    #Store array into list
    temp_masked_array.append(temp_masked_raster)
#Convert list of arrays into 3-dimensional array
temp_masked_array = np.concatenate([temp_masked_array_[None, :, :] for temp_masked_array_ in temp_masked_array])

#Accumulation for each timestep
temp_masked_accu = np.zeros(len(temp_masked_array))
for tstep in range(len(temp_masked_accu)):
    temp_masked_accu[tstep] = np.nansum(temp_masked_array[tstep])

#1-hour-period accumulations with moving window
moving_masked_accu = np.zeros(int(timeseries_size - window_size))
for i in range(0, (int(timeseries_size - window_size))):
    moving_masked_accu[i] = sum(temp_masked_accu[i:int(i+window_size-1)])
    
#Convert from mmm/h to mm
moving_masked_accu_mm = moving_masked_accu * (timestep/60)
#Weighting with whole area
temp_masked_accu_weighted = moving_masked_accu_mm / kokemaki_area_km2

#Index of max window
max_index = np.where(moving_masked_accu == np.amax(moving_masked_accu))
max_index = int(max_index[0])
#First and last radar image of max window
first_im = max_index
last_im = int(max_index + window_size-1)
# #Max 1-hour-period accumulation array from radar images
# moving_masked_max_accu = sum(temp_masked_array[first_im:last_im])
# show(moving_masked_max_accu, cmap="Blues")

#############
# TEST:
# #1-hour-window accumulation for whole river basin
# testi1_point = np.sum(moving_point_accu, axis=0)
# testi1_zonal = np.sum(moving_zonal_accu, axis=0)
# testi1_point_mm = testi1_point * (timestep/60)
# testi1_zonal_mm = testi1_zonal * (timestep/60)

# plt.figure()
# plt.plot(testi1_point_mm, label="point accumulation")
# plt.plot(testi1_zonal_mm, label="zonal accumulation")
# plt.legend()
# plt.title("Accumulation [mm] for whole basin")
#############

#1-hour-window accumulation for whole river basin: Weighting with area
for i in range(0, len(kokemaki_jako3)):
    #moving_point_accu[i] and moving_zonal_accu[i] == moving window accumulations for each subbasin i
    #kokemaki_jako3.iloc[i]["area/km2"] == area [km2] of each subbasin i
    #kokemaki_area_km2 == total area [km2] of the river basin
    temp_weighted_point = ((moving_point_accu[i]* (timestep/60)) * kokemaki_jako3.iloc[i]["area/km2"]) / kokemaki_area_km2
    temp_weighted_zonal = ((moving_zonal_accu[i]* (timestep/60)) * kokemaki_jako3.iloc[i]["area/km2"]) / kokemaki_area_km2
    if i == 0:
        temp_weighted_point_tot = temp_weighted_point.copy()
        temp_weighted_zonal_tot = temp_weighted_zonal.copy()
    else:
        temp_weighted_point_tot = temp_weighted_point_tot + temp_weighted_point
        temp_weighted_zonal_tot = temp_weighted_zonal_tot + temp_weighted_zonal

#Plotting 
plt.figure()
plt.plot(temp_weighted_zonal_tot, label="zonal")
plt.plot(temp_weighted_point_tot, label="point")
plt.plot(temp_masked_accu_weighted, label="radar", color="red", marker='.', linestyle="")
plt.legend()
plt.title("1-hour-window accumulations weighted with area for whole basin")
if save_moving_accu_ts == 1:
    plt.savefig(os.path.join(out_dir, "ts_moving_1hour_accumulations.png"))

##############################################################################
# TIME USED FOR SETTING UP EVERYTHING FOR CALCULATING ACCUMULATIONS: End timer

run_end_0 = time.perf_counter()
run_dur_0 = run_end_0 - run_start_0
print(run_dur_0, "sec, or", run_dur_0 / 60, "min")

##############################################################################
# SAVING GEODATAFRAMES AS GPKG-FILES (including accumulations)

if save_gdfs == 1:
    #Subbasins
    temp_gdf = kokemaki_jako3.copy()
    temp_gdf = temp_gdf.drop("centroid", axis = 1)
    temp_gdf.to_file(driver="GPKG", filename=os.path.join(out_dir,"kokemaki_subbasin.gpkg"))
    #Centroids
    temp_gdf = kokemaki_jako3.copy()
    temp_gdf["geometry"] = temp_gdf["centroid"]
    temp_gdf = temp_gdf.drop("centroid", axis = 1)
    temp_gdf.to_file(driver="GPKG", filename=os.path.join(out_dir,"kokemaki_centroid.gpkg"))
    
    #Gauges: rain/hour
    temp_gdf = mittarit_sade_h.copy()
    temp_gdf = temp_gdf.drop("temp_point", axis = 1)
    temp_gdf.to_file(driver="GPKG", filename=os.path.join(out_dir,"mittarit_rain_hour.gpkg"))
    #Gauges: rain/day
    temp_gdf = mittarit_sade_d.copy()
    # temp_gdf = temp_gdf.drop("temp_point", axis = 1)
    temp_gdf.to_file(driver="GPKG", filename=os.path.join(out_dir,"mittarit_rain_day.gpkg"))

##############################################################################
# SAVING DATA and RESULTS AS A NUMPY OBJECT?

# numpy.save(file, arr, allow_pickle=True, fix_imports=True) --- Save an array to a binary file in NumPy .npy format.
# numpy.savez(file, *args, **kwds) --- Save several arrays into a single file in uncompressed .npz format.
# numpy.savez_compressed(file, *args, **kwds) --- Save several arrays into a single file in compressed .npz format.

#Halutaan tallentaa binäärimuodossa ja pakattuna
