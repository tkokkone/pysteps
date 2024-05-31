# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:46:54 2022

@author: lindgrv1
"""

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

data_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/GIS-aineistot" #letter "r" needs to be added in front of the path in Windows
fp1 = os.path.join(data_dir, "syke-valuma", "Paajako.shp") #Syken valuma-alue rasteri
fp2 = os.path.join(data_dir, "syke-valuma", "Jako3.shp") #Syken kolmannen jakovaiheen valuma-alue rasteri
fp3 = os.path.join(data_dir, "FMI_stations_2022-03.gpkg") #Ilmatieteenlaitoksen sää- ja sadehavaintoasemat
fp4 = os.path.join(data_dir, "FMI_stations_2022-05_rain.csv") #Ilmatieteenlaitoksen sää- ja sadehavaintoasemat, jotka mittaa sadetta (toukokuu 2022)

# data_dir_location = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simulations_pysteps/Event3_new/Simulations"
data_dir_location = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Event1_new/Simulations"

dir_list = os.listdir(data_dir_location)

#for purpose of creating a bounding box
chosen_realization = dir_list[0]
data_dir2 = os.path.join(data_dir_location, chosen_realization, "Event_tiffs") #simulated event
fp5 = os.path.join(data_dir2, "test_0.tif")

files = os.listdir(data_dir2)

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

#Area outside of basin 
bbox_outside = bbox.difference(kokemaki_alue.iloc[0]["geometry"])

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
#FIND 3.LVL BASINS WITH LARGEST AND SMALLEST AREA

#3th lvl
np.max(kokemaki_jako3["area/km2"])
np.min(kokemaki_jako3["area/km2"])
np.min(kokemaki_jako3["area/km2"].nsmallest(2).iloc[-1])
#np.mean(kokemaki_jako3["area/km2"])

#Largest and smallest basins
basin_3lvl_max = float(kokemaki_jako3["Jako3Tunnu"][(kokemaki_jako3["area/km2"] == np.max(kokemaki_jako3["area/km2"]))])
basin_3lvl_min = float(kokemaki_jako3["Jako3Tunnu"][(kokemaki_jako3["area/km2"] == np.min(kokemaki_jako3["area/km2"]))])
#toiseksi pienin
basin_3lvl_min2 = float(kokemaki_jako3["Jako3Tunnu"][(kokemaki_jako3["area/km2"] == np.min(kokemaki_jako3["area/km2"].nsmallest(2).iloc[-1]))])

#Pick and try new area with wanted locations
basin_3lvl_max = float(35.111)
basin_3lvl_min2 = float(35.887)
############
# REMEMBER TO COMMENT ABOVE 2 ROWS 
############

#Largest
int(str(basin_3lvl_max)[3]) #fourth letter in string -> 1.lvl basin
int(str(basin_3lvl_max)[4]) #fifth letter in string -> 2.lvl basin
ind_2lvl_max = (int(str(basin_3lvl_max)[3]) - 1)*9 -1 +int(str(basin_3lvl_max)[4])
ind_1lvl_max = (int(str(basin_3lvl_max)[3]) - 1)

#2nd smallest
int(str(basin_3lvl_min2)[3]) #fourth letter in string -> 1.lvl basin
int(str(basin_3lvl_min2)[4]) #fifth letter in string -> 2.lvl basin
ind_2lvl_min2 =(int(str(basin_3lvl_min2)[3]) - 1)*9 -1 +int(str(basin_3lvl_min2)[4])
ind_1lvl_min2 = (int(str(basin_3lvl_min2)[3]) - 1)

# #Smallest
# int(str(basin_3lvl_min)[3]) #fourth letter in string -> 1.lvl basin
# int(str(basin_3lvl_min)[4]) #fifth letter in string -> 2.lvl basin
# ind_2lvl_min =(int(str(basin_3lvl_min)[3]) - 1)*9 -1 +int(str(basin_3lvl_min)[4])
# ind_1lvl_min = (int(str(basin_3lvl_min)[3]) - 1)

#Areas
print("Koko alue:", kokemaki_area_km2)
print("Suurin 3.vaiheen alue:", np.max(kokemaki_jako3["area/km2"]))
print("Suurimman 2.vaiheen alue:", areas_jako2[ind_2lvl_max])
print("Suurimman 1.vaiheen alue:", areas_jako1[ind_1lvl_max])
print("Toiseksi pienin 3.vaiheen alue:", np.min(kokemaki_jako3["area/km2"].nsmallest(2).iloc[-1]))
print("Toiseksi pienimmän 2.vaiheen alue:", areas_jako2[ind_2lvl_min2])
print("Toiseksi pienimmän 1.vaiheen alue:", areas_jako1[ind_1lvl_min2])

#Plot areas
fig, ax = plt.subplots()
kokemaki_jako3.plot(ax = ax, color="grey")
kokemaki_jako1_polys[ind_1lvl_max].plot(ax = ax, color="yellow")
kokemaki_jako2_polys[ind_2lvl_max].plot(ax = ax, color="orange")
kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"]==str(basin_3lvl_max)].plot(ax = ax, color="red")
kokemaki_jako1_polys[ind_1lvl_min2].plot(ax = ax, color="purple")
kokemaki_jako2_polys[ind_2lvl_min2].plot(ax = ax, color="green")
kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"]==str(basin_3lvl_min2)].plot(ax = ax, color="blue")

# kokemaki_jako1_polys[ind_1lvl_min].plot(ax = ax, color="purple")
# kokemaki_jako2_polys[ind_2lvl_min].plot(ax = ax, color="green")
# kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"]==str(basin_3lvl_min)].plot(ax = ax, color="blue")

#Plot subbasins in different levels
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
kokemaki_alue.plot(ax = ax1)
for i in range(len(kokemaki_jako1_polys)):
    kokemaki_jako1_polys[i].plot(ax = ax2)
for j in range(len(kokemaki_jako2_polys)):
    kokemaki_jako2_polys[j].plot(ax = ax3)
kokemaki_jako3.plot(ax = ax4)

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

# Cumulative rainfall [mm]:
# - Alueiden koko ja sijainti (alueet ei saa osua päällekkäin)
# - koko alueelle (b kuva alla).
# - 3 jakovaihe, 2 valuma-aluetta (iso ja pieni)
# - 2 jakovaihe (sisältää 3. vaiheen alueet)
# - 1 jakovaihe (sisältää 2. vaiheen alueet)

##############################################################################
# INDICES OF CHOCEN 3.LVL BASINS

# ind0_3lvl_max = kokemaki_jako3[kokemaki_jako3["area/km2"] == np.max(kokemaki_jako3["area/km2"])].index[0] #get index of largest subbasin
ind0_3lvl_max = kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"] == str(35.111)].index[0] #get index of wanted basin
############
# REMEMBER TO CHANGE ABOVE 2 ROWS OTHER WAY AROUND
############
ind_3lvl_max = kokemaki_jako3.index.get_loc(ind0_3lvl_max) #get row number of largest subbasin
kokemaki_jako3.iloc[ind_3lvl_max]

# ind0_3lvl_min2 = kokemaki_jako3[kokemaki_jako3["area/km2"] == np.min(kokemaki_jako3["area/km2"].nsmallest(2).iloc[-1])].index[0] #get index of 2nd smallest subbasin
ind0_3lvl_min2 = kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"] == str(35.887)].index[0] #get index of wanted basin
############
# REMEMBER TO CHANGE ABOVE 2 ROWS OTHER WAY AROUND
############
ind_3lvl_min2 = kokemaki_jako3.index.get_loc(ind0_3lvl_min2) #get row number of 2nd smallest subbasin
kokemaki_jako3.iloc[ind_3lvl_min2]

##############################################################################
#Z-R relationship: Z = a*R^b (Reflectivity)

a_R=223
b_R=1.53

##############################################################################
# ACCUMULATION TIME SERIES

#Empty arrays where to store results
accu_whole_basin = np.zeros((len(dir_list), timeseries_size)) #Empty array for whole basin wide 
accu_3_largest = np.zeros((len(dir_list), timeseries_size)) 
accu_3_2smallest = np.zeros((len(dir_list), timeseries_size))
accu_2_largest = np.zeros((len(dir_list), timeseries_size)) 
accu_2_2smallest = np.zeros((len(dir_list), timeseries_size)) 
accu_1_largest = np.zeros((len(dir_list), timeseries_size)) 
accu_1_2smallest = np.zeros((len(dir_list), timeseries_size)) 

accu_3_test_largest = np.zeros((len(dir_list), timeseries_size))
accu_3_test_2smallest = np.zeros((len(dir_list), timeseries_size))

accu_whole_basin_test = np.zeros((len(dir_list), timeseries_size))

accu_whole_grid = np.zeros((len(dir_list), timeseries_size))
accu_outside = np.zeros((len(dir_list), timeseries_size))

for member in range(len(dir_list)):
    print("########## ", member, " ##########")
    chosen_realization = dir_list[member]
    
    #Data location and files
    # data_dir_zonal = os.path.join(data_dir_location, chosen_realization, "Calculations_thresholded")
    data_dir_member = os.path.join(data_dir_location, chosen_realization, "Event_tiffs") #simulated event
    files = os.listdir(data_dir_member)
    
    ##############################################################################
    # #Whole basin
    # accus_whole_pd = pd.read_csv(os.path.join(data_dir_zonal, "accu_tss.csv"), delimiter=(","))
    # accus_whole = accus_whole_pd.to_numpy()
    # accus_whole = accus_whole[:, 1:]
    # accu_whole_basin[member] = accus_whole[1,] / kokemaki_area_km2
    
    # ##############################################################################
    # #3.lvl subbasins (larges and second to largest)
    # accus_3_pd = pd.read_csv(os.path.join(data_dir_zonal, "subbasin_zonal_means.csv"), delimiter=(","))
    # accus_3 = accus_3_pd.to_numpy()
    # accus_3 = accus_3[:, 1:]
    # accu_3_largest[member] = accus_3[ind_3lvl_max,]
    # accu_3_2smallest[member] =  accus_3[ind_3lvl_min2]

    ##############################################################################
    #2.lvl and 1.lvl subbasins (including larges and second to largest 3.lvl subbasins)
    
    #Import simulated event in raster-format and convert it into array
    event_sim_array  = []
    for i in range(len(files)):
        temp_raster = rasterio.open(os.path.join(data_dir_member, f"test_{i}.tif"))
        temp_array = temp_raster.read(1)
        event_sim_array.append(temp_array)
        if i == 0:
            event_affine = temp_raster.transform  
    event_sim_array = np.concatenate([event_sim_array_[None, :, :] for event_sim_array_ in event_sim_array])
    
    #Clear values over threshold of 45 dBZ
    event_sim_array[event_sim_array > 45] = 0.5*(45 + event_sim_array[event_sim_array > 45])
    #Event from dbz into mm/h
    event_sim_array_mmh = event_sim_array.copy()
    event_sim_array_mmh = 10**((event_sim_array_mmh-10*np.log10(a_R))/(10*b_R))
    #Values less than threshold to zero
    event_sim_array_mmh[event_sim_array_mmh < 0.1] = 0
    
    for j in range(len(event_sim_array_mmh)):
        print(j)
        ##############################################################################
        #2.lvl 
        #larges - ind_2lvl_max
        temp_zonal2_largest = zonal_stats(kokemaki_jako2_polys[ind_2lvl_max], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
        accu_2_largest[member, j] = float(temp_zonal2_largest[0]["mean"])
        #smallest - ind_2lvl_min2
        temp_zonal2_2smallest = zonal_stats(kokemaki_jako2_polys[ind_2lvl_min2], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
        accu_2_2smallest[member, j] = float(temp_zonal2_2smallest[0]["mean"])
        
        ##############################################################################
        #1.lvl
        #larges - ind_1lvl_max
        temp_zonal1_largest = zonal_stats(kokemaki_jako1_polys[ind_1lvl_max], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
        accu_1_largest[member, j] = float(temp_zonal1_largest[0]["mean"])
        #smallest - ind_1lvl_min2
        temp_zonal1_2smallest = zonal_stats(kokemaki_jako1_polys[ind_1lvl_min2], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
        accu_1_2smallest[member, j] = float(temp_zonal1_2smallest[0]["mean"])
        
        ##############################################################################
        #3.lvl
        #larges - ind_3lvl_max
        temp_zonal3_largest = zonal_stats(kokemaki_jako3.iloc[ind_3lvl_max]["geometry"], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
        accu_3_test_largest[member, j] = float(temp_zonal3_largest[0]["mean"])
        #smallest - ind_3lvl_min2
        temp_zonal3_2smallest = zonal_stats(kokemaki_jako3.iloc[ind_3lvl_min2]["geometry"], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
        accu_3_test_2smallest[member, j] = float(temp_zonal3_2smallest[0]["mean"])
        
        ##############################################################################
        # #whole basin
        # test_z = zonal_stats(kokemaki_alue, event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
        # accu_whole_basin_test[member, j] = float(test_z[0]["mean"])
        
        # ##############################################################################
        # #whole grid
        # test_whole_grid = zonal_stats(bbox, event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
        # accu_whole_grid[member, j] = float(test_whole_grid[0]["mean"])
        
        # ##############################################################################
        # #area outside of kokemäki river basin
        # test_outside = zonal_stats(bbox_outside, event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
        # accu_outside[member, j] = float(test_outside[0]["mean"])
#
accu_whole_basin_test2 = accu_whole_basin_test * (timestep/60)

##############################################################################
# OSAPOL

# Events used in the study:
# 1. last radar image: 201306271955 -> number of previous files: 141
# 3. last radar image: 201310290345 -> number of previous files: 115
# 6. last radar image: 201408071800 -> number of previous files: 97

#Read in the event with pySTEPS
date = datetime.strptime("201306271955", "%Y%m%d%H%M") #last radar image of the event
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
fns = pysteps.io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=141)

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

#Clear values over threshold of 45 dBZ
event_osapol_array_thr = event_osapol_array.copy()
event_osapol_array_thr[event_osapol_array_thr > 45] = 0.5*(45 + event_osapol_array_thr[event_osapol_array_thr > 45])

#Back to mm/h, 0.1 mm/h thresholding and converting to mm
event_osapol_array_mmh_thr = event_osapol_array_thr.copy()
event_osapol_array_mmh_thr = 10**((event_osapol_array_mmh_thr-10*np.log10(a_R))/(10*b_R))
event_osapol_array_mmh_thr[event_osapol_array_mmh_thr < 0.1] = 0

# event_osapol_array_mmh_thr[event_osapol_array_mmh_thr == np.nan] = 0
event_osapol_array_mmh_thr = np.nan_to_num(event_osapol_array_mmh_thr, nan=0.0)
aaaaa = np.zeros((1, len(event_osapol_array_mmh_thr)))
for i in range(len(event_osapol_array_mmh_thr)):
    aaaaa[0,i] = np.nanmean(event_osapol_array_mmh_thr[i])

aaaaa_cum = np.zeros((1, len(event_osapol_array_mmh_thr))) 
for j in range(len(event_osapol_array_mmh_thr)):
    if j == 0:
        aaaaa_cum[0,j] = aaaaa[0,j]
    else:
        aaaaa_cum[0,j] = aaaaa[0,j] + aaaaa_cum[0,j-1] 
        
#Accumulations
accu_osapol_3_largest = np.zeros((1, len(event_osapol_array_mmh_thr))) 
accu_osapol_3_2smallest = np.zeros((1, len(event_osapol_array_mmh_thr))) 
accu_osapol_2_largest = np.zeros((1, len(event_osapol_array_mmh_thr))) 
accu_osapol_2_2smallest = np.zeros((1, len(event_osapol_array_mmh_thr))) 
accu_osapol_1_largest = np.zeros((1, len(event_osapol_array_mmh_thr)))
accu_osapol_1_2smallest = np.zeros((1, len(event_osapol_array_mmh_thr)))
accu_osapol_whole_basin = np.zeros((1, len(event_osapol_array_mmh_thr)))

for j in range(len(event_osapol_array_mmh_thr)):
    ########################################
    #3.lvl
    #larges - ind_3lvl_max
    osapol_zonal3_largest = zonal_stats(kokemaki_jako3.iloc[ind_3lvl_max]["geometry"], event_osapol_array_mmh_thr[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_osapol_3_largest[0, j] = float(osapol_zonal3_largest[0]["mean"])
    #smallest - ind_3lvl_min2
    osapol_zonal3_2smallest = zonal_stats(kokemaki_jako3.iloc[ind_3lvl_min2]["geometry"], event_osapol_array_mmh_thr[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_osapol_3_2smallest[0, j] = float(osapol_zonal3_2smallest[0]["mean"])
    ########################################
    #2.lvl 
    #larges - ind_2lvl_max
    osapol_zonal2_largest = zonal_stats(kokemaki_jako2_polys[ind_2lvl_max], event_osapol_array_mmh_thr[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_osapol_2_largest[0, j] = float(osapol_zonal2_largest[0]["mean"])
    #smallest - ind_2lvl_min2
    osapol_zonal2_2smallest = zonal_stats(kokemaki_jako2_polys[ind_2lvl_min2], event_osapol_array_mmh_thr[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_osapol_2_2smallest[0, j] = float(osapol_zonal2_2smallest[0]["mean"])
    #########################################
    #1.lvl
    #larges - ind_1lvl_max
    osapol_zonal1_largest = zonal_stats(kokemaki_jako1_polys[ind_1lvl_max], event_osapol_array_mmh_thr[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_osapol_1_largest[0, j] = float(osapol_zonal1_largest[0]["mean"])
    #smallest - ind_1lvl_min2
    osapol_zonal1_2smallest = zonal_stats(kokemaki_jako1_polys[ind_1lvl_min2], event_osapol_array_mmh_thr[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_osapol_1_2smallest[0, j] = float(osapol_zonal1_2smallest[0]["mean"])
    ########################################
    #whole
    osapol_z = zonal_stats(kokemaki_alue, event_osapol_array_mmh_thr[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_osapol_whole_basin[0, j] = float(osapol_z[0]["mean"])

#Cumulative rainfall
accu_cum_osapol_3_largest = np.zeros((1, len(event_osapol_array_mmh_thr))) 
accu_cum_osapol_3_2smallest = np.zeros((1, len(event_osapol_array_mmh_thr))) 
accu_cum_osapol_2_largest = np.zeros((1, len(event_osapol_array_mmh_thr))) 
accu_cum_osapol_2_2smallest = np.zeros((1, len(event_osapol_array_mmh_thr))) 
accu_cum_osapol_1_largest = np.zeros((1, len(event_osapol_array_mmh_thr)))
accu_cum_osapol_1_2smallest = np.zeros((1, len(event_osapol_array_mmh_thr)))
accu_cum_osapol_whole_basin = np.zeros((1, len(event_osapol_array_mmh_thr)))

for j in range(len(event_osapol_array_mmh_thr)):
    if j == 0:
        accu_cum_osapol_3_largest[0,j] = accu_osapol_3_largest[0,j]
        accu_cum_osapol_3_2smallest[0,j] = accu_osapol_3_2smallest[0,j]
        accu_cum_osapol_2_largest[0,j] = accu_osapol_2_largest[0,j]
        accu_cum_osapol_2_2smallest[0,j] = accu_osapol_2_2smallest[0,j]
        accu_cum_osapol_1_largest[0,j] = accu_osapol_1_largest[0,j]
        accu_cum_osapol_1_2smallest[0,j] = accu_osapol_1_2smallest[0,j]
        accu_cum_osapol_whole_basin[0,j] = accu_osapol_whole_basin[0,j]
    else:
        accu_cum_osapol_3_largest[0,j] = accu_osapol_3_largest[0,j] + accu_cum_osapol_3_largest[0,j-1]
        accu_cum_osapol_3_2smallest[0,j] = accu_osapol_3_2smallest[0,j] + accu_cum_osapol_3_2smallest[0,j-1]
        accu_cum_osapol_2_largest[0,j] = accu_osapol_2_largest[0,j] + accu_cum_osapol_2_largest[0,j-1]
        accu_cum_osapol_2_2smallest[0,j] = accu_osapol_2_2smallest[0,j] + accu_cum_osapol_2_2smallest[0,j-1]
        accu_cum_osapol_1_largest[0,j] = accu_osapol_1_largest[0,j] + accu_cum_osapol_1_largest[0,j-1]
        accu_cum_osapol_1_2smallest[0,j] = accu_osapol_1_2smallest[0,j] + accu_cum_osapol_1_2smallest[0,j-1]
        accu_cum_osapol_whole_basin[0,j] = accu_osapol_whole_basin[0,j] + accu_cum_osapol_whole_basin[0,j-1]

##############################################################################
# CUMULATIVE ACCUMULATION

#Empty arrays
accu_cum_whole_basin = np.zeros((len(dir_list), timeseries_size))
accu_cum_whole_basin_test = np.zeros((len(dir_list), timeseries_size))

#Loop to calculate cumulate accumulations for each ensemble member
for i in range(accu_whole_basin.shape[0]):
    for j in range(accu_whole_basin.shape[1]):
        if j == 0:
            accu_cum_whole_basin[i,j] = accu_whole_basin[i,j]
            accu_cum_whole_basin_test[i,j] = accu_whole_basin_test2[i,j]
        else:
            accu_cum_whole_basin[i,j] = accu_cum_whole_basin[i,j-1] + accu_whole_basin[i,j]
            accu_cum_whole_basin_test[i,j] = accu_cum_whole_basin_test[i,j-1] + accu_whole_basin_test2[i,j]

#Ensemble mean cum accumulation time series
accu_cum_whole_basin_mean = np.zeros((1,timeseries_size))
accu_cum_whole_basin_test_mean = np.zeros((1,timeseries_size))
for k in range(accu_cum_whole_basin.shape[1]):
    accu_cum_whole_basin_mean[0,k] = np.mean(accu_cum_whole_basin[:,k])
    accu_cum_whole_basin_test_mean[0,k] = np.mean(accu_cum_whole_basin_test[:,k])

# #Plots
# plt.figure()
# for l in range(accu_whole_basin.shape[0]):
# # for l in range(0,20):
#     plt.plot(accu_cum_whole_basin[l], color="grey")
# plt.plot(accu_cum_whole_basin_mean[0], color="red")
# plt.title("Whole radar grid")
# # plt.ylim((0,55))
# plt.grid(True, which='major', axis='y')
# # plt.plot((aaaaa_cum[0]*(timestep/60)), color="blue")

plt.figure()
for m in range(accu_whole_basin_test.shape[0]):
# for m in range(0,20):
    plt.plot(accu_cum_whole_basin_test[m], color="grey")
plt.plot(accu_cum_whole_basin_test_mean[0], color="red")
plt.title("Kokemaki basin")
plt.ylim((0,130))
plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
# plt.plot((accu_cum_osapol_whole_basin[0]*(timestep/60)), color="blue")

##############################################################################
#Whole grid and area outside of Kokemäenjoki basin

#Empty arrays
accu_cum_whole_grid = np.zeros((len(dir_list), timeseries_size))
accu_cum_outside = np.zeros((len(dir_list), timeseries_size))
#Loop to calculate cumulate accumulations for each ensemble member
for i in range(accu_whole_grid.shape[0]):
    for j in range(accu_whole_grid.shape[1]):
        if j == 0:
            accu_cum_whole_grid[i,j] = accu_whole_grid[i,j]
            accu_cum_outside[i,j] = accu_outside[i,j]
        else:
            accu_cum_whole_grid[i,j] = accu_cum_whole_grid[i,j-1] + accu_whole_grid[i,j]
            accu_cum_outside[i,j] = accu_cum_outside[i,j-1] + accu_outside[i,j]

#Ensemble mean cum accumulation time series
accu_cum_whole_grid_mean = np.zeros((1,timeseries_size))
accu_cum_outside_mean = np.zeros((1,timeseries_size))
#Loop to calculate mean of each timestep
for k in range(accu_cum_whole_grid.shape[1]):
    accu_cum_whole_grid_mean[0,k] = np.mean(accu_cum_whole_grid[:,k])
    accu_cum_outside_mean[0,k] = np.mean(accu_cum_outside[:,k])

#Plots
plt.figure()
for l in range(accu_whole_grid.shape[0]):
    plt.plot(accu_cum_whole_grid[l]*(timestep/60), color="grey")
plt.plot(accu_cum_whole_grid_mean[0]*(timestep/60), color="red")
plt.title("Whole radar grid")
plt.ylim((0,130))
plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
# plt.plot((aaaaa_cum[0]*(timestep/60)), color="blue")
plt.plot((aaaaa_cum[0,0:-1]*(timestep/60)), color="blue")

plt.figure()
for m in range(accu_outside.shape[0]):
    plt.plot(accu_cum_outside[m]*(timestep/60), color="grey")
plt.plot(accu_cum_outside_mean[0]*(timestep/60), color="red")
plt.title("Area outside of Kokemakijoki basin")
plt.ylim((0,130))
plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')

##############################################################################
#Empty arrays
accu_cum_1_largest = np.zeros((len(dir_list), timeseries_size))
accu_cum_1_2smallest = np.zeros((len(dir_list), timeseries_size))
accu_cum_2_largest = np.zeros((len(dir_list), timeseries_size))
accu_cum_2_2smallest = np.zeros((len(dir_list), timeseries_size))
accu_cum_3_largest = np.zeros((len(dir_list), timeseries_size))
accu_cum_3_2smallest = np.zeros((len(dir_list), timeseries_size))
accu_cum_3_test_largest = np.zeros((len(dir_list), timeseries_size))
accu_cum_3_test_2smallest = np.zeros((len(dir_list), timeseries_size))

#Loop to calculate cumulate accumulations for each ensemble member
for i in range(accu_1_largest.shape[0]):
    for j in range(accu_1_largest.shape[1]):
        if j == 0:
            accu_cum_1_largest[i,j] = accu_1_largest[i,j]
            accu_cum_1_2smallest[i,j] = accu_1_2smallest[i,j]
            accu_cum_2_largest[i,j] = accu_2_largest[i,j]
            accu_cum_2_2smallest[i,j] = accu_2_2smallest[i,j]
            accu_cum_3_largest[i,j] = accu_3_largest[i,j]
            accu_cum_3_2smallest[i,j] = accu_3_2smallest[i,j]
            accu_cum_3_test_largest[i,j] = accu_3_test_largest[i,j]
            accu_cum_3_test_2smallest[i,j] = accu_3_test_2smallest[i,j]
        else:
            accu_cum_1_largest[i,j] = accu_1_largest[i,j] + accu_cum_1_largest[i,j-1]
            accu_cum_1_2smallest[i,j] = accu_1_2smallest[i,j] + accu_cum_1_2smallest[i,j-1]
            accu_cum_2_largest[i,j] = accu_2_largest[i,j] + accu_cum_2_largest[i,j-1]
            accu_cum_2_2smallest[i,j] = accu_2_2smallest[i,j] + accu_cum_2_2smallest[i,j-1]
            accu_cum_3_largest[i,j] = accu_3_largest[i,j] + accu_cum_3_largest[i,j-1]
            accu_cum_3_2smallest[i,j] = accu_3_2smallest[i,j] + accu_cum_3_2smallest[i,j-1]
            accu_cum_3_test_largest[i,j] = accu_3_test_largest[i,j] + accu_cum_3_test_largest[i,j-1]
            accu_cum_3_test_2smallest[i,j] = accu_3_test_2smallest[i,j] + accu_cum_3_test_2smallest[i,j-1]

#Ensemble mean cum accumulation time series
accu_cum_1_largest_mean = np.zeros((1,timeseries_size))
accu_cum_1_2smallest_mean = np.zeros((1,timeseries_size))
accu_cum_2_largest_mean = np.zeros((1,timeseries_size))
accu_cum_2_2smallest_mean = np.zeros((1,timeseries_size))
accu_cum_3_largest_mean = np.zeros((1,timeseries_size))
accu_cum_3_2smallest_mean = np.zeros((1,timeseries_size))
accu_cum_3_test_largest_mean = np.zeros((1,timeseries_size))
accu_cum_3_test_2smallest_mean = np.zeros((1,timeseries_size))
for k in range(accu_cum_1_largest.shape[1]):
    accu_cum_1_largest_mean[0,k] = np.mean(accu_cum_1_largest[:,k])
    accu_cum_1_2smallest_mean[0,k] = np.mean(accu_cum_1_2smallest[:,k])
    accu_cum_2_largest_mean[0,k] = np.mean(accu_cum_2_largest[:,k])
    accu_cum_2_2smallest_mean[0,k] = np.mean(accu_cum_2_2smallest[:,k])
    accu_cum_3_largest_mean[0,k] = np.mean(accu_cum_3_largest[:,k])
    accu_cum_3_2smallest_mean[0,k] = np.mean(accu_cum_3_2smallest[:,k])
    accu_cum_3_test_largest_mean[0,k] = np.mean(accu_cum_3_test_largest[:,k])
    accu_cum_3_test_2smallest_mean[0,k] = np.mean(accu_cum_3_test_2smallest[:,k])

#Ensemble std cum accumulation time series
accu_cum_1_largest_std = np.zeros((1,timeseries_size))
accu_cum_1_2smallest_std = np.zeros((1,timeseries_size))
accu_cum_2_largest_std = np.zeros((1,timeseries_size))
accu_cum_2_2smallest_std = np.zeros((1,timeseries_size))
accu_cum_3_largest_std = np.zeros((1,timeseries_size))
accu_cum_3_2smallest_std = np.zeros((1,timeseries_size))
accu_cum_3_test_largest_std = np.zeros((1,timeseries_size))
accu_cum_3_test_2smallest_std = np.zeros((1,timeseries_size))
for k in range(accu_cum_1_largest.shape[1]):
    accu_cum_1_largest_std[0,k] = np.std(accu_cum_1_largest[:,k])
    accu_cum_1_2smallest_std[0,k] = np.std(accu_cum_1_2smallest[:,k])
    accu_cum_2_largest_std[0,k] = np.std(accu_cum_2_largest[:,k])
    accu_cum_2_2smallest_std[0,k] = np.std(accu_cum_2_2smallest[:,k])
    accu_cum_3_largest_std[0,k] = np.std(accu_cum_3_largest[:,k])
    accu_cum_3_2smallest_std[0,k] = np.std(accu_cum_3_2smallest[:,k])
    accu_cum_3_test_largest_std[0,k] = np.std(accu_cum_3_test_largest[:,k])
    accu_cum_3_test_2smallest_std[0,k] = np.std(accu_cum_3_test_2smallest[:,k])

#Coefficient of Variation for ensemble cum accumulation time series
#np.std(x) / np.mean(x)
accu_cum_1_largest_cv = np.zeros((1,timeseries_size))
accu_cum_1_2smallest_cv = np.zeros((1,timeseries_size))
accu_cum_2_largest_cv = np.zeros((1,timeseries_size))
accu_cum_2_2smallest_cv = np.zeros((1,timeseries_size))
accu_cum_3_largest_cv = np.zeros((1,timeseries_size))
accu_cum_3_2smallest_cv = np.zeros((1,timeseries_size))
accu_cum_3_test_largest_cv = np.zeros((1,timeseries_size))
accu_cum_3_test_2smallest_cv = np.zeros((1,timeseries_size))
for k in range(accu_cum_1_largest.shape[1]):
    accu_cum_1_largest_cv[0,k] = np.std(accu_cum_1_largest[:,k]) / np.mean(accu_cum_1_largest[:,k])
    accu_cum_1_2smallest_cv[0,k] = np.std(accu_cum_1_2smallest[:,k]) / np.mean(accu_cum_1_2smallest[:,k])
    accu_cum_2_largest_cv[0,k] = np.std(accu_cum_2_largest[:,k]) / np.mean(accu_cum_2_largest[:,k])
    accu_cum_2_2smallest_cv[0,k] = np.std(accu_cum_2_2smallest[:,k]) / np.mean(accu_cum_2_2smallest[:,k])
    accu_cum_3_largest_cv[0,k] = np.std(accu_cum_3_largest[:,k]) / np.mean(accu_cum_3_largest[:,k])
    accu_cum_3_2smallest_cv[0,k] = np.std(accu_cum_3_2smallest[:,k]) / np.mean(accu_cum_3_2smallest[:,k])
    accu_cum_3_test_largest_cv[0,k] = np.std(accu_cum_3_test_largest[:,k]) / np.mean(accu_cum_3_test_largest[:,k])
    accu_cum_3_test_2smallest_cv[0,k] = np.std(accu_cum_3_test_2smallest[:,k]) / np.mean(accu_cum_3_test_2smallest[:,k])

#Ensemble median cum accumulation time series
accu_cum_1_largest_median = np.zeros((1,timeseries_size))
accu_cum_1_2smallest_median = np.zeros((1,timeseries_size))
accu_cum_2_largest_median = np.zeros((1,timeseries_size))
accu_cum_2_2smallest_median = np.zeros((1,timeseries_size))
accu_cum_3_largest_median = np.zeros((1,timeseries_size))
accu_cum_3_2smallest_median = np.zeros((1,timeseries_size))
accu_cum_3_test_largest_median = np.zeros((1,timeseries_size))
accu_cum_3_test_2smallest_median = np.zeros((1,timeseries_size))
for k in range(accu_cum_1_largest.shape[1]):
    accu_cum_1_largest_median[0,k] = np.median(accu_cum_1_largest[:,k])
    accu_cum_1_2smallest_median[0,k] = np.median(accu_cum_1_2smallest[:,k])
    accu_cum_2_largest_median[0,k] = np.median(accu_cum_2_largest[:,k])
    accu_cum_2_2smallest_median[0,k] = np.median(accu_cum_2_2smallest[:,k])
    accu_cum_3_largest_median[0,k] = np.median(accu_cum_3_largest[:,k])
    accu_cum_3_2smallest_median[0,k] = np.median(accu_cum_3_2smallest[:,k])
    accu_cum_3_test_largest_median[0,k] = np.median(accu_cum_3_test_largest[:,k])
    accu_cum_3_test_2smallest_median[0,k] = np.median(accu_cum_3_test_2smallest[:,k])



#Function to find closest value in array to a given value
def find_closest(arr, val):
       idx = np.abs(arr - val).argmin()
       return arr[idx]

#Plots
plt.figure()
for t in range(accu_cum_1_largest.shape[0]):
    plt.plot((accu_cum_1_largest[t]*(timestep/60)), color="grey")
plt.plot(accu_cum_1_largest_mean[0]*(timestep/60), color="red")
plt.plot((accu_cum_1_largest[np.argmax(accu_cum_1_largest[:,-1], axis=0)]*(timestep/60)), color="blue")
# plt.plot(accu_cum_1_largest_cv[0], color="blue")
# plt.plot(accu_cum_1_largest_std[0]*(timestep/60), color="yellow")
# plt.plot(accu_cum_1_largest_median[0]*(timestep/60), color="orange")
plt.plot((accu_cum_1_largest[int(np.where(accu_cum_1_largest[:,-1] == find_closest(accu_cum_1_largest[:,-1], accu_cum_1_largest_median[0,-1]))[0])]*(timestep/60)), color="orange")
plt.title("yellow")
plt.ylim((0,130))
plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
# plt.plot((accu_cum_osapol_1_largest[0]*(timestep/60)), color="blue")

plt.figure()
for t in range(accu_cum_1_largest.shape[0]):
    plt.plot((accu_cum_2_largest[t]*(timestep/60)), color="grey")
plt.plot(accu_cum_2_largest_mean[0]*(timestep/60), color="red")
plt.plot((accu_cum_2_largest[np.argmax(accu_cum_2_largest[:,-1], axis=0)]*(timestep/60)), color="blue")
# plt.plot(accu_cum_2_largest_cv[0], color="blue")
# plt.plot(accu_cum_2_largest_std[0]*(timestep/60), color="yellow")
# plt.plot(accu_cum_2_largest_median[0]*(timestep/60), color="orange")
plt.plot((accu_cum_2_largest[int(np.where(accu_cum_2_largest[:,-1] == find_closest(accu_cum_2_largest[:,-1], accu_cum_2_largest_median[0,-1]))[0])]*(timestep/60)), color="orange")
plt.title("orange")
plt.ylim((0,130))
plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
# plt.plot((accu_cum_osapol_2_largest[0]*(timestep/60)), color="blue")

plt.figure()
for t in range(accu_cum_1_largest.shape[0]):
    plt.plot((accu_cum_3_test_largest[t]*(timestep/60)), color="grey")
plt.plot(accu_cum_3_test_largest_mean[0]*(timestep/60), color="red")
plt.plot((accu_cum_3_test_largest[np.argmax(accu_cum_3_test_largest[:,-1], axis=0)]*(timestep/60)), color="blue")
# plt.plot(accu_cum_3_test_largest_cv[0], color="blue")
# plt.plot(accu_cum_3_test_largest_std[0]*(timestep/60), color="yellow")
# plt.plot(accu_cum_3_test_largest_median[0]*(timestep/60), color="orange")
plt.plot((accu_cum_3_test_largest[int(np.where(accu_cum_3_test_largest[:,-1] == find_closest(accu_cum_3_test_largest[:,-1], accu_cum_3_test_largest_median[0,-1]))[0])]*(timestep/60)), color="orange")
plt.title("red")
plt.ylim((0,130))
plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
# plt.plot((accu_cum_osapol_3_largest[0]*(timestep/60)), color="blue")

# plt.figure()
# for t in range(accu_cum_1_largest.shape[0]):
#     plt.plot((accu_cum_3_test_largest[t]*(timestep/60)), color="grey")
# plt.plot(accu_cum_3_test_largest_mean[0]*(timestep/60), color="red")
# plt.title("red - test")
# plt.ylim((0,55))
# plt.grid(True, which='major', axis='y')

plt.figure()
for t in range(accu_cum_1_largest.shape[0]):
    plt.plot((accu_cum_1_2smallest[t]*(timestep/60)), color="grey")
plt.plot(accu_cum_1_2smallest_mean[0]*(timestep/60), color="red")
plt.plot((accu_cum_1_2smallest[np.argmax(accu_cum_1_2smallest[:,-1], axis=0)]*(timestep/60)), color="blue")
# plt.plot(accu_cum_1_2smallest_cv[0], color="blue")
# plt.plot(accu_cum_1_2smallest_std[0]*(timestep/60), color="yellow")
# plt.plot(accu_cum_1_2smallest_median[0]*(timestep/60), color="orange")
plt.plot((accu_cum_1_2smallest[int(np.where(accu_cum_1_2smallest[:,-1] == find_closest(accu_cum_1_2smallest[:,-1], accu_cum_1_2smallest_median[0,-1]))[0])]*(timestep/60)), color="orange")
plt.title("purple")
plt.ylim((0,130))
plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
# plt.plot((accu_cum_osapol_1_2smallest[0]*(timestep/60)), color="blue")

plt.figure()
for t in range(accu_cum_1_largest.shape[0]):
    plt.plot((accu_cum_2_2smallest[t]*(timestep/60)), color="grey")
plt.plot(accu_cum_2_2smallest_mean[0]*(timestep/60), color="red")
plt.plot((accu_cum_2_2smallest[np.argmax(accu_cum_2_2smallest[:,-1], axis=0)]*(timestep/60)), color="blue")
# plt.plot(accu_cum_2_2smallest_cv[0], color="blue")
# plt.plot(accu_cum_2_2smallest_std[0]*(timestep/60), color="yellow")
# plt.plot(accu_cum_2_2smallest_median[0]*(timestep/60), color="orange")
plt.plot((accu_cum_2_2smallest[int(np.where(accu_cum_2_2smallest[:,-1] == find_closest(accu_cum_2_2smallest[:,-1], accu_cum_2_2smallest_median[0,-1]))[0])]*(timestep/60)), color="orange")
plt.title("green")
plt.ylim((0,130))
plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
# plt.plot((accu_cum_osapol_2_2smallest[0]*(timestep/60)), color="blue")

plt.figure()
for t in range(accu_cum_1_largest.shape[0]):
    plt.plot((accu_cum_3_test_2smallest[t]*(timestep/60)), color="grey")
plt.plot(accu_cum_3_test_2smallest_mean[0]*(timestep/60), color="red")
plt.plot((accu_cum_3_test_2smallest[np.argmax(accu_cum_3_test_2smallest[:,-1], axis=0)]*(timestep/60)), color="blue")
# plt.plot(accu_cum_3_test_2smallest_cv[0], color="blue")
# plt.plot(accu_cum_3_test_2smallest_std[0]*(timestep/60), color="yellow")
# plt.plot(accu_cum_3_test_2smallest_median[0]*(timestep/60), color="orange")
plt.plot((accu_cum_3_test_2smallest[int(np.where(accu_cum_3_test_2smallest[:,-1] == find_closest(accu_cum_3_test_2smallest[:,-1], accu_cum_3_test_2smallest_median[0,-1]))[0])]*(timestep/60)), color="orange")
plt.title("blue")
plt.ylim((0,130))
plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
# plt.plot((accu_cum_osapol_3_2smallest[0]*(timestep/60)), color="blue")

# plt.figure()
# for t in range(accu_cum_1_largest.shape[0]):
#     plt.plot((accu_cum_3_test_2smallest[t]*(timestep/60)), color="grey")
# plt.plot(accu_cum_3_test_2smallest_mean[0]*(timestep/60), color="red")
# plt.title("blue - test")
# plt.ylim((0,55))
# plt.grid(True, which='major', axis='y')

##############################################################################

# #find max of ensemble accu
# np.max(accu_cum_1_largest*(timestep/60))
# np.max(accu_cum_1_largest)
# np.max(accu_cum_1_2smallest*(timestep/60))
# np.max(accu_cum_1_2smallest)

# aa = np.where(accu_cum_1_largest == np.max(accu_cum_1_largest)) #Simulation_9067_6211_5943_1876
# aaa = np.where(accu_cum_1_2smallest == np.max(accu_cum_1_2smallest)) #Simulation_569_1509_7789_9010

##############################################################################
# SAVING RESULTS
    
#Output directory
# out_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simulations_pysteps/Event3_new/Accumulations_500"
out_dir = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Event3_new/Accumulations_500"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
#Save csvs
accu_1_2smallest_pd = pd.DataFrame(accu_1_2smallest)
pd.DataFrame(accu_1_2smallest_pd).to_csv(os.path.join(out_dir, "south_accu_1_2smallest.csv"))

# accu_1_largest
accu_1_largest_pd = pd.DataFrame(accu_1_largest)
pd.DataFrame(accu_1_largest_pd).to_csv(os.path.join(out_dir, "west_accu_1_largest.csv"))

# accu_2_2smallest
accu_2_2smallest_pd = pd.DataFrame(accu_2_2smallest)
pd.DataFrame(accu_2_2smallest_pd).to_csv(os.path.join(out_dir, "south_accu_2_2smallest.csv"))

# accu_2_largest
accu_2_largest_pd = pd.DataFrame(accu_2_largest)
pd.DataFrame(accu_2_largest_pd).to_csv(os.path.join(out_dir, "west_accu_2_largest.csv"))

# # accu_3_2smallest
# accu_3_2smallest_pd = pd.DataFrame(accu_3_2smallest)
# pd.DataFrame(accu_3_2smallest_pd).to_csv(os.path.join(out_dir, "accu_3_2smallest.csv"))

# # accu_3_largest
# accu_3_largest_pd = pd.DataFrame(accu_3_largest)
# pd.DataFrame(accu_3_largest_pd).to_csv(os.path.join(out_dir, "accu_3_largest.csv"))

# accu_3_test_2smallest
accu_3_test_2smallest_pd = pd.DataFrame(accu_3_test_2smallest)
pd.DataFrame(accu_3_test_2smallest_pd).to_csv(os.path.join(out_dir, "south_accu_3_test_2smallest.csv"))

# accu_3_test_largest
accu_3_test_largest_pd = pd.DataFrame(accu_3_test_largest)
pd.DataFrame(accu_3_test_largest_pd).to_csv(os.path.join(out_dir, "west_accu_3_test_largest.csv"))

# accu_whole_basin
accu_whole_basin_pd = pd.DataFrame(accu_whole_basin)
pd.DataFrame(accu_whole_basin_pd).to_csv(os.path.join(out_dir, "accu_whole_basin.csv"))

# accu_whole_basin_test
accu_whole_basin_test_pd = pd.DataFrame(accu_whole_basin_test)
pd.DataFrame(accu_whole_basin_test_pd).to_csv(os.path.join(out_dir, "accu_whole_basin_test.csv"))

# accu_whole_grid
accu_whole_grid_pd = pd.DataFrame(accu_whole_grid)
pd.DataFrame(accu_whole_grid_pd).to_csv(os.path.join(out_dir, "accu_whole_grid.csv"))

# accu_outside
accu_outside_pd = pd.DataFrame(accu_outside)
pd.DataFrame(accu_outside_pd).to_csv(os.path.join(out_dir, "accu_outside.csv"))

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

# Kolmannen vaiheen alueiden eventti akkumulaatiot
# - maksimi jokaiselle alueelle sadasta realisaatiosta
# - maksimi, minimi ja mean edellisen arvoista (+ sijainti ja realisaation numero)
# - verrataan vastaavien alueiden ja realisaatioiden mittariarvoihin 
# - samat laskut 1. ja 2. jakovaiheen alueille

##############################################################################

# TESTING 2 REALIZATIONS OF EVENT3

accu_cum_2_largest_mm = accu_cum_2_largest * (timestep/60)
# -> realizations 37 and 86
# = Simulation_4041_357_167_3268 and Simulation_8700_7030_4498_2879

#indices
ind_111 = kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"] == str(35.111)].index[0] #get index of wanted basin
ind_111_row = kokemaki_jako3.index.get_loc(ind_111) #get row number of largest subbasin
kokemaki_jako3.iloc[ind_111_row]

ind_112 = kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"] == str(35.112)].index[0] #get index of wanted basin
ind_112_row = kokemaki_jako3.index.get_loc(ind_112) #get row number of largest subbasin
kokemaki_jako3.iloc[ind_112_row]

ind_113 = kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"] == str(35.113)].index[0] #get index of wanted basin
ind_113_row = kokemaki_jako3.index.get_loc(ind_113) #get row number of largest subbasin
kokemaki_jako3.iloc[ind_113_row]

ind_114 = kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"] == str(35.114)].index[0] #get index of wanted basin
ind_114_row = kokemaki_jako3.index.get_loc(ind_114) #get row number of largest subbasin
kokemaki_jako3.iloc[ind_114_row]

#empty arrays
accu_3_test_111 = np.zeros((1, timeseries_size))
accu_3_test_112 = np.zeros((1, timeseries_size))
accu_3_test_113 = np.zeros((1, timeseries_size))
accu_3_test_114 = np.zeros((1, timeseries_size))

#calculate accumulations
chosen_realization = dir_list[37]
# chosen_realization = dir_list[86]

#Data location and files
data_dir_member = os.path.join(data_dir_location, chosen_realization, "Event_tiffs") #simulated event
files = os.listdir(data_dir_member)

#Import simulated event in raster-format and convert it into array
event_sim_array  = []
for i in range(len(files)):
    temp_raster = rasterio.open(os.path.join(data_dir_member, f"test_{i}.tif"))
    temp_array = temp_raster.read(1)
    event_sim_array.append(temp_array)
    if i == 0:
        event_affine = temp_raster.transform  
event_sim_array = np.concatenate([event_sim_array_[None, :, :] for event_sim_array_ in event_sim_array])

#Clear values over threshold of 45 dBZ
event_sim_array[event_sim_array > 45] = 0.5*(45 + event_sim_array[event_sim_array > 45])
#Event from dbz into mm/h
event_sim_array_mmh = event_sim_array.copy()
event_sim_array_mmh = 10**((event_sim_array_mmh-10*np.log10(a_R))/(10*b_R))
#Values less than threshold to zero
event_sim_array_mmh[event_sim_array_mmh < 0.1] = 0

for j in range(len(event_sim_array_mmh)):
    print(j)
    #3.lvl - 35.111
    temp_zonal3_111 = zonal_stats(kokemaki_jako3.iloc[ind_111_row]["geometry"], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_3_test_111[0, j] = float(temp_zonal3_111[0]["mean"])
    #3.lvl - 35.112
    temp_zonal3_112 = zonal_stats(kokemaki_jako3.iloc[ind_112_row]["geometry"], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_3_test_112[0, j] = float(temp_zonal3_112[0]["mean"])
    #3.lvl - 35.113
    temp_zonal3_113 = zonal_stats(kokemaki_jako3.iloc[ind_113_row]["geometry"], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_3_test_113[0, j] = float(temp_zonal3_113[0]["mean"])
    #3.lvl - 35.114
    temp_zonal3_114 = zonal_stats(kokemaki_jako3.iloc[ind_114_row]["geometry"], event_sim_array_mmh[j], affine=event_affine, nodata=-999, stats=["mean"])
    accu_3_test_114[0, j] = float(temp_zonal3_114[0]["mean"])
    
plt.plot(accu_3_test_111[0])
plt.plot(accu_3_test_112[0])
plt.plot(accu_3_test_113[0])
plt.plot(accu_3_test_114[0])


    
        



