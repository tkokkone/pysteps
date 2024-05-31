# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:06:53 2023

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


data_dir_location = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Event1_new/Simulations"
# data_dir_location = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Event3_new/Simulations"

data_dir_location = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Figures/To_be_submitted/review/bl_test" #data from brokenline tests

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

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
# kokemaki_alue.plot(ax = ax1)
# kokemaki_jako3.plot(ax = ax2)

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

#Z-R relationship: Z = a*R^b (Reflectivity)
a_R=223
b_R=1.53

#dBZ transformation for mm/h-data
event_osapol_array_dbz, event_osapol_metadata_dbz = pysteps.utils.conversion.to_reflectivity(event_osapol_array_mmh, event_osapol_metadata_mmh, zr_a=a_R, zr_b=b_R)

##############################################################################

# plt.figure()
# plt.imshow(event_osapol_array_mmh[0], cmap="Blues")
# plt.figure()
# plt.imshow(event_osapol_array_dbz[0], cmap="Blues")

##############################################################################
#accumulation field: OSAPOL
event_osapol_array_mmh_accu = np.nansum(event_osapol_array_mmh, 0)
event_osapol_array_dbz_accu = np.nansum(event_osapol_array_dbz, 0)
event_osapol_array_mm_accu = event_osapol_array_mmh_accu * (timestep/60)

# plt.figure()
# plt.imshow(event_osapol_array_mmh_accu, cmap="Blues")
# plt.figure()
# plt.imshow(event_osapol_array_dbz_accu, cmap="Blues")
# plt.figure()
# plt.imshow(event_osapol_array_mm_accu, cmap="Blues")

##############################################################################
#accumulation fields: ensemble members

# dir_list_slice = dir_list[0:8]

# accu_members = []

# for i in range(len(dir_list_slice)):
#     chosen_realization = dir_list_slice[i]
#     accu_location = os.path.join(data_dir_location, chosen_realization, "Calculations_500") 
#     accu_file = [file for file in os.listdir(accu_location) if file.endswith("_mm_accu.csv")]
#     accu_file = str(accu_file)
#     accu_file = accu_file[2:-2]
#     fp_accu = os.path.join(accu_location, accu_file)
    
#     accu_member = pd.read_csv(fp_accu, delimiter=(","))
#     accu_member = accu_member.to_numpy()
#     accu_member = accu_member[:,1:]
    
#     accu_members.append(accu_member)

# ##############################################################################
# #plot accumulation maps

# # plt.figure()
# # plt.imshow(accu_members[2], cmap="Blues")

# scale_min = 0.0
# scale_max = np.max([np.max(event_osapol_array_mm_accu),np.max(accu_members[0]),np.max(accu_members[1]),np.max(accu_members[2]),np.max(accu_members[3]),
#        np.max(accu_members[4]),np.max(accu_members[5]),np.max(accu_members[6]),np.max(accu_members[7])])

# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3)
# ax1.imshow(event_osapol_array_mm_accu, cmap="nipy_spectral", vmin=scale_min, vmax=scale_max)
# ax1.title.set_text("Observed event")
# ax2.imshow(accu_members[0], cmap="nipy_spectral", vmin=scale_min, vmax=scale_max)
# ax3.imshow(accu_members[1], cmap="nipy_spectral", vmin=scale_min, vmax=scale_max)
# ax4.imshow(accu_members[2], cmap="nipy_spectral", vmin=scale_min, vmax=scale_max)
# ax5.imshow(accu_members[3], cmap="nipy_spectral", vmin=scale_min, vmax=scale_max)
# ax6.imshow(accu_members[4], cmap="nipy_spectral", vmin=scale_min, vmax=scale_max)
# ax7.imshow(accu_members[5], cmap="nipy_spectral", vmin=scale_min, vmax=scale_max)
# ax8.imshow(accu_members[6], cmap="nipy_spectral", vmin=scale_min, vmax=scale_max)
# ax9.imshow(accu_members[7], cmap="nipy_spectral", vmin=scale_min, vmax=scale_max)

# norm_accu = mpl.colors.Normalize(vmin=scale_min,vmax=scale_max)
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_accu), ax=ax1, location="right", orientation="vertical", ticks=np.linspace(scale_min,scale_max,6))
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_accu), ax=ax2, location="right", orientation="vertical", ticks=np.linspace(scale_min,scale_max,6))
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_accu), ax=ax3, location="right", orientation="vertical", ticks=np.linspace(scale_min,scale_max,6))
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_accu), ax=ax4, location="right", orientation="vertical", ticks=np.linspace(scale_min,scale_max,6))
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_accu), ax=ax5, location="right", orientation="vertical", ticks=np.linspace(scale_min,scale_max,6))
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_accu), ax=ax6, location="right", orientation="vertical", ticks=np.linspace(scale_min,scale_max,6))
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_accu), ax=ax7, location="right", orientation="vertical", ticks=np.linspace(scale_min,scale_max,6))
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_accu), ax=ax8, location="right", orientation="vertical", ticks=np.linspace(scale_min,scale_max,6))
# fig.colorbar(plt.cm.ScalarMappable(cmap="nipy_spectral", norm=norm_accu), ax=ax9, location="right", orientation="vertical", ticks=np.linspace(scale_min,scale_max,6))

# # plt.savefig(os.path.join(r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Review_edits", "accumulations_maps_event1.png"))
# # plt.savefig(os.path.join(r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Review_edits", "accumulations_maps_event1.pdf"), format="pdf", bbox_inches="tight")
# # plt.savefig(os.path.join(r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Review_edits", "accumulations_maps_event2.png"))
# # plt.savefig(os.path.join(r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Review_edits", "accumulations_maps_event2.pdf"), format="pdf", bbox_inches="tight")

# ##############################################################################

# import matplotlib.colors as mcolors

# # draw filled contours.
# clevs = [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40,
#          50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 750]

# # clevs = [0.0,0.10,0.25,0.40,0.65,1,1.6,2.5,4,6.3,10,16,25,40,65,100,160]

# # In future MetPy
# # norm, cmap = ctables.registry.get_with_boundaries('precipitation', clevs)
# cmap_data = [(1.0, 1.0, 1.0),
#              (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
#              (0.0, 1.0, 1.0),
#              (0.0, 0.8784313797950745, 0.501960813999176),
#              (0.0, 0.7529411911964417, 0.0),
#              (0.501960813999176, 0.8784313797950745, 0.0),
#              (1.0, 1.0, 0.0),
#              (1.0, 0.6274510025978088, 0.0),
#              (1.0, 0.0, 0.0),
#              (1.0, 0.125490203499794, 0.501960813999176),
#              (0.9411764740943909, 0.250980406999588, 1.0),
#              (0.501960813999176, 0.125490203499794, 1.0),
#              (0.250980406999588, 0.250980406999588, 1.0),
#              (0.125490203499794, 0.125490203499794, 0.501960813999176),
#              (0.125490203499794, 0.125490203499794, 0.125490203499794),
#              (0.501960813999176, 0.501960813999176, 0.501960813999176),
#              (0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
#              (0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
#              (0.8549019694328308, 0.6509804129600525, 0.47058823704719543),
#              (0.6274510025978088, 0.42352941632270813, 0.23529411852359772),
#              (0.4000000059604645, 0.20000000298023224, 0.0)]

# cmap_new = mcolors.ListedColormap(cmap_data, 'precipitation')
# norm_new = mcolors.BoundaryNorm(clevs, cmap_new.N)

# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3)
# ax1.imshow(event_osapol_array_mm_accu, cmap=cmap_new, vmin=scale_min, vmax=scale_max)
# ax1.title.set_text("Observed event")
# ax2.imshow(accu_members[0], cmap=cmap_new, vmin=scale_min, vmax=scale_max)
# ax3.imshow(accu_members[1], cmap=cmap_new, vmin=scale_min, vmax=scale_max)
# ax4.imshow(accu_members[2], cmap=cmap_new, vmin=scale_min, vmax=scale_max)
# ax5.imshow(accu_members[3], cmap=cmap_new, vmin=scale_min, vmax=scale_max)
# ax6.imshow(accu_members[4], cmap=cmap_new, vmin=scale_min, vmax=scale_max)
# ax7.imshow(accu_members[5], cmap=cmap_new, vmin=scale_min, vmax=scale_max)
# ax8.imshow(accu_members[6], cmap=cmap_new, vmin=scale_min, vmax=scale_max)
# ax9.imshow(accu_members[7], cmap=cmap_new, vmin=scale_min, vmax=scale_max)

# fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_new, norm=norm_new), ax=ax1, location="right", orientation="vertical")
# fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_new, norm=norm_new), ax=ax2, location="right", orientation="vertical")
# fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_new, norm=norm_new), ax=ax3, location="right", orientation="vertical")
# fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_new, norm=norm_new), ax=ax4, location="right", orientation="vertical")
# fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_new, norm=norm_new), ax=ax5, location="right", orientation="vertical")
# fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_new, norm=norm_new), ax=ax6, location="right", orientation="vertical")
# fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_new, norm=norm_new), ax=ax7, location="right", orientation="vertical")
# fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_new, norm=norm_new), ax=ax8, location="right", orientation="vertical")
# fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_new, norm=norm_new), ax=ax9, location="right", orientation="vertical")

# ##############################################################################
# #total accumulation within the radar grid

# tot_accus = np.zeros(9)
# tot_accus[0] = np.sum(event_osapol_array_mm_accu)
# tot_accus[1] = np.sum(accu_members[0])
# tot_accus[2] = np.sum(accu_members[1])
# tot_accus[3] = np.sum(accu_members[2])
# tot_accus[4] = np.sum(accu_members[3])
# tot_accus[5] = np.sum(accu_members[4])
# tot_accus[6] = np.sum(accu_members[5])
# tot_accus[7] = np.sum(accu_members[6])
# tot_accus[8] = np.sum(accu_members[7])

# ##############################################################################
# ##############################################################################
# ##############################################################################
# ##############################################################################
# ##############################################################################

# #zonal accumulation timeseries from observed events for each 3rd level subbasin
# accu_mean_ts = np.zeros((len(kokemaki_jako3), len(event_osapol_array_mmh)))
# accu_sum_ts = np.zeros((len(kokemaki_jako3), len(event_osapol_array_mmh)))

# temp_raster = rasterio.open(fp5)
# event_affine = temp_raster.transform

# for step in range(len(event_osapol_array_mmh)):
#     accu_temp = zonal_stats(kokemaki_jako3["geometry"], event_osapol_array_mmh[step], affine=event_affine, nodata=-999, stats=["mean", "sum"])
#     for subbasin in range(len(accu_temp)):
#         accu_mean_ts[subbasin,step] = float(accu_temp[subbasin]["mean"])
#         accu_sum_ts[subbasin,step] = float(accu_temp[subbasin]["sum"])
        
# accu_mean_ts_cum = np.zeros((len(kokemaki_jako3), len(event_osapol_array_mmh)))
# accu_mean_ts_cum[:,0] = accu_mean_ts[:,0]
# for steps in range(1,len(event_osapol_array_mmh)):
#     accu_mean_ts_cum[:,steps] = accu_mean_ts[:,steps] + accu_mean_ts_cum[:,steps-1]
# accu_mean_ts_cum = accu_mean_ts_cum * (timestep/60)

# accu_mean_ts_cum_mean = np.zeros((1,timeseries_size+1))
# for k in range(accu_mean_ts_cum.shape[1]):
#     accu_mean_ts_cum_mean[0,k] = np.mean(accu_mean_ts_cum[:,k])

# accu_mean_ts_cum_median = np.zeros((1,timeseries_size+1))
# for k in range(accu_mean_ts_cum.shape[1]):
#     accu_mean_ts_cum_median[0,k] = np.median(accu_mean_ts_cum[:,k])

# #Function to find closest value in array to a given value
# def find_closest(arr, val):
#        idx = np.abs(arr - val).argmin()
#        return arr[idx]

# plt.figure()
# for i in range(len(accu_mean_ts_cum)): 
#     if i == 0:
#         plt.plot(accu_mean_ts_cum[i], color="grey", alpha=0.1, label="Ensemble members")
#     else:
#         plt.plot(accu_mean_ts_cum[i], color="grey", alpha=0.1)
# plt.plot(accu_mean_ts_cum_mean[0], color="red", label="Ensemble mean")
# plt.plot((accu_mean_ts_cum[np.argmax(accu_mean_ts_cum[:,-1], axis=0)]), color="orange", label="Ensemble max")
# plt.plot((accu_mean_ts_cum[int(np.where(accu_mean_ts_cum[:,-1] == find_closest(accu_mean_ts_cum[:,-1], accu_mean_ts_cum_median[0,-1]))[0])]), color="purple", label="Ensemble median")
# plt.xlabel("Time (min)")
# plt.ylabel("Cumulative rainfall (mm)")
# plt.ylim((0,70))
# plt.yticks(np.arange(0,71,10))
# plt.legend()
# # plt.savefig(os.path.join(r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Review_edits", "accumulations_3rd_osapol_event1_new70.png"))
# # plt.savefig(os.path.join(r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Review_edits", "accumulations_3rd_osapol_event1_new70.pdf"), format="pdf", bbox_inches="tight")
# # plt.savefig(os.path.join(r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Review_edits", "accumulations_3rd_osapol_event2_new1.png"))
# # plt.savefig(os.path.join(r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Review_edits", "accumulations_3rd_osapol_event2_new1.pdf"), format="pdf", bbox_inches="tight")

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

#import all areal mean rainfall bl-timeseries

bl_scaled = np.zeros((500, 142))
bl_org = np.zeros((500, 142))
for i in range(len(dir_list)):
    chosen_realization = dir_list[i]
    bl_location = os.path.join(data_dir_location, chosen_realization, "sim_brokenlines.csv")
    bl_member = pd.read_csv(bl_location, delimiter=(","))
    
    bl_member = bl_member.to_numpy()
    bl_member_scaled = bl_member[0,1:]
    bl_member_org = bl_member[1,1:]
    
    bl_scaled[i] = bl_member_scaled
    bl_org[i] = bl_member_org

bl_scaled_means = np.zeros((500, 1))
bl_org_means = np.zeros((500, 1))
for j in range(len(dir_list)):
    bl_scaled_means[j] = np.mean(bl_scaled[j])
    bl_org_means[j] = np.mean(bl_org[j])

##############################################################################
#Import simulated event in raster-format and convert it into array
event1 = 10
event2 = 100
event_sim_array_1  = []
# chosen_realization = dir_list[event1]
chosen_realization = dir_list[0]
tif_location = os.path.join(data_dir_location, chosen_realization, "Event_tiffs")
for i in range(len(files)):
    temp_raster = rasterio.open(os.path.join(tif_location, f"test_{i}.tif"))
    temp_array = temp_raster.read(1)
    event_sim_array_1.append(temp_array)
    if i == 0:
        event_affine = temp_raster.transform  
event_sim_array_1 = np.concatenate([event_sim_array_1_[None, :, :] for event_sim_array_1_ in event_sim_array_1])

event_sim_array_2  = []
# chosen_realization = dir_list[event2]
chosen_realization = dir_list[1]
tif_location = os.path.join(data_dir_location, chosen_realization, "Event_tiffs")
for i in range(len(files)):
    temp_raster = rasterio.open(os.path.join(tif_location, f"test_{i}.tif"))
    temp_array = temp_raster.read(1)
    event_sim_array_2.append(temp_array)
    if i == 0:
        event_affine = temp_raster.transform  
event_sim_array_2 = np.concatenate([event_sim_array_2_[None, :, :] for event_sim_array_2_ in event_sim_array_2])

#field means
event_sim_array_1_means = np.zeros((1, 142))
event_sim_array_2_means = np.zeros((1, 142))
for i in range(len(files)):
    event_sim_array_1_means[0,i] = np.mean(event_sim_array_1[i])
    event_sim_array_2_means[0,i] = np.mean(event_sim_array_2[i])
    
#mean of field mean
np.mean(event_sim_array_1_means)
np.mean(event_sim_array_2_means)

#######################################

event_sim_array_1_45 = event_sim_array_1.copy()
event_sim_array_2_45 = event_sim_array_2.copy()
#Clear values over threshold of 45 dBZ
# event_sim_array_1_45[event_sim_array_1_45 > 45] = 0.5*(45 + event_sim_array_1_45[event_sim_array_1_45 > 45])
# event_sim_array_2_45[event_sim_array_2_45 > 45] = 0.5*(45 + event_sim_array_2_45[event_sim_array_2_45 > 45])

# #field means
# event_sim_array_1_45_means = np.zeros((1, 142))
# event_sim_array_2_45_means = np.zeros((1, 142))
# for i in range(len(files)):
#     event_sim_array_1_45_means[0,i] = np.mean(event_sim_array_1_45[i])
#     event_sim_array_2_45_means[0,i] = np.mean(event_sim_array_2_45[i])
    
# #mean of field mean
# np.mean(event_sim_array_1_45_means)
# np.mean(event_sim_array_2_45_means)

#######################################

event_sim_array_1_45_mmh = event_sim_array_1_45.copy()
event_sim_array_2_45_mmh = event_sim_array_2_45.copy()
#unit conversion
a_R=223
b_R=1.53
event_sim_array_1_45_mmh = 10**((event_sim_array_1_45_mmh-10*np.log10(a_R))/(10*b_R))
event_sim_array_2_45_mmh = 10**((event_sim_array_2_45_mmh-10*np.log10(a_R))/(10*b_R))

#field means
event_sim_array_1_45_mmh_means = np.zeros((1, 142))
event_sim_array_2_45_mmh_means = np.zeros((1, 142))
for i in range(len(files)):
    event_sim_array_1_45_mmh_means[0,i] = np.mean(event_sim_array_1_45_mmh[i])
    event_sim_array_2_45_mmh_means[0,i] = np.mean(event_sim_array_2_45_mmh[i])
    
#mean of field mean
np.mean(event_sim_array_1_45_mmh_means)
np.mean(event_sim_array_2_45_mmh_means)

#######################################

event_sim_array_1_45_mmh_01 = event_sim_array_1_45_mmh.copy()
event_sim_array_2_45_mmh_01 = event_sim_array_2_45_mmh.copy()
#Values less than threshold to zero
event_sim_array_1_45_mmh_01[event_sim_array_1_45_mmh_01 < 0.1] = 0
event_sim_array_2_45_mmh_01[event_sim_array_2_45_mmh_01 < 0.1] = 0

#field means
event_sim_array_1_45_mmh_01_means = np.zeros((1, 142))
event_sim_array_2_45_mmh_01_means = np.zeros((1, 142))
for i in range(len(files)):
    event_sim_array_1_45_mmh_01_means[0,i] = np.mean(event_sim_array_1_45_mmh_01[i])
    event_sim_array_2_45_mmh_01_means[0,i] = np.mean(event_sim_array_2_45_mmh_01[i])
    
#mean of field mean
np.mean(event_sim_array_1_45_mmh_01_means)
np.mean(event_sim_array_2_45_mmh_01_means)
