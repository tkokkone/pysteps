# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:37:13 2023

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

##############################################################################
# DATA DIRECTORIES

data_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/GIS-aineistot" #letter "r" needs to be added in front of the path in Windows
fp1 = os.path.join(data_dir, "syke-valuma", "Paajako.shp") #Syken valuma-alue rasteri
fp2 = os.path.join(data_dir, "syke-valuma", "Jako3.shp") #Syken kolmannen jakovaiheen valuma-alue rasteri
fp3 = os.path.join(data_dir, "FMI_stations_2022-03.gpkg") #Ilmatieteenlaitoksen sää- ja sadehavaintoasemat
fp4 = os.path.join(data_dir, "FMI_stations_2022-05_rain.csv") #Ilmatieteenlaitoksen sää- ja sadehavaintoasemat, jotka mittaa sadetta (toukokuu 2022)

# data_dir_location = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simulations_pysteps/Event1_new/Simulations"
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

# #Plot geodataframes
# alue_paajako.plot()
# alue_jako3.plot()

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

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2) #, figsize=(20,6.5), constrained_layout = True
# alue_paajako.plot(ax = ax1)
# kokemaki_alue.plot(ax = ax1, color="red")
# kokemaki_jako3.plot(ax = ax2)
# fig.suptitle("Study area: Kokemäenjoki riverbasin", fontsize=16)
# if save_study_area == 1:
#     plt.savefig(os.path.join(out_dir, "map_study_area.png"))

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

#Tutkat
tutkat_sijainti = gpd.read_file(os.path.join(data_dir, "radars", "radars2_CopyFeatures1_Projected.shp"))
tutkat_alue = gpd.read_file(os.path.join(data_dir, "radars", "radars2_CopyFeatures1_Projected_Buffer.shp"))

tutkat_lista2013 = [101312, 101234, 100926, 101001]
tutkat_2013 = tutkat_sijainti[tutkat_sijainti["FMISID"].isin(tutkat_lista2013)]
tutkat_2013_alue = tutkat_alue[tutkat_alue["FMISID"].isin(tutkat_lista2013)]
tutkat_lista2014 = [101582, 101872]
tutkat_2014 = tutkat_sijainti[tutkat_sijainti["FMISID"].isin(tutkat_lista2014)]
tutkat_2014_alue = tutkat_alue[tutkat_alue["FMISID"].isin(tutkat_lista2014)]
tutkat_lista2015 = [100690, 101939]
tutkat_2015 = tutkat_sijainti[tutkat_sijainti["FMISID"].isin(tutkat_lista2015)]
tutkat_2015_alue = tutkat_alue[tutkat_alue["FMISID"].isin(tutkat_lista2015)]

# #Purkupiste
# purkupiste = gpd.read_file(os.path.join(data_dir, "syke-valuma", "Purkupiste.shp"))
# purkupiste = gpd.clip(purkupiste, bbox_df)
# purkupiste = purkupiste[purkupiste["PurkuTaso"]==1]
# purkupiste = purkupiste[purkupiste["ValumaTunn"]=="35.111"]

#Suomi
suomi = gpd.read_file(os.path.join(data_dir, "mml-hallintorajat_10k", "2021", "SuomenValtakunta_2021_10k.shp"))

#world
#http://www.naturalearthdata.com/downloads/10m-cultural-vectors/
countries = gpd.read_file(os.path.join(data_dir, "ne_10m_admin_0_countries", "ne_10m_admin_0_countries.shp"))
countries.crs
# countries = countries.to_crs(kokemaki_alue.crs)
# countries.crs

countries_finland = countries[countries["SOVEREIGNT"]=="Finland"]
countries_sweden = countries[countries["SOVEREIGNT"]=="Sweden"]
countries_norway = countries[countries["SOVEREIGNT"]=="Norway"]
countries_estonia = countries[countries["SOVEREIGNT"]=="Estonia"]
countries_russia = countries[countries["SOVEREIGNT"]=="Russia"]

#Set projection of all layers to WGS84
alue_paajako = alue_paajako.to_crs(countries.crs)
suomi = suomi.to_crs(countries.crs)
kokemaki_alue = kokemaki_alue.to_crs(countries.crs)
tutkat_2015_alue = tutkat_2015_alue.to_crs(countries.crs)
tutkat_2015 = tutkat_2015.to_crs(countries.crs)
tutkat_2014_alue = tutkat_2014_alue.to_crs(countries.crs)
tutkat_2014 = tutkat_2014.to_crs(countries.crs)
tutkat_2013_alue = tutkat_2013_alue.to_crs(countries.crs)
tutkat_2013 = tutkat_2013.to_crs(countries.crs)
kokemaki_jako3 = kokemaki_jako3.to_crs(countries.crs)
bbox_df = bbox_df.to_crs(countries.crs)
mittarit_sade_h = mittarit_sade_h.to_crs(countries.crs)


#AX_1
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2) #, gridspec_kw={'height_ratios': [1], "width_ratios": [1,2]}
# ax1.set_xlim(0, 800000)
# ax1.set_ylim(6500000, 7850000)
ax1.set_xlim(16, 33)
ax1.set_ylim(58.5, 71)
countries_sweden.plot(ax = ax1, color="gainsboro")
countries_norway.plot(ax = ax1, color="gainsboro")
countries_estonia.plot(ax = ax1, color="gainsboro")
countries_russia.plot(ax = ax1, color="gainsboro")

alue_paajako.plot(ax = ax1, color="darkgray", ec="black", linewidth=0.5)
suomi.plot(ax = ax1, fc="none", ec="black", linewidth=2)
kokemaki_alue.plot(ax = ax1, fc="lightblue", ec="black", linewidth=2)

tutkat_2014_alue.plot(ax = ax1, fc="green", ec="green", alpha=0.1)
tutkat_2014_alue.plot(ax = ax1, fc="none", ec="green")
tutkat_2014.plot(ax = ax1, color="green", marker="o", edgecolor="black", linewidth=1)

tutkat_2015_alue.plot(ax = ax1, fc="green", ec="green", alpha=0.1)
tutkat_2015_alue.plot(ax = ax1, fc="none", ec="green")
tutkat_2015.plot(ax = ax1, color="green", marker="o", edgecolor="black", linewidth=1)

tutkat_2013_alue.plot(ax = ax1, fc="green", ec="green", alpha=0.1)
tutkat_2013_alue.plot(ax = ax1, fc="none", ec="green")
tutkat_2013.plot(ax = ax1, color="green", marker="o", edgecolor="black", linewidth=1)

ax1.set_title("a)", loc="right")

x, y, arrow_length = 0.1, 0.95, 0.1
ax1.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black'),
            ha='center', va='center',
            xycoords=ax1.transAxes)

scalebar1 = ScaleBar(1, "km", length_fraction=0.4, location="lower right", scale_loc="top")
ax1.add_artist(scalebar1)

#AX_2
# ax2.set_xlim(185000, 475000)
# ax2.set_ylim(6700000, 6990000)
ax2.set_xlim(21, 26.75)
ax2.set_ylim(60.25, 63.25)
alue_paajako.plot(ax = ax2, fc="darkgray", ec="black", linewidth=0.5)
kokemaki_jako3.plot(ax = ax2, fc="lightblue", ec="black", linewidth=0.5)
kokemaki_alue.plot(ax = ax2, fc="none", ec="black", linewidth=2)
bbox_df.plot(ax = ax2, fc="none", ec="black", linewidth=2)

# kokemaki_jako1_polys[ind_1lvl_min2].plot(ax = ax2, color="yellow")
# kokemaki_jako2_polys[ind_2lvl_min2].plot(ax = ax2, color="orange")
# kokemaki_jako3[kokemaki_jako3["Jako3Tunnu"]==str(basin_3lvl_min2)].plot(ax = ax2, color="red")
# kokemaki_jako3.plot(ax = ax2, fc="none", ec="black", linewidth=0.5)

mittarit_sade_h.plot(ax = ax2, color="blue", marker="o", edgecolor="black", linewidth=1)

ax2.set_title("b)", loc="right")

x, y, arrow_length = 0.9, 0.90, 0.1 #width=5, headwidth=15)
ax2.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black'),
            ha='center', va='center',
            xycoords=ax2.transAxes)

scalebar2 = ScaleBar(1, "km", length_fraction=0.4, location="lower right", scale_loc="top")
ax2.add_artist(scalebar2)

fig.tight_layout()

out_figs = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Figures"
plt.savefig(os.path.join(out_figs,"figure_1test.pdf"), format="pdf", bbox_inches="tight")
