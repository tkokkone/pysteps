# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:00:47 2022

@author: lindgrv1
"""

##############################################################################

#Packages
import os
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import pysteps
from datetime import datetime

#Data location
data_dir_location = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simulations_pysteps/Event1_new/Simulations/Simulation_2087_4095_3282_3135" # Simulation_2087_4095_3282_3135
data_dir = os.path.join(data_dir_location, "Event_tiffs") #simulated event
files = os.listdir(data_dir)

##############################################################################

#Import simulated event in raster-format and convert it into array in dbz
event_sim_array  = []
for i in range(len(files)):
    temp_raster = rasterio.open(os.path.join(data_dir, f"test_{i}.tif"))
    temp_array = temp_raster.read(1)
    event_sim_array.append(temp_array)
    if i == 0:
        event_affine = temp_raster.transform  
event_sim_array = np.concatenate([event_sim_array_[None, :, :] for event_sim_array_ in event_sim_array])

##############################################################################

#Clear values over threshold of 45 dBZ
event_sim_array[event_sim_array > 45] = 0.5*(45 + event_sim_array[event_sim_array > 45])

#Event from dbz into mm/h
a_R=223
b_R=1.53
event_sim_array_mmh = event_sim_array.copy()
event_sim_array_mmh = 10**((event_sim_array_mmh-10*np.log10(a_R))/(10*b_R))

#Values less than threshold to zero
event_sim_array_mmh[event_sim_array_mmh<0.1] = 0

##############################################################################

#Accumulation
event_sim_array_mmh_accu = sum(event_sim_array_mmh)
event_sim_array_mm_accu = event_sim_array_mmh_accu * (5/60)

plt.figure()
plt.imshow(event_sim_array_mm_accu)

np.max(event_sim_array_mm_accu) #[255,144]

#rows: 227 - 255
#columns: 137 - 182

result_accu = np.where(event_sim_array_mm_accu == np.amax(event_sim_array_mm_accu))

##############################################################################

clipped_area_mmh = event_sim_array_mmh[:, 238:255, 138:149]

clipped_area_dbz = event_sim_array[:, 235:255, 135:150]

##############################################################################

#Max cell value
print(np.max(event_sim_array)) #68.71488581575663
print(np.max(event_sim_array_mmh)) #904.3347675296155

10**(((56.5)-10*np.log10(a_R))/(10*b_R)) #64

#Indices of max value
result_sim = np.where(event_sim_array_mmh == np.amax(event_sim_array_mmh))

##############################################################################

#Plot
plt.figure()
plt.imshow(event_sim_array_mmh[int(result_sim[0])])
plt.colorbar()
plt.title("mm/h")

plt.figure()
plt.imshow(event_sim_array[int(result_sim[0])])
plt.colorbar()
plt.title("dbz")

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

#Max cell value Osapol
print(np.nanmax(event_osapol_array))
print(np.nanmax(event_osapol_array_mmh))

#Indices of max value Osapol
result_osapol = np.where(event_osapol_array_mmh == np.nanmax(event_osapol_array_mmh))

##############################################################################

#Plot
plt.figure()
plt.imshow(event_osapol_array_mmh[int(result_osapol[0])])
plt.colorbar()
plt.title("osapol mm/h")

plt.figure()
plt.imshow(event_osapol_array[int(result_osapol[0])])
plt.colorbar()
plt.title("osapol dbz")

plt.figure()
plt.imshow(event_osapol_array_mmh[22, 60:80, 215:235])
plt.colorbar()
plt.figure()
plt.imshow(event_osapol_array_mmh[23, 60:80, 215:235])
plt.colorbar()
plt.figure()
plt.imshow(event_osapol_array_mmh[24, 60:80, 215:235]) #max
plt.colorbar()
plt.figure()
plt.imshow(event_osapol_array_mmh[25, 60:80, 215:235])
plt.colorbar()
plt.figure()
plt.imshow(event_osapol_array_mmh[26, 60:80, 215:235])
plt.colorbar()

##############################################################################

#Accumulation
event_osapol_array_mmh_accu = sum(event_osapol_array_mmh)
plt.imshow(event_osapol_array_mmh_accu * (5/60))

##############################################################################

event_osapol_array_mmh[24]

plt.hist(event_osapol_array[24])

plt.hist(event_sim_array[87])

counts, bins = np.histogram(event_sim_array[80], bins=25) #87

event_osapol_array[60] = np.nan_to_num(event_osapol_array[60], nan=0.0)
counts_o, bins_o = np.histogram(event_osapol_array[60], bins=25)

plt.figure()
plt.hist(event_sim_array[87], range=(-70,70))
plt.figure()
plt.hist(event_sim_array[87], range=(-70,70), density=True)
plt.figure()
plt.hist(event_sim_array[87], bins=10, range=(-70,70), density=True)
plt.figure()
plt.hist(event_sim_array[87], bins=16, range=(-70,70), density=True)
plt.figure()
plt.hist(event_sim_array[87], bins=20, range=(-70,70), density=True)

##############################################################################

# Replase high values with nan
event_sim_array_copy = event_sim_array.copy()
event_sim_array_copy[event_sim_array_copy > 50] = np.nan

plt.figure()
plt.imshow(event_sim_array[87])
plt.figure()
plt.imshow(event_sim_array_copy[87])

nans = np.zeros(len(event_sim_array_copy))
for i in range(len(event_sim_array_copy)):
    nans[i] = np.count_nonzero(np.isnan(event_sim_array_copy[i]))
    
plt.figure()
plt.plot(nans)

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

# https://www.jstor.org/stable/pdf/26173732.pdf?refreqid=excelsior%3Aed7ef4870135ed53da63ee80b6ebe3eb&ab_segments=&origin=&acceptTC=1
# https://radarscope.zendesk.com/hc/en-us/articles/4411519812498-Finnish-Products
# https://journals.ametsoc.org/view/journals/apme/49/1/2009jamc2116.1.xml

