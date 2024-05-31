# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:14:55 2022

@author: lindgrv1
"""

# IMPORT PACKAGES

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
import pysteps
from pysteps import extrapolation
from datetime import datetime
import statsmodels.api as sm
import scipy.stats as sp
from pysteps.visualization import plot_precip_field, animate_interactive

#
AR_length = 8

# 1. last radar image: 201306271955 -> number of previous files: 141
# 3. last radar image: 201310290345 -> number of previous files: 115

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
R, quality, metadata = pysteps.io.read_timeseries(fns, importer, **importer_kwargs)
del quality  #delete quality variable because it is not used

###############################
# Ville added:
R[R<0.1] = 0

#Add unit to metadata
metadata["unit"] = "mm/h"

metadata["zerovalue"] = 0
metadata["threshold"] = 0.1

a_R=223
b_R=1.53
R, metadata = pysteps.utils.conversion.to_reflectivity(R, metadata, zr_a=a_R, zr_b=b_R)

###############################

for i in range(R.shape[0]):
    R[i, ~np.isfinite(R[i, :])] = np.nanmin(R[i])

#Choose the method to calculate advection
oflow_method = "LK" #Lukas-Kanade
oflow_advection = pysteps.motion.get_method(oflow_method) 

nx_field = R[0].shape[1]
ny_field = R[0].shape[0]
x_values, y_values = np.meshgrid(np.arange(nx_field), np.arange((ny_field)))
xy_coords = np.stack([x_values, y_values])


extrap_method = "semilagrangian"
extrapolator_method = extrapolation.get_method(extrap_method)
extrap_kwargs = dict()
extrap_kwargs["xy_coords"] = xy_coords
extrap_kwargs["allow_nonfinite_values"] = True

gamma = np.zeros((len(R)-2,AR_length))
for i in range(AR_length, len(R)):
    R2 = []
    for j in range(1,AR_length+1):
        R_tmp = R[i-AR_length+j-1]
        for k in range(1,AR_length-j+2): 
            V = oflow_advection(np.stack([R_tmp,R[i-AR_length+j-1+k]],axis=0))
            R_tmp = extrapolator_method(R_tmp , V, 1, "min", **extrap_kwargs)[-1]
        R2.append(R_tmp)
    R2.append(R[i])
    gamma[i-2,:] = pysteps.timeseries.correlation.temporal_autocorrelation(
            np.stack(R2), domain="spatial")
gamma_mean = gamma.mean(axis=0)
x = [i for i in range(1,AR_length+1)]
plt.figure()
plt.plot(x,gamma_mean)