# -*- coding: utf-8 -*-

# Events used in the study:
# 1. last radar image: 201306271955 -> number of previous files: 141
# 3. last radar image: 201310290345 -> number of previous files: 115
# 6. last radar image: 201408071800 -> number of previous files: 97
# Events are trimmed to start when areal mean is 5 dbz or higher, and to end when it is lower than 5 dbz.

##############################################################################
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

grid_on = False
colorbar_on =True 
predefined_value_range = False
cmap='Blues'

#Number of cascade levels
n_cascade_levels = 6




##############################################################################
# TIME USED FOR PARAMETER ESTIMATION: Start timer

run_start_0 = time.perf_counter()

##############################################################################
# INPUT DATA

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
R, quality, metadata = pysteps.io.read_timeseries(fns, importer, **importer_kwargs)
del quality  #delete quality variable because it is not used

#Add unit to metadata
metadata["unit"] = "mm/h"

#Replace non-finite values with zero
#Ehkä myöhemmin? Nannit maskataan optical flow:ssa
#for i in range(R.shape[0]):
#    R[i, ~np.isfinite(R[i, :])] = 0


#Values less than threshold to zero
R[R<0.1] = 0


##############################################################################
# DATA TRANSFORMATION INTO UNIT OF DBZ

metadata["zerovalue"] = 0
metadata["threshold"] = 0.1

#dBZ transformation for mm/h-data
#Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with size. J. Meteor., 5, 165â€“166.
#Z-R relationship: Z = a*R^b (Reflectivity)
#dBZ conversion: dBZ = 10 * log10(Z) (Decibels of Reflectivity)
a_R=223
b_R=1.53
#Leinonen et al. 2012 - A Climatology of Disdrometer Measurements of Rainfall in Finland over Five Years with Implications for Global Radar Observations
#https://doi.org/10.1175/JAMC-D-11-056.1

R, metadata = pysteps.utils.conversion.to_reflectivity(R, metadata, zr_a=a_R, zr_b=b_R)

#Bandpass filter
bp_filter = pysteps.cascade.bandpass_filters.filter_gaussian(R[0].shape, n_cascade_levels)

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

#Variables for magnitude and direction of advection
Vx = np.zeros(len(R)-1)
Vy = np.zeros(len(R)-1)

#Loop to calculate average x- and y-components of the advection, as well as magnitude and direction in xy-dir

R_plot = []
R_plot2 = []
ar_order = 2
gamma = np.empty((n_cascade_levels, ar_order))
gamma_sum = np.zeros((n_cascade_levels, ar_order))
count = 0
for i in range(2, len(R)-1):
    count = count + 1
    R_cur_d = []
    R_prev_adv_d = []
    R_prev2_adv_d = []
    R_cur = R[i, :, :]
    R_prev = R[i-1, :, :]
    R_prev2 = R[i-2, :, :]
    V1 = oflow_advection(np.stack([R_prev,R_cur],axis=0))
    V2 = oflow_advection(np.stack([R_prev2,R_prev],axis=0))
    R_prev_adv = extrapolator_method(R_prev, V1, 1, "min", **extrap_kwargs)[-1]
    R_prev2_adv = extrapolator_method(R_prev2, V2, 1, "min", **extrap_kwargs)[-1]
    R_prev2_adv = extrapolator_method(R_prev2_adv, V1, 1, "min", **extrap_kwargs)[-1]
    R_cur[~np.isfinite(R_cur)] = 0
    R_prev_adv[~np.isfinite(R_prev_adv)] = 0
    R_prev2_adv[~np.isfinite(R_prev2_adv)] = 0
    R_cur_d.append(pysteps.cascade.decomposition.decomposition_fft(R_cur, bp_filter, normalize=True, compute_stats=True))
    R_prev_adv_d.append(pysteps.cascade.decomposition.decomposition_fft(R_prev_adv, bp_filter, normalize=True, compute_stats=True))
    R_prev2_adv_d.append(pysteps.cascade.decomposition.decomposition_fft(R_prev2_adv, bp_filter, normalize=True, compute_stats=True))
    R_cur_c = pysteps.nowcasts.utils.stack_cascades(R_cur_d, n_cascade_levels, convert_to_full_arrays=True)
    R_prev_adv_c = pysteps.nowcasts.utils.stack_cascades(R_prev_adv_d, n_cascade_levels, convert_to_full_arrays=True)
    R_prev2_adv_c = pysteps.nowcasts.utils.stack_cascades(R_prev2_adv_d, n_cascade_levels, convert_to_full_arrays=True)
    for j in range(n_cascade_levels):
        gamma[j, :] = pysteps.timeseries.correlation.temporal_autocorrelation(
            np.stack([R_prev2_adv_c[j],R_prev_adv_c[j],R_cur_c[j]]), domain="spatial")
    level = 2
    gamma_sum = gamma_sum + gamma
    #R_plot2.append(R_cur_c[level,0,:,:])
    #R_plot.append(R_cur_c[level,0,:,:])
    #R_plot.append(R_prev_adv_c[level,0,:,:])
    #R_plot.append(R_prev2_adv_c[level,0,:,:])
    R_plot2.append(R_cur)
    R_plot.append(R_cur)
    R_plot.append(R_prev_adv)
    R_plot.append(R_prev2_adv)    

gamma_mean = gamma_sum / count
R_plot = np.concatenate([R_[None, :, :] for R_ in R_plot])
R_plot2 = np.concatenate([R_[None, :, :] for R_ in R_plot2])
ani = animate_interactive(R_plot,grid_on,colorbar_on, predefined_value_range,cmap)
ani2 = animate_interactive(R_plot2,grid_on,colorbar_on, predefined_value_range,cmap)


