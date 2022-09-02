# IMPORT PACKAGES

import os
import re
import numpy as np
import pandas as pd
from osgeo import gdal 
import matplotlib.pyplot as plt
import time
import pysteps
from pysteps import extrapolation
from datetime import datetime
import statsmodels.api as sm
import scipy.stats as sp
from pysteps.visualization import plot_precip_field, animate_interactive

#
AR_length = 20
thold = 0.1
no_prev_files = 76 #97
tstep = 5
data_type = 2 #1: observations, 2: simulations
#in_dir: directory where simulation files are, assumes that radar tiffs are in
#a subdirectory called Event_tiffs
in_dir = "W:/Opinnaytteet/Vaitoskirjat/Ville/Data/Simulations/Simulations/Event1/Simulation_162_6321_1408_5133/"
adv_file = "sim_brokenlines.csv"

if data_type == 1:
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
    fns = pysteps.io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext, timestep=tstep, num_prev_files=no_prev_files)

    #Select the importer
    importer = pysteps.io.get_method(importer_name, "importer")

    #Read the radar composites
    R, quality, metadata = pysteps.io.read_timeseries(fns, importer, **importer_kwargs)
    del quality  #delete quality variable because it is not used
    #Add unit to metadata
    metadata["unit"] = "mm/h"
    for i in range(R.shape[0]):
        R[i, ~np.isfinite(R[i, :])] = 0

if data_type == 2:
    tiff_dir = in_dir + 'Event_tiffs/'
    files = [f for f in os.listdir(tiff_dir) if os.path.isfile(os.path.join(tiff_dir, f))]
    files.sort(key=lambda test_string : list( 
        map(int, re.findall(r'\d+', test_string)))[0])
    R_list = []
    for img_file in files:
        if img_file.endswith(".tif"):
            img_path = os.path.join(tiff_dir, img_file)
            ds = gdal.Open(img_path)
            R_list.append(np.array(ds.GetRasterBand(1).ReadAsArray()))
    R = np.concatenate([R_[None, :, :] for R_ in R_list])
#    for i in range(0, len(files)):   
#        img_path = os.path.join(tiff_dir, inFile)

    #Read advection velocities
    adv_file_path = in_dir + adv_file
    vel_csv_df = pd.read_csv(adv_file_path)
    vx_sim = vel_csv_df.iloc[4,1:]
    vy_sim = vel_csv_df.iloc[5,1:]

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

gamma = np.zeros((len(R)-AR_length,AR_length))
rain_pixels = np.zeros((len(R)-AR_length,AR_length))
for i in range(AR_length, len(R)):
    print("computing time step: " + str(i) + "\n")
    for j in range(1,AR_length+1):
        MASK_thr = np.ones(R[0].shape, dtype=bool)
        R2 = []
        R_tmp = R[i-AR_length+j-1]
        for k in range(1,AR_length-j+2):
            if data_type == 1:
                V = oflow_advection(np.stack([R_tmp,R[i-AR_length+j-1+k]],axis=0))
            else:
                 vx = vx_sim[i-AR_length+j-2+k]
                 vy = vy_sim[i-AR_length+j-2+k]
                 V = [vx*np.ones(R[0].shape),vy*np.ones(R[0].shape)]
                 V = np.concatenate([V_[None, :, :] for V_ in V])
            R_tmp = extrapolator_method(R_tmp , V, 1, "min", **extrap_kwargs)[-1]
        R2.append(R_tmp)
        MASK_thr[R_tmp < thold] = False
        R2.append(R[i])
        MASK_thr[R[i] < thold] = False
        gamma[i-AR_length,AR_length-j] = pysteps.timeseries.correlation.temporal_autocorrelation(
            np.stack(R2), mask=MASK_thr, domain="spatial")[0]
        rain_pixels[i-AR_length,AR_length-j] = MASK_thr.sum()
gamma_mean = gamma.mean(axis=0)
rain_pixels_mean = rain_pixels.mean(axis=0)
x = [i for i in range(1,AR_length+1)]
plt.figure()
plt.plot(x,gamma_mean)