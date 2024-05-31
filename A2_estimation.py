# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:17:00 2023

@author: lindgrv1
"""

# -*- coding: utf-8 -*-
"""
kiira
"""

##############################################################################
# IMPORT PACKAGES

import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import pysteps
from pysteps import extrapolation
from datetime import datetime
import statsmodels.api as sm
import scipy.stats as sp
import os
import math
import random

##############################################################################
# OUTDIR

out_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

##############################################################################
# OPTIONS TO SAVE FIGURES, TABLES, AND PARAMETERS

save_figs = 0
save_csvs = 0
save_animation = 0

##############################################################################
# INPUT DATA

# -> Start of the event: timestep when areal mean rainfall > 5 dBZ is 9 -> first image is 14:45
# -> End of the event: timestep when areal mean rainfall < 5 dBZ is 97 -> last image is 22:00

# radar_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\ARTIKKELI_2\data_a2\kiira_radar" #short event
input_dir = r"W:\lindgrv1\Kiira_whole_day\kiira_14-22_dbz"
radar_dir = input_dir

file_list = os.listdir(radar_dir)
file_list = [x for x in file_list if ".tif" in x]
file_list = [f for f in file_list if f.endswith(".tif")]

file_list = file_list[8:]

radar_kiira = []
for i in range(len(file_list)):
    src = rasterio.open(os.path.join(radar_dir, file_list[i]))
    array = src.read(1)
    radar_kiira.append(array)
    
radar_kiira = np.concatenate([radar_kiira_[None, :, :] for radar_kiira_ in radar_kiira])

#This have to be added for new event data: Remove last column from each layer
radar_kiira = radar_kiira[:,:,:-1]

#The following data is available for Finnish radar composite: radar reflectivity (dbz), conversion: Z[dBZ] = 0.5 * pixel value - 32
radar_kiira = (radar_kiira * 0.5) - 32

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

# np.max(radar_kiira)
# np.min(radar_kiira)
# np.min(radar_kiira[np.nonzero(radar_kiira)])

#Values less than threshold to wanted value, which represents no-rain
radar_kiira[radar_kiira < 10] = 3.1830486304816077

#Clear values over threshold of 45 dBZ -> This is not done for observed event, but just for simulated events
# radar_kiira[radar_kiira > 45] = 0.5*(45 + radar_kiira[radar_kiira > 45]) 

plt.figure()
plt.imshow(radar_kiira[25])

# ani_blue = pysteps.visualization.animations.animate_interactive(radar_kiira,False,True,False,"Blues")
# ani_pysteps = pysteps.visualization.animations.animate_interactive(radar_kiira,False,True,False,None)

if save_animation == 1:
    #Create datetime object
    starttime = "14:45:00"
    title_time = pd.date_range(starttime, periods=len(radar_kiira), freq="5Min")
    #If needed, add folder where to save the animation
    out_dir2 = os.path.join(out_dir, "Input_animation")
    if not os.path.exists(out_dir2):
        os.makedirs(out_dir2)
    #Save pngs
    for im in range(len(radar_kiira)):
        plt.figure()
        test_im = plt.imshow(radar_kiira[im], cmap="Blues", vmin=0, vmax=round(np.nanmax(radar_kiira)+0.5))
        plt.colorbar(test_im, spacing="uniform", extend="max", shrink=0.8, cax=None, label="Precipitation intensity [dBZ]")
        plt.title("Time: %s"% str(title_time[im])[11:16])
        plt.savefig(os.path.join(out_dir2, f"kiira_dbz_{im}.png"))
        plt.close()
        
##############################################################################
# 2048x2048 SIZED FFT-FILTER

fft_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"

fft_kiira = []
src_fft = rasterio.open(os.path.join(fft_dir, "fft_filter_tif_kiira.tif"))
array_fft = src_fft.read(1)
fft_kiira.append(array_fft) 
fft_kiira = np.concatenate([fft_kiira_[None, :, :] for fft_kiira_ in fft_kiira])
fft_kiira = fft_kiira[:,1:-1,:]

fft_kiira_nan = fft_kiira.copy()
print(fft_kiira_nan.max())
fft_kiira_nan = np.where(fft_kiira_nan==255, np.nan, fft_kiira_nan)
fft_kiira_nan = (fft_kiira_nan * 0.5) - 32
fft_kiira_nan[fft_kiira_nan < 10] = 3.1830486304816077

plt.figure()
plt.imshow(fft_kiira_nan[0])

fft_kiira_nan_temp = fft_kiira_nan.copy()
fft_kiira_nan_temp[0, ~np.isfinite(fft_kiira_nan[0, :])] = np.nanmin(fft_kiira_nan_temp[0, :])

plt.figure()
plt.imshow(fft_kiira_nan_temp[0])

Fnp = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(fft_kiira_nan_temp[0])

##############################################################################
# TIME SERIES OF AREAL MEAN R, STD, AND WAR IN UNIT OF DBZ

R = radar_kiira.copy()

#Areal mean rainfall in dbz
areal_rainfall_ts = np.zeros(len(file_list))
for i in range (len(file_list)):
    areal_rainfall_ts[i] = np.nanmean(R[i])
plt.figure()
plt.plot(areal_rainfall_ts)
plt.title("Areal mean rainfall (dBZ)")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "ts_meanR_dbz.png"))

#Wetted area ratio in dbz
war_ts = np.zeros(len(file_list))
for i in range (len(file_list)):
    war_ts[i] = (np.count_nonzero(R[i][R[i] > 3.1830486304816077])) / np.count_nonzero(~np.isnan(R[i]))
plt.figure()
plt.plot(war_ts)
plt.title("Wetted area ratio (-)")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "ts_WAR_dbz.png"))

#Standard deviation in dbz
std_ts = np.zeros(len(file_list))
for i in range (len(file_list)):
    std_ts[i] = np.nanstd(R[i])
plt.figure()
plt.plot(std_ts)
plt.title("Standard deviation (dBZ)")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "ts_STD_dbz.png"))

#Save time series from input data
if save_csvs == 1:
    data_temp = [areal_rainfall_ts, std_ts, war_ts]
    fft_params = pd.DataFrame(data_temp, index=['mean', 'std', 'war'])
    pd.DataFrame(fft_params).to_csv(os.path.join(out_dir, "data_tss.csv"))
    
##############################################################################
# CORRECT MAR, WAR, AND STD TIME SERIES

#plot mar time series
plt.figure()
plt.plot(areal_rainfall_ts)
plt.title("Areal mean rainfall (dBZ)")
plt.axvline(x = 17, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 29, color = "black", linestyle = "-", linewidth=0.5)
plt.axvline(x = 36, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 44, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 45, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 50, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 51, color = "black", linestyle = "-", linewidth=0.5)

#plot war time series
plt.figure()
plt.plot(war_ts)
plt.title("Wetted area ratio (-)")
plt.axvline(x = 17, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 29, color = "black", linestyle = "-", linewidth=0.5)
plt.axvline(x = 36, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 44, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 45, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 50, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 51, color = "black", linestyle = "-", linewidth=0.5)

#plot std time series
plt.figure()
plt.plot(std_ts)
plt.title("Standard deviation (dBZ)")
plt.axvline(x = 17, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 29, color = "black", linestyle = "-", linewidth=0.5)
plt.axvline(x = 36, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 44, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 45, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 50, color = "black", linestyle = "-", linewidth=0.5)
# plt.axvline(x = 51, color = "black", linestyle = "-", linewidth=0.5)

#take a copy from all time series
corrected_areal_rainfall_ts = areal_rainfall_ts.copy()
corrected_war_ts = war_ts.copy()
corrected_std_ts = std_ts.copy()

#correct all time series
x_obs_17 = [16, 18]
y_mar_obs_17 = [areal_rainfall_ts[16], areal_rainfall_ts[18]]
y_war_obs_17 = [war_ts[16], war_ts[18]]
y_std_obs_17 = [std_ts[16], std_ts[18]]
x_new_17 = 17
y_mar_new_17 = np.interp(x_new_17, x_obs_17, y_mar_obs_17)
y_war_new_17 = np.interp(x_new_17, x_obs_17, y_war_obs_17)
y_std_new_17 = np.interp(x_new_17, x_obs_17, y_std_obs_17)
corrected_areal_rainfall_ts[17] = y_mar_new_17
corrected_war_ts[17] = y_war_new_17
corrected_std_ts[17] = y_std_new_17

x_obs_36 = [34, 37]
y_mar_obs_36 = [areal_rainfall_ts[34], areal_rainfall_ts[37]]
y_war_obs_36 = [war_ts[34], war_ts[37]]
y_std_obs_36 = [std_ts[34], std_ts[37]]
x_new_36 = [35, 36]
y_mar_new_36 = np.interp(x_new_36, x_obs_36, y_mar_obs_36)
y_war_new_36 = np.interp(x_new_36, x_obs_36, y_war_obs_36)
y_std_new_36 = np.interp(x_new_36, x_obs_36, y_std_obs_36)
corrected_areal_rainfall_ts[35] = y_mar_new_36[0]
corrected_areal_rainfall_ts[36] = y_mar_new_36[1]
corrected_war_ts[35] = y_war_new_36[0]
corrected_war_ts[36] = y_war_new_36[1]
corrected_std_ts[35] = y_std_new_36[0]
corrected_std_ts[36] = y_std_new_36[1]

x_obs_46 = [45, 50]
y_mar_obs_46 = [areal_rainfall_ts[45], areal_rainfall_ts[50]]
y_war_obs_46 = [war_ts[45], war_ts[50]]
y_std_obs_46 = [std_ts[45], std_ts[50]]
x_new_46 = [46, 47, 48, 49]
y_mar_new_46 = np.interp(x_new_46, x_obs_46, y_mar_obs_46)
y_war_new_46 = np.interp(x_new_46, x_obs_46, y_war_obs_46)
y_std_new_46 = np.interp(x_new_46, x_obs_46, y_std_obs_46)
corrected_areal_rainfall_ts[46] = y_mar_new_46[0]
corrected_areal_rainfall_ts[47] = y_mar_new_46[1]
corrected_areal_rainfall_ts[48] = y_mar_new_46[2]
corrected_areal_rainfall_ts[49] = y_mar_new_46[3]
corrected_war_ts[46] = y_war_new_46[0]
corrected_war_ts[47] = y_war_new_46[1]
corrected_war_ts[48] = y_war_new_46[2]
corrected_war_ts[49] = y_war_new_46[3]
corrected_std_ts[46] = y_std_new_46[0]
corrected_std_ts[47] = y_std_new_46[1]
corrected_std_ts[48] = y_std_new_46[2]
corrected_std_ts[49] = y_std_new_46[3]

#plot
plt.figure()
plt.plot(corrected_areal_rainfall_ts)
plt.title("Areal mean rainfall (dBZ)")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "ts_meanR_dbz_corrected.png"))
plt.figure()
plt.plot(corrected_war_ts)
plt.title("Wetted area ratio (-)")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "ts_WAR_dbz_corrected.png"))
plt.figure()
plt.plot(corrected_std_ts)
plt.title("Standard deviation (dBZ)")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "ts_STD_dbz_corrected.png"))

#Save time series from input data
if save_csvs == 1:
    data_temp = [corrected_areal_rainfall_ts, corrected_std_ts, corrected_war_ts]
    fft_params = pd.DataFrame(data_temp, index=['mean', 'std', 'war'])
    pd.DataFrame(fft_params).to_csv(os.path.join(out_dir, "data_tss_corrected.csv"))

##############################################################################
# CALCULATE ADVECTION TIME SERIES

#Choose the method to calculate advection
oflow_method = "LK" #Lukas-Kanade
oflow_advection = pysteps.motion.get_method(oflow_method) 

#Variables for x- and y-components of the advection
Vx = np.zeros(len(R)-1)
Vy = np.zeros(len(R)-1)

#Loop to calculate average x- and y-components of the advection, as well as magnitude and direction in xy-dir
V_of = []
for i in range(0, len(R)-1):
    V_of.append(oflow_advection(R[i:i+2, :, :]))
    Vx[i] = np.nanmean(V_of[i][0]) #field mean in x-direction
    Vy[i] = np.nanmean(V_of[i][1]) #field mean in y-direction
    
plt.figure()
plt.plot(Vx)
plt.title("V_x")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "V_x.png"))

plt.figure()
plt.plot(Vy)
plt.title("V_y")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "V_y.png"))
    
#Save advection time series
if save_csvs == 1:
    data_temp = [Vx, Vy]
    fft_params = pd.DataFrame(data_temp, index=['Vx', 'Vy'])
    pd.DataFrame(fft_params).to_csv(os.path.join(out_dir, "advection_tss.csv"))

#Variables for magnitude and direction of advection
Vmag = np.zeros(len(R)-1)
Vdir_rad = np.zeros(len(R)-1)

#Loop to calculate average x- and y-components of the advection, as well as magnitude and direction in xy-dir
for i in range(len(R)-1):
    Vmag[i] = np.sqrt(Vx[i]**2 + Vy[i]**2) #total magnitude of advection
    Vdir_rad[i] = np.arctan(Vy[i]/Vx[i]) #direction of advection in xy-dir in radians
    #The inverse of tan, so that if y = tan(x) then x = arctan(y).
    #Range of values of the function: −pi/2 < y < pi/2 (-1.5707963267948966 < y < 1.5707963267948966) -> np.pi gives value of pi
Vdir_deg = (Vdir_rad/(2*np.pi))*360 #direction of advection in xy-dir in degrees

#Turn advection direction 90 degs clochwise
Vdir_deg_new = Vdir_deg + 90
Vdir_rad_new = (Vdir_deg_new/360)*(2*np.pi)
#Advection components after changing direction
Vx_turn = np.cos(Vdir_rad_new) * Vmag
Vy_turn = np.sin(Vdir_rad_new / 360 * 2 * np.pi) * Vmag

#Make advection slower
Vmag_new = Vmag / 2
#Advection components after changing speed
Vx_slower = np.cos(Vdir_rad / 360 * 2 * np.pi) * Vmag_new
Vy_slower = np.sin(Vdir_rad / 360 * 2 * np.pi) * Vmag_new

##############################################################################
# GENERAL BROKEN LINE PARAMETERS

q = 0.8
noBLs = 1
var_tol = 0.3
mar_tol = 0.7

tStep = float(5) #timestep [min]
tSerieLength = (len(R) - 1) * tStep #timeseries length [min]
x_values = np.arange(0, tSerieLength + tStep, tStep)
euler = 2.71828

##############################################################################
# BROKEN LINE PARAMETERS FOR AREAL MEAN RAINFALL

#Mean and variance of input time series
mu_z = float(np.mean(corrected_areal_rainfall_ts)) #mean of input time series [dBz]
sigma2_z = float(np.var(corrected_areal_rainfall_ts)) #variance of input time series [dBz]

#Correlations and a_zero of time series
event_cors = sm.tsa.acf(corrected_areal_rainfall_ts, nlags=(len(corrected_areal_rainfall_ts) - 1))
plt.figure()
plt.plot(event_cors)
plt.axhline(y = 0, color = 'black', linestyle = '-', linewidth=0.5)
plt.title("Autocorrelations: mean R")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "autocor_meanR.png"))

#Timesteps for no autocorrelation
no_autocor = np.where(event_cors < 0) #first this was the condition: < 1/euler
a_zero = (no_autocor[0][0]-1)*tStep #timesteps in which correlation values are > 1/e = 0.36787968862663156

#Power spectrum, beta, and H of time series
power_spectrum = np.abs(np.fft.fft(corrected_areal_rainfall_ts))**2 #absolute value of the complex components of fourier transform of data
freqs = np.fft.fftfreq(corrected_areal_rainfall_ts.size, tStep) #frequencies

freqs_sliced = freqs[1:]
freqs_sliced = freqs_sliced[freqs_sliced>0]
power_spectrum_sliced = power_spectrum[1:]
power_spectrum_sliced = power_spectrum_sliced[0:len(freqs_sliced)]

#Calculate power spectrum exponent beta and H from beta using linear fitting of the data
slope, intercept, r_value, p_value, std_err = sp.linregress(np.log(freqs_sliced[0:len(freqs_sliced)]), np.log(power_spectrum_sliced[0:len(freqs_sliced)]))
y_values = slope*np.log(freqs_sliced[0:len(freqs_sliced)])+intercept

plt.figure()
plt.plot(np.log(freqs_sliced), np.log(power_spectrum_sliced))
plt.plot(np.log(freqs_sliced[0:len(freqs_sliced)]), y_values)
plt.title("Freqs vs. power spectrum -fit: mean R")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "powerspectr_meanR.png"))

H = (-slope-1)/2

##############################################################################
# FFT-FILTER

#Time step of maximum areal mean rainfall
max_idx = np.where(corrected_areal_rainfall_ts == corrected_areal_rainfall_ts.max())
max_idx = int(max_idx[0])

# #Moving averages of areal mean rainfall and timestep of maximum moving average
# window_size = 10
# moving_averages = []
# i = 0
# while i < len(areal_rainfall_ts) - window_size + 1:
#     window = areal_rainfall_ts[i : i + window_size]
#     window_average = round(sum(window) / window_size, 2)
#     moving_averages.append(window_average)
#     i += 1
# moving_averages = np.array(moving_averages)

# max_idx2 = np.where(moving_averages == moving_averages.max())
# max_idx2 = int(max_idx2[0])

#Replace non-finite values with the minimum value
R2 = R.copy()
for i in range(R2.shape[0]):
    R2[i, ~np.isfinite(R[i, :])] = np.nanmin(R2[i, :])
# R3 = R2.copy()  
# R3 = R3[max_idx2:max_idx2+window_size]

#Power law filters using only one field (max areal mean rainfall) and average of 10 fields (max moving average)
Fp = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(R2[max_idx])
Fnp = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2[max_idx])
# Fp_ave = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(R3)
# Fnp_ave = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R3)

# TEST THE FILTER
seed = 1234
num_realizations = 3
# Generate noise
Np = []
Nnp = []
# Np_ave = []
# Nnp_ave = []
for k in range(num_realizations):
    Np.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fp, seed=seed + k))
    Nnp.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp, seed=seed + k))
    # Np_ave.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fp_ave, seed=seed + k))
    # Nnp_ave.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_ave, seed=seed + k))

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
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "fft_filters-par_vs_nonpar.png"))

# fig, ax = plt.subplots(nrows=2, ncols=3)
# # parametric noise
# ax[0, 0].imshow(Np_ave[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[0, 1].imshow(Np_ave[1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[0, 2].imshow(Np_ave[2], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# # nonparametric noise
# ax[1, 0].imshow(Nnp_ave[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[1, 1].imshow(Nnp_ave[1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[1, 2].imshow(Nnp_ave[2], cmap=cm.RdBu_r, vmin=-3, vmax=3)

# for i in range(2):
#     for j in range(3):
#         ax[i, j].set_xticks([])
#         ax[i, j].set_yticks([])
# plt.tight_layout()
# plt.show()

# #TEST: filters from different timesteps
# Fnp_field_0 = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2[0])
# Fnp_field_5 = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2[5])
# Fnp_field_10 = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2[10])
# Fnp_field_20 = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2[20])
# Fnp_field_30 = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2[30])
# Fnp_field_36 = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2[36])
# Fnp_ave_all = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2)
# seed = 1234
# Nnp_field_0 = []
# Nnp_field_5 = []
# Nnp_field_10 = []
# Nnp_field_20 = []
# Nnp_field_30 = []
# Nnp_field_36 = []
# Nnp_ave_all = []
# Nnp_ave_10 = []
# Nnp_field_0.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_field_0, seed=seed))
# Nnp_field_5.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_field_5, seed=seed))
# Nnp_field_10.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_field_10, seed=seed))
# Nnp_field_20.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_field_20, seed=seed))
# Nnp_field_30.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_field_30, seed=seed))
# Nnp_field_36.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_field_36, seed=seed))
# Nnp_ave_all.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_ave_all, seed=seed))
# Nnp_ave_10.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_ave, seed=seed))

# # Plot the generated noise fields
# fig, ax = plt.subplots(nrows=3, ncols=3)
# ax[0, 0].imshow(Nnp_field_0[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[0, 1].imshow(Nnp_field_5[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[0, 2].imshow(Nnp_field_10[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[1, 0].imshow(Nnp_field_20[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[1, 1].imshow(Nnp_field_30[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[1, 2].imshow(Nnp_field_36[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[2, 0].imshow(Nnp[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[2, 1].imshow(Nnp_ave_10[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# ax[2, 2].imshow(Nnp_ave_all[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# fig.suptitle("i = [0, 5, 10, 20, 30, 36, max, ave_10, ave_all")

# for i in range(2):
#     for j in range(3):
#         ax[i, j].set_xticks([])
#         ax[i, j].set_yticks([])
# plt.tight_layout()
# plt.show()
# if save_figs == 1:
#     plt.savefig(os.path.join(out_dir, "fft_filters.png"))

##############################################################################
# CREATE GAUSSIAN BANDPASS FILTER

#Number of cascade levels
n_cascade_levels = 6

#Bandpass filter
bp_filter = pysteps.cascade.bandpass_filters.filter_gaussian(R2[0].shape, n_cascade_levels)

#Spatial scale in each cascade level in km
#https://gmd.copernicus.org/articles/12/4185/2019/, figure1
L_k = np.zeros([n_cascade_levels, 1])
L_k[0] = 256
for i in range(1, len(L_k)):
    L_k[i] = (L_k[0]/2)/(bp_filter["central_wavenumbers"][i])

##############################################################################
# ESTIMATE AR(2)-MODEL PARAMETERS

#The Lagrangian temporal evolution of each normalised cascade level is modelled using an AR(2)-model
ar_order = 2

#Create variables that include all gammas and phis of all cascade levels for each time step
gamma_all = np.zeros([n_cascade_levels*2, len(R2)-1])
phi_all = np.zeros([n_cascade_levels*3, len(R2)-1])

nx_field = R2[0].shape[1]
ny_field = R2[0].shape[0]
x_values, y_values = np.meshgrid(np.arange(nx_field), np.arange((ny_field)))
xy_coords = np.stack([x_values, y_values])
extrap_method = "semilagrangian"
extrapolator_method = extrapolation.get_method(extrap_method)
extrap_kwargs = dict()
extrap_kwargs["xy_coords"] = xy_coords
extrap_kwargs["allow_nonfinite_values"] = True

#Compute the cascade decompositions of the input precipitation fields
R_prev_adv = []
R_prev2_adv = []
R_prev_adv.append(0)
R_prev_adv.append(0)
R_prev2_adv.append(0)
R_prev2_adv.append(0)
for i in range(2,len(R2)-1): 
    R_prev_adv.append(extrapolator_method(R2[i-1], V_of[i-1], 1, "min", **extrap_kwargs)[-1])
    R_prev2_tmp = extrapolator_method(R2[i-2], V_of[i-2], 1, "min", **extrap_kwargs)[-1]
    R_prev2_adv.append(extrapolator_method(R_prev2_tmp, V_of[i-1], 1, "min", **extrap_kwargs)[-1])
  
gamma = np.zeros((n_cascade_levels, ar_order))    
for i in range(2,len(R2)-1):    
    R_cur_d = []
    R_prev_adv_d = []
    R_prev2_adv_d = []

    R_cur_d.append(pysteps.cascade.decomposition.decomposition_fft(R2[i], bp_filter, normalize=True, compute_stats=True))
    R_prev_adv_d.append(pysteps.cascade.decomposition.decomposition_fft(R_prev_adv[i], bp_filter, normalize=True, compute_stats=True))
    R_prev2_adv_d.append(pysteps.cascade.decomposition.decomposition_fft(R_prev2_adv[i], bp_filter, normalize=True, compute_stats=True))
    R_cur_c = pysteps.nowcasts.utils.stack_cascades(R_cur_d, n_cascade_levels, convert_to_full_arrays=True)
    R_prev_adv_c = pysteps.nowcasts.utils.stack_cascades(R_prev_adv_d, n_cascade_levels, convert_to_full_arrays=True)
    R_prev2_adv_c = pysteps.nowcasts.utils.stack_cascades(R_prev2_adv_d, n_cascade_levels, convert_to_full_arrays=True)
    
    for j in range(n_cascade_levels):
        gamma[j, :] = pysteps.timeseries.correlation.temporal_autocorrelation(
            np.stack([R_prev2_adv_c[j],R_prev_adv_c[j],R_cur_c[j]]), domain="spatial")
    
    #Adjust the lag-2 correlation coefficient to ensure that the AR(p) process is stationary
    for l in range(n_cascade_levels):
        gamma[l, 1] = pysteps.timeseries.autoregression.adjust_lag2_corrcoef2(gamma[l, 0], gamma[l, 1])
        
    #Estimate the parameters of the AR(p)-model from the autocorrelation coefficients
    phi = np.empty((n_cascade_levels, ar_order + 1))
    for m in range(n_cascade_levels):
        phi[m, :] = pysteps.timeseries.autoregression.estimate_ar_params_yw(gamma[m, :])
        
    #Fill gamma_all and phi_all after each timestep
    gamma_all[0:n_cascade_levels, i] = gamma[:, 0]
    gamma_all[n_cascade_levels:len(gamma_all), i] = gamma[:, 1]
    phi_all[0:n_cascade_levels, i] = phi[:, 0]
    phi_all[n_cascade_levels:(2*n_cascade_levels), i] = phi[:, 1]
    phi_all[(2*n_cascade_levels):len(phi_all), i] = phi[:, 2]

##############################################################################
# PLOT EXAMPLE OF CASCADE DECOMPOSITION

decomp = pysteps.cascade.decomposition.decomposition_fft(R2[max_idx], bp_filter, compute_stats=True)
# Plot the normalized cascade levels
for i in range(n_cascade_levels):
    mu = decomp["means"][i]
    sigma = decomp["stds"][i]
    decomp["cascade_levels"][i] = (decomp["cascade_levels"][i] - mu) / sigma
    
fig, ax = plt.subplots(nrows=2, ncols=4) 
ax[0, 0].imshow(R2[max_idx], cmap=cm.RdBu_r, vmin=-5, vmax=5)
ax[0, 1].imshow(decomp["cascade_levels"][0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 2].imshow(decomp["cascade_levels"][1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 3].imshow(decomp["cascade_levels"][2], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 0].imshow(decomp["cascade_levels"][3], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 1].imshow(decomp["cascade_levels"][4], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 2].imshow(decomp["cascade_levels"][5], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 0].set_title("Observed: step %i" % max_idx)
ax[0, 1].set_title("Level 1")
ax[0, 2].set_title("Level 2")
ax[0, 3].set_title("Level 3")
ax[1, 0].set_title("Level 4")
ax[1, 1].set_title("Level 5")
ax[1, 2].set_title("Level 6")
ax[1, 3].set_title("Level 7")
for i in range(2):
    for j in range(4):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.tight_layout()
plt.show()
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "example_cascades_max-ts.png"))

##############################################################################
# CALCULATE GAMMA1, GAMMA2, AND TAU_K

#Mean of gamma1 for each cascade level
gamma1_mean = np.zeros([n_cascade_levels, 1])
for i in range(0, len(gamma1_mean)):
    gamma1_mean[i] = np.mean(abs(gamma_all[i,2:])) #Fixed the calculation skip first two valies, because those are zero

#Mean of gamma2 for each cascade level
gamma2_mean = np.zeros([n_cascade_levels, 1])
for i in range(len(gamma2_mean), len(gamma_all)):
    gamma2_mean[i-len(gamma1_mean)] = np.mean(abs(gamma_all[i,2:])) #Fixed the calculation skip first two valies, because those are zero

timestep = 5
#Lagrangian temporal autocorrelation
tau_k = np.zeros([n_cascade_levels, 1])
for i in range(0, len(tau_k)):
    tau_k[i] = -timestep/np.log(gamma1_mean[i])

##############################################################################
# ESTIMATING AR-PARAMETERS: tlen_a, tlen_b, and tlen_c

#Estimate tlen_a and tlen_b based on linear fit of tau_k and L_k
slope_tlen, intercept_tlen, r_value_tlen, p_value_tlen, std_err_tlen = sp.linregress(np.log(L_k[1:n_cascade_levels,0]), np.log(tau_k[1:n_cascade_levels,0]))
y_values_tlen = slope_tlen*np.log(L_k[1:n_cascade_levels])+intercept_tlen

tlen_b = slope_tlen
tlen_a = np.exp(intercept_tlen)
print(intercept_tlen, tlen_a) #tlen_a
print(slope_tlen) #tlen_b

plt.figure()
plt.plot(np.log(L_k[0:n_cascade_levels,0]), np.log(tau_k[0:n_cascade_levels,0]), marker='s')
plt.plot(np.log(L_k[1:n_cascade_levels]), y_values_tlen)
plt.title("L_k vs. Tau_k -fit - AR-pars: tlen_a (%1.4f) and tlen_b (%1.4f)" % (tlen_a, tlen_b))
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "ar_pars-a_b.png"))

#Estimate tlen_c based on linear fit of gamma1 and gamma2
slope_tlen_c, intercept_tlen_c, r_value_tlen_c, p_value_tlen_c, std_err_tlen_c = sp.linregress(np.log(gamma1_mean[0:n_cascade_levels-1,0]), np.log(gamma2_mean[0:n_cascade_levels-1,0]))
y_values_tlen_c = slope_tlen_c*np.log(gamma1_mean[0:n_cascade_levels-1,0])+intercept_tlen_c

tlen_c = slope_tlen_c
print(slope_tlen_c) #tlen_c

plt.figure()
plt.plot(np.log(gamma1_mean[0:n_cascade_levels,0]), np.log(gamma2_mean[0:n_cascade_levels,0]), marker='s')
plt.plot(np.log(gamma1_mean[0:n_cascade_levels-1]), y_values_tlen_c)
plt.title("Gamma1 vs. Gamma2 -fit - AR-pars: tlen_c (%1.4f)" % tlen_c)
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "ar_pars-c.png"))

##############################################################################
# FIT VARIABLES WITH FIELD MEAN

#polynomial fit with degree = 2
std_fit = np.poly1d(np.polyfit(areal_rainfall_ts, std_ts, 2))
print(std_fit)
std_fit_str = str(std_fit)
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, std_ts)
plt.plot(polyline, std_fit(polyline))
plt.title("mean vs. std: %s" % std_fit_str)
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "fit_STD.png"))

#polynomial fit with degree = 2
war_fit = np.poly1d(np.polyfit(areal_rainfall_ts, war_ts, 2))
print(war_fit)
war_fit_str = str(war_fit)
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, war_ts)
plt.plot(polyline, war_fit(polyline))
plt.title("mean vs. war: %s" % war_fit_str)
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "fit_WAR.png"))
    
#SAME WITH CORRECTED TIME SERIES

#polynomial fit with degree = 2
std_fit_corrected = np.poly1d(np.polyfit(corrected_areal_rainfall_ts, corrected_std_ts, 2))
print(std_fit_corrected)
std_fit_corrected_str = str(std_fit_corrected)
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(corrected_areal_rainfall_ts), max(corrected_areal_rainfall_ts), 400)
plt.scatter(corrected_areal_rainfall_ts, corrected_std_ts)
plt.plot(polyline, std_fit_corrected(polyline))
plt.title("corrected mean vs. std: %s" % std_fit_corrected_str)
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "fit_STD_corrected.png"))

#polynomial fit with degree = 2
war_fit_corrected = np.poly1d(np.polyfit(corrected_areal_rainfall_ts, corrected_war_ts, 2))
print(war_fit_corrected)
war_fit_corrected_str = str(war_fit_corrected)
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(corrected_areal_rainfall_ts), max(corrected_areal_rainfall_ts), 400)
plt.scatter(corrected_areal_rainfall_ts, corrected_war_ts)
plt.plot(polyline, war_fit_corrected(polyline))
plt.title("corrected mean vs. war: %s" % war_fit_corrected_str)
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, "fit_WAR_corrected.png"))

##############################################################################
# SAVE PARAMETERS AS CSV-FILES

#AR-parameter time series
if save_csvs == 1:
    ar_params1 = pd.DataFrame(gamma1_mean)
    ar_params1[1] = gamma2_mean
    ar_params1[2] = tau_k
    ar_params1[3] = L_k
    ar_params1.rename(columns = {0 : 'gamma1_mean', 1 : 'gamma2_mean', 2 : 'tau_k', 3 : 'L_k'}, inplace = True)
    pd.DataFrame(ar_params1).to_csv(os.path.join(out_dir, "ar_params.csv"))
    
#Estimatted simulation parameters
if save_csvs == 1:
    data_temp = [tlen_a, tlen_b, tlen_c, std_fit[0], std_fit[1], std_fit[2], war_fit[0], war_fit[1], war_fit[2], 
                 mu_z, sigma2_z, H, a_zero, q, noBLs, var_tol, mar_tol]
    all_params = pd.DataFrame(data_temp, index=['tlen_a', 'tlen_b', 'tlen_c', 'a_v', 'b_v', 'c_v', 'a_war', 'b_war', 'c_war',
                                                'mu_z', 'sigma2_z', 'h_val_z', 'a_zero_z', 'q', 'no_bls', 'var_tol', 'mar_tol'])
    pd.DataFrame(all_params).to_csv(os.path.join(out_dir, "all_params.csv"))
    
if save_csvs == 1:
    data_temp = [tlen_a, tlen_b, tlen_c, std_fit_corrected[0], std_fit_corrected[1], std_fit_corrected[2], war_fit_corrected[0], war_fit_corrected[1], war_fit_corrected[2], 
                 mu_z, sigma2_z, H, a_zero, q, noBLs, var_tol, mar_tol]
    all_params = pd.DataFrame(data_temp, index=['tlen_a', 'tlen_b', 'tlen_c', 'a_v', 'b_v', 'c_v', 'a_war', 'b_war', 'c_war',
                                                'mu_z', 'sigma2_z', 'h_val_z', 'a_zero_z', 'q', 'no_bls', 'var_tol', 'mar_tol'])
    pd.DataFrame(all_params).to_csv(os.path.join(out_dir, "all_params_corrected.csv"))

##############################################################################
# FUNCTION TO CREATE BROKEN LINES

def create_broken_lines(mu_z, sigma2_z, H, q, a_zero, tStep, tSerieLength, noBLs, var_tol, mar_tol, areal_rainfall_ts):
    """A function to create multiplicative broken lines

    Input Parameters:
        :mu_z -- Mean of input time series [dBz]:
        :sigma2_z -- Variance of input time series [dBz]:
        :H -- Structure function exponent [-]:
        :q -- Scale ratio between levels n and n+1 (constant) [-]:
        :a_zero -- Time series decorrelation time [min]:
        :tStep -- Timestep [min]:
        :tSerieLength -- Length of the time series in units of a_zero [min]:
        :noBLs -- Number of broken lines [-]:
        :var_tol -- Acceptable tolerance for variance as ratio of input variance [-]:
        :mar_tol -- Acceptable value for first and last elements of the final broken line as ratio of input mean [-]:

    Returns:
        :blines_final -- A matrix including chosen number of broken lines:

    Reference:
        :Seed et al. 2000 - A multiplicative broken-line model for time series of mean areal rainfall:
    """

    # Parameters
    H2 = 2*H  # Why 2*H is needed?
    # X-values of final time series
    x_values = np.arange(0, tSerieLength + tStep, tStep)
    iPoints = len(x_values)  # Number of data points in the final time series

    # Statistics for the logarithmic time series
    sigma2_y = np.log(((math.sqrt(sigma2_z)/mu_z)**2.0) +
                      1.0)  # Variance of Y_t: Eq. 15
    mu_y = np.log(mu_z) - 0.5*sigma2_y  # Mean of Y_t: Eq. 14
    # Number of simple broken lines (ksii): Eq. 11
    iN = int((np.log(tStep/a_zero)/np.log(q)) + 1)
    # Variance of the broken line at the outer scale: modified Eq. 12
    sigma2_0 = ((1-pow(q, H2))/(1-pow(q, H2*(iN+1))))*sigma2_y

    # Broken line statistics for individual levels
    a_p = np.zeros(iN)
    sigma2_p = np.zeros(iN)
    sigma_p = np.zeros(iN)
    for p in range(0, iN):
        # The time lag between vertices of the broken line on level p: Eq. 11
        a_p[p] = a_zero * pow(q, p)
        # The variance of the broken line on level p: Eq. 10
        sigma2_p[p] = sigma2_0 * pow(q, p*H2)
        # The standard deviation of the broken line on level p
        sigma_p[p] = math.sqrt(sigma2_p[p])

    # Limits for variance and mean of final sum broken line
    var_min = sigma2_z - (var_tol * sigma2_z)  # acceptable minimum of variance
    var_max = sigma2_z + (var_tol* sigma2_z)  # acceptable maximum of variance
    mar_max = mar_tol * mu_z  # acceptable maximum of mean areal rainfall
    
    ### NEW: ACCEPTABLE LIMITS FOR VARIANCE
    var_tol_fixed = 0.2
    var_min = sigma2_z - (var_tol_fixed * sigma2_z) 
    var_max = sigma2_z + (var_tol_fixed * sigma2_z) 
    
    ### NEW: CONDITION REQUIREMENT FOR FIRST AND LAST VALUES OF BROKEN LINE TIME SERIES
    lim_min_first = areal_rainfall_ts[0] -1
    lim_max_first = areal_rainfall_ts[0] +1
    lim_min_last = areal_rainfall_ts[-1] -1
    lim_max_last = areal_rainfall_ts[-1] +1

    # Empty matrices
    blines = np.zeros((iPoints, iN))  # all simple broken lines (ksii)
    # all sum broken lines (final broken lines)
    blines_final = np.zeros((iPoints, noBLs))

    # Loop to create noBLs number of sum broken lines
    noAccepted_BLs = 0
    while noAccepted_BLs < noBLs:
        bline_sum = np.zeros(iPoints)

        # Loop to create one sum broken line from iN number of simple broken lines
        for p in range(0, iN):

            # Generate a multiplicative broken line time series
            eeta = []
            ksii = []

            # Create random variates for this simple broken line
            level = p
            # Number of vertices (iid variables) for this level (+ 2 to cover the ends + 1 for trimming of decimals)
            N = int(tSerieLength/a_p[level] + 3)
            eeta = np.random.normal(0.0, 1.0, N)

            # Interpolate values of this simple broken line at every time step
            k = np.random.uniform(0.0, 1.0)
            a = a_p[level]
            # Location first eeta for this simple broken line
            x0 = float(-k * a)
            xp = np.linspace(x0, N * a, num=N)
            yp = eeta
            ksii = np.interp(x_values, xp, yp)

            # Force to correct mean (0) and standard deviation (dSigma_p) for this simple broken line
            ksii_mean = float(np.mean(ksii))  # mean of ksii
            ksii_std = float(np.std(ksii))  # standard deviation of ksii
            ksii_scaled = ((ksii - ksii_mean) / ksii_std) * \
                sigma_p[int(level)]  # scaling of array

            # Add values of this simple broken line to sum array
            bline_sum = bline_sum + ksii_scaled

            # Create matrix including all simple broken lines
            blines[:, p] = ksii_scaled

        # Set corrections for sum broken line and create final broken line
        # Set correct mean for the sum array (as we have 0 sum)
        bline_sum_corrected = bline_sum + mu_y
        # Exponentiate the sum array to get the final broken line
        bline_sum_exp = np.exp(bline_sum_corrected)

        # Accept sum brokenlines that fulfil desired conditions of mean and variance:
        #var_min < var_bline_sum_exp < var_max
        # bline_sum_exp[0] (first value) and bline_sum_exp[-1] (last value) have to be less than mar_max
        # variance of sum broken line
        var_bline_sum_exp = np.var(bline_sum_exp)

        ### NEW: CONDITION REQUIREMENT FOR FIRST AND LAST VALUES OF BROKEN LINE TIME SERIES
        # if (var_min < var_bline_sum_exp and var_bline_sum_exp < var_max and bline_sum_exp[0] < mar_max and bline_sum_exp[-1] < mar_max):
        if (var_min < var_bline_sum_exp and var_bline_sum_exp < var_max and bline_sum_exp[0] < lim_max_first and bline_sum_exp[0] > lim_min_first 
            and bline_sum_exp[-1] < lim_max_last and bline_sum_exp[-1] > lim_min_last):

            # Create matrix including all accepted sum broken lines
            blines_final[:, noAccepted_BLs] = bline_sum_exp

            # If sum broken is accapted, save corresponding simple broken lines as csv
            # pd.DataFrame(blines).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/tests/blines_%d.csv" %(noAccepted_BLs))
            print("Created %d." % noAccepted_BLs)
            noAccepted_BLs = noAccepted_BLs + 1
        else:
            print(str(noAccepted_BLs) + ". not accepted.")

    return blines_final

##############################################################################
# OPTIMIZING A_ZERO FOR AREAL MEAN RAINFALL

#Initialize variables
mu_z = mu_z #mean of mean areal reflectivity over the simulation period
sigma2_z = sigma2_z #variance of mean areal reflectivity over the simulation period
h_val_z = H #structure function exponent
q_val_z = q #scale ratio between levels n and n+1 (constant) [-]
a_zero_z = a_zero #time series decorrelation time [min]
no_bls_test = 100 #number of broken lines
var_tol_z = var_tol #0.3
mar_tol_z = mar_tol #0.7
timestep = 5
n_timesteps = len(R)  # number of timesteps

a_zeros = []
sses = []
test_autocors_ensemble_means = []

R_mean_kiira = []
for i in range (len(file_list)):
    R_mean_kiira.append(np.nanmean(R2[i]))
autocors_kiira = sm.tsa.acf(R_mean_kiira, nlags=(len(R_mean_kiira)))

#Loop to calculate a_zeros
seed_test_bls = random.randint(1, 10000)
np.random.seed(seed_test_bls)  #set seed
for i in range(int(a_zero_z/timestep), (len(R)-1)): #a_zero_z = 80 (len(R)-1)
    azero_0 = (i*5)
    
    test_r_mean = create_broken_lines(mu_z, sigma2_z, h_val_z, q_val_z, azero_0, timestep, (n_timesteps-1) * timestep, no_bls_test, var_tol_z, mar_tol_z, areal_rainfall_ts)
    
    test_r_mean_t = np.transpose(test_r_mean)
    
    test_autocors_ensemble = []
    for i in range(len(test_r_mean_t)):
        test_autocors_ensemble.append(sm.tsa.acf(test_r_mean_t[i], nlags=(len(test_r_mean_t[i])))) #autocors for each broken line time series
    test_autocors_ensemble = np.vstack(test_autocors_ensemble)
    
    test_autocors_ensemble_mean = []
    for i in range(test_autocors_ensemble.shape[1]):
        test_autocors_ensemble_mean.append(np.mean(test_autocors_ensemble[:,i]))

    #SSE
    sse = sum((test_autocors_ensemble_mean[0:int(a_zero_z/timestep)] - autocors_kiira[0:int(a_zero_z/timestep)])**2.0)
    print(azero_0, sse)
    a_zeros.append(azero_0)
    sses.append(sse)
    test_autocors_ensemble_means.append(test_autocors_ensemble_mean)

a_zeros = np.vstack(a_zeros)
a_zeros = np.hstack(a_zeros)
sses = np.vstack(sses)
sses = np.hstack(sses)

a_zero_vs_sse = np.empty([len(a_zeros), 3])
for i in range(0,len(a_zero_vs_sse)):
    a_zero_vs_sse[i,0] = i
a_zero_vs_sse[:,1] = a_zeros
a_zero_vs_sse[:,2] = sses

a_zero_min = int(a_zero_vs_sse[:,1][a_zero_vs_sse[:,2] == np.min(a_zero_vs_sse[:,2])])
a_zero_nro = int(a_zero_vs_sse[:,0][a_zero_vs_sse[:,1] == a_zero_min])

#Plot whole time series
plt.figure()
plt.axhline(y = 0, color = 'black', linestyle = '-', linewidth=0.5)
plt.axvline(x = int(a_zero_z/timestep), color = 'black', linestyle = '-', linewidth=0.5)
for bl in range(len(test_autocors_ensemble_means)):
    plt.plot(test_autocors_ensemble_means[bl], color="Gray")
plt.plot(test_autocors_ensemble_means[a_zero_nro], color="Red", label=f"a_zero = {a_zero_min} ({a_zero_z}) min")
plt.plot(autocors_kiira, color="Blue", label="kiira")
plt.legend()
plt.title(f"Optimized a_zero vs. kiira (SSE = {a_zero_vs_sse[:,2][a_zero_vs_sse[:,2] == np.min(a_zero_vs_sse[:,2])]}, seed = {seed_test_bls})")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, f"optimized_azero_whole_seed{seed_test_bls}.png"))

#Plot just part used for optimizing
plt.figure()
plt.axhline(y = 0, color = 'black', linestyle = '-', linewidth=0.5)
for i in range(len(test_autocors_ensemble_means)):
    plt.plot(test_autocors_ensemble_means[i][0:int(a_zero_z/timestep)], color="Gray")
plt.plot(test_autocors_ensemble_means[a_zero_nro][0:int(a_zero_z/timestep)], color="Red", label=f"a_zero = {a_zero_min} ({a_zero_z}) min")
plt.plot(autocors_kiira[0:int(a_zero_z/timestep)], color="Blue", label="kiira")
plt.legend()
plt.title(f"Optimized a_zero vs. kiira (SSE = {a_zero_vs_sse[:,2][a_zero_vs_sse[:,2] == np.min(a_zero_vs_sse[:,2])]}, seed = {seed_test_bls})")
if save_figs == 1:
    plt.savefig(os.path.join(out_dir, f"optimized_azero_seed{seed_test_bls}.png"))
    
#Save a_zero csvs
if save_csvs == 1:
    data_temp = [a_zero_vs_sse[:,0], a_zero_vs_sse[:,1], a_zero_vs_sse[:,2]]
    data_temp_save = pd.DataFrame(data_temp, index=['nro', 'azero', 'sse'])
    pd.DataFrame(data_temp_save).to_csv(os.path.join(out_dir, f"azero_autocors_seed{seed_test_bls}.csv"))

if save_csvs == 1:
    csv_autocors = np.hstack(test_autocors_ensemble_means)
    data_temp2 = [csv_autocors]
    data_temp2_save = pd.DataFrame(data_temp2, index=['autocor_means'])
    pd.DataFrame(data_temp2_save).to_csv(os.path.join(out_dir, f"azero_autocor_means_seed{seed_test_bls}.csv"))
    csv_autocors = np.vstack(test_autocors_ensemble_means)
    
# ##############################################################################  
# # TESTING BROKEN LINES WITH MULTIPLE H AND A_ZERO VALUERS 

# seed_bl_testing = 1234 #6508
# no_bls_testing = 100
# for i in range(1,2):
#     np.random.seed(seed_bl_testing) #set seed
#     #Changing a_zero
#     a_zero_z = all_params[12]
#     a_zero_z = 175
#     # a_zero_z = a_zero_z - (i-1)*5
#     #Changing H
#     h_val_z = all_params[11]
#     # h_val_z = 0.67
#     # h_val_z = h_val_z + (i-1)*0.10
#     test_r_mean = create_broken_lines(mu_z, sigma2_z, h_val_z, q_val_z, a_zero_z, timestep, (n_timesteps-1) * timestep, no_bls_testing, var_tol_z, mar_tol_z, areal_rainfall_ts)
#     plt.figure()
#     for j in range(len(test_r_mean[0])):
#         if j == 0:
#             plt.plot(test_r_mean[:,j], color="Gray", label="brokenlines")
#         else:
#             plt.plot(test_r_mean[:,j], color="Gray")  
#     plt.plot(areal_rainfall_ts, label="observed", linestyle="--", color="Red")
#     plt.title(f"a_zero = {a_zero_z} and \nH = {h_val_z}")
#     plt.legend()
# if save_figs == 0:
#     plt.savefig(os.path.join(out_dir, f"optimizing_H({h_val_z})_and_azero({a_zero_z})_{seed_bl_testing}.png"))

############################################################################## 

# LONGER MAR AND ADVECTION TIME SERIES

radar_dir = r"W:\lindgrv1\Kiira_whole_day\kiira_14-22_dbz"
file_list = os.listdir(radar_dir)
file_list = [x for x in file_list if ".tif" in x]
file_list = [f for f in file_list if f.endswith(".tif")]

file_list = file_list[7:]

radar_kiira = []
for i in range(len(file_list)):
    src = rasterio.open(os.path.join(radar_dir, file_list[i]))
    array = src.read(1)
    radar_kiira.append(array)
    
radar_kiira = np.concatenate([radar_kiira_[None, :, :] for radar_kiira_ in radar_kiira])

#This have to be added for new event data: Remove last column from each layer
radar_kiira = radar_kiira[:,:,:-1]

#The following data is available for Finnish radar composite: radar reflectivity (dbz), conversion: Z[dBZ] = 0.5 * pixel value - 32
radar_kiira = (radar_kiira * 0.5) - 32

#Values less than threshold to wanted value, which represents no-rain
rain_zero_value = 3.1830486304816077 # no rain pixels assigned with this value
radar_kiira[radar_kiira < 10] = rain_zero_value

#Areal mean rainfall of observed event in dbz
areal_rainfall_ts = np.zeros(len(file_list))
for i in range (len(file_list)):
    areal_rainfall_ts[i] = np.nanmean(radar_kiira[i])
areal_rainfall_ts2 = areal_rainfall_ts.copy()
#Time step of maximum areal mean rainfall
max_idx = np.where(areal_rainfall_ts == areal_rainfall_ts.max())
max_idx = int(max_idx[0])
#Plot the areal rainfall time series
plt.figure()
plt.plot(areal_rainfall_ts)
plt.axvline(x = max_idx, color = "black", linestyle = "-", linewidth=0.5)

#correct all time series
x_obs_17 = [17, 19]
y_mar_obs_17 = [areal_rainfall_ts[17], areal_rainfall_ts[19]]
x_new_17 = 18
y_mar_new_17 = np.interp(x_new_17, x_obs_17, y_mar_obs_17)
areal_rainfall_ts[18] = y_mar_new_17

x_obs_36 = [35, 38]
y_mar_obs_36 = [areal_rainfall_ts[35], areal_rainfall_ts[38]]
x_new_36 = [36, 37]
y_mar_new_36 = np.interp(x_new_36, x_obs_36, y_mar_obs_36)
areal_rainfall_ts[36] = y_mar_new_36[0]
areal_rainfall_ts[37] = y_mar_new_36[1]

x_obs_46 = [46, 51]
y_mar_obs_46 = [areal_rainfall_ts[46], areal_rainfall_ts[51]]
x_new_46 = [47, 48, 49, 50]
y_mar_new_46 = np.interp(x_new_46, x_obs_46, y_mar_obs_46)
areal_rainfall_ts[47] = y_mar_new_46[0]
areal_rainfall_ts[48] = y_mar_new_46[1]
areal_rainfall_ts[49] = y_mar_new_46[2]
areal_rainfall_ts[50] = y_mar_new_46[3]

#Plot the corrected areal rainfall time series
plt.figure()
plt.plot(areal_rainfall_ts)
plt.axvline(x = max_idx, color = "black", linestyle = "-", linewidth=0.5)

r_mean = areal_rainfall_ts

#########
##### MAR
#########

test_array = r_mean.copy()
test_new_steps = []
for i in range(0,len(test_array)-1):
    test_array_int = np.interp(i+0.5, [i, i+1], [test_array[i], test_array[i+1]])
    test_new_steps.append(test_array_int)
test_new_steps = np.concatenate([test_new_steps_[None] for test_new_steps_ in test_new_steps])

test_new_array = np.zeros((len(test_array)+len(test_new_steps), ))
for num in range(len(test_new_array)):
    if (num % 2) == 0:
        test_new_array[num] = test_array[int(num-(num/2))]
    else:
        test_new_array[num] = test_new_steps[int(num-(num/2))]
        
plt.figure()
plt.plot(test_array)
plt.figure()
plt.plot(test_new_array)
    
###############
##### ADVECTION
###############

oflow_method = "LK" #Lukas-Kanade
oflow_advection = pysteps.motion.get_method(oflow_method) 

R = radar_kiira.copy()

#Variables for x- and y-components of the advection
Vx = np.zeros(len(R)-1)
Vy = np.zeros(len(R)-1)

#Loop to calculate average x- and y-components of the advection, as well as magnitude and direction in xy-dir
V_of = []
for i in range(0, len(R)-1):
    V_of.append(oflow_advection(R[i:i+2, :, :]))
    Vx[i] = np.nanmean(V_of[i][0]) #field mean in x-direction
    Vy[i] = np.nanmean(V_of[i][1]) #field mean in y-direction
    
#advection in x-dir
test_array_vx = Vx.copy()
test_new_steps_vx = []
for i in range(0,len(test_array_vx)-1):
    test_array_vx_int = np.interp(i+0.5, [i, i+1], [test_array_vx[i], test_array_vx[i+1]])
    test_new_steps_vx.append(test_array_vx_int)
test_new_steps_vx = np.concatenate([test_new_steps_vx_[None] for test_new_steps_vx_ in test_new_steps_vx])

test_new_array_vx = np.zeros((len(test_array_vx)+len(test_new_steps_vx), ))
for num in range(len(test_new_array_vx)):
    if (num % 2) == 0:
        test_new_array_vx[num] = test_array_vx[int(num-(num/2))]
    else:
        test_new_array_vx[num] = test_new_steps_vx[int(num-(num/2))]
        
plt.figure()
plt.plot(test_array_vx)
plt.figure()
plt.plot(test_new_array_vx)
    
test_new_steps_vx2 = []
for i in range(0,len(test_new_array_vx)-1):
    test_array_vx2_int = np.interp(i+0.5, [i, i+1], [test_new_array_vx[i], test_new_array_vx[i+1]])
    test_new_steps_vx2.append(test_array_vx2_int)
test_new_steps_vx2 = np.concatenate([test_new_steps_vx2_[None] for test_new_steps_vx2_ in test_new_steps_vx2])

test_new_array_vx2 = np.zeros((len(test_new_array_vx)+len(test_new_steps_vx2), ))
for num in range(len(test_new_array_vx2)):
    if (num % 2) == 0:
        test_new_array_vx2[num] = test_new_array_vx[int(num-(num/2))]
    else:
        test_new_array_vx2[num] = test_new_steps_vx2[int(num-(num/2))]
        
plt.figure()
plt.plot(test_array_vx)
plt.figure()
plt.plot(test_new_array_vx)
plt.figure()
plt.plot(test_new_array_vx2)

#advection in y-dir
test_array_vy = Vy.copy()
test_new_steps_vy = []
for i in range(0,len(test_array_vy)-1):
    test_array_vy_int = np.interp(i+0.5, [i, i+1], [test_array_vy[i], test_array_vy[i+1]])
    test_new_steps_vy.append(test_array_vy_int)
test_new_steps_vy = np.concatenate([test_new_steps_vy_[None] for test_new_steps_vy_ in test_new_steps_vy])

test_new_array_vy = np.zeros((len(test_array_vy)+len(test_new_steps_vy), ))
for num in range(len(test_new_array_vy)):
    if (num % 2) == 0:
        test_new_array_vy[num] = test_array_vy[int(num-(num/2))]
    else:
        test_new_array_vy[num] = test_new_steps_vy[int(num-(num/2))]
        
plt.figure()
plt.plot(test_array_vy)
plt.figure()
plt.plot(test_new_array_vy)
    
test_new_steps_vy2 = []
for i in range(0,len(test_new_array_vy)-1):
    test_array_vy2_int = np.interp(i+0.5, [i, i+1], [test_new_array_vy[i], test_new_array_vy[i+1]])
    test_new_steps_vy2.append(test_array_vy2_int)
test_new_steps_vy2 = np.concatenate([test_new_steps_vy2_[None] for test_new_steps_vy2_ in test_new_steps_vy2])

test_new_array_vy2 = np.zeros((len(test_new_array_vy)+len(test_new_steps_vy2), ))
for num in range(len(test_new_array_vy2)):
    if (num % 2) == 0:
        test_new_array_vy2[num] = test_new_array_vy[int(num-(num/2))]
    else:
        test_new_array_vy2[num] = test_new_steps_vy2[int(num-(num/2))]
        
plt.figure()
plt.plot(test_array_vy)
plt.figure()
plt.plot(test_new_array_vy)
plt.figure()
plt.plot(test_new_array_vy2)

#Save longer time series of MAR, Vx, and Vy
ts_long_mar = test_new_array.copy()
ts_long_mar = ts_long_mar[2:]
ts_long_mar = ts_long_mar[:-2]

ts_long_vx = test_new_steps_vx2.copy()
ts_long_vx = ts_long_vx[1:]
ts_long_vx = ts_long_vx[:-1]

ts_long_vy = test_new_steps_vy2.copy()
ts_long_vy = ts_long_vy[1:]
ts_long_vy = ts_long_vy[:-1]

out_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
save_new_ts_csv = False
if save_new_ts_csv:
    data_temp = [ts_long_mar, ts_long_vx, ts_long_vy]
    fft_params = pd.DataFrame(data_temp, index=['mar_long', 'vx_long', 'vy_long'])
    pd.DataFrame(fft_params).to_csv(os.path.join(out_dir, "data_tss_long.csv"))