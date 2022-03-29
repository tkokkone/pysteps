# -*- coding: utf-8 -*-
"""
pysteps_parameter_estimation.py

Scrip to estimate all the parameters for the model, as well as statistics and time series for several variables:
    - Time series:
        - areal mean rainfall
        - wetted area ratio
        - standard deviation
        - advection components, magnitude, and direction
        - fft-parameters: beta1, beta2, and w0 (scale break)

    - Parameters:
        - broken line parameters: q, no_bls, var_tol, and mar_tol
        - broken line parameters for field mean: mu_z, sigma2_z, h_val_z, and a_zero_z
        - broken line parameters for velocity magnitude: mu_vmag, sigma2_vmag, h_val_vmag, and a_zero_vmag 
        - broken line parameters for velocity direction: mu_vdir, sigma2_vdir, h_val_vdir, and a_zero_vdir 
        - ar-parameters for all cascade levels: gamma1, gamma2, Tau_k, and L_k
        - ar-parameters: tlen_a, tlen_b, and tlen_c
        - scale break parameters: a_w0, b_w0, and c_w0
        - beta1 parameters: a_1, b_1, and c_1
        - beta2 parameters: a_2, b_2, and c_2
        - std parameters: a_v, b_v, and c_v
        - war parameters: a_war, b_war, and c_war

Requirements:
    - pysteps
    - numpy
    - pandas
    - matplotlib.pyplot
    - matplotlib.cm
    - time
    - datetime
    - statsmodels.api
    - scipy
    - os

References:
    - Leinonen et al. 2012: A climatology of disdrometer measurements of rainfall in Finland over five years with implications for global radar observations
    - Marshall and Palmer. 1948: The distribution of raindrops with size
    - Seed et al. 2000: A multiplicative broken-line model for time series of mean areal rainfall
    - Seed et al. 2014: Stochastic simulation of space-time rainfall patterns for the Brisbane River catchment

Created on Tue Mar 22 08:07:57 2022

@author: Ville Lindgren
"""

# Events used in the study:
# 1. last radar image: 201306271955 -> number of previous files: 141
# 3. last radar image: 201310290345 -> number of previous files: 115
# 6. last radar image: 201408071800 -> number of previous files: 97
# Events are trimmed to start when areal mean is 5 dbz or higher, and to end when it is lower than 5 dbz.
# //home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/events
# C:/Users/lindgrv1/pysteps-data/radar/osapol

##############################################################################
# IMPORT PACKAGES

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import pysteps
from pysteps import extrapolation
from datetime import datetime
import statsmodels.api as sm
import scipy.stats as sp
import os

##############################################################################
# TIME USED FOR PARAMETER ESTIMATION: Start timer

run_start_0 = time.perf_counter()

##############################################################################
# OUTDIR

out_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/events/Test_event"

##############################################################################
# OPTIONS TO SAVE PLOTS, ANIMATIONS, AND TABLES

plots_mmh = 0 #areal mean R, WAR, and std time series in mm/h
plots_dbz = 0 #areal mean R, WAR, and std time series in dBZ
plot_bl_pars = 0 #broken line parameters for areal mean R and advection (magnitude and direction): a_zero and H
plot_estimated_ar_pars = 0 #linear fits from which ar-parameters tlen_a, tlen_b, and tlen_c are estimated
plot_fit_vs_mean = 0 #polynomial fits against areal mean R: WAR, std, beta1, beta2, and scale break

plot_2dfft = 1 #plot example of 2d fourier spectrum
save_2dfft = 0 #save example of 2d fourier spectrum
plot_bpf_weights = 1 #plot bandpass filter weights
save_bpf_weights = 0 #save example of 2d fourier spectrum
plot_decomp_example = 1 #plot example of cascade decomposition
save_decomp_example = 0 #save example of 2d fourier spectrum

animation_mmh = 0 #show animation of input radar images in mm/h
save_animation_mmh = 0 #save animation of input radar images in mm/h (as png-images)
animation_dbz = 0 #show animation of input radar images in dBZ
save_animation_dbz = 0 #save animation of input radar images in dBZ (as png-images)

csv_input_ts = 0 #areal mean R, WAR, and std time series in mm/h and dBZ
csv_advection_ts = 0 # advection components, magnitude, and direction
csv_fft_par_ts = 0 #fft-parameters: beta1, beta2, and scale break time series
csv_ar_par_ts = 0 #ar-parameters: gamma1, gamma2, Tau_k, and L_k for each cascade level
csv_sim_pars = 0 #all estimated parameters needed for simulations

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

#Values less than threshold to zero
R[R<0.1] = 0

#Plot event as a animation
if animation_mmh == 1:
    plt.figure()
    if save_animation_mmh == 1:
        path_outputs1 = os.path.join(out_dir, "input_pngs")
        if not os.path.exists(path_outputs1):
            os.makedirs(path_outputs1)
        path_outputs2 = os.path.join(path_outputs1, "mmh")
        if not os.path.exists(path_outputs2):
            os.makedirs(path_outputs2)
        pysteps.visualization.animations.animate(R, savefig=True, path_outputs=path_outputs2)
    else:
        pysteps.visualization.animations.animate(R, savefig=False)

##############################################################################
# TIME SERIES OF AREAL MEAN R, STD, AND WAR IN UNIT OF MM/H

#Areal mean rainfall in mm/h
areal_rainfall_mmh = np.zeros(len(fns[1]))
for i in range (0, len(fns[1])):
    areal_rainfall_mmh[i] = np.nanmean(R[i])
plt.figure()
plt.plot(areal_rainfall_mmh)
plt.title("Areal mean rainfall (mm/h)")
if plots_mmh == 1:
    plt.savefig(os.path.join(out_dir, "ts_meanR_mmh.png"))

#Wetted area ratio in mm/h
war_mmh = np.zeros(len(fns[1]))
for i in range (0, len(fns[1])):
    war_mmh[i] = (np.count_nonzero(R[i][R[i] > 0.1])) / np.count_nonzero(~np.isnan(R[i]))
plt.figure()
plt.plot(war_mmh)
plt.title("Wetted area ratio (mm/h)")
if plots_mmh == 1:
    plt.savefig(os.path.join(out_dir, "ts_war_mmh.png"))

#Standard deviation in mm/h
std_mmh = np.zeros(len(fns[1]))
for i in range (0, len(fns[1])):
    std_mmh[i] = np.nanstd(R[i])
plt.figure()
plt.plot(std_mmh)
plt.title("Standard deviation (mm/h)")
if plots_mmh == 1:
    plt.savefig(os.path.join(out_dir, "ts_std_mmh.png"))

##############################################################################
# DATA TRANSFORMATION INTO UNIT OF DBZ

metadata["zerovalue"] = 0
metadata["threshold"] = 0.1

#dBZ transformation for mm/h-data
#Z-R relationship: Z = a*R^b (Reflectivity)
#dBZ conversion: dBZ = 10 * log10(Z) (Decibels of Reflectivity)
#Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with size. J. Meteor., 5, 165–166.
a_R=223
b_R=1.53
#Leinonen et al. 2012 - A Climatology of Disdrometer Measurements of Rainfall in Finland over Five Years with Implications for Global Radar Observations
#https://doi.org/10.1175/JAMC-D-11-056.1

R, metadata = pysteps.utils.conversion.to_reflectivity(R, metadata, zr_a=a_R, zr_b=b_R)

#Plot event as a animation
if animation_dbz == 1:
    plt.figure()
    if save_animation_dbz == 1:
        path_outputs3 = os.path.join(out_dir, "input_pngs")
        if not os.path.exists(path_outputs3):
            os.makedirs(path_outputs3)
        path_outputs4 = os.path.join(path_outputs3, "dbz")
        if not os.path.exists(path_outputs4):
            os.makedirs(path_outputs4)
        pysteps.visualization.animations.animate(R, savefig=True, path_outputs=path_outputs4)
    else:
        pysteps.visualization.animations.animate(R, savefig=False)

##############################################################################
# TIME SERIES OF AREAL MEAN R, STD, AND WAR IN UNIT OF DBZ

#Areal mean rainfall in dbz
areal_rainfall_ts = np.zeros(len(fns[1]))
for i in range (0, len(fns[1])):
    areal_rainfall_ts[i] = np.nanmean(R[i])
plt.figure()
plt.plot(areal_rainfall_ts)
plt.title("Areal mean rainfall (dBZ)")
if plots_dbz == 1:
    plt.savefig(os.path.join(out_dir, "ts_meanR_dbz.png"))

#Wetted area ratio in dbz
war_ts = np.zeros(len(fns[1]))
for i in range (0, len(fns[1])):
    war_ts[i] = (np.count_nonzero(R[i][R[i] > 3.1830486304816077])) / np.count_nonzero(~np.isnan(R[i]))
plt.figure()
plt.plot(war_ts)
plt.title("Wetted area ratio (dBZ)")
if plots_dbz == 1:
    plt.savefig(os.path.join(out_dir, "ts_war_dbz.png"))
    
#Standard deviation in dbz
std_ts = np.zeros(len(fns[1]))
for i in range (0, len(fns[1])):
    std_ts[i] = np.nanstd(R[i])
plt.figure()
plt.plot(std_ts)
plt.title("Standard deviation (dBZ)")
if plots_dbz == 1:
    plt.savefig(os.path.join(out_dir, "ts_std_dbz.png"))

##############################################################################
# CALCULATE ADVECTION TIME SERIES

#Choose the method to calculate advection
oflow_method = "LK" #Lukas-Kanade
oflow_advection = pysteps.motion.get_method(oflow_method) 

#Variables for magnitude and direction of advection
Vx = np.zeros(len(R)-1)
Vy = np.zeros(len(R)-1)
Vxy = np.zeros(len(R)-1)
Vdir_rad = np.zeros(len(R)-1)

#Loop to calculate average x- and y-components of the advection, as well as magnitude and direction in xy-dir
V_of = []
for i in range(0, len(R)-1):
    V_of.append(oflow_advection(R[i:i+2, :, :]))
    #Vtemp[0,:,:] contains the x-components of the motion vectors.
    #Vtemp[1,:,:] contains the y-components of the motion vectors.
    #The velocities are in units of pixels/timestep.
    Vx[i] = np.nanmean(V_of[i][0]) #field mean in x-direction
    Vy[i] = np.nanmean(V_of[i][1]) #field mean in y-direction
    Vxy[i] = np.sqrt(Vx[i]**2 + Vy[i]**2) #total magnitude of advection
    Vdir_rad[i] = np.arctan(Vy[i]/Vx[i]) #direction of advection in xy-dir in radians
    #The inverse of tan, so that if y = tan(x) then x = arctan(y).
    #Range of values of the function: −pi/2 < y < pi/2 (-1.5707963267948966 < y < 1.5707963267948966) -> np.pi gives value of pi
Vdir_deg = (Vdir_rad/(2*np.pi))*360 #direction of advection in xy-dir in degrees

#Change of direction
Vdir_change = np.zeros(len(Vdir_deg))
for i in range(1, len(Vdir_deg)):
    Vdir_change[i] = Vdir_deg[i]-Vdir_deg[i-1]

for i in range(1, len(Vdir_change)):
    if Vdir_change[i] < -180:
        Vdir_change[i] = Vdir_change[i] + 360
    elif Vdir_change[i] > 180:
        Vdir_change[i] = Vdir_change[i] - 360
    
#True direction in degs
Vdir_deg_adj = (Vdir_rad/(2*np.pi))*360
for j in range(0, len(R)-1):
    if Vx[j]<0 and Vy[j]>0:
        Vdir_deg_adj[j] = Vdir_deg_adj[j]+180
    elif Vx[j]>0 and Vy[j]<0:
        Vdir_deg_adj[j] = Vdir_deg_adj[j]+360
    elif Vx[j]<0 and Vy[j]<0:
        Vdir_deg_adj[j] = Vdir_deg_adj[j]+180
    else:
        Vdir_deg_adj[j] = Vdir_deg_adj[j]

#True change of direction  
Vdir_change2 = np.zeros(len(Vdir_deg_adj))
for i in range(1, len(Vdir_deg_adj)):
    Vdir_change2[i] = Vdir_deg_adj[i]-Vdir_deg_adj[i-1]

for i in range(1, len(Vdir_change2)):
    if Vdir_change2[i] < -180:
        Vdir_change2[i] = Vdir_change2[i] + 360
    elif Vdir_change2[i] > 180:
        Vdir_change2[i] = Vdir_change2[i] - 360
        
##############################################################################
# GENERAL BROKEN LINE PARAMETERS

q = 0.8
noBLs = 1
var_tol = 1
mar_tol = 1

tStep = float(5) #timestep [min]
tSerieLength = (len(R) - 1) * tStep #timeseries length [min]
x_values = np.arange(0, tSerieLength + tStep, tStep)
euler = 2.71828

##############################################################################
# BROKEN LINE PARAMETERS FOR AREAL MEAN RAINFALL

#Mean and variance of input time series
mu_z = float(np.mean(areal_rainfall_ts)) #mean of input time series [dBz]
sigma2_z = float(np.var(areal_rainfall_ts)) #variance of input time series [dBz]

#Correlations and a_zero of time series
event_cors = sm.tsa.acf(areal_rainfall_ts, nlags=(len(areal_rainfall_ts) - 1))
plt.figure()
plt.plot(event_cors)
plt.title("Autocorrelations: mean R")
if plot_bl_pars == 1:
    plt.savefig(os.path.join(out_dir, "auto_cors_meanR.png"))

#Timesteps for no autocorrelation
no_autocor = np.where(event_cors < 1/euler)
a_zero = (no_autocor[0][0]-1)*tStep #timesteps in which correlation values are > 1/e = 0.36787968862663156

#Power spectrum, beta, and H of time series
power_spectrum = np.abs(np.fft.fft(areal_rainfall_ts))**2 #absolute value of the complex components of fourier transform of data
freqs = np.fft.fftfreq(areal_rainfall_ts.size, tStep) #frequencies

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
if plot_bl_pars == 1:
    plt.savefig(os.path.join(out_dir, "freq_vs_ps_meanR.png"))

H = (-slope-1)/2

##############################################################################
# BROKEN LINE PARAMETERS FOR ADVECTION MAGNITUDE

#Mean and variance of input time series
mu_z_Vxy = float(np.mean(Vxy)) #mean of input time series
sigma2_z_Vxy = float(np.var(Vxy)) #variance of input time series

#Correlations and a_zero of time series
event_cors_Vxy = sm.tsa.acf(Vxy, nlags=(len(Vxy) - 1))
plt.figure()
plt.plot(event_cors_Vxy)
plt.title("Autocorrelations: advection mag")
if plot_bl_pars == 1:
    plt.savefig(os.path.join(out_dir, "auto_cors_adv-mag.png"))

#Timesteps for no autocorrelation
no_autocor_Vxy = np.where(event_cors_Vxy < 1/euler)
a_zero_Vxy = (no_autocor_Vxy[0][0]-1)*tStep #time steps in which correlation values are > 1/e = 0.36787968862663156

#Power spectrum, beta, and H of time series
power_spectrum_Vxy = np.abs(np.fft.fft(Vxy))**2 #absolute value of the complex components of fourier transform of data
freqs_Vxy = np.fft.fftfreq(Vxy.size, tStep) #frequencies

freqs_Vxy_sliced = freqs_Vxy[1:]
freqs_Vxy_sliced = freqs_Vxy_sliced[freqs_Vxy_sliced>0]
power_spectrum_Vxy_sliced = power_spectrum_Vxy[1:]
power_spectrum_Vxy_sliced = power_spectrum_Vxy_sliced[0:len(freqs_Vxy_sliced)]

#Calculate power spectrum exponent beta and H from beta using linear fitting of the data
slope_Vxy, intercept_Vxy, r_value_Vxy, p_value_Vxy, std_err_Vxy = sp.linregress(np.log(freqs_Vxy_sliced[0:len(freqs_Vxy_sliced)]), np.log(power_spectrum_Vxy_sliced[0:len(freqs_Vxy_sliced)]))
y_values_Vxy = slope_Vxy*np.log(freqs_Vxy_sliced[0:len(freqs_Vxy_sliced)])+intercept_Vxy

plt.figure()
plt.plot(np.log(freqs_Vxy_sliced), np.log(power_spectrum_Vxy_sliced))
plt.plot(np.log(freqs_Vxy_sliced[0:len(freqs_Vxy_sliced)]), y_values_Vxy)
plt.title("Freqs vs. power spectrum -fit: advection mag")
if plot_bl_pars == 1:
    plt.savefig(os.path.join(out_dir, "freq_vs_ps_adv-mag.png"))

H_Vxy = (-slope_Vxy-1)/2

##############################################################################
# BROKEN LINE PARAMETERS FOR ADVECTION DIRECTION

#Mean and variance of input time series
mu_z_Vdir_deg_adj = float(np.mean(Vdir_deg_adj)) #mean of input time series
sigma2_z_Vdir_deg_adj = float(np.var(Vdir_deg_adj)) #variance of input time series

#Correlations and a_zero of time series
event_cors_Vdir_deg_adj = sm.tsa.acf(Vdir_deg_adj, nlags=(len(Vdir_deg_adj) - 1))
plt.figure()
plt.plot(event_cors_Vdir_deg_adj)
plt.title("Autocorrelations: advection dir")
if plot_bl_pars == 1:
    plt.savefig(os.path.join(out_dir, "auto_cors_adv-dir.png"))

#Timesteps for no autocorrelation
no_autocor_Vdir_deg_adj = np.where(event_cors_Vdir_deg_adj < 1/euler)
a_zero_Vdir_deg_adj = (no_autocor_Vdir_deg_adj[0][0]-1)*tStep #time steps in which correlation values are > 1/e = 0.36787968862663156

#Power spectrum, beta, and H of time series
power_spectrum_Vdir_deg_adj = np.abs(np.fft.fft(Vdir_deg_adj))**2 #absolute value of the complex components of fourier transform of data
freqs_Vdir_deg_adj = np.fft.fftfreq(Vdir_deg_adj.size, tStep) #frequencies

freqs_Vdir_deg_adj_sliced = freqs_Vdir_deg_adj[1:]
freqs_Vdir_deg_adj_sliced = freqs_Vdir_deg_adj_sliced[freqs_Vdir_deg_adj_sliced>0]
power_spectrum_Vdir_deg_adj_sliced = power_spectrum_Vdir_deg_adj[1:]
power_spectrum_Vdir_deg_adj_sliced = power_spectrum_Vdir_deg_adj_sliced[0:len(freqs_Vdir_deg_adj_sliced)]

#Calculate power spectrum exponent beta and H from beta using linear fitting of the data
slope_Vdir_deg_adj, intercept_Vdir_deg_adj, r_value_Vdir_deg_adj, p_value_Vdir_deg_adj, std_err_Vdir_deg_adj = sp.linregress(np.log(freqs_Vdir_deg_adj_sliced[0:len(freqs_Vdir_deg_adj_sliced)]), np.log(power_spectrum_Vdir_deg_adj_sliced[0:len(freqs_Vdir_deg_adj_sliced)]))
y_values_Vdir_deg_adj = slope_Vdir_deg_adj*np.log(freqs_Vdir_deg_adj_sliced[0:len(freqs_Vdir_deg_adj_sliced)])+intercept_Vdir_deg_adj

plt.figure()
plt.plot(np.log(freqs_Vdir_deg_adj_sliced), np.log(power_spectrum_Vdir_deg_adj_sliced))
plt.plot(np.log(freqs_Vdir_deg_adj_sliced[0:len(freqs_Vdir_deg_adj_sliced)]), y_values_Vdir_deg_adj)
plt.title("Freqs vs. power spectrum -fit: advection dir")
if plot_bl_pars == 1:
    plt.savefig(os.path.join(out_dir, "freq_vs_ps_adv-dir.png"))

H_Vdir_deg_adj = (-slope_Vdir_deg_adj-1)/2

##############################################################################
# ESTIMATE FFT-PARAMETERS

#Replace non-finite values with the minimum value
R2 = R.copy()
for i in range(R2.shape[0]):
    R2[i, ~np.isfinite(R[i, :])] = np.nanmin(R2[i, :])

#https://pysteps.readthedocs.io/en/latest/auto_examples/plot_noise_generators.html
#Fit the parametric PSD to the observation

beta1s = np.zeros(len(R2))
beta2s = np.zeros(len(R2))
w0s = np.zeros(len(R2))

scale_break = 18  # scale break in km
scale_break_wn = np.log(max(R2.shape[1],R2.shape[2])/scale_break)
for i in range(0, len(R2)):
    Fp = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(R2[i], scale_break=scale_break_wn)

    #Compute the observed and fitted 1D PSD
    L = np.max(Fp["input_shape"])
    if L % 2 == 0:
        wn = np.arange(0, int(L / 2) + 1)
    else:
        wn = np.arange(0, int(L / 2))
    iR_, freq = pysteps.utils.rapsd(R2[i], fft_method=np.fft, return_freq=True)
    f = np.exp(Fp["model"](np.log(wn), *Fp["pars"]))

    # Extract the scaling break in km, beta1 and beta2
    w0s[i] = 18
    beta1s[i] = Fp["pars"][1]
    beta2s[i] = Fp["pars"][2]
    
##############################################################################  
# PLOT EXAMPLE OF 2D FOURIER SPRECTRUM

if plot_2dfft == 1:
    # Compute the Fourier transform of the input field
    F80 = abs(np.fft.fftshift(np.fft.fft2(R2[80])))
    
    # Plot the power spectrum
    M, N = F80.shape
    fig, ax = plt.subplots()
    im = ax.imshow(
        np.log(F80**2), vmin=4, vmax=24, cmap=cm.jet, extent=(-N / 2, N / 2, -M / 2, M / 2)
    )
    cb = fig.colorbar(im)
    ax.set_xlabel("Wavenumber $k_x$")
    ax.set_ylabel("Wavenumber $k_y$")
    ax.set_title("Log-power spectrum of R")
    plt.show()
    if save_2dfft == 1:
        plt.savefig(os.path.join(out_dir, "example_2dfft.png"))

##############################################################################
# CREATE GAUSSIAN BANDPASS FILTER

#Number of cascade levels
n_cascade_levels = 7

#Bandpass filter
bp_filter = pysteps.cascade.bandpass_filters.filter_gaussian(R2[0].shape, n_cascade_levels)

#Spatial scale in each cascade level in km
#https://gmd.copernicus.org/articles/12/4185/2019/, figure1
L_k = np.zeros([n_cascade_levels, 1])
L_k[0] = 264
for i in range(1, len(L_k)):
    L_k[i] = (L_k[0]/2)/(bp_filter["central_wavenumbers"][i])

##############################################################################
# PLOT BANDPASS FILTER WEIGHTS

if plot_bpf_weights == 1:
    # Plot the bandpass filter weights
    L = np.max(Fp["input_shape"])
    fig, ax = plt.subplots()
    for k in range(n_cascade_levels):
        ax.semilogx(
            np.linspace(0, L / 2, len(bp_filter["weights_1d"][k, :])),
            bp_filter["weights_1d"][k, :],
            "k-",
            base=pow(0.5 * L / 3, 1.0 / (n_cascade_levels - 2)),
        )
    ax.set_xlim(1, L / 2)
    ax.set_ylim(0, 1)
    xt = np.hstack([[1.0], bp_filter["central_wavenumbers"][1:]])
    ax.set_xticks(xt)
    ax.set_xticklabels(["%.2f" % cf for cf in bp_filter["central_wavenumbers"]])
    ax.set_xlabel("Radial wavenumber $|\mathbf{k}|$")
    ax.set_ylabel("Normalized weight")
    ax.set_title("Bandpass filter weights")
    plt.show()
    if save_bpf_weights == 1:
        plt.savefig(os.path.join(out_dir, "example_bpf_weights.png"))

##############################################################################
# ESTIMATE AR(2)-MODEL PARAMETERS

#The Lagrangian temporal evolution of each normalised cascade level is modelled using an AR(2)-model
ar_order = 2

#Create variables that include all gammas and phis of all cascade levels for each time step
gamma_all = np.zeros([n_cascade_levels*2, len(R2)-2])
phi_all = np.zeros([n_cascade_levels*3, len(R2)-2])


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
R_prev_adv2 = []
Ylläoleviin listoihin pari alkiota alkuun, niin löytyvät indesksistä samalta kohtaa...
for i in range(2,len(R2)-1): 
    
    R_prev_adv[i] = extrapolator_method(R2[i-1], V_of[i-1], 1, "min", **extrap_kwargs)[-1]
    R_prev2_tmp = extrapolator_method(R2[i-2], V_of[i-2], 1, "min", **extrap_kwargs)[-1]
    R_prev2_adv[i] = extrapolator_method(R_prev2_tmp, V_of[i-1], 1, "min", **extrap_kwargs)[-1]
    
    
    R_d = []
    
    for j in range(i,i+ar_order+1):
        R_ = pysteps.cascade.decomposition.decomposition_fft(R2[j, :, :], bp_filter, normalize=True, compute_stats=True)
        R_d.append(R_)
        
    #Rearrange the cascade levels into a four-dimensional array of shape
    #(n_cascade_levels,ar_order+1,m,n) for the autoregressive model
    R_c = pysteps.nowcasts.utils.stack_cascades(R_d, n_cascade_levels, convert_to_full_arrays=True)
    
    #Compute lag-l temporal autocorrelation coefficients for each cascade level
    gamma = np.empty((n_cascade_levels, ar_order))
    MASK_thr = None
    for k in range(n_cascade_levels):
        gamma[k, :] = pysteps.timeseries.correlation.temporal_autocorrelation(R_c[k], domain="spatial", mask=MASK_thr)
    
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

if plot_decomp_example == 1:
    decomp = pysteps.cascade.decomposition.decomposition_fft(R2[80], bp_filter, compute_stats=True)
    # Plot the normalized cascade levels
    for i in range(n_cascade_levels):
        mu = decomp["means"][i]
        sigma = decomp["stds"][i]
        decomp["cascade_levels"][i] = (decomp["cascade_levels"][i] - mu) / sigma
    
    fig, ax = plt.subplots(nrows=2, ncols=4)
    
    ax[0, 0].imshow(R2[80], cmap=cm.RdBu_r, vmin=-5, vmax=5)
    ax[0, 1].imshow(decomp["cascade_levels"][0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[0, 2].imshow(decomp["cascade_levels"][1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[0, 3].imshow(decomp["cascade_levels"][2], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[1, 0].imshow(decomp["cascade_levels"][3], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[1, 1].imshow(decomp["cascade_levels"][4], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[1, 2].imshow(decomp["cascade_levels"][5], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[1, 3].imshow(decomp["cascade_levels"][6], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    
    ax[0, 0].set_title("Observed")
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
    if save_decomp_example == 1:
        plt.savefig(os.path.join(out_dir, "example_decomp.png"))

##############################################################################
# CALCULATE GAMMA1, GAMMA2, AND TAU_K

#Mean of gamma1 for each cascade level
gamma1_mean = np.zeros([n_cascade_levels, 1])
for i in range(0, len(gamma1_mean)):
    gamma1_mean[i] = np.mean(abs(gamma_all[i,:]))

#Mean of gamma2 for each cascade level
gamma2_mean = np.zeros([n_cascade_levels, 1])
for i in range(len(gamma2_mean), len(gamma_all)):
    gamma2_mean[i-len(gamma1_mean)] = np.mean(abs(gamma_all[i,:]))

#Lagrangian temporal autocorrelation
tau_k = np.zeros([n_cascade_levels, 1])
for i in range(0, len(tau_k)):
    tau_k[i] = -timestep/np.log(gamma1_mean[i])

##############################################################################
# ESTIMATING AR-PARAMETERS: tlen_a, tlen_b, and tlen_c

#Estimate tlen_a and tlen_b based on linear fit of tau_k and L_k
slope_tlen, intercept_tlen, r_value_tlen, p_value_tlen, std_err_tlen = sp.linregress(np.log(L_k[0:n_cascade_levels-1,0]), np.log(tau_k[0:n_cascade_levels-1,0]))
y_values_tlen = slope_tlen*np.log(L_k[0:n_cascade_levels-1])+intercept_tlen

plt.figure()
plt.plot(np.log(L_k[0:n_cascade_levels,0]), np.log(tau_k[0:n_cascade_levels,0]), marker='s')
plt.plot(np.log(L_k[0:n_cascade_levels-1]), y_values_tlen)
plt.title("L_k vs. Tau_k -fit - AR-pars: tlen_a and tlen_b")
if plot_estimated_ar_pars == 1:
    plt.savefig(os.path.join(out_dir, "ar-par_Lk_vs_Tauk.png"))

tlen_b = slope_tlen
tlen_a = np.exp(intercept_tlen)

#Estimate tlen_c based on linear fit of gamma1 and gamma2
slope_tlen_c, intercept_tlen_c, r_value_tlen_c, p_value_tlen_c, std_err_tlen_c = sp.linregress(np.log(gamma1_mean[0:n_cascade_levels-1,0]), np.log(gamma2_mean[0:n_cascade_levels-1,0]))
y_values_tlen_c = slope_tlen_c*np.log(gamma1_mean[0:n_cascade_levels-1,0])+intercept_tlen_c

plt.figure()
plt.plot(np.log(gamma1_mean[0:n_cascade_levels,0]), np.log(gamma2_mean[0:n_cascade_levels,0]), marker='s')
plt.plot(np.log(gamma1_mean[0:n_cascade_levels-1]), y_values_tlen_c)
plt.title("Gamma1 vs. Gamma2 -fit - AR-pars: tlen_c")
if plot_estimated_ar_pars == 1:
    plt.savefig(os.path.join(out_dir, "ar-par_Gamma1_vs_Gamma2.png"))

tlen_c = slope_tlen_c

##############################################################################
# FIT VARIABLES WITH FIELD MEAN

#polynomial fit with degree = 2
std_fit = np.poly1d(np.polyfit(areal_rainfall_ts, std_ts, 2))
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, std_ts)
plt.plot(polyline, std_fit(polyline))
plt.title("mean vs. std")
if plot_fit_vs_mean == 1:
    plt.savefig(os.path.join(out_dir, "fit_mean_vs_std.png"))
print(std_fit)

#polynomial fit with degree = 2
war_fit = np.poly1d(np.polyfit(areal_rainfall_ts, war_ts, 2))
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, war_ts)
plt.plot(polyline, war_fit(polyline))
plt.title("mean vs. war")
if plot_fit_vs_mean == 1:
    plt.savefig(os.path.join(out_dir, "fit_mean_vs_war.png"))
print(war_fit)

#polynomial fit with degree = 2
beta1_fit = np.poly1d(np.polyfit(areal_rainfall_ts, beta1s, 2))
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, beta1s)
plt.plot(polyline, beta1_fit(polyline))
plt.title("mean vs. beta1")
if plot_fit_vs_mean == 1:
    plt.savefig(os.path.join(out_dir, "fit_mean_vs_beta1.png"))
print(beta1_fit)

#polynomial fit with degree = 2
beta2_fit = np.poly1d(np.polyfit(areal_rainfall_ts, beta2s, 2))
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, beta2s)
plt.plot(polyline, beta2_fit(polyline))
plt.title("mean vs. beta2")
if plot_fit_vs_mean == 1:
    plt.savefig(os.path.join(out_dir, "fit_mean_vs_beta2.png"))
print(beta2_fit)

#polynomial fit with degree = 2
w0_fit = np.poly1d(np.polyfit(areal_rainfall_ts, w0s, 2))
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, w0s)
plt.plot(polyline, w0_fit(polyline))
plt.title("mean vs. w0")
if plot_fit_vs_mean == 1:
    plt.savefig(os.path.join(out_dir, "fit_mean_vs_w0.png"))
print(w0_fit)

##############################################################################
# SAVE PARAMETERS AS CSV-FILES

if csv_input_ts == 1:
    #Time series from input data 
    data_temp = [areal_rainfall_ts, std_ts, war_ts, areal_rainfall_mmh, std_mmh, war_mmh]
    fft_params = pd.DataFrame(data_temp, index=['mean', 'std', 'war', 'mean_mmh', 'std_mmh', 'war_mmh'])
    pd.DataFrame(fft_params).to_csv(os.path.join(out_dir, "data_tss.csv"))
    
if csv_advection_ts == 1:
    #Advection time series
    data_temp = [Vx, Vy, Vxy, Vdir_rad, Vdir_deg, Vdir_deg_adj]
    fft_params = pd.DataFrame(data_temp, index=['Vx', 'Vy', 'Vmag', 'Vdir_rad', 'Vdir_deg', 'Vdir_deg_true'])
    pd.DataFrame(fft_params).to_csv(os.path.join(out_dir, "advection_tss.csv"))

if csv_fft_par_ts == 1:
    #FFT-parameter time series
    data_temp = [beta1s, beta2s, w0s]
    fft_params = pd.DataFrame(data_temp, index=['beta1', 'beta2', 'w0'])
    pd.DataFrame(fft_params).to_csv(os.path.join(out_dir, "fft_params.csv"))
    
if csv_ar_par_ts == 1:
    #AR-parameter time series
    ar_params1 = pd.DataFrame(gamma1_mean)
    ar_params1[1] = gamma2_mean
    ar_params1[2] = tau_k
    ar_params1[3] = L_k
    ar_params1.rename(columns = {0 : 'gamma1_mean', 1 : 'gamma2_mean', 2 : 'tau_k', 3 : 'L_k'}, inplace = True)
    pd.DataFrame(ar_params1).to_csv(os.path.join(out_dir, "ar_params.csv"))
    
if csv_sim_pars == 1:
    #Estimatted simulation parameters
    data_temp = [w0_fit[0], w0_fit[1], w0_fit[2], beta1_fit[0], beta1_fit[1], beta1_fit[2], beta2_fit[0], beta2_fit[1], beta2_fit[2], 
                  tlen_a, tlen_b, tlen_c, std_fit[0], std_fit[1], std_fit[2], war_fit[0], war_fit[1], war_fit[2], 
                  mu_z, sigma2_z, H, a_zero, mu_z_Vxy, sigma2_z_Vxy, H_Vxy, a_zero_Vxy, mu_z_Vdir_deg_adj, sigma2_z_Vdir_deg_adj, H_Vdir_deg_adj, a_zero_Vdir_deg_adj, 
                  q, noBLs, var_tol, mar_tol]
    
    all_params = pd.DataFrame(data_temp, index=['a_w0', 'b_w0', 'c_w0', 'a_1', 'b_1', 'c_1', 'a_2', 'b_2', 'c_2', 'tlen_a', 'tlen_b', 'tlen_c', 'a_v', 'b_v', 'c_v', 'a_war', 'b_war', 'c_war', 
                                                'mu_z', 'sigma2_z', 'h_val_z', 'a_zero_z', 'mu_vmag', 'sigma2_vmag', 'h_val_vmag', 'a_zero_vmag', 'mu_vdir', 'sigma2_vdir', 'h_val_vdir', 'a_zero_vdir', 
                                                'q', 'no_bls', 'var_tol', 'mar_tol'])
    pd.DataFrame(all_params).to_csv(os.path.join(out_dir, "all_params.csv"))

##############################################################################
# TIME USED FOR PARAMETER ESTIMATION: End timer

run_end_0 = time.perf_counter()
run_dur_0 = run_end_0 - run_start_0
print(run_dur_0 / 60)
