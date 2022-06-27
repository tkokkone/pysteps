# -*- coding: utf-8 -*-
"""
pysteps_event_simulation.py

Scrip to simulate a stochastic rainfall event using estimated parameters.

Outputs:
    - Simulated event as pngs
    - Time series:
        - areal mean rainfall
        - wetted area ratio
        - standard deviation

Requirements:
    - time
    - os
    - numpy
    - numpy.genfromtxt
    - math
    - random
    - matplotlib.pyplot
    - pandas
    - pysteps
    - gdal
    - rasterio
    - rasterio.transform.from_origin

References:
    - Leinonen et al. 2012 - A Climatology of Disdrometer Measurements of Rainfall in Finland over Five Years with Implications for Global Radar Observations
    - Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with size
    - Seed et al. 2014: Stochastic simulation of space-time rainfall patterns for the Brisbane River catchment

Created on Tue Mar 22 08:11:05 2022

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

import time
import os
from numpy import genfromtxt
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import pysteps
from pysteps.utils import rapsd
from pysteps.visualization import plot_spectrum1d
from pysteps import nowcasts, noise, utils
from pysteps.utils import conversion
from pysteps.utils.dimension import clip_domain
from pysteps.postprocessing.probmatching import set_stats
from pysteps import extrapolation
from osgeo import gdal
import rasterio
from rasterio.transform import from_origin
from fiona.crs import from_epsg

##############################################################################
# TIME USED FOR PARAMETER ESTIMATION: Start timer

run_start_0 = time.perf_counter()

##############################################################################
# INDIR

in_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simuloinnit/Event_6"
    
##############################################################################
# OPTIONS TO SAVE PLOTS, ANIMATIONS, AND TABLES

plot_simulated_beta1 = 0
plot_simulated_beta2 = 0
plot_simulated_w0 = 0
plot_stat_correction = 0
plot_simulated_vs_osapol = 0

plot_1dps = 0 #plot example of 1d power spectrum
save_1dps = 0 #save example of 1d power spectrum

animation_dbz = 0 #show animation of input radar images in dBZ
save_animation_dbz = 0 #save animation of input radar images in dBZ (as png-images)

animation_blues = 0 #show animation of input radar images in dBZ with Blues-colormap
save_animation_blues = 0 #save animation of input radar images in dBZ with Blues-colormap (as png-images)

animation_mmh = 0 #show animation of input radar images in mm/h
save_animation_mmh = 0 #save animation of input radar images in mm/h (as png-images)

csv_sim_fft_par_ts = 0 #simulated fft-parameters: beta1, beta2, and scale break time series

save_tiff = 0 #save simulated event in tif format

csv_arrays_to_save = 0 #save broken line timeseries

##############################################################################
# INPUT DATA

#Read in estimated parameters
all_params = genfromtxt(fname=os.path.join(in_dir, "all_params.csv"), delimiter=',', skip_header=1, usecols=1)

#Read in time fft parameters
fft_params = genfromtxt(fname=os.path.join(in_dir, "fft_params.csv"), delimiter=',', skip_header=1)
fft_params = np.delete(fft_params, 0, axis=1)

#Read in time series from input data
data_tss = genfromtxt(fname=os.path.join(in_dir, "data_tss.csv"), delimiter=',', skip_header=1)
data_tss = np.delete(data_tss, 0, axis=1)

advection_tss = genfromtxt(fname=os.path.join(in_dir, "advection_tss.csv"), delimiter=',', skip_header=1)
advection_tss = np.delete(advection_tss, 0, axis=1)
#'Vx', 'Vy', 'Vmag', 'Vdir_rad', 'Vdir_deg', 'Vdir_deg_true'

##############################################################################
# SIMULATION PARAMETERS AND VARIABLES

#Initialisation of variables
stats_kwargs = dict()
metadata = dict()

#Set general simulation parameters
n_cascade_levels = 7 #number of cascade levels
ar_order = 2 #order of ar-model
n_timesteps = 97 #number of timesteps
timestep = 5 #timestep length
seed1 = random.randint(1, 10000)#456#1234 #random.randint(1, 10000) #seed number for generation of the first precipitation field
seed2 = random.randint(1, 10000)#567#2345 #random.randint(1, 10000) #seed number for generation of the first innovation field
seed_bl = random.randint(1, 10000)#1234 #random.randint(1, 10000) #seed for generating broken lines
seed_random = random.randint(1, 10000)#678#4321 #random.randint(1, 10000) #seed for generating random fields

nx_field = 512 #number of columns in precip fields
ny_field = 512 #number of rows in precip fields
kmperpixel = 1.0 #grid resolution
domain = "spatial" #spatial or spectral
metadata["x1"] = 0.0 #x-coordinate of lower left
metadata["y1"] = 0.0 #y-coordinate of lower left
metadata["x2"] = nx_field * kmperpixel #x-coordinate of upper right
metadata["y2"] = ny_field * kmperpixel #y-coordinate of upper right
metadata["xpixelsize"] = kmperpixel #grid resolution
metadata["ypixelsize"] = kmperpixel #grid resolution
metadata["zerovalue"] = 0.0
metadata["yorigin"] = "lower" #location of grid origin (y), lower or upper

#bounding box coordinates for the extracted middel part of the entire domain
extent = [0, 0, 0, 0]
extent[0] = nx_field / 4 * kmperpixel
extent[1] = 3 * nx_field / 4 * kmperpixel
extent[2] = ny_field / 4 * kmperpixel
extent[3] = 3 * ny_field / 4 * kmperpixel

# scale_ratio = (2.0 / max(nx_field,ny_field)) ** (1.0/(n_cascade_levels-1))
# while scale_ratio < 0.42:
#     n_cascade_levels += 1
#     scale_ratio = (2.0 / max(nx_field,ny_field)) ** (1.0/(n_cascade_levels-1))

noise_method = "parametric_sim"
fft_method = "numpy"
extrap_method = "semilagrangian_wrap" #set method for advection
simulation_method = "steps_sim"

scale_break = 18  # scale break in km
scale_break_wn = np.log(nx_field/scale_break)

#scale break parameters a_w0, b_w0, and c_w0 from fitted polynomial line
a_w0 = all_params[0]
b_w0 = all_params[1]
c_w0 = all_params[2]
#beta1 parameters a_1, b_1, and c_1 from fitted polynomial line
a_1 = all_params[3]  # a_1...c_2 see Seed et al. 2014
b_1 = all_params[4]
c_1 = all_params[5]
#beta2 parameters a_2, b_2, and c_2 from fitted polynomial line
a_2 = all_params[6]
b_2 = all_params[7]
c_2 = all_params[8]

# a_w0 = 46.2277778840522
# b_w0 = -6.05904086447522
# c_w0 = 0.308206831707891
# a_1 = 8.33786814
# b_1 = -0.357822145
# c_1 = 0.024345381
# a_2 = -0.20895615
# b_2 = -0.37098595
# c_2 = 0.019951611

#initialization of power law filter parameter array
p_pow = np.array([scale_break_wn, 0.0, -1.5, -3.5])

#Initialization of ar-parameter array, Seed et al. 2014 eqs. 9-11
tlen_a = all_params[9] #lower tlen_a makes result more random, higher tlen_a makes evolution make more sense
tlen_b = all_params[10] #lower tlen_b makes result more random, higher tlen_b makes evolution make more sense
tlen_c = all_params[11] #changing this doesnt change the result much 

# tlen_a = 3.04351624305851
# tlen_b = 1.43679592839916
# tlen_c = 2.16748333115863

ar_par = np.array([tlen_a, tlen_b, tlen_c])  #order: at, bt, ct

#Set std and WAR parameters, Seed et al. eq. 4
a_v = all_params[12]
b_v = all_params[13]
c_v = all_params[14]
a_war = all_params[15]
b_war = all_params[16]
c_war = all_params[17]

#Broken line parameters for field mean
mu_z = all_params[18] #mean of mean areal reflectivity over the simulation period
sigma2_z = all_params[19] #variance of mean areal reflectivity over the simulation period
h_val_z = all_params[20] #structure function exponent
q_val_z = all_params[30] #scale ratio between levels n and n+1 (constant) [-]
a_zero_z = all_params[21] #time series decorrelation time [min]
no_bls = 1 #number of broken lines
var_tol_z = all_params[32] #acceptable tolerance for variance as ratio of input variance [-]
mar_tol_z = all_params[33] #acceptable value for first and last elements of the final broken line as ratio of input mean -> maybe should be less than 1 -> 0.2

#Broken line parameters for velocity magnitude
mu_vmag = all_params[22] #mean of mean areal reflectivity over the simulation period
sigma2_vmag = all_params[23] #variance of mean areal reflectivity over the simulation period
h_val_vmag = all_params[24]  #structure function exponent
q_val_vmag = all_params[30] #scale ratio between levels n and n+1 (constant) [-]
a_zero_vmag = all_params[25]  #time series decorrelation time [min]
no_bls = 1 #number of broken lines
var_tol_vmag = all_params[32] #acceptable tolerance for variance as ratio of input variance [-]
mar_tol_vmag = all_params[33] #acceptable value for first and last elements of the final broken line as ratio of input mean

#Broken line parameters for velocity direction
mu_vdir = all_params[26] #mean of mean areal reflectivity over the simulation period
sigma2_vdir = all_params[27] #variance of mean areal reflectivity over the simulation period
h_val_vdir = all_params[28]  #structure function exponent
q_val_vdir = all_params[30] #scale ratio between levels n and n+1 (constant) [-]
a_zero_vdir = all_params[29]  #time series decorrelation time [min]
no_bls = 1 #number of broken lines
var_tol_vdir = all_params[32] #acceptable tolerance for variance as ratio of input variance [-]
mar_tol_vdir = all_params[33] #acceptable value for first and last elements of the final broken line as ratio of input mean:

#Z-R relationship: Z = a*R^b (Reflectivity)
#dBZ conversion: dBZ = 10 * log10(Z) (Decibels of Reflectivity)
#Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with size. J. Meteor., 5, 165–166.
a_R=223
b_R=1.53
#Leinonen et al. 2012 - A Climatology of Disdrometer Measurements of Rainfall in Finland over Five Years with Implications for Global Radar Observations

##############################################################################
# OUTDIR

out_dir = os.path.join(in_dir, "Simulations")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

out_dir = os.path.join(out_dir, f"Simulation_{seed1}_{seed2}_{seed_bl}_{seed_random}")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

##############################################################################
# FUNCTION TO CREATE BROKEN LINES

def create_broken_lines(mu_z, sigma2_z, H, q, a_zero, tStep, tSerieLength, noBLs, var_tol, mar_tol):
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
    var_max = sigma2_z + (var_tol * sigma2_z)  # acceptable maximum of variance
    mar_max = mar_tol * mu_z  # acceptable maximum of mean areal rainfall
    # mu_z - (mar_tol * mu_z) ???
    
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

        if (var_min < var_bline_sum_exp < var_max and bline_sum_exp[0] < mar_max and bline_sum_exp[-1] < mar_max):

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

# # TEEMU: Muutettu autokorrealatiokertoimien laskenta käyttäen parametreja
# # a-c (Seed at al., 2014, kaavat 9-11). Parametrit annetaan argumentteina
# # forecast -funktioon.
# # compute lag-l temporal autocorrelation coefficients for each cascade level
# GAMMA = np.empty((n_cascade_levels, ar_order))
# L_k = max(nx_field,ny_field) * kmperpixel
# Lk_all = np.zeros([n_cascade_levels, 1])
# tauk_all = np.zeros([n_cascade_levels, 1])
# for i in range(n_cascade_levels):
#     tau_k = ar_par[0] * L_k ** ar_par[1]
#     GAMMA[i,0] = np.exp(-timestep/tau_k)
#     GAMMA[i,1] = GAMMA[i,0] ** ar_par[2]
#     Lk_all[i] = L_k
#     tauk_all[i] = tau_k
#     L_k *= scale_ratio

# gamma_all = GAMMA

# nowcast_utils.print_corrcoefs(GAMMA)

# if ar_order == 2:
#     # adjust the lag-2 correlation coefficient to ensure that the AR(p)
#     # process is stationary
#     for i in range(n_cascade_levels):
#         GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef2(GAMMA[i, 0], GAMMA[i, 1])

# # estimate the parameters of the AR(p) model from the autocorrelation
# # coefficients
# PHI = np.empty((n_cascade_levels, ar_order + 1))
# for i in range(n_cascade_levels):
#     PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])

# phi_all = PHI

# nowcast_utils.print_ar_params(PHI)

##############################################################################
# GENERATING BROKEN LINE TIME SERIES FOR MEAN R AND ADVECTION

np.random.seed(seed_bl)
#Create the field mean for the requested number of simulation time steps
r_mean = create_broken_lines(mu_z, sigma2_z, h_val_z, q_val_z, a_zero_z, timestep, (n_timesteps-1) * timestep, no_bls, var_tol_z, mar_tol_z)

np.random.seed(seed_bl)
#Create velocity magnitude for the requested number of simulation time steps
v_mag = create_broken_lines(mu_vmag, sigma2_vmag, h_val_vmag, q_val_vmag, a_zero_vmag, timestep, (n_timesteps-1) * timestep, no_bls, var_tol_vmag, mar_tol_vmag)

np.random.seed(seed_bl)
#Create velocity direction (deg) for the requested number of simulation time steps
v_dir = create_broken_lines(mu_vdir, sigma2_vdir, h_val_vdir, q_val_vdir, a_zero_vdir, timestep, (n_timesteps-1) * timestep, no_bls, var_tol_vdir, mar_tol_vdir)

#Compute advection variables in x- and y-directions using broken lines
vx = np.cos(v_dir / 360 * 2 * np.pi) * v_mag
vy = np.sin(v_dir / 360 * 2 * np.pi) * v_mag

##############################################################################

#Tämä vain kokeilua varten, V joka paikassa 1 tai 0
V = [np.zeros((ny_field, nx_field)), np.zeros((ny_field, nx_field))]
# V = [np.ones((ny_field, nx_field)),np.ones((ny_field, nx_field))]
# V[0] = V[0] * vx[0,0]
# V[1] = V[1] * vy[0,0]
V = np.concatenate([V_[None, :, :] for V_ in V])

x_values, y_values = np.meshgrid(np.arange(nx_field), np.arange((ny_field)))
xy_coords = np.stack([x_values, y_values])

# No advection
# vx = V[0]
# vy = V[1]

# Advection x2
# vx = vx * 2
# vy = vy * 2

###############################################################################
# CREATING FIRST TWO PRECIPITATION FIELDS

#Create the first two precipitation fields, the number is determined by the order of the ar process, in this example it is two. 
#The second one is a copy of the first one. The first velocity magnitude is zero.

v_mag[0] = 0.0
R_ini = []

np.random.seed(seed_random)
R_ini.append(np.random.normal(0.0, 1.0, size=(ny_field, nx_field)))
R_ini.append(np.random.normal(0.0, 1.0, size=(ny_field, nx_field)))
#Change the type of R to align with pySTEPS
R_ini = np.concatenate([R_[None, :, :] for R_ in R_ini])

fft = utils.get_method(fft_method, shape=(ny_field, nx_field), n_threads=1)
init_noise, generate_noise = noise.get_method(noise_method)
noise_kwargs = dict()

p_pow[0] = np.log(nx_field/(a_w0 + b_w0 * r_mean[0] + c_w0 * r_mean[0] ** 2)) #+ 0.15  #scale break
# p_pow[0] = scale_break_wn
p_pow[2] = (a_1 + b_1 * r_mean[0] + c_1 * r_mean[0] ** 2) - 0.5 #beta1
p_pow[3] = (a_2 + b_2 * r_mean[0] + c_2 * r_mean[0] ** 2) - 0.5 #beta2

# #simulate straight with estimated beta1, beta, and w0 values
# p_pow[0] = np.log(nx_field/(fft_params[2,0])) #w0
# p_pow[2] = fft_params[0,0] #beta1
# p_pow[3] = fft_params[1,0] #beta2

p_pow_w0 = np.zeros([n_timesteps, 1])
p_pow_b1 = np.zeros([n_timesteps, 1])
p_pow_b2 = np.zeros([n_timesteps, 1])
w0_sim = np.zeros([n_timesteps, 1])
w0_sim_km = np.zeros([n_timesteps, 1])
beta1_sim = np.zeros([n_timesteps, 1])
beta2_sim = np.zeros([n_timesteps, 1])

p_pow_w0[0] = p_pow[0]
p_pow_b1[0] = p_pow[2]
p_pow_b2[0] = p_pow[3]

pp = init_noise(R_ini, p_pow, fft_method=fft, **noise_kwargs)
R = []
R_0 = generate_noise(pp, randstate=None, seed=seed1, fft_method=fft, domain=domain)
R.append(R_0)
R.append(R_0)

extrapolator_method = extrapolation.get_method(extrap_method)
extrap_kwargs = dict()
extrap_kwargs["xy_coords"] = xy_coords
extrap_kwargs["allow_nonfinite_values"] = True
R_1 = extrapolator_method(R[0], V, 1, "min", **extrap_kwargs)[-1]
R.append(R_1)
#Generate the first innovation field and append it as the last term in R
R.append(generate_noise(pp, randstate=None, seed=seed2, fft_method=fft, domain=domain))
R = np.concatenate([R_[None, :, :] for R_ in R])

#Set missing values with the fill value. Maybe not even needed
R[~np.isfinite(R)] = -15.0

###############################################################################
# SIMULATION LOOP WITH STEPS

nowcast_method = nowcasts.get_method(simulation_method)
R_sim = []
for i in range(1, n_timesteps):
    p_pow[0] = np.log(nx_field/(a_w0 + b_w0 * r_mean[i] + c_w0 * r_mean[i] ** 2)) #+ 0.15  #scale break
    # p_pow[0] = scale_break_wn
    p_pow[2] = (a_1 + b_1 * r_mean[i] + c_1 * r_mean[i] ** 2) - 0.5 #beta1
    p_pow[3] = (a_2 + b_2 * r_mean[i] + c_2 * r_mean[i] ** 2) - 0.5 #beta2
    
    # #simulate straight with estimated beta1, beta, and w0 values
    # p_pow[0] = np.log(nx_field/(fft_params[2,i])) #w0
    # p_pow[2] = fft_params[0,i] #beta1
    # p_pow[3] = fft_params[1,i] #beta2
    
    pp = init_noise(R_ini, p_pow, fft_method=fft, **noise_kwargs)
    #R_prev needs to be saved as R is advected in STEPS loop
    R_prev = R[1].copy()
    R_new = nowcast_method(
        R,
        vx[i],
        vy[i],
        ar_par,
        n_cascade_levels=6,
        R_thr=-10.0, #pitäiskö tän olla eri? 8.1830486304816077?
        kmperpixel=kmperpixel,
        timestep=timestep,
        noise_method=noise_method,
        vel_pert_method="bps",
        mask_method="incremental",
    )

    Fp = noise.fftgenerators.initialize_param_2d_fft_filter(R_new)

    w0_km = nx_field / np.exp(Fp["pars"][0])
    R[0] = R_prev

    p_pow_w0[i] = p_pow[0]
    p_pow_b1[i] = p_pow[2]
    p_pow_b2[i] = p_pow[3]
    w0_sim[i-1] = Fp["pars"][0]
    w0_sim_km[i-1] = nx_field / np.exp(w0_sim[i-1])
    beta1_sim[i-1] = Fp["pars"][2]
    beta2_sim[i-1] = Fp["pars"][3]

    R[1] = R_new
    R[2] = generate_noise(pp, randstate=None, fft_method=fft, domain=domain)
    
    stats_kwargs["mean"] = r_mean[i]
    stats_kwargs["std"] = a_v + b_v * r_mean[i] + c_v * r_mean[i] ** 2
    stats_kwargs["war"] = a_war + b_war * r_mean[i] + c_war * r_mean[i] ** 2
    R_new = set_stats(R_new, stats_kwargs) #VILLE: To make this work, a and b values in probmatching.py (lines 45 and 46) were changed to -135 and 135 from 0 and 50.
    R_new, metadata_clip = clip_domain(R_new, metadata, extent)
    R_sim.append(R_new)

R_sim = np.concatenate([R_[None, :, :] for R_ in R_sim])

#Plot event as a animation
if animation_dbz == 1:
    plt.figure()
    if save_animation_dbz == 1:
        path_outputs1 = os.path.join(out_dir, "Simulated_pngs")
        if not os.path.exists(path_outputs1):
            os.makedirs(path_outputs1)
        path_outputs2 = os.path.join(path_outputs1, "dbz")
        if not os.path.exists(path_outputs2):
            os.makedirs(path_outputs2)
        pysteps.visualization.animations.animate(R_sim, savefig=True, path_outputs=path_outputs2)
    else:
        pysteps.visualization.animations.animate(R_sim, savefig=False)

# #Plot example
# plt.figure()
# pysteps.visualization.plot_precip_field(R_sim[70])

#Plot event as a animation using possibility to jump between timesteps
if animation_blues == 1:
    #ani = animate_interactive(R_sim,grid_on,colorbar_on, predefined_value_range,cmap) #cmap=None is pysteps colorscale
    ani = pysteps.visualization.animations.animate_interactive(R_sim, False, True, False, "Blues")

if save_animation_blues == 1:
    #Create datetime object
    starttime = "00:00:00"
    title_time = pd.date_range(starttime, periods=len(R_sim), freq="5Min")
    #If needed, add folder where to save the animation
    out_dir2 = os.path.join(out_dir, "Output_animation_blues")
    if not os.path.exists(out_dir2):
        os.makedirs(out_dir2)
    #Save pngs
    for im in range(0,len(R_sim)):
        plt.figure()
        testi_im = plt.imshow(R_sim[im], cmap="Blues", vmin=0, vmax=round(np.nanmax(R_sim)+0.5))
        plt.colorbar(testi_im, spacing="uniform", extend="max", shrink=0.8, cax=None, label="Precipitation intensity [dBZ]")
        plt.title("Time: %s"% str(title_time[im])[11:16])
        plt.savefig(os.path.join(out_dir2, f"test_{im}.png"))
        plt.close()

#Convert simulated event to mm/h
metadata["transform"] = "dB"
metadata["unit"] = "dBZ"
metadata ["zerovalue"] = 3.1830486304816077
R_sim_mmh, metadata_mmh = conversion.to_rainrate(R_sim, metadata, zr_a=a_R, zr_b=b_R)
R_sim_mmh[R_sim_mmh < 0.1] = 0 #Values less than threshold to zero

#Plot event as a animation
if animation_mmh == 1:
    plt.figure()
    if save_animation_mmh == 1:
        path_outputs3 = os.path.join(out_dir, "Simulated_pngs")
        if not os.path.exists(path_outputs3):
            os.makedirs(path_outputs3)
        path_outputs4 = os.path.join(path_outputs3, "mmh")
        if not os.path.exists(path_outputs4):
            os.makedirs(path_outputs4)
        pysteps.visualization.animations.animate(R_sim_mmh, savefig=True, path_outputs=path_outputs4)
    else:
        pysteps.visualization.animations.animate(R_sim_mmh, savefig=False)

# plt.figure()
# plt.imshow(R_sim[0])
# plt.figure()
# plt.imshow(R_sim_mmh[0])

##############################################################################
# TABLES OF SIMULATED FFT-PARAMETERS

if csv_sim_fft_par_ts == 1:
    beta1_sim0 = np.zeros(n_timesteps)
    beta2_sim0 = np.zeros(n_timesteps)
    w0_sim_km0 = np.zeros(n_timesteps)
    for i in range(n_timesteps):
        beta1_sim0[i] = beta1_sim[i]
        beta2_sim0[i] = beta2_sim[i]
        w0_sim_km0[i] = w0_sim_km[i]
    data_temp = [beta1_sim0, beta2_sim0, w0_sim_km0]
    fft_params_sim = pd.DataFrame(data_temp, index=['beta1', 'beta2', 'w0'])
    pd.DataFrame(fft_params_sim).to_csv(os.path.join(out_dir, "fft_params_sim.csv"))

##############################################################################
# PLOT SIMULATED FFT-PARAMETERS

#Plot timeseries of simulated beta1
plt.figure()
plt.plot(p_pow_b1[:-1, 0], label="in")
plt.plot(beta1_sim[:-1, 0], label="out")
plt.plot(fft_params[0, :], label="osapol")
plt.legend()
plt.title("Beta 1")
if plot_simulated_beta1 == 1:
    plt.savefig(os.path.join(out_dir, "simulated_beta1.png"))

#Plot timeseries of simulated beta2
plt.figure()
plt.plot(p_pow_b2[:-1, 0], label="in")
plt.plot(beta2_sim[:-1, 0], label="out")
plt.plot(fft_params[1, :], label="osapol")
plt.legend()
plt.title("Beta 2")
if plot_simulated_beta2 == 1:
    plt.savefig(os.path.join(out_dir, "simulated_beta2.png"))

#Plot timeseries of simulated scale break
plt.figure()
plt.plot(p_pow_w0[:-1, 0], label="in")
plt.plot(w0_sim[:-1, 0], label="out")
plt.legend()
plt.title("Scale break, w0")
if plot_simulated_w0 == 1:
    plt.savefig(os.path.join(out_dir, "simulated_w0.png"))
    
plt.figure()
plt.plot(fft_params[2, :], label="osapol")
plt.plot(w0_sim_km[:-1, 0], label="sim")
plt.legend()
plt.title("Scale break: Osapol vs. simulation")

##############################################################################
# PLOT STATISTICS BEFORE AND AFTER STAT_CORRECTION IN SIMULATION

#Plot timeseries of mean
ts_mean = np.zeros(len(R_sim))
for i in range(0, len(R_sim)):
    ts_mean[i] = np.nanmean(R_sim[i])
plt.figure()
plt.plot(r_mean, label="in")
plt.plot(ts_mean, label="out")
plt.plot(data_tss[0, :], label="osapol")  #measured osapol-event
plt.legend()
plt.title("Mean")
if plot_stat_correction == 1:
    plt.savefig(os.path.join(out_dir, "corrected_mean.png"))

#Plot timeseries of standard deviation
r_std = np.zeros(len(R_sim))
for i in range(0, len(R_sim)):
    r_std[i] = a_v + b_v * r_mean[i] + c_v * r_mean[i] ** 2
ts_std = np.zeros(len(R_sim))
for i in range(0, len(R_sim)):
    ts_std[i] = np.nanstd(R_sim[i])
plt.figure()
plt.plot(r_std, label="in")
plt.plot(ts_std, label="out")
plt.plot(data_tss[1, :], label="osapol")  #measured osapol-event
plt.legend()
plt.title("Std")
if plot_stat_correction == 1:
    plt.savefig(os.path.join(out_dir, "corrected_std.png"))

#Plot timeseries of WAR
r_war = np.zeros(len(R_sim))
for i in range(0, len(R_sim)):
    r_war[i] = a_war + b_war * r_mean[i] + c_war * r_mean[i] ** 2
ts_war = np.zeros(len(R_sim))
for i in range(0, len(R_sim)):
    ts_war[i] = (np.count_nonzero(R_sim[i][R_sim[i] > 3.1830486304816077])) / \
        np.count_nonzero(~np.isnan(R_sim[i]))
plt.figure()
plt.plot(r_war, label="in")
plt.plot(ts_war, label="out")
plt.plot(data_tss[2, :], label="osapol")  #measured osapol-event
plt.legend()
plt.title("WAR")
if plot_stat_correction == 1:
    plt.savefig(os.path.join(out_dir, "corrected_war.png"))

##############################################################################
# PLOT SIMULATED VARIABLES AGAINST OSAPOL-INPUT

ts_mean_mmh = np.zeros(len(R_sim_mmh))
for i in range(0, len(R_sim_mmh)):
    ts_mean_mmh[i] = np.nanmean(R_sim_mmh[i])

# Means of mm/h timeseries
np.mean(ts_mean_mmh) #mm/h simulated event
np.mean(data_tss[3, :]) #mm/h osapol input

plt.figure()
plt.plot(data_tss[3, :], label="osapol", color="blue")
plt.plot(ts_mean_mmh, label="simulated", color="orange")
plt.axhline(y=np.mean(data_tss[3, :]), xmin=0, xmax=1, color="blue", linestyle = "dashed")
plt.axhline(y=np.mean(ts_mean_mmh), xmin=0, xmax=1, color="orange", linestyle = "dashed")
plt.legend()
plt.title("Mean (mm/h)")
if plot_simulated_vs_osapol == 1:
    plt.savefig(os.path.join(out_dir, "osapol_vs_mean.png"))

# Means of dbz timeseries
ts_mean_dbz = np.zeros(len(R_sim))
for i in range(0, len(R_sim)):
    ts_mean_dbz[i] = np.nanmean(R_sim[i])

mean_dbz_osapol = np.mean(data_tss[0, :]) #dbz osapol input
mean_dbz_bl = np.mean(r_mean) #dbz brokenline
mean_dbz_sim = np.mean(ts_mean_dbz) #mm/h simulated event

plt.figure()
plt.plot(data_tss[0, :], label="osapol", color="blue")
plt.plot(r_mean, label="broken line", color="orange")
plt.plot(ts_mean_dbz, label="simulated", color="green")
plt.axhline(y=mean_dbz_osapol, xmin=0, xmax=1, color="blue", linestyle = "dashed")
plt.axhline(y=mean_dbz_bl, xmin=0, xmax=1, color="orange", linestyle = "dashed")
plt.axhline(y=mean_dbz_sim, xmin=0, xmax=1, color="green", linestyle = "dashed")
plt.legend()
plt.title("Mean (dBZ)")

##########################

ts_std_mmh = np.zeros(len(R_sim_mmh))
for i in range(0, len(R_sim_mmh)):
    ts_std_mmh[i] = np.nanstd(R_sim_mmh[i])
    
np.mean(ts_std_mmh) #mm/h simulated event
np.mean(data_tss[4, :]) #mm/h osapol input
    
plt.figure()
plt.plot(data_tss[4, :], label="osapol", color="blue")
plt.plot(ts_std_mmh, label="simulated", color="orange")
plt.axhline(y=np.mean(data_tss[4, :]), xmin=0, xmax=1, color="blue", linestyle = "dashed")
plt.axhline(y=np.mean(ts_std_mmh), xmin=0, xmax=1, color="orange", linestyle = "dashed")
plt.legend()
plt.title("Std (mm/h)")

if plot_simulated_vs_osapol == 1:
    plt.savefig(os.path.join(out_dir, "osapol_vs_std.png"))

ts_std_dbz = np.zeros(len(R_sim))
for i in range(0, len(R_sim)):
    ts_std_dbz[i] = np.nanstd(R_sim[i])
    
mean_std_sim = np.mean(ts_std_dbz) #dbz simulated event
mean_std_osapol = np.mean(data_tss[1, :]) #dbz osapol input

plt.figure()
plt.plot(data_tss[1, :], label="osapol", color="blue")
plt.plot(ts_std_dbz, label="simulated", color="orange")
plt.axhline(y=mean_std_osapol, xmin=0, xmax=1, color="blue", linestyle = "dashed")
plt.axhline(y=mean_std_sim, xmin=0, xmax=1, color="orange", linestyle = "dashed")
plt.legend()
plt.title("Std (dBZ)")

##########################

ts_war_mmh = np.zeros(len(R_sim_mmh))
for i in range(0, len(R_sim_mmh)):
    ts_war_mmh[i] = (np.count_nonzero(R_sim_mmh[i][R_sim_mmh[i] > 0.1])) / \
        np.count_nonzero(~np.isnan(R_sim_mmh[i]))  #poista nimittäjästä nan solut
    #Threshold-arvo oli aiemmin 0, jolloin ts_war oli joka aika-askeleella 1.
    #Muutettu nyt vastaamaan dbz-datassa esiintyvää satamattoman pixelin arvoa.
    
ts_war_dbz = np.zeros(len(R_sim))
for i in range (0, len(R_sim)):
    ts_war_dbz[i] = (np.count_nonzero(R_sim[i][R_sim[i] > 3.1830486304816077])) / np.count_nonzero(~np.isnan(R_sim[i]))

np.mean(ts_war_dbz) #dbz simulated event
np.mean(ts_war_mmh) #mm/h simulated event
np.mean(data_tss[5, :]) #mm/h osapol input
np.mean(data_tss[2, :]) #dbz osapol input

plt.figure()
plt.plot(data_tss[5, :], label="osapol", color="blue")
plt.plot(ts_war_mmh, label="simulated", color="orange")
plt.axhline(y=np.mean(data_tss[5, :]), xmin=0, xmax=1, color="blue", linestyle = "dashed")
plt.axhline(y=np.mean(ts_war_mmh), xmin=0, xmax=1, color="orange", linestyle = "dashed")
plt.legend()
plt.title("WAR (mm/h)")
if plot_simulated_vs_osapol == 1:
    plt.savefig(os.path.join(out_dir, "osapol_vs_war.png"))

##########################

plt.figure()
plt.plot(advection_tss[2, :], label="osapol: calculated after optical flow")
plt.plot(v_mag, label="broken line")
plt.legend()
plt.title("Advection: magnitude")

plt.figure()
plt.plot(advection_tss[5, :], label="osapol: calculated after optical flow")
plt.plot(v_dir, label="broken line")
plt.legend()
plt.title("Advection: direction")

plt.figure()
plt.plot(advection_tss[0, :], label="osapol: using optical flow")
plt.plot(vx, label="calculated using broken lines")
plt.legend()
plt.title("Advection: x-direction")

plt.figure()
plt.plot(advection_tss[1, :], label="osapol: using optical flow")
plt.plot(vy, label="calculated using broken lines")
plt.legend()
plt.title("Advection: y-direction")

plt.figure()
plt.scatter(vx, vy, c ="blue", label="sim")
plt.scatter(advection_tss[0, :], advection_tss[1, :], c ="red", label="osapol")
plt.xlabel('Advection in x-direction')
plt.ylabel('Advection: y-direction')
plt.legend()

##############################################################################
# PLOT THE OBSERVED 1D POWER SPECTRUM AND THE MODEL

#The parametric model uses a piece-wise linear function with two spectral slopes (beta1 and beta2) and one breaking point
#https://pysteps.readthedocs.io/en/latest/auto_examples/plot_noise_generators.html

if plot_1dps == 1:
    #Compute the observed and fitted 1D PSD
    L = np.max(Fp["input_shape"])
    if L % 2 == 1:
        wn = np.arange(0, int(L / 2) + 1)
    else:
        wn = np.arange(0, int(L / 2))
    R_, freq = rapsd(R[2], fft_method=np.fft, return_freq=True)
    f = np.exp(Fp["model"](np.log(wn), *Fp["pars"]))
    
    #Extract the scaling break in km, beta1 and beta2
    w0 = L / np.exp(Fp["pars"][0])
    b1 = Fp["pars"][2]
    b2 = Fp["pars"][3]
    
    fig, ax = plt.subplots()
    plot_scales = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
    plot_spectrum1d(
        freq,
        R_,
        x_units="km",
        y_units="dBZ",
        color="k",
        ax=ax,
        label="Observed",
        wavelength_ticks=plot_scales,
    )
    plot_spectrum1d(
        freq[:-1],
        f,
        x_units="km",
        y_units="dBZ",
        color="r",
        ax=ax,
        label="Fit",
        wavelength_ticks=plot_scales,
    )
    plt.legend()
    ax.set_title(
        "Radially averaged log-power spectrum of R\n"
        r"$\omega_0=%.0f km, \beta_1=%.1f, \beta_2=%.1f$" % (w0, b1, b2)
    )
    plt.show()
    if save_1dps == 1:
        plt.savefig(os.path.join(out_dir, "example_1dps.png"))
        
##############################################################################
# CALCULATE ERROR BETWEEN ESTIMATED BP-FILTER PARAMETERS AND PARAMETERS FROM SIMULATED FIELD

# #Simulation is done by using estimated beta1, beta, and w0 values without linking them into mean areal rainfall bl-timeseries

# beta1_error = p_pow_b1 - beta1_sim
# beta1_error_osapol = fft_params[0, :] - beta1_sim

# beta2_error = p_pow_b2 - beta2_sim
# beta2_error_osapol = fft_params[1, :] - beta2_sim

# w0_error = p_pow_w0 - w0_sim
# w0_error_osapol = np.log(nx_field/fft_params[2, :]) - w0_sim

# beta1_mean_error = np.mean(beta1_error[:-1])
# beta1_mean_error_osapol = np.mean(beta1_error_osapol[:-1])
# beta2_mean_error = np.mean(beta2_error[:-1])
# beta2_mean_error_osapol = np.mean(beta2_error_osapol[:-1])
# w0_mean_error = np.mean(w0_error[:-1])
# w0_mean_error_osapol = np.mean(w0_error_osapol[:-1])

# print(beta1_mean_error)
# print(beta2_mean_error)
# print(w0_mean_error)
# print(beta1_mean_error_osapol)
# print(beta2_mean_error_osapol)
# print(w0_mean_error_osapol)
        
##############################################################################
# TIME USED FOR PARAMETER ESTIMATION: End timer

run_end_0 = time.perf_counter()
run_dur_0 = run_end_0 - run_start_0
print(run_dur_0 / 60)

##############################################################################
# SAVE SIMULATED EVENT IN TIFF-FORMAT

if save_tiff == 1:
    out_dir3 = os.path.join(out_dir, "Event_tiffs")
    if not os.path.exists(out_dir3):
        os.makedirs(out_dir3)
    crs = from_epsg(3067) 
    # Coordinates of clipped osapol data: xmin = 214000, xmax = 470000, ymin = 6720500, ymax = 6976500
    transform = from_origin(214000, 6976500, 1000, 1000) #rasterio.transform.from_origin(west, north, xsize, ysize)
    for i in range(0,len(R_sim)):
        arr = R_sim[i]
        new_dataset = rasterio.open(os.path.join(out_dir3, f"test_{i}.tif"), "w", driver="GTiff",
                                    height = arr.shape[0], width = arr.shape[1],
                                    count=1, dtype=str(arr.dtype),
                                    crs=crs,
                                    transform=transform)
        new_dataset.write(arr, 1)
        new_dataset.close()

##############################################################################
# TABLES OF SIMULATED FFT-PARAMETERS

if csv_arrays_to_save == 1:
    data_temp2 = [r_mean, v_mag, v_dir, vx, vy]
    arrays_to_save = pd.DataFrame(data_temp2, index=['r_mean', 'v_mag', 'v_dir', 'vx', 'vy'])
    pd.DataFrame(arrays_to_save).to_csv(os.path.join(out_dir, "arrays_to_save.csv"))
    