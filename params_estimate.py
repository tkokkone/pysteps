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

#Choose the method to calculate advection
oflow_method = "LK" #Lukas-Kanade
oflow_advection = pysteps.motion.get_method(oflow_method) 


extrap_method = "semilagrangian"
extrapolator_method = extrapolation.get_method(extrap_method)
extrap_kwargs["xy_coords"] = xy_coords
extrap_kwargs["allow_nonfinite_values"] = True

#Variables for magnitude and direction of advection
Vx = np.zeros(len(R)-1)
Vy = np.zeros(len(R)-1)

#Loop to calculate average x- and y-components of the advection, as well as magnitude and direction in xy-dir
for i in range(2, len(R)-1):
    R_cur = R[i, :, :]
    R_prev = R[i-1, :, :]
    R_prev2 = R[i-2, :, :]
    V1 = oflow_advection(np.stack([R_prev,R_cur],axis=0))
    V2 = oflow_advection(np.stack([R_prev2,R_prev,R_cur],axis=0))
    R_prev_adv = extrapolator_method(R_prev, V1, 1, "min", **extrap_kwargs)[
            -1
        ]
   # V1test = oflow_advection(R[i-1:i, :, :])
   # V2 = oflow_advection(R[i-2:i, :, :])
    
    #Vx = V[0,:,:] #contains the x-components of the motion vectors.
    #y = V[1,:,:] #contains the y-components of the motion vectors.
    #The velocities are in units of pixels/timestep.

##############################################################################
# ESTIMATE AR(2)-MODEL PARAMETERS

#The Lagrangian temporal evolution of each normalised cascade level is modelled using an AR(2)-model
ar_order = 2

#Create variables that include all gammas and phis of all cascade levels for each time step
gamma_all = np.zeros([n_cascade_levels*2, len(R2)-2])
phi_all = np.zeros([n_cascade_levels*3, len(R2)-2])

#Compute the cascade decompositions of the input precipitation fields
for i in range(0,len(R2)-2): 
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
slope_tlen, intercept_tlen, r_value_tlen, p_value_tlen, std_err_tlen = sp.linregress(np.log(L_k[1:n_cascade_levels-1,0]), np.log(tau_k[1:n_cascade_levels-1,0]))
y_values_tlen = slope_tlen*np.log(L_k[1:n_cascade_levels-1])+intercept_tlen

plt.figure()
plt.plot(np.log(L_k[0:n_cascade_levels,0]), np.log(tau_k[0:n_cascade_levels,0]), marker='s')
plt.plot(np.log(L_k[1:n_cascade_levels-1]), y_values_tlen)
plt.title("L_k vs. Tau_k -fit - AR-pars: tlen_a and tlen_b")
plt.savefig("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/events/Test_event/ar-par_Lk_vs_Tauk.png")

tlen_b = slope_tlen
tlen_a = np.exp(intercept_tlen)

#Estimate tlen_c based on linear fit of gamma1 and gamma2
slope_tlen_c, intercept_tlen_c, r_value_tlen_c, p_value_tlen_c, std_err_tlen_c = sp.linregress(np.log(gamma1_mean[0:n_cascade_levels-1,0]), np.log(gamma2_mean[0:n_cascade_levels-1,0]))
y_values_tlen_c = slope_tlen_c*np.log(gamma1_mean[0:n_cascade_levels-1,0])+intercept_tlen_c

plt.figure()
plt.plot(np.log(gamma1_mean[0:n_cascade_levels,0]), np.log(gamma2_mean[0:n_cascade_levels,0]), marker='s')
plt.plot(np.log(gamma1_mean[0:n_cascade_levels-1]), y_values_tlen_c)
plt.title("Gamma1 vs. Gamma2 -fit - AR-pars: tlen_c")
plt.savefig("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/events/Test_event/ar-par_Gamma1_vs_Gamma2.png")

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
print(std_fit)

#polynomial fit with degree = 2
war_fit = np.poly1d(np.polyfit(areal_rainfall_ts, war_ts, 2))
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, war_ts)
plt.plot(polyline, war_fit(polyline))
plt.title("mean vs. war")
print(war_fit)

#polynomial fit with degree = 2
beta1_fit = np.poly1d(np.polyfit(areal_rainfall_ts, beta1s, 2))
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, beta1s)
plt.plot(polyline, beta1_fit(polyline))
plt.title("mean vs. beta1")
print(beta1_fit)

#polynomial fit with degree = 2
beta2_fit = np.poly1d(np.polyfit(areal_rainfall_ts, beta2s, 2))
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, beta2s)
plt.plot(polyline, beta2_fit(polyline))
plt.title("mean vs. beta2")
print(beta2_fit)

#polynomial fit with degree = 2
w0_fit = np.poly1d(np.polyfit(areal_rainfall_ts, w0s, 2))
#add fitted polynomial line to scatterplot
plt.figure()
polyline = np.linspace(min(areal_rainfall_ts), max(areal_rainfall_ts), 400)
plt.scatter(areal_rainfall_ts, w0s)
plt.plot(polyline, w0_fit(polyline))
plt.title("mean vs. w0")
print(w0_fit)

##############################################################################
# SAVE PARAMETERS AS CSV-FILES

#Time series from input data 
data_temp = [areal_rainfall_ts, std_ts, war_ts, areal_rainfall_mmh, std_mmh, war_mmh]
fft_params = pd.DataFrame(data_temp, index=['mean', 'std', 'war', 'mean_mmh', 'std_mmh', 'war_mmh'])
pd.DataFrame(fft_params).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/events/Test_event/data_tss.csv")

#FFT-parameters
data_temp = [beta1s, beta2s, w0s]
fft_params = pd.DataFrame(data_temp, index=['beta1', 'beta2', 'w0'])
pd.DataFrame(fft_params).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/events/Test_event/fft_params.csv")

#AR-parameter time series
ar_params1 = pd.DataFrame(gamma1_mean)
ar_params1[1] = gamma2_mean
ar_params1[2] = tau_k
ar_params1[3] = L_k
ar_params1.rename(columns = {0 : 'gamma1_mean', 1 : 'gamma2_mean', 2 : 'tau_k', 3 : 'L_k'}, inplace = True)
pd.DataFrame(ar_params1).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/events/Test_event/ar_params.csv")

#All needed parameters
data_temp = [w0_fit[0], w0_fit[1], w0_fit[2], beta1_fit[0], beta1_fit[1], beta1_fit[2], beta2_fit[0], beta2_fit[1], beta2_fit[2], 
              tlen_a, tlen_b, tlen_c, std_fit[0], std_fit[1], std_fit[2], war_fit[0], war_fit[1], war_fit[2], 
              mu_z, sigma2_z, H, a_zero, mu_z_Vxy, sigma2_z_Vxy, H_Vxy, a_zero_Vxy, mu_z_Vdir_deg_adj, sigma2_z_Vdir_deg_adj, H_Vdir_deg_adj, a_zero_Vdir_deg_adj, 
              q, noBLs, var_tol, mar_tol]

all_params = pd.DataFrame(data_temp, index=['a_w0', 'b_w0', 'c_w0', 'a_1', 'b_1', 'c_1', 'a_2', 'b_2', 'c_2', 'tlen_a', 'tlen_b', 'tlen_c', 'a_v', 'b_v', 'c_v', 'a_war', 'b_war', 'c_war', 
                                            'mu_z', 'sigma2_z', 'h_val_z', 'a_zero_z', 'mu_vmag', 'sigma2_vmag', 'h_val_vmag', 'a_zero_vmag', 'mu_vdir', 'sigma2_vdir', 'h_val_vdir', 'a_zero_vdir', 
                                            'q', 'no_bls', 'var_tol', 'mar_tol'])
pd.DataFrame(all_params).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/events/Test_event/all_params.csv")

##############################################################################
# TIME USED FOR PARAMETER ESTIMATION: End timer

run_end_0 = time.perf_counter()
run_dur_0 = run_end_0 - run_start_0
print(run_dur_0 / 60)
