# -*- coding: utf-8 -*-
"""

pysteps_params.py

Scrip to estimate all the parameters for the model:
    - 

Requirements:
    - 

References:
    - 

Created on Fri Feb  26 06:58:42 2021

@author: Ville Lindgren
"""

##############################################################################
# IMPORT PACKAGES

import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import cm
# import fiona
# import rasterio
# import rasterio.mask
# import rasterio.plot
import time
import pysteps
from datetime import datetime
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
import scipy.stats as sp
import math
from scipy import optimize
from pysteps import utils
from pysteps.utils import conversion, transformation
from osgeo import gdal

##############################################################################
# INPUT DATA

#Read in the event with pySTEPS
date = datetime.strptime("201408071855", "%Y%m%d%H%M") #last radar image of the event
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
fns = pysteps.io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=134)

#Select the importer
importer = pysteps.io.get_method(importer_name, "importer")

#Read the radar composites
R, quality, metadata = pysteps.io.read_timeseries(fns, importer, **importer_kwargs)
del quality  #delete quality variable because it is not used

#Add unit to metadata
metadata["unit"] = "mm/h"

#Plot example
# plt.figure()
# pysteps.visualization.plot_precip_field(R[100], title=date)

#Plot event as a animation
# plt.figure()
# pysteps.visualization.animations.animate(R)

##############################################################################
# DATA TRANSFORMATION

#Values less than threshold to zero
R[R<0.1] = 0
metadata["threshold"] = 0.1

#Add value 1 to each cell so after dB-transformation there are no negative cells
R = R+1

#Areal mean rainfall in mm/h
#This is just for testing purposes
# areal_rainfall_mmh = np.zeros(len(fns[1])) #time series of arithmetic means, ignoring NaNs
# for i in range (0, len(fns[1])):
#     #print(i)
#     areal_rainfall_mmh[i] = np.nanmean(R[i])
# plt.figure()
# plt.plot(areal_rainfall_mmh)

# Log-transform the data to unit of dBR
R, metadata = transformation.dB_transform(R, metadata, threshold=None, zerovalue=None)

#Plot event as a animation
# plt.figure()
# pysteps.visualization.animations.animate(dBR[0])

#dBZ transformation for mm/h-data (Cannot use dBR transformation.)
#Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with size. J. Meteor., 5, 165–166.
#Z-R relationship: Z = a*R^b (Reflectivity)
#dBZ conversion: dBZ = 10 * log10(Z) (Decibels of Reflectivity)
a_R=223
b_R=1.53
#Leinonen et al. 2012 - A Climatology of Disdrometer Measurements of Rainfall in Finland over Five Years with Implications for Global Radar Observations
#https://doi.org/10.1175/JAMC-D-11-056.1

R, metadata = pysteps.utils.conversion.to_reflectivity(R, metadata, zr_a=a_R, zr_b=b_R)

R[R==metadata["zerovalue"]] = 0
metadata["zerovalue"] = 0

#This is doing same than function above, but with zerovalue of 0
# dBZ=R.copy()
# dBZ = a_R * (dBZ**b_R)
# dBZ = 10 * np.log10(dBZ)


#Set missing values with the fill value
#R[~np.isfinite(R)] = -15.0

##############################################################################
# CALCULATE AREAL VARIABLES

#Areal mean rainfall
areal_rainfall_ts = np.zeros(len(fns[1])) #time series of arithmetic means, ignoring NaNs
for i in range (0, len(fns[1])):
    #print(i)
    areal_rainfall_ts[i] = np.nanmean(R[i])
plt.figure()
plt.plot(areal_rainfall_ts)

#Wetted area ratio
war_ts = np.zeros(len(fns[1]))
for i in range (0, len(fns[1])):
    #print(i)
    #war_ts[i] = (np.count_nonzero(R[i][R[i] > -15])) / (264*264) #poista nimittäjästä nan solut
    war_ts[i] = (np.count_nonzero(R[i][R[i] > 0])) / np.count_nonzero(~np.isnan(R[i])) #poista nimittäjästä nan solut
plt.figure()
plt.plot(war_ts)
    
#Standard deviation 
std_ts = np.zeros(len(fns[1]))
for i in range (0, len(fns[1])):
    #print(i)
    std_ts[i] = np.nanstd(R[i])
# plt.figure()
# plt.plot(std_ts)

##############################################################################
# CALCULATE ADVECTION

#Advection
oflow_method = pysteps.motion.get_method("LK") #method: Lukas-Kanade

#Loop to calculate average x- and y-components of the advection, as well as magnitude and direction in xy-dir
#run_start_2 = time.perf_counter()
Vx = np.zeros(len(R)-1)
Vy = np.zeros(len(R)-1)
Vxy = np.zeros(len(R)-1)
Vdir_rad = np.zeros(len(R)-1)
for i in range(0, len(R)-1):
    Vtemp = oflow_method(R[i:i+2, :, :])
    #Vtemp[0,:,:] contains the x-components of the motion vectors.
    #Vtemp[1,:,:] contains the y-components of the motion vectors.
    #The velocities are in units of pixels/timestep.
    Vx[i] = np.nanmean(Vtemp[0]) #field mean in x-direction
    Vy[i] = np.nanmean(Vtemp[1]) #field mean in y-direction
    Vxy[i] = np.sqrt(Vx[i]**2 + Vy[i]**2) #total magnitude of advection
    Vdir_rad[i] = np.arctan(Vy[i]/Vx[i]) #direction of advection in xy-dir in radians
    #The inverse of tan, so that if y = tan(x) then x = arctan(y).
    #Range of values of the funktion: −pi/2 < y < pi/2 (-1.5707963267948966 < y < 1.5707963267948966)
    #np.pi gives value of pi
Vdir_deg = (Vdir_rad/(2*np.pi))*360 #direction of advection in xy-dir in degrees
#run_end_2 = time.perf_counter()
#run_dur_2 = run_end_2 - run_start_2

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
        
# plt.figure()
# plt.plot(Vx)
# plt.figure()
# plt.plot(Vy)
# plt.figure()
# plt.plot(Vxy)
# plt.figure()
# plt.plot(Vdir_deg_adj)

##############################################################################
# BROKEN LINE PARAMETERS FOR AREAL MEAN RAINFALL

tStep = float(5) #timestep [min]
tSerieLength = (len(R) - 1) * tStep #timeseries length [min]
x_values = np.arange(0, tSerieLength + tStep, tStep)
euler = 2.71828

#Mean and variance of input time series
mu_z = float(np.mean(areal_rainfall_ts)) #mean of input time series [dBz]
sigma2_z = float(np.var(areal_rainfall_ts)) #variance of input time series [dBz]

#Correlations and a_zero of time series
event_cors = sm.tsa.acf(areal_rainfall_ts, nlags=(len(areal_rainfall_ts) - 1))
plt.figure()
plt.plot(event_cors)
# tsaplots.plot_acf(areal_rainfall_ts, lags=(len(areal_rainfall_ts) - 1), alpha=1)

#Timesteps for no autocorrelation
no_autocor = np.where(event_cors < 1/euler)
a_zero = (no_autocor[0][0]-1)*tStep #timesteps in which correlation values are > 1/e = 0.36787968862663156
#For now, this have to be checked from plots above

#Power spectrum, beta, and H of time series
power_spectrum = np.abs(np.fft.fft(areal_rainfall_ts))**2 #absolute value of the complex components of fourier transform of data
freqs = np.fft.fftfreq(areal_rainfall_ts.size, tStep) #frequencies

freqs_sliced = freqs[1:]
freqs_sliced = freqs_sliced[freqs_sliced>0]
power_spectrum_sliced = power_spectrum[1:]
power_spectrum_sliced = power_spectrum_sliced[0:len(freqs_sliced)]

# plt.figure()
# plt.plot(np.log(freqs_sliced), np.log(power_spectrum_sliced))
#For now, beginning of the "flat tale" have to be checked from plot above and insert the value manually

#Calculate power spectrum exponent beta and H from beta using linear fitting of the data
slope, intercept, r_value, p_value, std_err = sp.linregress(np.log(freqs_sliced[0:len(freqs_sliced)]), np.log(power_spectrum_sliced[0:len(freqs_sliced)]))
y_values = slope*np.log(freqs_sliced[0:len(freqs_sliced)])+intercept

plt.figure()
plt.plot(np.log(freqs_sliced), np.log(power_spectrum_sliced))
plt.plot(np.log(freqs_sliced[0:len(freqs_sliced)]), y_values)

H = (-slope-1)/2

##############################################################################
# BROKEN LINE PARAMETERS FOR ADVECTION MAGNITUDE

mu_z_Vxy = float(np.mean(Vxy)) #mean of input time series
sigma2_z_Vxy = float(np.var(Vxy)) #variance of input time series

#Correlations and a_zero of time series
event_cors_Vxy = sm.tsa.acf(Vxy, nlags=(len(Vxy) - 1))
plt.figure()
plt.plot(event_cors_Vxy)
#tsaplots.plot_acf(Vxy, lags=(len(Vxy) - 1), alpha=1)

no_autocor_Vxy = np.where(event_cors_Vxy < 1/euler)
a_zero_Vxy = (no_autocor_Vxy[0][0]-1)*tStep #time steps in which correlation values are > 1/e = 0.36787968862663156
#For now, this have to be checked from plots above

#Power spectrum, beta, and H of time series
power_spectrum_Vxy = np.abs(np.fft.fft(Vxy))**2 #absolute value of the complex components of fourier transform of data
freqs_Vxy = np.fft.fftfreq(Vxy.size, tStep) #frequencies

freqs_Vxy_sliced = freqs_Vxy[1:]
freqs_Vxy_sliced = freqs_Vxy_sliced[freqs_Vxy_sliced>0]
power_spectrum_Vxy_sliced = power_spectrum_Vxy[1:]
power_spectrum_Vxy_sliced = power_spectrum_Vxy_sliced[0:len(freqs_Vxy_sliced)]

# plt.figure()
# plt.plot(np.log(freqs_Vxy_sliced), np.log(power_spectrum_Vxy_sliced))
#For now, beginning of the "flat tale" have to be checked from plot above and insert the value manually

#Calculate power spectrum exponent beta and H from beta using linear fitting of the data
slope_Vxy, intercept_Vxy, r_value_Vxy, p_value_Vxy, std_err_Vxy = sp.linregress(np.log(freqs_Vxy_sliced[0:len(freqs_Vxy_sliced)]), np.log(power_spectrum_Vxy_sliced[0:len(freqs_Vxy_sliced)]))
y_values_Vxy = slope_Vxy*np.log(freqs_Vxy_sliced[0:len(freqs_Vxy_sliced)])+intercept_Vxy

plt.figure()
plt.plot(np.log(freqs_Vxy_sliced), np.log(power_spectrum_Vxy_sliced))
plt.plot(np.log(freqs_Vxy_sliced[0:len(freqs_Vxy_sliced)]), y_values_Vxy)

H_Vxy = (-slope_Vxy-1)/2

##############################################################################
# BROKEN LINE PARAMETERS FOR ADVECTION DIRECTION

mu_z_Vdir_deg_adj = float(np.mean(Vdir_deg_adj)) #mean of input time series
sigma2_z_Vdir_deg_adj = float(np.var(Vdir_deg_adj)) #variance of input time series

#Correlations and a_zero of time series
event_cors_Vdir_deg_adj = sm.tsa.acf(Vdir_deg_adj, nlags=(len(Vdir_deg_adj) - 1))
plt.figure()
plt.plot(event_cors_Vdir_deg_adj)
#tsaplots.plot_acf(Vdir_deg_adj, lags=(len(Vdir_deg_adj) - 1), alpha=1)

#Timesteps for no autocorrelation
no_autocor_Vdir_deg_adj = np.where(event_cors_Vdir_deg_adj < 1/euler)
a_zero_Vdir_deg_adj = (no_autocor_Vdir_deg_adj[0][0]-1)*tStep #time steps in which correlation values are > 1/e = 0.36787968862663156
#For now, this have to be checked from plots above

#Power spectrum, beta, and H of time series
power_spectrum_Vdir_deg_adj = np.abs(np.fft.fft(Vdir_deg_adj))**2 #absolute value of the complex components of fourier transform of data
freqs_Vdir_deg_adj = np.fft.fftfreq(Vdir_deg_adj.size, tStep) #frequencies

freqs_Vdir_deg_adj_sliced = freqs_Vdir_deg_adj[1:]
freqs_Vdir_deg_adj_sliced = freqs_Vdir_deg_adj_sliced[freqs_Vdir_deg_adj_sliced>0]
power_spectrum_Vdir_deg_adj_sliced = power_spectrum_Vdir_deg_adj[1:]
power_spectrum_Vdir_deg_adj_sliced = power_spectrum_Vdir_deg_adj_sliced[0:len(freqs_Vdir_deg_adj_sliced)]

# plt.figure()
# plt.plot(np.log(freqs_Vdir_deg_adj_sliced), np.log(power_spectrum_Vdir_deg_adj_sliced))
#For now, beginning of the "flat tale" have to be checked from plot above and insert the value manually

#Calculate power spectrum exponent beta and H from beta using linear fitting of the data
slope_Vdir_deg_adj, intercept_Vdir_deg_adj, r_value_Vdir_deg_adj, p_value_Vdir_deg_adj, std_err_Vdir_deg_adj = sp.linregress(np.log(freqs_Vdir_deg_adj_sliced[0:len(freqs_Vdir_deg_adj_sliced)]), np.log(power_spectrum_Vdir_deg_adj_sliced[0:len(freqs_Vdir_deg_adj_sliced)]))
y_values_Vdir_deg_adj = slope_Vdir_deg_adj*np.log(freqs_Vdir_deg_adj_sliced[0:len(freqs_Vdir_deg_adj_sliced)])+intercept_Vdir_deg_adj

plt.figure()
plt.plot(np.log(freqs_Vdir_deg_adj_sliced), np.log(power_spectrum_Vdir_deg_adj_sliced))
plt.plot(np.log(freqs_Vdir_deg_adj_sliced[0:len(freqs_Vdir_deg_adj_sliced)]), y_values_Vdir_deg_adj)

H_Vdir_deg_adj = (-slope_Vdir_deg_adj-1)/2

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
    
    #Parameters
    H2 = 2*H #Why 2*H is needed?
    x_values = np.arange(0, tSerieLength + tStep, tStep) #X-values of final time series
    iPoints = len(x_values) #Number of data points in the final time series
    
    #Statistics for the logarithmic time series
    sigma2_y = np.log(((math.sqrt(sigma2_z)/mu_z)**2.0)+1.0) #Variance of Y_t: Eq. 15
    mu_y = np.log(mu_z) - 0.5*sigma2_y #Mean of Y_t: Eq. 14
    iN = int((np.log(tStep/a_zero)/np.log(q)) + 1) #Number of simple broken lines (ksii): Eq. 11
    sigma2_0 = ((1-pow(q,H2))/(1-pow(q,H2*(iN+1))))*sigma2_y #Variance of the broken line at the outer scale: modified Eq. 12
    
    #Broken line statistics for individual levels
    a_p = np.zeros(iN)
    sigma2_p = np.zeros(iN)
    sigma_p = np.zeros(iN)
    for p in range(0, iN):
        a_p[p] = a_zero * pow(q, p) #The time lag between vertices of the broken line on level p: Eq. 11
        sigma2_p[p] = sigma2_0 * pow(q, p*H2) #The variance of the broken line on level p: Eq. 10
        sigma_p[p] = math.sqrt(sigma2_p[p]) #The standard deviation of the broken line on level p
        
    #Limits for variance and mean of final sum broken line
    var_min = sigma2_z - (var_tol * sigma2_z) #acceptable minimum of variance
    var_max = sigma2_z + (var_tol * sigma2_z) #acceptable maximum of variance
    mar_max = mar_tol * mu_z #acceptable maximum of mean areal rainfall
    
    #Empty matrices
    blines = np.zeros((iPoints, iN)) #all simple broken lines (ksii)
    blines_final = np.zeros((iPoints, noBLs)) #all sum broken lines (final broken lines)
    
    #Loop to create noBLs number of sum broken lines
    noAccepted_BLs = 0
    while noAccepted_BLs < noBLs:
        bline_sum = np.zeros(iPoints)
        
        #Loop to create one sum broken line from iN number of simple broken lines
        for p in range(0, iN):
            
            #Generate a multiplicative broken line time series
            eeta = []
            ksii = []
            
            #Create random variates for this simple broken line
            level = p
            N = int(tSerieLength/a_p[level] + 3) #Number of vertices (iid variables) for this level (+ 2 to cover the ends + 1 for trimming of decimals)
            eeta = np.random.normal(0.0, 1.0, N)
            
            #Interpolate values of this simple broken line at every time step
            k = np.random.uniform(0.0, 1.0)
            a = a_p[level]
            x0 = float(-k * a) #Location first eeta for this simple broken line
            xp = np.linspace(x0, N * a, num=N)
            yp = eeta
            ksii = np.interp(x_values, xp, yp)
        
            #Force to correct mean (0) and standard deviation (dSigma_p) for this simple broken line
            ksii_mean = float(np.mean(ksii)) #mean of ksii
            ksii_std = float(np.std(ksii)) #standard deviation of ksii
            ksii_scaled = ((ksii - ksii_mean) / ksii_std) * sigma_p[int(level)] #scaling of array
            
            #Add values of this simple broken line to sum array
            bline_sum = bline_sum + ksii_scaled
            
            #Create matrix including all simple broken lines
            blines[:, p] = ksii_scaled
            
        #Set corrections for sum broken line and create final broken line
        bline_sum_corrected = bline_sum + mu_y #Set correct mean for the sum array (as we have 0 sum)
        bline_sum_exp = np.exp(bline_sum_corrected) #Exponentiate the sum array to get the final broken line
        
        #Accept sum brokenlines that fulfil desired conditions of mean and variance:
        #var_min < var_bline_sum_exp < var_max
        #bline_sum_exp[0] (first value) and bline_sum_exp[-1] (last value) have to be less than mar_max
        var_bline_sum_exp = np.var(bline_sum_exp) #variance of sum broken line
        
        if (var_min < var_bline_sum_exp < var_max and bline_sum_exp[0] < mar_max and bline_sum_exp[-1] < mar_max):
            
            #Create matrix including all accepted sum broken lines
            blines_final[:, noAccepted_BLs] = bline_sum_exp
            
            #If sum broken is accapted, save corresponding simple broken lines as csv
            # pd.DataFrame(blines).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/tests/blines_%d.csv" %(noAccepted_BLs))
            print("Created %d." %noAccepted_BLs)
            noAccepted_BLs = noAccepted_BLs + 1
        else:
            print(str(noAccepted_BLs) + ". not accepted.")
    
    #Save final broken lines as csv
    # pd.DataFrame(blines_final).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/tests/blines_final.csv")
    
    #Plot final broken lines and save as png and svg
    # fig = plt.figure()
    # for i in range(0, noBLs):
    #     plt.plot(blines_final[:,i], label=i)
    # plt.title("final broken lines")
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # fig.savefig("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/tests/blines_final.png")
    # fig.savefig("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/tests/blines_final.svg")
    
    return blines_final

##############################################################################
# CREATE BROKEN LINES

#create_broken_lines(mu_z, sigma2_z, H, q, a_zero, tStep, tSerieLength, noBLs, var_tol, mar_tol)
np.random.seed(1234)
areal_rainfall_bl = create_broken_lines(mu_z, sigma2_z, H, 0.8, a_zero, tStep, tSerieLength, 3, 1, 1)
Vxy_bl = create_broken_lines(mu_z_Vxy, sigma2_z_Vxy, H_Vxy, 0.8, a_zero_Vxy, tStep, tSerieLength, 3, 1, 1)
Vdir_deg_adj_bl = create_broken_lines(mu_z_Vdir_deg_adj, sigma2_z_Vdir_deg_adj, H_Vdir_deg_adj, 0.8, a_zero_Vdir_deg_adj, tStep, tSerieLength, 3, 1, 1)

# plt.figure()
# for i in range(0, 3):
#     plt.plot(areal_rainfall_bl[:,i], label=i)
# plt.plot(areal_rainfall_ts)

# plt.figure()
# for i in range(0, 3):
#     plt.plot(Vxy_bl[:,i], label=i)
# plt.plot(Vxy)

# plt.figure()
# for i in range(0, 3):
#     plt.plot(Vdir_deg_adj_bl[:,i], label=i)
# plt.plot(Vdir_deg_adj)

##############################################################################
# COMPUTE X- AND Y-DIRECTION ADVECTION VARIABLES 

#direction in radians
Vdir_rad_bl = (Vdir_deg_adj_bl / 360) * (2*np.pi)

#test plot
# plt.figure()
# for i in range(0, 3):
#     plt.plot(Vdir_rad_bl[:,i], label=i)
# plt.plot(Vdir_rad)

#x-component
Vx_bl = np.cos(Vdir_rad_bl) * Vxy_bl

# plt.figure()
# plt.plot(Vx_bl)

#y-component
Vy_bl = np.sin(Vdir_rad_bl) * Vxy_bl

# plt.figure()
# plt.plot(Vy_bl)

##############################################################################
# FUNCTION TO ESTIMATE PARAMETERS FOR POWER LAW FILTER

# #Estimation of beta1 and beta2 
# def estimate_fft_filter_parameters(field, **kwargs):
#     """Takes one or more 2d input fields and fits two spectral slopes, beta1 and beta2
    
#     This function is lines 52-196 from file fftgenerators.py in pysteps.
    
#     """
    
#     if len(field.shape) < 2 or len(field.shape) > 3:
#         raise ValueError("the input is not two- or three-dimensional array")
#     if np.any(~np.isfinite(field)):
#         raise ValueError("field contains non-finite values")

#     #defaults
#     win_fun = kwargs.get("win_fun", None)
#     model = kwargs.get("model", "power-law")
#     weighted = kwargs.get("weighted", False)
#     rm_rdisc = kwargs.get("rm_rdisc", False)
#     fft = kwargs.get("fft_method", "numpy")
#     if type(fft) == str:
#         fft_shape = field.shape if len(field.shape) == 2 else field.shape[1:]
#         fft = utils.get_method(fft, shape=fft_shape)

#     field = field.copy()

#     #remove rain/no-rain discontinuity
#     if rm_rdisc:
#         field[field > field.min()] -= field[field > field.min()].min() - field.min()

#     #dims
#     if len(field.shape) == 2:
#         field = field[None, :, :]
#     nr_fields = field.shape[0]
#     M, N = field.shape[1:]

#     if win_fun is not None:
#         tapering = utils.tapering.compute_window_function(M, N, win_fun)

#         #make sure non-rainy pixels are set to zero
#         field -= field.min(axis=(1, 2))[:, None, None]
#     else:
#         tapering = np.ones((M, N))

#     if model.lower() == "power-law":

#         #compute average 2D PSD
#         F = np.zeros((M, N), dtype=complex)
#         for i in range(nr_fields):
#             F += fft.fftshift(fft.fft2(field[i, :, :] * tapering))
#         F /= nr_fields
#         F = abs(F) ** 2 / F.size

#         #compute radially averaged 1D PSD
#         psd = utils.spectral.rapsd(F)
#         L = max(M, N)

#         #wavenumbers
#         if L % 2 == 0:
#             wn = np.arange(0, int(L / 2) + 1)
#         else:
#             wn = np.arange(0, int(L / 2))

#         #compute single spectral slope beta as first guess
#         if weighted:
#             p0 = np.polyfit(np.log(wn[1:]), np.log(psd[1:]), 1, w=np.sqrt(psd[1:]))
#         else:
#             p0 = np.polyfit(np.log(wn[1:]), np.log(psd[1:]), 1)
#         beta = p0[0]

#         #create the piecewise function with two spectral slopes beta1 and beta2
#         #and scaling break x0
#         def piecewise_linear(x, x0, y0, beta1, beta2):
#             return np.piecewise(
#                 x,
#                 [x < x0, x >= x0],
#                 [
#                     lambda x: beta1 * x + y0 - beta1 * x0,
#                     lambda x: beta2 * x + y0 - beta2 * x0,
#                 ],
#             )

#         #fit the two betas and the scaling break
#         p0 = [2.0, 0, beta, beta]  # first guess
#         bounds = (
#             [2.0, 0, -4, -4],
#             [5.0, 20, -1.0, -1.0],
#         )  #TODO: provide better bounds
#         if weighted:
#             p, e = optimize.curve_fit(
#                 piecewise_linear,
#                 np.log(wn[1:]),
#                 np.log(psd[1:]),
#                 p0=p0,
#                 bounds=bounds,
#                 sigma=1 / np.sqrt(psd[1:]),
#             )
#         else:
#             p, e = optimize.curve_fit(
#                 piecewise_linear, np.log(wn[1:]), np.log(psd[1:]), p0=p0, bounds=bounds
#             )   
#     return p

##############################################################################
# ESTIMATE FFT PARAMETERS

# #Log-transform the data
# R_log, metadata_log = pysteps.utils.transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)
# #Assign the fill value to all the Nans
# R_log[~np.isfinite(R_log)] = metadata_log["zerovalue"]

#Replace non-finite values with the minimum value
R2 = R.copy()
for i in range(R2.shape[0]):
    R2[i, ~np.isfinite(R[i, :])] = np.nanmin(R2[i, :])

# #Estimate fft parameters for each timestep of input event
# fft_pars = [ [] for _ in range(len(R2)) ]
# for i in range(0, len(R2)):
#     fft_pars[i] = estimate_fft_filter_parameters(R2[i])

# #Create time series of beta1 and beta2
# beta1_ts = np.zeros(len(fft_pars))
# beta2_ts = np.zeros(len(fft_pars))
# for i in range(0, len(fft_pars)):
#     beta1_ts[i] = fft_pars[i][2]
#     beta2_ts[i] = fft_pars[i][3]

# plt.figure()
# plt.scatter(areal_rainfall_bl[:,0], beta1_ts)

# plt.figure()
# plt.scatter(areal_rainfall_bl[:,0], beta2_ts)

###########
#https://pysteps.readthedocs.io/en/latest/auto_examples/plot_noise_generators.html
# Fit the parametric PSD to the observation

beta1s = np.zeros(len(R2))
beta2s = np.zeros(len(R2))
w0s = np.zeros(len(R2))

for i in range(0, len(R2)):
    Fp = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(R2[i])

    #Compute the observed and fitted 1D PSD
    L = np.max(Fp["input_shape"])
    if L % 2 == 0:
        wn = np.arange(0, int(L / 2) + 1)
    else:
        wn = np.arange(0, int(L / 2))
    iR_, freq = pysteps.utils.rapsd(R2[i], fft_method=np.fft, return_freq=True)
    f = np.exp(Fp["model"](np.log(wn), *Fp["pars"]))

    # Extract the scaling break in km, beta1 and beta2
    w0s[i] = L / np.exp(Fp["pars"][0])
    beta1s[i] = Fp["pars"][2]
    beta2s[i] = Fp["pars"][3]

##############################################################################
# CREATE GAUSSIAN BANDPASS FILTER

n_cascade_levels = 6
bp_filter = pysteps.cascade.bandpass_filters.filter_gaussian(R2[0].shape, n_cascade_levels)

##############################################################################
# ESTIMATE AR(2)-MODEL PARAMETERS

#The Lagrangian temporal evolution of each normalised cascade level is modelled using an AR(2) model
ar_order = 2

#Create variables that include all gammas and phis of all cascade levels for each time step
gamma_all = np.zeros([n_cascade_levels*2, len(R2)-2])
phi_all = np.zeros([n_cascade_levels*3, len(R2)-2])

#Compute the cascade decompositions of the input precipitation fields
# R_d = pysteps.cascade.decomposition.decomposition_fft(R2[0, :, :], bp_filter, normalize=True, compute_stats=True)

for i in range(0,len(R2)-2): 
    R_d = []
    
    for j in range(i,i+ar_order+1):
    #for j in range(ar_order + 1): 
    #for j in range(0, len(R2)):
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
        # gamma[k, :] = pysteps.timeseries.correlation.temporal_autocorrelation(R_c[k], domain="spectral", x_shape=R.shape[1:])
    
    # #
    # R_c = pysteps.nowcasts.utils.stack_cascades(R_d, n_cascade_levels, convert_to_full_arrays=False)
    
    # #
    # R_d = R_d[-1]
    
    #Adjust the lag-2 correlation coefficient to ensure that the AR(p) process is stationary
    for l in range(n_cascade_levels):
        gamma[l, 1] = pysteps.timeseries.autoregression.adjust_lag2_corrcoef2(gamma[l, 0], gamma[l, 1])
    
    #Print correlation coefficients
    #pysteps.nowcasts.utils.print_corrcoefs(gamma)
    
    #Estimate the parameters of the AR(p) model from the autocorrelation coefficients
    phi = np.empty((n_cascade_levels, ar_order + 1))
    for m in range(n_cascade_levels):
        phi[m, :] = pysteps.timeseries.autoregression.estimate_ar_params_yw(gamma[m, :])
    
    #Print AR parameters for cascade levels
    #pysteps.nowcasts.utils.print_ar_params(phi)
    
    # #Discard all except the p-1 last cascades because they are not needed for the AR(p) model
    # R_c = [R_c[i][-ar_order:] for i in range(n_cascade_levels)]
    
    #Fill gamma_all and phi_all after each timestep
    gamma_all[0:n_cascade_levels, i] = gamma[:, 0]
    gamma_all[n_cascade_levels:len(gamma_all), i] = gamma[:, 1]
    phi_all[0:n_cascade_levels, i] = phi[:, 0]
    phi_all[n_cascade_levels:(2*n_cascade_levels), i] = phi[:, 1]
    phi_all[(2*n_cascade_levels):len(phi_all), i] = phi[:, 2]

##############################################################################
# LINEAR FIT OF TAU_K AND L_K

#Mean of gamma1 for each cascade level
gamma1_mean = np.zeros([n_cascade_levels, 1])
for i in range(0, len(gamma1_mean)):
    gamma1_mean[i] = np.mean(abs(gamma_all[i,:]))

#Mean of gamma2 for each cascade level
gamma2_mean = np.zeros([n_cascade_levels, 1])
for i in range(len(gamma2_mean), len(gamma_all)):
    gamma2_mean[i-len(gamma1_mean)] = np.mean(abs(gamma_all[i,:]))

#Lagrangian temporal autocorrelation
#tau_k = np.zeros([7, 1])
tau_k = np.zeros([n_cascade_levels, 1])
for i in range(0, len(tau_k)):
    tau_k[i] = -timestep/np.log(gamma1_mean[i])

#Spatial scale in each cascade level in km
#https://gmd.copernicus.org/articles/12/4185/2019/, figure1
#264>=L_k>=1
#L_k = np.zeros([7, 1])
L_k = np.zeros([n_cascade_levels, 1])
L_k[0] = 264
for i in range(1, len(L_k)):
    L_k[i] = (L_k[0]/2)/(bp_filter["central_wavenumbers"][i])

#Estimate tlen_a and tlen_b based on linear fit of tau_k and L_k
slope_tlen, intercept_tlen, r_value_tlen, p_value_tlen, std_err_tlen = sp.linregress(np.log(L_k[0:n_cascade_levels-1,0]), np.log(tau_k[0:n_cascade_levels-1,0]))
y_values_tlen = slope_tlen*np.log(L_k[0:n_cascade_levels-1])+intercept_tlen

plt.figure()
plt.plot(np.log(L_k[0:n_cascade_levels,0]), np.log(tau_k[0:n_cascade_levels,0]), marker='s')
plt.plot(np.log(L_k[0:n_cascade_levels-1]), y_values_tlen)

tlen_b = slope_tlen
tlen_a = np.exp(intercept_tlen)

#Estimate tlen_a and tlen_b based on linear fit of tau_k and L_k
slope_tlen_c, intercept_tlen_c, r_value_tlen_c, p_value_tlen_c, std_err_tlen_c = sp.linregress(np.log(gamma1_mean[0:n_cascade_levels-1,0]), np.log(gamma2_mean[0:n_cascade_levels-1,0]))
y_values_tlen_c = slope_tlen_c*np.log(gamma1_mean[0:n_cascade_levels-1,0])+intercept_tlen_c

plt.figure()
plt.plot(np.log(gamma1_mean[0:n_cascade_levels,0]), np.log(gamma2_mean[0:n_cascade_levels,0]), marker='s')
plt.plot(np.log(gamma1_mean[0:n_cascade_levels-1]), y_values_tlen_c)
#TODO: need to add zero forcing?

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

#Broken line parameters
q = 0.8
noBLs = 1
var_tol = 1
mar_tol = 1

data_temp = [mu_z, sigma2_z, H, q, a_zero, tStep, tSerieLength, noBLs, var_tol, mar_tol, mu_z_Vxy, sigma2_z_Vxy, H_Vxy, a_zero_Vxy, mu_z_Vdir_deg_adj, sigma2_z_Vdir_deg_adj, H_Vdir_deg_adj, a_zero_Vdir_deg_adj]
bl_params = pd.DataFrame(data_temp, index=['mu_z', 'sigma2_z', 'H', 'q', 'a_zero', 'tStep', 'tSerieLength', 'noBLs', 'var_tol', 'mar_tol', 'mu_z_Vxy', 'sigma2_z_Vxy', 'H_Vxy', 'a_zero_Vxy', 'mu_z_Vdir_deg_adj', 'sigma2_z_Vdir_deg_adj', 'H_Vdir_deg_adj', 'a_zero_Vdir_deg_adj'])
#bl_params.rename(columns = {0 : 'param'}, inplace = True)
#pd.DataFrame(bl_params).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters/bl_params.csv")
pd.DataFrame(bl_params).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters_log/bl_params.csv")

#Broken lines
#pd.DataFrame(areal_rainfall_bl).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters/areal_rainfall_bl.csv")
#pd.DataFrame(Vx_bl).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters/Vx_bl.csv")
#pd.DataFrame(Vy_bl).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters/Vy_bl.csv")
pd.DataFrame(areal_rainfall_bl).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters_log/areal_rainfall_bl.csv")
pd.DataFrame(Vx_bl).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters_log/Vx_bl.csv")
pd.DataFrame(Vy_bl).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters_log/Vy_bl.csv")

#FFT parameters
data_temp = [beta1s, beta2s, w0s]
fft_params = pd.DataFrame(data_temp, index=['beta1', 'beta2', 'w0'])
#pd.DataFrame(fft_params).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters/fft_params.csv")
pd.DataFrame(fft_params).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters_log/fft_params.csv")

#AR parameters
ar_params1 = pd.DataFrame(gamma1_mean)
ar_params1[1] = gamma2_mean
ar_params1[2] = tau_k
ar_params1[3] = L_k
ar_params1.rename(columns = {0 : 'gamma1_mean', 1 : 'gamma2_mean', 2 : 'tau_k', 3 : 'L_k'}, inplace = True)
#pd.DataFrame(ar_params1).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters/ar_params1.csv")
pd.DataFrame(ar_params1).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters_log/ar_params1.csv")

data_temp = [tlen_a, tlen_b, tlen_c]
ar_params2 = pd.DataFrame(data_temp, index=['tlen_a', 'tlen_b', 'tlen_c'])
#ar_params2.rename(columns = {0 : 'param'}, inplace = True)
#pd.DataFrame(ar_params2).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters/ar_params2.csv")
pd.DataFrame(ar_params2).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters_log/ar_params2.csv")

#Fitted curves
data_temp = [std_fit, war_fit, beta1_fit, beta2_fit, w0_fit]
curves = pd.DataFrame(data_temp, index=['std', 'war', 'beta1', 'beta2', 'w0'])
#curves.rename(columns = {0 : 'x2', 1 : 'x', 2 : '-'}, inplace = True)
#pd.DataFrame(curves).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters/curves.csv")
pd.DataFrame(curves).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters_log/curves.csv")

##############################################################################
# ALL NEEDED PARAMETERS IN ONE CSV-FILES

data_temp = [w0_fit[0], w0_fit[1], w0_fit[2], beta1_fit[0], beta1_fit[1], beta1_fit[2], beta2_fit[0], beta2_fit[1], beta2_fit[2], 
             tlen_a, tlen_b, tlen_c, std_fit[0], std_fit[1], std_fit[2], war_fit[0], war_fit[1], war_fit[2], 
             mu_z, sigma2_z, H, a_zero, mu_z_Vxy, sigma2_z_Vxy, H_Vxy, a_zero_Vxy, mu_z_Vdir_deg_adj, sigma2_z_Vdir_deg_adj, H_Vdir_deg_adj, a_zero_Vdir_deg_adj, 
             q, noBLs, var_tol, mar_tol]

all_params = pd.DataFrame(data_temp, index=['a_w0', 'b_w0', 'c_w0', 'a_1', 'b_1', 'c_1', 'a_2', 'b_2', 'c_2', 'tlen_a', 'tlen_b', 'tlen_c', 'a_v', 'b_v', 'c_v', 'a_war', 'b_war', 'c_war', 
                                           'mu_z', 'sigma2_z', 'h_val_z', 'a_zero_z', 'mu_vmag', 'sigma2_vmag', 'h_val_vmag', 'a_zero_vmag', 'mu_vdir', 'sigma2_vdir', 'h_val_vdir', 'a_zero_vdir', 
                                           'q', 'no_bls', 'var_tol', 'mar_tol'])
pd.DataFrame(all_params).to_csv("//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/parameters_log/all_params.csv")
