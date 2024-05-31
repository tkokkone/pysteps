# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 09:26:30 2022

@author: lindgrv1
"""

##############################################################################
# IMPORT PACKAGES

import os
import numpy as np
from numpy import genfromtxt
import math
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime
import pysteps
from pysteps import noise
from pysteps.utils import rapsd
from pysteps.visualization import plot_spectrum1d
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import random
import time

##############################################################################
# TIME USED FOR PARAMETER ESTIMATION: Start timer

run_start_0 = time.perf_counter()

##############################################################################
# CREATE FUNCTION TO ROUND A FLOAT UP TO THE NEAREST 0.5

def round_up_to_nearest_half_int(num):
    return math.ceil(num * 2) / 2

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
    print(iN)
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

        if (var_min < var_bline_sum_exp and var_bline_sum_exp < var_max and bline_sum_exp[0] < mar_max and bline_sum_exp[-1] < mar_max):

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
# READ IN OSAPOL ADVECTIONS AND STATISTICS

# indir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simulations_pysteps/Event3_new"
indir = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Event1_new"
osapol_adv = genfromtxt(fname=os.path.join(indir, "advection_tss.csv"), delimiter=',', skip_header=1)
osapol_adv = np.delete(osapol_adv, 0, axis=1)
vx_osapol = osapol_adv[0,:]
vy_osapol = osapol_adv[1,:]

osapol_stats = genfromtxt(fname=os.path.join(indir, "data_tss.csv"), delimiter=',', skip_header=1)
osapol_stats = np.delete(osapol_stats, 0, axis=1)

##############################################################################
# OUTDIR

outdir = os.path.join(indir, "Model_performance_500")
if not os.path.exists(outdir):
    os.makedirs(outdir)

out_figs = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Figures"
# plt.savefig(os.path.join(out_figs,"figure_1.pdf"), format="pdf", bbox_inches="tight")

##############################################################################
# READ IN BROKENLINE TIME SERIES FOR EACH REALIZATION

data_dir_location = os.path.join(indir, "Simulations")
dir_list = os.listdir(data_dir_location)

vx = []
vy = []
r_mean = []

for i in range(len(dir_list)):

    realization_dir = os.path.join(data_dir_location, dir_list[i])
    
    brokenlines = genfromtxt(fname=os.path.join(realization_dir, "sim_brokenlines.csv"), delimiter=',', skip_header=1)
    brokenlines = np.delete(brokenlines, 0, axis=1)
    
    vx.append(brokenlines[4,:])
    vy.append(brokenlines[5,:])
    r_mean.append(brokenlines[0,:])

vx_array = np.vstack(vx)
vy_array = np.vstack(vy)
r_mean_array = np.vstack(r_mean)

##############################################################################
# ADVECTION PLOT - FIGURE 3.d (Niemi et al. 2016)

#find axis-limits for the plot
vx_osapol_max = abs(np.max(vx_osapol))
vy_osapol_max = abs(np.max(vy_osapol))
vx_osapol_min = abs(np.min(vx_osapol))
vy_osapol_min = abs(np.min(vy_osapol))
vx_max = abs(np.max(vx_array))
vy_max = abs(np.max(vy_array))
vx_min = abs(np.min(vx_array))
vy_min = abs(np.min(vy_array))

adv_plot_lims = np.max([vx_osapol_max, vy_osapol_max, vx_osapol_min, vy_osapol_min, vx_max, vy_max, vx_min, vy_min])
if adv_plot_lims == 0:
    adv_plot_lims += 0.5
else:
    adv_plot_lims = round_up_to_nearest_half_int(adv_plot_lims)
    
# adv_plot_lims = adv_plot_lims + 0.5 #event3 = 9.0
adv_plot_lims = 8.5 #event1

#Create scatter plot
plt.figure()
plt.scatter(vx_array, vy_array, color="Red", label="Ensemble members")
plt.scatter(vx_osapol, vy_osapol, color="Black", label="Estimated from observed event")
# plt.scatter(vx_array[0], vy_array[0], c=np.arange(len(vx_array[0])), cmap="Reds", label="Ensemble members")
# plt.scatter(vx_osapol, vy_osapol, c=np.arange(len(vx_osapol)), cmap="gray", label="Estimated from observed event")
plt.xlim(-adv_plot_lims, adv_plot_lims)
plt.ylim(adv_plot_lims, -adv_plot_lims)
# plt.xticks(np.arange(-adv_plot_lims, adv_plot_lims, 1.0))
plt.xlabel("West-East")
plt.ylabel("South-North")
plt.legend()
plt.title("Advection velocities to southern and eastern directions \nfor all the time steps of the ensemble members")
plt.savefig(os.path.join(outdir, "fig_d1.png"))

# TODO: Add marginal distributions of the velocities
#https://www.geeksforgeeks.org/scatter-plot-with-marginal-histograms-in-python-with-seaborn/

#definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

colors = len(vx_array[0]) * ["Gray"]
colors2 = len(vx_array[0]) * ["none"]

# start with a rectangular Figure
plt.figure(figsize=(8, 8))

ax_scatter = plt.axes(rect_scatter)
ax_scatter.tick_params(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', right=True, labelbottom=False)
ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', right=True, labelleft=False)

circle1 = plt.Circle((0, 0), 1, color="Black", fill=False, linestyle="--", linewidth=0.5)
circle2 = plt.Circle((0, 0), 2, color="Black", fill=False, linestyle="--", linewidth=0.5)
circle3 = plt.Circle((0, 0), 3, color="Black", fill=False, linestyle="--", linewidth=0.5)
circle4 = plt.Circle((0, 0), 4, color="Black", fill=False, linestyle="--", linewidth=0.5)
circle5 = plt.Circle((0, 0), 5, color="Black", fill=False, linestyle="--", linewidth=0.5)
circle6 = plt.Circle((0, 0), 6, color="Black", fill=False, linestyle="--", linewidth=0.5)
circle7 = plt.Circle((0, 0), 7, color="Black", fill=False, linestyle="--", linewidth=0.5)
circle8 = plt.Circle((0, 0), 8, color="Black", fill=False, linestyle="--", linewidth=0.5)
ax_scatter.add_patch(circle1)
ax_scatter.add_patch(circle2)
ax_scatter.add_patch(circle3)
ax_scatter.add_patch(circle4)
ax_scatter.add_patch(circle5)
ax_scatter.add_patch(circle6)
ax_scatter.add_patch(circle7)
ax_scatter.add_patch(circle8)

ax_scatter.axhline(y=0.0, color="Black", linestyle="--", linewidth=0.5)
ax_scatter.axvline(x=0.0, color="Black", linestyle="--", linewidth=0.5)

# the scatter plot:
ax_scatter.scatter(vx_array, vy_array, marker=".", color="Gray", alpha=0.1, label="ensemble members") #, marker="."
ax_scatter.scatter(vx_osapol, vy_osapol, marker=".", color="Blue", label="osapol")

# now determine nice limits by hand:
binwidth = 0.10
lim = np.ceil(np.abs([vx_array, vy_array]).max() / binwidth) * binwidth
lim = adv_plot_lims + binwidth 
ax_scatter.set_xlim((-lim, lim))
ax_scatter.set_ylim((lim, -lim))

bins = np.arange(-lim, lim + binwidth, binwidth)
# ax_histx.hist(vx_array, bins=bins, color=colors, density=True, stacked=True)
# ax_histy.hist(vy_array, bins=bins, color=colors, orientation='horizontal', density=True, stacked=True)
n_x, bins_x, patches_x = ax_histx.hist(vx_array, bins=bins, color=colors2, density=False, stacked=True)
n_y, bins_y, patches_y = ax_histy.hist(vy_array, bins=bins, color=colors2, density=False, stacked=True, orientation="horizontal")

bin_centers_x = 0.5*(bins_x[1:]+bins_x[:-1])
bin_centers_y = 0.5*(bins_y[1:]+bins_y[:-1])

ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histy.set_ylim(ax_scatter.get_ylim())

ax_histx.plot(bin_centers_x, n_x[-1,:], color="Gray")
ax_histy.plot(n_y[-1,:], -bin_centers_y, color="Gray")

ax_histx.fill_between(bin_centers_x, n_x[-1,:], 0, color="Gray") #, alpha=0.1
ax_histy.fill_between(n_y[-1,:], -bin_centers_y, -lim*2, color="Gray") #, alpha=0.1


ax_histx.set_ylim((0,1+binwidth))
ax_histy.set_xlim((0,1+binwidth))
ax_histx.set_xlim(((-lim, lim)))
ax_histy.set_ylim(((-lim, lim)))

ax_histy.set_xlim(((0, 7500)))
ax_histy.set_xticks((0,1500,3000,4500,6000,7500))
ax_histy.set_xticklabels(ax_histy.get_xticks(), rotation = 90)
ax_histx.set_ylim(((0, 7500)))
ax_histx.set_yticks((0,1500,3000,4500,6000,7500))

ax_histy.axvline(x=1500, color="Black", linestyle="--", linewidth=0.5)
ax_histy.axvline(x=3000, color="Black", linestyle="--", linewidth=0.5)
ax_histy.axvline(x=4500, color="Black", linestyle="--", linewidth=0.5)
ax_histy.axvline(x=6000, color="Black", linestyle="--", linewidth=0.5)

ax_histx.axhline(y=1500, color="Black", linestyle="--", linewidth=0.5)
ax_histx.axhline(y=3000, color="Black", linestyle="--", linewidth=0.5)
ax_histx.axhline(y=4500, color="Black", linestyle="--", linewidth=0.5)
ax_histx.axhline(y=6000, color="Black", linestyle="--", linewidth=0.5)

ax_scatter.set_xlabel("West-East [m/s]")
ax_scatter.set_ylabel("South-North [m/s]")
ax_histx.set_ylabel("Count [-]")
ax_histy.set_xlabel("Count [-]")

ax_scatter.legend()

plt.show()
plt.savefig(os.path.join(outdir, "fig_d12.png"))

plt.savefig(os.path.join(out_figs,"figure_2g.pdf"), format="pdf", bbox_inches="tight")

##############################################################################
# 1D POWER SPECTRUMS

R_powers_means = []
R_freqs_means = []

for i in range(len(dir_list)):
    Fps = []
    R_powers = []
    R_freqs = []
    R_powers_mean = []
    R_freqs_mean = []
    
    # chosen_realization = dir_list[0]
    chosen_realization = dir_list[i]
    data_dir2 = os.path.join(data_dir_location, chosen_realization, "Event_tiffs")
    files = os.listdir(data_dir2)
    
    event_sim_array  = []
    for i in range(len(files)):
        temp_raster = rasterio.open(os.path.join(data_dir2, f"test_{i}.tif"))
        temp_array = temp_raster.read(1)
        event_sim_array.append(temp_array)
        if i == 0:
            event_affine = temp_raster.transform  
    event_sim_array = np.concatenate([event_sim_array_[None, :, :] for event_sim_array_ in event_sim_array])
    
    nx_field = 512
    scale_break = 18  # constant scale break in km
    scale_break_wn = np.log(nx_field/scale_break)
    
    for k in range(len(event_sim_array)):
        Fp = noise.fftgenerators.initialize_param_2d_fft_filter(event_sim_array[k], scale_break=scale_break_wn)
        R_, freq = rapsd(event_sim_array[k], fft_method=np.fft, return_freq=True)
        Fps.append(Fp)
        R_powers.append(R_)
        R_freqs.append(freq)
    
    R_powers = np.vstack(R_powers)
    R_freqs = np.vstack(R_freqs)
    
    for i in range(R_powers.shape[1]):
        R_powers_mean.append(np.mean(R_powers[:,i]))
        R_freqs_mean.append(np.mean(R_freqs[:,i]))
        
    R_powers_means.append(R_powers_mean)
    R_freqs_means.append(R_freqs_mean)
    
R_powers_means = np.vstack(R_powers_means)    
R_freqs_means = np.vstack(R_freqs_means)

R_powers_means_mean = []
R_freqs_means_mean = []
for i in range(R_powers_means.shape[1]):
    R_powers_means_mean.append(np.mean(R_powers_means[:,i]))
    R_freqs_means_mean.append(np.mean(R_freqs_means[:,i]))
    
#Osapol
R_powers_osapol = []
R_freqs_osapol = []
R_powers_osapol_mean = []
R_freqs_osapol_mean = []

# 1. last radar image: 201306271955 -> number of previous files: 141
# 3. last radar image: 201310290345 -> number of previous files: 115
# 6. last radar image: 201408071800 -> number of previous files: 97
date = datetime.strptime("201310290345", "%Y%m%d%H%M") #last radar image of the event
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
fns = pysteps.io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=115)
#Select the importer
importer = pysteps.io.get_method(importer_name, "importer")
#Read the radar composites
R, quality, metadata = pysteps.io.read_timeseries(fns, importer, **importer_kwargs)
del quality  #delete quality variable because it is not used
#Add unit to metadata
metadata["unit"] = "mm/h"
#Values less than threshold to zero
R[R<0.1] = 0
metadata["zerovalue"] = 0
metadata["threshold"] = 0.1
a_R=223
b_R=1.53
R, metadata = pysteps.utils.conversion.to_reflectivity(R, metadata, zr_a=a_R, zr_b=b_R)
R2 = R.copy()
#Replace non-finite values with the minimum value
for i in range(R.shape[0]):
    R[i, ~np.isfinite(R[i, :])] = np.nanmin(R[i, :])

for k in range(len(R)):
    R_, freq = rapsd(R[k], fft_method=np.fft, return_freq=True)
    R_powers_osapol.append(R_)
    R_freqs_osapol.append(freq)
    
R_powers_osapol = np.vstack(R_powers_osapol)
R_freqs_osapol = np.vstack(R_freqs_osapol)

for i in range(R_powers_osapol.shape[1]):
    R_powers_osapol_mean.append(np.mean(R_powers_osapol[:,i]))
    R_freqs_osapol_mean.append(np.mean(R_freqs_osapol[:,i]))

#Save csv
R_powers_means_csv = np.vstack(R_powers_means)
R_powers_means_csv = np.hstack(R_powers_means_csv)
R_freqs_means_csv = np.vstack(R_freqs_means)
R_freqs_means_csv = np.hstack(R_freqs_means_csv)
R_powers_means_mean_csv = np.vstack(R_powers_means_mean)
R_freqs_means_mean_csv = np.vstack(R_freqs_means_mean)
R_powers_osapol_csv = np.hstack(R_powers_osapol)
R_freqs_osapol_csv = np.hstack(R_freqs_osapol)
R_powers_osapol_mean_csv = np.vstack(R_powers_osapol_mean)
R_freqs_osapol_mean_csv = np.vstack(R_freqs_osapol_mean)

data_temp_powers = [R_powers_means_csv, R_freqs_means_csv]
data_temp_save = pd.DataFrame(data_temp_powers, index=['pow', 'freq'])
pd.DataFrame(data_temp_save).to_csv(os.path.join(outdir, "fig_a_pows_freqs.csv"))

data_temp_powers_mean = [R_powers_means_mean_csv[:,0], R_freqs_means_mean_csv[:,0]]
data_temp_save = pd.DataFrame(data_temp_powers_mean, index=['pow', 'freq'])
pd.DataFrame(data_temp_save).to_csv(os.path.join(outdir, "fig_a_pows_freqs_mean.csv"))

data_temp_osapol_powers = [R_powers_osapol_csv, R_freqs_osapol_csv]
data_temp_save = pd.DataFrame(data_temp_osapol_powers, index=['pow', 'freq'])
pd.DataFrame(data_temp_save).to_csv(os.path.join(outdir, "fig_a_osapol_pows_freqs.csv"))

data_temp_osapol_powers_mean = [R_powers_osapol_mean_csv[:,0], R_freqs_osapol_mean_csv[:,0]]
data_temp_save = pd.DataFrame(data_temp_osapol_powers_mean, index=['pow', 'freq'])
pd.DataFrame(data_temp_save).to_csv(os.path.join(outdir, "fig_a_osapol_pows_freqs_mean.csv"))

##############################################################################
# READ IN ABOVE CSVS
 
pows_freqs = genfromtxt(fname=os.path.join(outdir, "fig_a_pows_freqs.csv"), delimiter=',', skip_header=1)

p_pows_mean = pows_freqs[0,:]
p_pows_mean = np.delete(p_pows_mean, 0)
p_pows_mean = np.vstack(p_pows_mean)
p_pows_mean = np.transpose(p_pows_mean)
p_pows_mean = np.split(p_pows_mean, 500, axis=1)
p_pows_mean = np.vstack(p_pows_mean)
R_powers_means = p_pows_mean

p_freqs_mean = pows_freqs[1,:]
p_freqs_mean= np.delete(p_freqs_mean, 0)
p_freqs_mean = np.vstack(p_freqs_mean)
p_freqs_mean = np.transpose(p_freqs_mean)
p_freqs_mean = np.split(p_freqs_mean, 500, axis=1)
p_freqs_mean = np.vstack(p_freqs_mean)
R_freqs_means = p_freqs_mean

pows_freqs_means = genfromtxt(fname=os.path.join(outdir, "fig_a_pows_freqs_mean.csv"), delimiter=',', skip_header=1)
pows_freqs_means = np.delete(pows_freqs_means, 0, axis=1)
R_powers_means_mean = pows_freqs_means[0,:]
R_freqs_means_mean = pows_freqs_means[1,:]

##############################################################################
# PLOT 1D POWER SPECTRUMS - FIGURE 3.a (Niemi et al. 2016)

#Simulated realizations
fig, ax = plt.subplots()
plot_scales = [256, 128, 64, 32, 16, 8, 4, 2]
for l in range(len(R_powers_means)):
    plot_spectrum1d(
        R_freqs_means[l],
        R_powers_means[l],
        x_units="km",
        y_units="dBZ",
        color="k",
        ax=ax,
        label="",
        wavelength_ticks=plot_scales,
    )
plot_spectrum1d(
    R_freqs_means_mean,
    R_powers_means_mean,
    x_units="km",
    y_units="dBZ",
    color="r",
    ax=ax,
    label="",
    wavelength_ticks=plot_scales,
)
# plt.legend()
plt.show()
ax.set_title("Ensemble members")
plt.savefig(os.path.join(outdir, "fig_a1.png"))

#Osapol
fig, ax = plt.subplots()
plot_scales = [256, 128, 64, 32, 16, 8, 4, 2]
for l in range(len(R_freqs_osapol)):
    plot_spectrum1d(
        R_freqs_osapol[l],
        R_powers_osapol[l],
        x_units="km",
        y_units="dBZ",
        color="k",
        ax=ax,
        label="",
        wavelength_ticks=plot_scales,
    )
plot_spectrum1d(
    R_freqs_osapol_mean,
    R_powers_osapol_mean,
    x_units="km",
    y_units="dBZ",
    color="b",
    ax=ax,
    label="",
    wavelength_ticks=plot_scales,
)
# plt.legend()
plt.show()
ax.set_title("OSAPOL")
plt.savefig(os.path.join(outdir, "fig_a2.png"))

#Realizations vs. Osapol
fig, ax = plt.subplots()
plot_scales = [256, 128, 64, 32, 16, 8, 4, 2]
for l in range(len(R_powers_means)):
    if l == 0:
        plot_spectrum1d(
            R_freqs_means[l],
            R_powers_means[l],
            x_units="km",
            y_units="dBZ",
            color="Gray",
            ax=ax,
            label="ensemble members",
            wavelength_ticks=plot_scales,
            alpha=0.1,
        )
    else:
        plot_spectrum1d(
            R_freqs_means[l],
            R_powers_means[l],
            x_units="km",
            y_units="dBZ",
            color="Gray",
            ax=ax,
            wavelength_ticks=plot_scales,
            alpha=0.1,
        )
plot_spectrum1d(
    R_freqs_means_mean,
    R_powers_means_mean,
    x_units="km",
    y_units="dBZ",
    color="r",
    ax=ax,
    label="ensemble mean",
    wavelength_ticks=plot_scales,
)
plot_spectrum1d(
    R_freqs_osapol_mean,
    R_powers_osapol_mean,
    x_units="km",
    y_units="dBZ",
    color="b",
    ax=ax,
    label="osapol",
    wavelength_ticks=plot_scales,
)
plt.legend()
ax.set_ylim(-1, 61)
# ax.set_title("Means")
plt.show()
plt.savefig(os.path.join(outdir, "fig_a7.png"))

plt.savefig(os.path.join(out_figs,"figure_2b.pdf"), format="pdf", bbox_inches="tight")

##############################################################################
# PLOT OSAPOL_MEAN TS and BROKENLINES
R_mean_osapol = []
for i in range (len(fns[1])):
    R_mean_osapol.append(np.nanmean(R2[i]))

plt.figure()
for i in range(len(r_mean_array)):
    if i == 0:
        plt.plot(r_mean_array[i,:], color="Gray", alpha=0.1, label="ensemble members")
    else:
        # plt.plot(r_mean_array[i,:], color="Black")
        plt.plot(r_mean_array[i,:], color="Gray", alpha=0.1)
plt.plot(R_mean_osapol, color="Blue", label="osapol")
plt.ylim(0, 30)
plt.xlabel("Time [min]")
plt.ylabel("Reflectivity [dBZ]")
plt.legend()
plt.savefig(os.path.join(outdir, "fig_timeseries8.png"))

plt.savefig(os.path.join(out_figs,"figure_2f.pdf"), format="pdf", bbox_inches="tight")

plt.figure()
for i in range(len(r_mean_array)):
    plt.plot(r_mean_array[i,:])
plt.plot(R_mean_osapol, color="Black", linestyle="--")
# plt.ylim(0, 30)
plt.savefig(os.path.join(outdir, "fig_timeseries2.png"))

##############################################################################

autocors_osapol = sm.tsa.acf(R_mean_osapol, nlags=(len(R_mean_osapol)))

autocors_ensemble = []
for i in range(len(r_mean_array)):
    autocors_ensemble.append(sm.tsa.acf(r_mean_array[i], nlags=(len(r_mean_array[i]))))
autocors_ensemble = np.vstack(autocors_ensemble)

autocors_ensemble_mean = []
for i in range(autocors_ensemble.shape[1]):
    autocors_ensemble_mean.append(np.mean(autocors_ensemble[:,i]))

plt.figure()
for i in range(len(autocors_ensemble)):
    if i == 0:
        plt.plot(autocors_ensemble[i,:], color="Gray", label="ensemble members", alpha=0.1)
    else:
        plt.plot(autocors_ensemble[i,:], color="Gray", alpha=0.1)
plt.plot(autocors_ensemble_mean[:], color="Red", label="ensemble mean")
plt.plot(autocors_osapol[:], color="Blue", label="osapol")
plt.ylim(-0.8, 1.1)
plt.axhline(y = 0, color = 'black', linestyle = '-', linewidth=0.5)
plt.legend()
plt.xlabel("Time [min]")
plt.ylabel("Correlation [-]")
# plt.title("Autocors")
plt.savefig(os.path.join(outdir, "fig_b3.png"))
# plt.savefig(os.path.join(outdir, "fig_b1_scaled.png"))
# plt.savefig(os.path.join(outdir, "fig_b1_scaled_line.png"))

plt.savefig(os.path.join(out_figs,"figure_2d.pdf"), format="pdf", bbox_inches="tight")

plt.figure()
for i in range(len(autocors_ensemble)):
    if i == 0:
        plt.plot(autocors_ensemble[i,0:18], color="Gray", label="realizations")
    else:
        plt.plot(autocors_ensemble[i,0:18], color="Gray")
plt.plot(autocors_ensemble_mean[0:18], color="Red", label="realization mean")
plt.plot(autocors_osapol[0:18], color="Blue", label="osapol")
# plt.ylim(0, 1.1)
plt.legend()
plt.title("Autocors")
plt.savefig(os.path.join(outdir, "fig_b2.png"))
# plt.savefig(os.path.join(outdir, "fig_b2_scaled.png"))

##############################################################################
# TEST: 
all_params = genfromtxt(fname=os.path.join(indir, "all_params.csv"), delimiter=',', skip_header=1, usecols=1)

# Broken line parameters for field mean
mu_z = all_params[18] #mean of mean areal reflectivity over the simulation period
sigma2_z = all_params[19] #variance of mean areal reflectivity over the simulation period
h_val_z = all_params[20] #structure function exponent
q_val_z = all_params[30] #scale ratio between levels n and n+1 (constant) [-]
a_zero_z = all_params[21] #time series decorrelation time [min]
#170
no_bls = 100 #number of broken lines
var_tol_z = all_params[32] #acceptable tolerance for variance as ratio of input variance [-]
mar_tol_z = all_params[33]
# event 1. last radar image: 201306271955 -> number of previous files: 141
# event 3. last radar image: 201310290345 -> number of previous files: 115
# event 6. last radar image: 201408071800 -> number of previous files: 97
# -> Number of timesteps in simulations:
    # event 1: 142
    # event 3: 116
    # event 6: 98
n_timesteps = 98  # number of timesteps
timestep = 5 

# # Create the field mean for the requested number of simulation time steps
# seed_bl = 8271
# np.random.seed(seed_bl) #set seed
# test_r_mean = create_broken_lines(mu_z, sigma2_z, h_val_z, q_val_z, a_zero_z, timestep, (n_timesteps-1) * timestep, no_bls, var_tol_z, mar_tol_z)

# test_r_mean_t = np.transpose(test_r_mean)

# test_autocors_ensemble = []
# for i in range(len(test_r_mean_t)):
#     test_autocors_ensemble.append(sm.tsa.acf(test_r_mean_t[i], nlags=(len(test_r_mean_t[i]))))
# test_autocors_ensemble = np.vstack(test_autocors_ensemble)

# test_autocors_ensemble_mean = []
# for i in range(test_autocors_ensemble.shape[1]):
#     test_autocors_ensemble_mean.append(np.mean(test_autocors_ensemble[:,i]))
    
# # plt.figure()
# # for i in range(len(test_autocors_ensemble)):
# #     plt.plot(test_autocors_ensemble[i,:], color="Gray")
# # plt.plot(test_autocors_ensemble_mean[:], color="Red")
# # plt.plot(autocors_osapol[:], color="Blue")

# plt.figure()
# for i in range(len(test_autocors_ensemble)):
#     plt.plot(test_autocors_ensemble[i,0:20], color="Gray")
# plt.plot(test_autocors_ensemble_mean[0:20], color="Red")
# plt.plot(autocors_osapol[0:20], color="Blue")

# plt.figure()
# plt.plot(test_r_mean[:,0:10])
# plt.plot(R_mean_osapol, color="Red", linestyle="--", label="osapol")

##############################################################################
# A_ZERO FITTING:
# This is done already for parameter estimation to find optimal a_zero value 
a_zeros = []
sses = []
test_autocors_ensemble_means = []

seed_bl = random.randint(1, 10000)
np.random.seed(seed_bl)  #set seed
for i in range(int(a_zero_z/timestep), (len(R)-1)): #a_zero_z = 90 (len(R)-1)
    azero = (i*5)
    
    test_r_mean = create_broken_lines(mu_z, sigma2_z, h_val_z, q_val_z, azero, timestep, (n_timesteps-1) * timestep, no_bls, var_tol_z, mar_tol_z)
    
    test_r_mean_t = np.transpose(test_r_mean)
    
    test_autocors_ensemble = []
    for i in range(len(test_r_mean_t)):
        test_autocors_ensemble.append(sm.tsa.acf(test_r_mean_t[i], nlags=(len(test_r_mean_t[i]))))
    test_autocors_ensemble = np.vstack(test_autocors_ensemble)
    
    test_autocors_ensemble_mean = []
    for i in range(test_autocors_ensemble.shape[1]):
        test_autocors_ensemble_mean.append(np.mean(test_autocors_ensemble[:,i]))

    #SSE
    sse = sum((test_autocors_ensemble_mean[0:int(a_zero_z/timestep)] - autocors_osapol[0:int(a_zero_z/timestep)])**2.0)
    print(azero, sse)
    a_zeros.append(azero)
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

#Plot
plt.figure()
for i in range(len(test_autocors_ensemble_means)):
    plt.plot(test_autocors_ensemble_means[i][0:int(a_zero_z/timestep)], color="Gray")
plt.plot(test_autocors_ensemble_means[a_zero_nro][0:int(a_zero_z/timestep)], color="Red", label=f"a_zero = {a_zero_min} ({a_zero_z}) min")
plt.plot(autocors_osapol[0:int(a_zero_z/timestep)], color="Blue", label="osapol")
plt.legend()
plt.title(f"Optimized a_zero vs. osapol (SSE = {a_zero_vs_sse[:,2][a_zero_vs_sse[:,2] == np.min(a_zero_vs_sse[:,2])]}, seed = {seed_bl})")
plt.savefig(os.path.join(outdir, f"optimized_azero_seed{seed_bl}.png"))

#save csv
data_temp = [a_zero_vs_sse[:,0], a_zero_vs_sse[:,1], a_zero_vs_sse[:,2]]
data_temp_save = pd.DataFrame(data_temp, index=['nro', 'azero', 'sse'])
pd.DataFrame(data_temp_save).to_csv(os.path.join(outdir, f"azero_autocors_seed{seed_bl}.csv"))

csv_autocors = np.hstack(test_autocors_ensemble_means)
data_temp2 = [csv_autocors]
data_temp2_save = pd.DataFrame(data_temp2, index=['autocor_means'])
pd.DataFrame(data_temp2_save).to_csv(os.path.join(outdir, f"autocor_means_seed{seed_bl}.csv"))
csv_autocors = np.vstack(test_autocors_ensemble_means)

##############################################################################
# #TEST: same seed, and changing a_zero and H

# # bl0 = []
# # bl1 = []
# # bl2 = []

# seed_bl = 8271 #6508
# no_bls = 100
# for i in range(0,1):
#     np.random.seed(seed_bl) #set seed
#     a_zero_z = all_params[21]
#     a_zero_z = a_zero_z +35 #+i*5
#     a_zero_z = 175 #= a_zero_min
#     # h_val_z = all_params[20]
#     # h_val_z = h_val_z + (i*0.20)
#     # var_tol_z = all_params[32]
#     # var_tol_z = var_tol_z - (i*0.1)
#     test_r_mean = create_broken_lines(mu_z, sigma2_z, h_val_z, q_val_z, a_zero_z, timestep, (n_timesteps-1) * timestep, no_bls, var_tol_z, mar_tol_z)
#     # test_r_mean[:,0]
#     plt.figure()
#     for j in range(len(test_r_mean[0])):
#         if j == 0:
#             # plt.plot(test_r_mean)
#             plt.plot(test_r_mean[:,j], color="Gray", label="brokenlines")
#         else:
#             plt.plot(test_r_mean[:,j], color="Gray")  
#     plt.plot(R_mean_osapol, color="Red", label="osapol")
#     # plt.plot(R_mean_osapol, color="Red", linestyle="--", label="osapol")
#     plt.title(f"a_zero = {a_zero_z} and \nH = {h_val_z}")
#     plt.legend()
    
# #     bl0.append(test_r_mean[:,0])
# #     bl1.append(test_r_mean[:,1])
# #     bl2.append(test_r_mean[:,2])

# # bl0 = np.vstack(bl0)
# # bl1 = np.vstack(bl1)
# # bl2 = np.vstack(bl2)

# # plt.figure()
# # for i in range(len(bl0)):
# #     # plt.plot(bl0[i,:], label=f"{i}: H={h_val_z - ((7-i)*0.41)}")
# #     plt.plot(bl0[i,:], label=f"{i}: var_tol={var_tol_z + ((7-i)*0.1)}")
# #     ## var_min = sigma2_z - (var_tol * sigma2_z)  # acceptable minimum of variance
# #     ## var_max = sigma2_z + (var_tol * sigma2_z)  # acceptable maximum of variance
# # plt.plot(R_mean_osapol, color="Red", linestyle="--", label="osapol")
# # plt.legend()

# # plt.figure()
# # for i in range(len(bl1)):
# #     plt.plot(bl1[i,:], label=f"{i}: H={h_val_z - ((7-i)*0.41)}")
# # plt.plot(R_mean_osapol, color="Red", linestyle="--", label="osapol")
# # plt.legend()

# # plt.figure()
# # for i in range(len(bl2)):
# #     plt.plot(bl2[i,:], label=f"{i}: H={h_val_z - ((7-i)*0.41)}")
# # plt.plot(R_mean_osapol, color="Red", linestyle="--", label="osapol")
# # plt.legend()

##############################################################################
# TIME USED FOR PARAMETER ESTIMATION: End timer

run_end_0 = time.perf_counter()
run_dur_0 = run_end_0 - run_start_0
print(run_dur_0 / 60)
