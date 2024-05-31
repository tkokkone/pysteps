# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 08:27:43 2023

@author: lindgrv1
"""

import time
import os
import pandas as pd
from numpy import genfromtxt
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import rasterio
from rasterio.transform import from_origin
from fiona.crs import from_epsg

from pprint import pprint

import pysteps
from pysteps import nowcasts, noise, utils
from pysteps.utils import conversion, dimension, transformation
from pysteps.utils.dimension import clip_domain
from pysteps.visualization import plot_precip_field, animate, animate_interactive, get_colormap
from pysteps.postprocessing.probmatching import set_stats
from pysteps.utils import rapsd
from pysteps.visualization import plot_spectrum1d
from pysteps.utils import conversion
from pysteps import extrapolation

from scipy import stats

##############################################################################
#pySTEPS colormap

def get_colormap_own(ptype="intensity", units="dBZ", colorscale="pysteps"):
    """Function to generate a colormap (cmap) and norm
    Returns
    -------
    cmap : Colormap instance
        colormap
    norm : colors.Normalize object
        Colors norm
    clevs: list(float)
        List of precipitation values defining the color limits.
    clevs_str: list(str)
        List of precipitation values defining the color limits (with correct
        number of decimals).
    """
    if ptype in ["intensity", "depth"]:
        # Get list of colors
        color_list, clevs, clevs_str = _get_colorlist_own(units, colorscale)

        cmap = colors.LinearSegmentedColormap.from_list(
            "cmap", color_list, len(clevs) - 1
        )

        if colorscale == "BOM-RF3":
            cmap.set_over("black", 1)
        if colorscale == "pysteps":
            cmap.set_over("darkred", 1)
        if colorscale == "STEPS-BE":
            cmap.set_over("black", 1)
        norm = colors.BoundaryNorm(clevs, cmap.N)

        cmap.set_bad("gray", alpha=0.5)
        cmap.set_under("none")

        return cmap, norm, clevs, clevs_str

    # if ptype == "prob":
    #     cmap = copy.copy(plt.get_cmap("OrRd", 10))
    #     cmap.set_bad("gray", alpha=0.5)
    #     cmap.set_under("none")
    #     clevs = np.linspace(0, 1, 11)
    #     clevs[0] = 1e-3  # to set zeros to transparent
    #     norm = colors.BoundaryNorm(clevs, cmap.N)
    #     clevs_str = [f"{clev:.1f}" for clev in clevs]
    #     return cmap, norm, clevs, clevs_str

    return cm.get_cmap("jet"), colors.Normalize(), None, None

def _get_colorlist_own(units="dBZ", colorscale="pysteps"):
    """
    Returns
    -------
    color_list : list(str)
        List of color strings.

    clevs : list(float)
        List of precipitation values defining the color limits.

    clevs_str : list(str)
        List of precipitation values defining the color limits
        (with correct number of decimals).
    """

    if colorscale == "BOM-RF3":
        color_list = np.array(
            [
                (255, 255, 255),  # 0.0
                (245, 245, 255),  # 0.2
                (180, 180, 255),  # 0.5
                (120, 120, 255),  # 1.5
                (20, 20, 255),  # 2.5
                (0, 216, 195),  # 4.0
                (0, 150, 144),  # 6.0
                (0, 102, 102),  # 10
                (255, 255, 0),  # 15
                (255, 200, 0),  # 20
                (255, 150, 0),  # 30
                (255, 100, 0),  # 40
                (255, 0, 0),  # 50
                (200, 0, 0),  # 60
                (120, 0, 0),  # 75
                (40, 0, 0),
            ]
        )  # > 100
        color_list = color_list / 255.0
        if units == "mm/h":
            clevs = [
                0.0,
                0.2,
                0.5,
                1.5,
                2.5,
                4,
                6,
                10,
                15,
                20,
                30,
                40,
                50,
                60,
                75,
                100,
                150,
            ]
        elif units == "mm":
            clevs = [
                0.0,
                0.2,
                0.5,
                1.5,
                2.5,
                4,
                5,
                7,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
            ]
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)
    elif colorscale == "pysteps":
        # pinkHex = '#%02x%02x%02x' % (232, 215, 242)
        redgrey_hex = "#%02x%02x%02x" % (156, 126, 148)
        color_list = [
            redgrey_hex,
            "#640064",
            "#AF00AF",
            "#DC00DC",
            "#3232C8",
            "#0064FF",
            "#009696",
            "#00C832",
            "#64FF00",
            "#96FF00",
            "#C8FF00",
            "#FFFF00",
            "#FFC800",
            "#FFA000",
            "#FF7D00",
            "#E11900",
        ]
        if units in ["mm/h", "mm"]:
            clevs = [
                0.08,
                0.16,
                0.25,
                0.40,
                0.63,
                1,
                1.6,
                2.5,
                4,
                6.3,
                10,
                16,
                25,
                40,
                63,
                100,
                160,
            ]
        elif units == "dBZ":
            clevs = np.arange(10, 65, 5)
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)
    elif colorscale == "STEPS-BE":
        color_list = [
            "cyan",
            "deepskyblue",
            "dodgerblue",
            "blue",
            "chartreuse",
            "limegreen",
            "green",
            "darkgreen",
            "yellow",
            "gold",
            "orange",
            "red",
            "magenta",
            "darkmagenta",
        ]
        if units in ["mm/h", "mm"]:
            clevs = [0.1, 0.25, 0.4, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100]
        elif units == "dBZ":
            clevs = np.arange(10, 65, 5)
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)

    else:
        print("Invalid colorscale", colorscale)
        raise ValueError("Invalid colorscale " + colorscale)

    # Generate color level strings with correct amount of decimal places
    clevs_str = _dynamic_formatting_floats_own(clevs)

    return color_list, clevs, clevs_str

def _dynamic_formatting_floats_own(float_array, colorscale="pysteps"):
    """Function to format the floats defining the class limits of the colorbar."""
    float_array = np.array(float_array, dtype=float)

    labels = []
    for label in float_array:
        if 0.1 <= label < 1:
            if colorscale == "pysteps":
                formatting = ",.2f"
            else:
                formatting = ",.1f"
        elif 0.01 <= label < 0.1:
            formatting = ",.2f"
        elif 0.001 <= label < 0.01:
            formatting = ",.3f"
        elif 0.0001 <= label < 0.001:
            formatting = ",.4f"
        elif label >= 1 and label.is_integer():
            formatting = "i"
        else:
            formatting = ",.1f"

        if formatting != "i":
            labels.append(format(label, formatting))
        else:
            labels.append(str(int(label)))

    return labels

#cmap
cmap, norm, clevs, clevs_str= get_colormap_own("intensity","dBZ","pysteps")

# #plot_precip_field(R_sim[0])
# plt.figure()
# plt.imshow(R_sim[20], cmap=cmap, norm=norm)

##############################################################################
# OBSERVED EVENT AND POWER-LAW FILTER FROM IT

# radar_dir = r"\\home.org.aalto.fi\lindgrv1\data\Desktop\Vaitoskirjaprojekti\ARTIKKELI_2\data_a2\kiira_radar"
radar_dir = r"W:\lindgrv1\Kiira_whole_day\kiira_14-22_dbz"
file_list = os.listdir(radar_dir)
file_list = [x for x in file_list if ".tif" in x]
file_list = [f for f in file_list if f.endswith(".tif")]

file_list = file_list[8:-1]

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
rain_zero_value = 3.1830486304816077  # no rain pixels assigned with this value
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

#Wetted area ratio of observed event
war_ts = np.zeros(len(file_list))
for i in range (len(file_list)):
    war_ts[i] = (np.count_nonzero(radar_kiira[i][radar_kiira[i] > rain_zero_value])) / np.count_nonzero(~np.isnan(radar_kiira[i]))
war_ts2 = war_ts.copy()
plt.figure()
plt.plot(war_ts)
plt.title("Wetted area ratio (-)")

#Standard deviation of observed event in dbz
std_ts = np.zeros(len(file_list))
for i in range (len(file_list)):
    std_ts[i] = np.nanstd(radar_kiira[i])
std_ts2 = std_ts.copy()
plt.figure()
plt.plot(std_ts)
plt.title("Standard deviation (dBZ)")

#correct all time series
x_obs_17 = [16, 18]
y_mar_obs_17 = [areal_rainfall_ts[16], areal_rainfall_ts[18]]
y_war_obs_17 = [war_ts[16], war_ts[18]]
y_std_obs_17 = [std_ts[16], std_ts[18]]
x_new_17 = 17
y_mar_new_17 = np.interp(x_new_17, x_obs_17, y_mar_obs_17)
y_war_new_17 = np.interp(x_new_17, x_obs_17, y_war_obs_17)
y_std_new_17 = np.interp(x_new_17, x_obs_17, y_std_obs_17)
areal_rainfall_ts[17] = y_mar_new_17
war_ts[17] = y_war_new_17
std_ts[17] = y_std_new_17

x_obs_36 = [34, 37]
y_mar_obs_36 = [areal_rainfall_ts[34], areal_rainfall_ts[37]]
y_war_obs_36 = [war_ts[34], war_ts[37]]
y_std_obs_36 = [std_ts[34], std_ts[37]]
x_new_36 = [35, 36]
y_mar_new_36 = np.interp(x_new_36, x_obs_36, y_mar_obs_36)
y_war_new_36 = np.interp(x_new_36, x_obs_36, y_war_obs_36)
y_std_new_36 = np.interp(x_new_36, x_obs_36, y_std_obs_36)
areal_rainfall_ts[35] = y_mar_new_36[0]
areal_rainfall_ts[36] = y_mar_new_36[1]
war_ts[35] = y_war_new_36[0]
war_ts[36] = y_war_new_36[1]
std_ts[35] = y_std_new_36[0]
std_ts[36] = y_std_new_36[1]

x_obs_46 = [45, 50]
y_mar_obs_46 = [areal_rainfall_ts[45], areal_rainfall_ts[50]]
y_war_obs_46 = [war_ts[45], war_ts[50]]
y_std_obs_46 = [std_ts[45], std_ts[50]]
x_new_46 = [46, 47, 48, 49]
y_mar_new_46 = np.interp(x_new_46, x_obs_46, y_mar_obs_46)
y_war_new_46 = np.interp(x_new_46, x_obs_46, y_war_obs_46)
y_std_new_46 = np.interp(x_new_46, x_obs_46, y_std_obs_46)
areal_rainfall_ts[46] = y_mar_new_46[0]
areal_rainfall_ts[47] = y_mar_new_46[1]
areal_rainfall_ts[48] = y_mar_new_46[2]
areal_rainfall_ts[49] = y_mar_new_46[3]
war_ts[46] = y_war_new_46[0]
war_ts[47] = y_war_new_46[1]
war_ts[48] = y_war_new_46[2]
war_ts[49] = y_war_new_46[3]
std_ts[46] = y_std_new_46[0]
std_ts[47] = y_std_new_46[1]
std_ts[48] = y_std_new_46[2]
std_ts[49] = y_std_new_46[3]

# Save time series plots
corrected_ts_out_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
#Plot the areal rainfall time series
plt.figure()
plt.plot(areal_rainfall_ts2)
plt.plot(areal_rainfall_ts)
plt.title("Areal mean rainfall (dBZ)")
# plt.savefig(os.path.join(corrected_ts_out_dir, "corrected_r_mean.png"))

plt.figure()
plt.plot(war_ts2)
plt.plot(war_ts)
plt.title("Wetted area ratio (-)")
# plt.savefig(os.path.join(corrected_ts_out_dir, "corrected_war.png"))

plt.figure()
plt.plot(std_ts2)
plt.plot(std_ts)
plt.title("Standard deviation (dBZ)")
# plt.savefig(os.path.join(corrected_ts_out_dir, "corrected_std.png"))

#Replace non-finite values with the minimum value
R_temp = radar_kiira.copy()
for i in range(R_temp.shape[0]):
    R_temp[i, ~np.isfinite(radar_kiira[i, :])] = np.nanmin(R_temp[i, :])

#Power law filters using the observed field with maximum areal mean rainfall of the event
Fnp = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R_temp[max_idx])

# # Compute the Fourier transform of the input field
# F_max_idx = abs(np.fft.fftshift(np.fft.fft2(R_temp[max_idx])))
# # Plot the power spectrum
# M, N = F_max_idx.shape
# fig, ax = plt.subplots()
# im = ax.imshow(np.log(F_max_idx**2), vmin=4, vmax=24, cmap=cm.jet, extent=(-N / 2, N / 2, -M / 2, M / 2))
# cb = fig.colorbar(im)
# ax.set_xlabel("Wavenumber $k_x$")
# ax.set_ylabel("Wavenumber $k_y$")
# ax.set_title("Log-power spectrum of R")
# plt.show()

##############################################################################

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
# plt.savefig(os.path.join(corrected_ts_out_dir, "fft_filter_raster.png"))

fft_kiira_nan_temp = fft_kiira_nan.copy()
fft_kiira_nan_temp[0, ~np.isfinite(fft_kiira_nan[0, :])] = np.nanmin(fft_kiira_nan_temp[0, :])

plt.figure()
plt.imshow(fft_kiira_nan_temp[0])
# plt.savefig(os.path.join(corrected_ts_out_dir, "fft_filter_raster_nan.png"))

Fnp_2048 = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(fft_kiira_nan_temp[0])

# Compute the Fourier transform of the input field
F_2048 = abs(np.fft.fftshift(np.fft.fft2(fft_kiira_nan_temp[0])))
# Plot the power spectrum
M, N = F_2048.shape
fig, ax = plt.subplots()
im = ax.imshow(np.log(F_2048**2), vmin=4, vmax=24, cmap=cm.jet, extent=(-N / 2, N / 2, -M / 2, M / 2))
cb = fig.colorbar(im)
ax.set_xlabel("Wavenumber $k_x$")
ax.set_ylabel("Wavenumber $k_y$")
ax.set_title("Log-power spectrum of R")
plt.show()
# plt.savefig(os.path.join(corrected_ts_out_dir, "fft_filter_powerspect_nan.png"))

##############################################################################

# KANSIKUVAA VARTEN

kansi_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\KANSI"

kansi_kiira = fft_kiira_nan_temp[0].copy()

plt.figure()
plt.imshow(kansi_kiira, cmap=cmap, norm=norm)
# plt.savefig(os.path.join(kansi_dir, "kansi_1.png"))
# plt.savefig(os.path.join(kansi_dir,"kansi_1.pdf"), format="pdf", bbox_inches="tight")

plt.figure()
plt.imshow(kansi_kiira, cmap=cmap)
# plt.savefig(os.path.join(kansi_dir, "kansi_2.png"))
# plt.savefig(os.path.join(kansi_dir,"kansi_2.pdf"), format="pdf", bbox_inches="tight")

plt.figure()
pysteps.visualization.plot_precip_field(kansi_kiira, cmap=cmap, norm=norm)
# plt.savefig(os.path.join(kansi_dir, "kansi_3.png"))
# plt.savefig(os.path.join(kansi_dir,"kansi_3.pdf"), format="pdf", bbox_inches="tight")

kansikuva = []
src_kansikuva= rasterio.open(os.path.join(fft_dir, "kansi.tif"))
array_kansikuva = src_kansikuva.read(1)
kansikuva.append(array_kansikuva) 
kansikuva = np.concatenate([kansikuva_[None, :, :] for kansikuva_ in kansikuva])
kansikuva = kansikuva[:,1:-1,:]

kansikuva_nan = kansikuva.copy()
print(kansikuva_nan.max())
kansikuva_nan = np.where(kansikuva_nan==255, np.nan, kansikuva_nan)
kansikuva_nan = (kansikuva_nan * 0.5) - 32
kansikuva_nan[kansikuva_nan < 10] = 3.1830486304816077

kansikuva_nan_temp = kansikuva_nan.copy()
kansikuva_nan_temp[0, ~np.isfinite(kansikuva_nan[0, :])] = np.nanmin(kansikuva_nan_temp[0, :])

plt.figure()
plt.imshow(kansikuva_nan_temp[0], cmap=cmap, norm=norm)
# plt.savefig(os.path.join(kansi_dir, "kansi_4.png"))
# plt.savefig(os.path.join(kansi_dir,"kansi_4.pdf"), format="pdf", bbox_inches="tight")

plt.figure()
plt.imshow(kansikuva_nan_temp[0], cmap=cmap)
# plt.savefig(os.path.join(kansi_dir, "kansi_5.png"))
# plt.savefig(os.path.join(kansi_dir,"kansi_5.pdf"), format="pdf", bbox_inches="tight")

plt.figure()
pysteps.visualization.plot_precip_field(kansikuva_nan_temp[0], cmap=cmap, norm=norm)
# plt.savefig(os.path.join(kansi_dir, "kansi_6.png"))
# plt.savefig(os.path.join(kansi_dir,"kansi_6.pdf"), format="pdf", bbox_inches="tight")

##############################################################################
#Wanted number of simulations
n_simulations = 1
n_sim = 0
while n_sim < n_simulations:

    ##############################################################################
    # OPTIONS TO SAVE PLOTS, ANIMATIONS, AND TABLES
    
    animation_blues = False #show animation of input radar images in dBZ with Blues-colormap
    save_animation_blues = True #save animation of input radar images in dBZ with Blues-colormap (as png-images)
    
    save_animation_mmh = True #save animation of input radar images in mm/h
    
    save_figs = True
    
    save_tiff = True #save simulated event in tif format
    
    csv_brokenlines = True #save broken line timeseries
    csv_betas = True #simulated fft-parameters: beta1 and beta2
    
    csv_locations = True
    
    ##############################################################################
    # INDIR AND INPUT DATA
    
    # in_dir = r"W:/lindgrv1/Simuloinnit/Simulations_kiira"
    in_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
        
    #Read in estimated parameters
    all_params = genfromtxt(fname=os.path.join(in_dir, "all_params_corrected.csv"), delimiter=',', skip_header=1, usecols=1)
    
    #Read in time series from input data
    # data_tss = genfromtxt(fname=os.path.join(in_dir, "data_tss.csv"), delimiter=',', skip_header=1)
    data_tss = genfromtxt(fname=os.path.join(in_dir, "data_tss_corrected.csv"), delimiter=',', skip_header=1)
    data_tss = np.delete(data_tss, 0, axis=1)
    
    #Read in advection time series: Vx and Vy
    advection_tss = genfromtxt(fname=os.path.join(in_dir, "advection_tss.csv"), delimiter=',', skip_header=1)
    advection_tss = np.delete(advection_tss, 0, axis=1)
    
    #Read in input time series for longer simulation
    data_tss_long = genfromtxt(fname=os.path.join(in_dir, "data_tss_long.csv"), delimiter=',', skip_header=1)
    data_tss_long = np.delete(data_tss_long, 0, axis=1)

    ##############################################################################
    # SET CONTROL PARAMETERS FOR SIMULATION
    
    #Visualization
    grid_on = False #gridlines on or off
    colorbar_on = True #colorbar on or off
    cmap = 'Blues' #None means pysteps color mapping
    
    predefined_value_range = False #predefined value range in colorbar on or off
    #This sets vmin and vmax parameters in imshow and sets normalization True/False
    
    show_in_out_param_figs = True #in and out parameter value plots on/off
    save_in_out_param_figs = True
    
    show_cumul_plot = True #cumulative plot on/off
    save_cumul_plot = True
    
    #Statistics
    # Set mean, std, waar at the end of the simulation to supplied values
    set_stats_active = True
    #Normalize rain fields to zero mean one std before applying set_stats
    normalize_field = True
    
    # AR_mode
    # 0: pure innovation
    # 1: pure advection 
    # 2: AR using parameters given below
    AR_mode = 2
    
    # Advection_mode
    # 0: no advection
    # 1: constant advection with velocities given below
    # 2: constant advection with velocities given below, initial field
    #    a rectangular one block with zero backcround, block coords below 
    # 3: dynamic advection using parameters given below
    advection_mode = 3
    const_v_x = 0
    const_v_y = 1.5
    block_ulr = 243 #upper left row of one block
    block_ulc = 100 #upper left column of one block
    block_xsize = 20 #size of one block in x direction (no of grid cells)
    block_ysize = 10 #size of one block in y direction (no of grid cells)
    
    ##############################################################################
    # SET SIMULATION PARAMETERS (estimated from osapol-event)
    # AND INITIALIZE SOME VARIABLES
    
    # Initialisation of variables
    stats_kwargs = dict()
    metadata = dict()
    
    # Set general simulation parameters
    n_timesteps = len(radar_kiira)  # number of timesteps -> 88
    # n_timesteps = len(data_tss_long[0]) #FOR LONGER SIMULATION
    timestep = 5  # timestep length
    nx_field = fft_kiira_nan_temp.shape[1]  # number of columns in precip fields -> 2048
    ny_field = fft_kiira_nan_temp.shape[2]  # number of rows in precip fields -> 2048
    # nx_field = radar_kiira.shape[1] # number of columns in precip fields -> 1024
    # ny_field = radar_kiira.shape[2] # number of rows in precip fields -> 1024
    kmperpixel = 0.250  # grid resolution
    domain = "spatial"  # spatial or spectral
    metadata["x1"] = 0.0  # x-coordinate of lower left
    metadata["y1"] = 0.0  # y-coordinate of lower left
    metadata["x2"] = nx_field * kmperpixel  # x-coordinate of upper right
    metadata["y2"] = ny_field * kmperpixel  # y-coordinate of upper right
    metadata["xpixelsize"] = kmperpixel  # grid resolution
    metadata["ypixelsize"] = kmperpixel  # grid resolution
    metadata["zerovalue"] = 0.0  # maybe not in use?
    metadata["yorigin"] = "lower"  # location of grid origin (y), lower or upper
    
    war_thold = rain_zero_value + 5  # below this value no rain
    
    # bounding box coordinates for the extracted middel part of the entire domain
    extent = [0, 0, 0, 0]
    extent[0] = nx_field / 4 * kmperpixel
    extent[1] = 3 * nx_field / 4 * kmperpixel
    extent[2] = ny_field / 4 * kmperpixel
    extent[3] = 3 * ny_field / 4 * kmperpixel
    
    # Set power filter parameters
    # where power filter pareameters given not estimated
    # noise_method = "parametric_sim"
    noise_method = "nonparametric"
    fft_method = "numpy"
    simulation_method = "steps_sim"
    
    scale_break = 18  # constant scale break in km
    scale_break_wn = np.log(nx_field/scale_break)
    constant_betas = False
    beta1 = -1.9
    beta2 = -3.5
    
    # #scale break parameters a_w0, b_w0, and c_w0 from fitted polynomial line
    # a_w0 = all_params[0]
    # b_w0 = all_params[1]
    # c_w0 = all_params[2]
    # #beta1 parameters a_1, b_1, and c_1 from fitted polynomial line
    # a_1 = all_params[3]  # a_1...c_2 see Seed et al. 2014
    # b_1 = all_params[4]
    # c_1 = all_params[5]
    # #beta2 parameters a_2, b_2, and c_2 from fitted polynomial line
    # a_2 = all_params[6]
    # b_2 = all_params[7]
    # c_2 = all_params[8]  
    
    # p_pow = np.array([scale_break_wn, 0.0, beta1, beta2]) # initialization
    
    # Initialise AR parameter array, Seed et al. 2014 eqs. 9-11
    tlen_a = all_params[0] #lower tlen_a makes result more random, higher tlen_a makes evolution make more sense
    tlen_b = all_params[1] #lower tlen_b makes result more random, higher tlen_b makes evolution make more sense
    tlen_c = all_params[2] #changing this doesnt change the result much 
    ar_par = np.array([tlen_a, tlen_b, tlen_c]) 
    
    # Set std and WAR parameters, Seed et al. eq. 4
    a_v = all_params[3]
    b_v = all_params[4]
    c_v = all_params[5]
    a_war = all_params[6]
    b_war = all_params[7]
    c_war = all_params[8]
    min_war = 0.1
    
    # Broken line parameters for field mean
    mu_z = all_params[9] #mean of mean areal reflectivity over the simulation period
    sigma2_z = all_params[10] #variance of mean areal reflectivity over the simulation period
    h_val_z = all_params[11] #structure function exponent
    q_val_z = all_params[13] #scale ratio between levels n and n+1 (constant) [-]
    a_zero_z = all_params[12] #time series decorrelation time [min]
    a_zero_z = 175
    no_bls = all_params[14] #number of broken lines
    var_tol_z = all_params[15] #acceptable tolerance for variance as ratio of input variance [-]
    mar_tol_z = all_params[16] #acceptable value for first and last elements of the final broken line as ratio of input mean -> maybe should be less than 1 -> 0.2
    
    # # Broken line parameters for velocity magnitude
    # mu_vmag = all_params[22] #mean of mean areal reflectivity over the simulation period
    # sigma2_vmag = all_params[23] #variance of mean areal reflectivity over the simulation period
    # h_val_vmag = all_params[24]  #structure function exponent
    # q_val_vmag = all_params[30] #scale ratio between levels n and n+1 (constant) [-]
    # a_zero_vmag = all_params[25]  #time series decorrelation time [min]
    # no_bls = 1 #number of broken lines
    # var_tol_vmag = all_params[32] #acceptable tolerance for variance as ratio of input variance [-]
    # mar_tol_vmag = all_params[33] #acceptable value for first and last elements of the final broken line as ratio of input mean 
    
    # # Broken line parameters for velocity direction
    # mu_vdir = all_params[26] #mean of mean areal reflectivity over the simulation period
    # sigma2_vdir = all_params[27] #variance of mean areal reflectivity over the simulation period
    # h_val_vdir = all_params[28]  #structure function exponent
    # q_val_vdir = all_params[30] #scale ratio between levels n and n+1 (constant) [-]
    # a_zero_vdir = all_params[29]  #time series decorrelation time [min]
    # no_bls = 1 #number of broken lines
    # var_tol_vdir = all_params[32] #acceptable tolerance for variance as ratio of input variance [-]
    # mar_tol_vdir = all_params[33] #acceptable value for first and last elements of the final broken line as ratio of input mean:
    
    # a_zero_z = azero_opt # change a_zero_z to correspond optimized a_zero value    
    # var_tol_vmag = 1
    # mar_tol_vmag = 10000
    # var_tol_vdir = 1
    # mar_tol_vdir = 10000
    
    # Set method for advection
    extrap_method = "semilagrangian_wrap"
    
    #Z-R relationship: Z = a*R^b (Reflectivity)
    #dBZ conversion: dBZ = 10 * log10(Z) (Decibels of Reflectivity)
    #Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with size. J. Meteor., 5, 165–166.
    a_R=223
    b_R=1.53
    #Leinonen et al. 2012 - A Climatology of Disdrometer Measurements of Rainfall in Finland over Five Years with Implications for Global Radar Observations
    
    R_sim = []
    mean_in = []
    std_in = []
    war_in = []
    # beta1_in = []
    # beta2_in = []
    # scaleb_in = []
    mean_out = []
    std_out = []
    war_out = []
    # beta1_out = []
    # beta2_out = []
    # scaleb_out = []
    
    # #Ville: added
    # beta1_out1 = []
    # beta2_out1 = []
    # beta1_out2 = []
    # beta2_out2 = []
    # war_out2 = []
    # beta1_out_Rnew2 = []
    # beta2_out_Rnew2 = []
    # beta1_out_Rnew3 = []
    # beta2_out_Rnew3 = []
    # all_Fp = []
    # all_R_ = []
    # all_freq = []
    # all_f = []
    # all_b1 = []
    # all_b2 = []
    
    # all_R_fp1_ = []
    # all_freq_fp1 = []
    # all_f_fp1 = []
    
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
    # SET SEEDS
    
    seed1 = random.randint(1, 100000) # seed number 1: seed number for generation of the first precipitation field
    seed2 = random.randint(1, 100000) # seed number 2: seed number for generation of the first innovation field
    seed_bl = random.randint(1, 100000) #seed for generating broken lines
    seed_random = random.randint(1, 100000) #seed for generating random fields
    
    # seed1 = 55056
    # seed2 = 1737
    # seed_bl = 87845
    # seed_random = 85735
    
    ##############################################################################
    # TESTING AR-PARAMETERS
    
    # tlen_a = tlen_a #lower tlen_a makes result more random, higher tlen_a makes evolution make more sense - originally: 4.810440239686069
    # tlen_b = tlen_b #lower tlen_b makes result more random, higher tlen_b makes evolution make more sense - originally: 0.8413854949124062
    # tlen_c = tlen_c #changing this doesnt change the result much - originally: 1.0790183398028317
    # ar_par = np.array([tlen_a, tlen_b, tlen_c]) 
    
    # -> tlen_a kasvattaminen smoothaa karttoja ja tekee hidastaa rakenteiden evoluutiota
    # -> tlen_b kasvattaminen smoothaa karttoja ja tekee hidastaa rakenteiden evoluutiota
    # -> tlen_b ei voi olla suurempi kuin tlen_a
    # -> tlen_b ei voi olla negatiivinen
    # -> tlen_c ei voi olla negatiivinen
    
    ##############################################################################
    # OUTDIR
    
    out_dir = os.path.join(in_dir, "Simulations")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_dir = os.path.join(out_dir, f"Simulation_{seed1}_{seed2}_{seed_bl}_{seed_random}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    ##############################################################################
    # GENERATING BROKEN LINE TIME SERIES FOR MEAN R AND ADVECTION
    
    # seed_bl = random.randint(1, 100000)
    no_bls = int(no_bls)
    # Create the field mean for the requested number of simulation time steps
    np.random.seed(seed_bl) #set seed
    
    # a_zero_z = 175 #55
    # h_val_z = 1 #0.3704526021049618
    
    # r_mean = create_broken_lines(mu_z, sigma2_z, h_val_z, q_val_z,
    #                              a_zero_z, timestep, (n_timesteps-1) * timestep,
    #                              no_bls, var_tol_z, mar_tol_z, areal_rainfall_ts)
    
    r_mean = areal_rainfall_ts
    # r_mean = data_tss_long[0] #FOR LONGER SIMULATION
    
    r_mean_org = r_mean.copy()
    #Scaling to have mean value of mu_z
    r_mean_scaled = (r_mean - np.mean(r_mean)) + mu_z
    r_mean = r_mean_scaled.copy()
    
    plt.figure()
    plt.plot(r_mean_org, label="bl_org")
    plt.plot(r_mean, label="bl_scaled") 
    plt.plot(data_tss[0,:], label="observed", linestyle="--", color="Red")
    plt.plot(areal_rainfall_ts, label="observed2", linestyle="--", color="Yellow")
    plt.legend()
    plt.title("Mean")
    if save_figs:
        plt.savefig(os.path.join(out_dir, "bl_r_mean.png"))
        plt.close()
    
    ###############################
    
    # #mar
    # test_array = r_mean.copy()
    # test_new_steps = []
    # for i in range(0,len(test_array)-1):
    #     test_array_int = np.interp(i+0.5, [i, i+1], [test_array[i], test_array[i+1]])
    #     test_new_steps.append(test_array_int)
    # test_new_steps = np.concatenate([test_new_steps_[None] for test_new_steps_ in test_new_steps])
    
    # test_new_array = np.zeros((len(test_array)+len(test_new_steps), ))
    # for num in range(len(test_new_array)):
    #     if (num % 2) == 0:
    #         test_new_array[num] = test_array[int(num-(num/2))]
    #     else:
    #         test_new_array[num] = test_new_steps[int(num-(num/2))]
            
    # plt.figure()
    # plt.plot(test_array)
    # plt.figure()
    # plt.plot(test_new_array)

    # #advection in x_dir
    # test_array_vx = vx.copy()
    # test_new_steps_vx = []
    # for i in range(0,len(test_array_vx)-1):
    #     test_array_vx_int = np.interp(i+0.5, [i, i+1], [test_array_vx[i], test_array_vx[i+1]])
    #     test_new_steps_vx.append(test_array_vx_int)
    # test_new_steps_vx = np.concatenate([test_new_steps_vx_[None] for test_new_steps_vx_ in test_new_steps_vx])
    
    # test_new_array_vx = np.zeros((len(test_array_vx)+len(test_new_steps_vx), ))
    # for num in range(len(test_new_array_vx)):
    #     if (num % 2) == 0:
    #         test_new_array_vx[num] = test_array_vx[int(num-(num/2))]
    #     else:
    #         test_new_array_vx[num] = test_new_steps_vx[int(num-(num/2))]
            
    # plt.figure()
    # plt.plot(test_array_vx)
    # plt.figure()
    # plt.plot(test_new_array_vx)
    
    # test_new_steps_vx2 = []
    # for i in range(0,len(test_new_array_vx)-1):
    #     test_array_vx2_int = np.interp(i+0.5, [i, i+1], [test_new_array_vx[i], test_new_array_vx[i+1]])
    #     test_new_steps_vx2.append(test_array_vx2_int)
    # test_new_steps_vx2 = np.concatenate([test_new_steps_vx2_[None] for test_new_steps_vx2_ in test_new_steps_vx2])
    
    # test_new_array_vx2 = np.zeros((len(test_new_array_vx)+len(test_new_steps_vx2), ))
    # for num in range(len(test_new_array_vx2)):
    #     if (num % 2) == 0:
    #         test_new_array_vx2[num] = test_new_array_vx[int(num-(num/2))]
    #     else:
    #         test_new_array_vx2[num] = test_new_steps_vx2[int(num-(num/2))]
            
    # plt.figure()
    # plt.plot(test_array_vx)
    # plt.figure()
    # plt.plot(test_new_array_vx)
    # plt.figure()
    # plt.plot(test_new_array_vx2)
        
    ###############################    
    
    # # Create velocity magnitude for the requested number of simulation time steps
    # np.random.seed(seed_bl)
    # v_mag = create_broken_lines(mu_vmag, sigma2_vmag, h_val_vmag, q_val_vmag,
    #                             a_zero_vmag, timestep, (n_timesteps-1) * timestep,
    #                             no_bls, var_tol_vmag, mar_tol_vmag)
    # plt.figure()
    # plt.plot(v_mag, label="bl", color="Gray")
    # plt.plot(advection_tss[2,:], label="osapol", linestyle="--", color="Red")
    # plt.legend()
    # plt.title("V_mag")
    # plt.savefig(os.path.join(out_dir, "bl_v_mag.png"))
    # plt.close()
    
    # # Create velocity direction (deg) for the requested number of simulation time steps
    # np.random.seed(seed_bl)
    # v_dir = create_broken_lines(mu_vdir, sigma2_vdir, h_val_vdir, q_val_vdir,
    #                             a_zero_vdir, timestep, (n_timesteps-1) * timestep,
    #                             no_bls, var_tol_vdir, mar_tol_vdir)
    # plt.figure()
    # plt.plot(v_dir, label="bl", color="Gray")
    # plt.plot(advection_tss[5,:], label="osapol", linestyle="--", color="Red")
    # plt.legend()
    # plt.title("V_dir")
    # plt.savefig(os.path.join(out_dir, "bl_v_dir.png"))
    # plt.close()
    
    # # Compute advection variables in x- and y-directions
    # vx = np.cos(v_dir / 360 * 2 * np.pi) * v_mag
    # vy = np.sin(v_dir / 360 * 2 * np.pi) * v_mag
    
    # plt.figure()
    # plt.plot(vx, label="bl", color="Gray")
    # plt.plot(advection_tss[0,:], label="osapol", linestyle="--", color="Red")
    # plt.legend()
    # plt.title("Vx")
    # plt.savefig(os.path.join(out_dir, "bl_vx.png"))
    # plt.close()
    
    # plt.figure()
    # plt.plot(vy, label="bl", color="Gray")
    # plt.plot(advection_tss[1,:], label="osapol", linestyle="--", color="Red")
    # plt.legend()
    # plt.title("Vy")
    # plt.savefig(os.path.join(out_dir, "bl_vy.png"))
    # plt.close()
    
    vx = advection_tss[0]
    vy = advection_tss[1]
    
    # # FOR LONGER SIMULATION
    # vx = data_tss_long[1] #FOR LONGER SIMULATION
    # vx = vx[:-1]
    # vy = data_tss_long[2] #FOR LONGER SIMULATION
    # vy = vy[:-1]
    
    if advection_mode == 0:
        vx[:] = 0
        vy[:] = 0
    elif advection_mode == 1 or advection_mode == 2:
        vx[:] = const_v_x
        vy[:] = const_v_y
        
    #Variables for magnitude and direction of advection
    Vmag = np.zeros(len(vx))
    Vdir_rad = np.zeros(len(vx))

    #Loop to calculate average x- and y-components of the advection, as well as magnitude and direction in xy-dir
    for i in range(len(vx)):
        Vmag[i] = np.sqrt(vx[i]**2 + vy[i]**2) #total magnitude of advection
        Vdir_rad[i] = np.arctan(vy[i]/vx[i]) #direction of advection in xy-dir in radians
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
    
    # Onko kahdella seuraavalla rivillä mitään virkaa? xy_coords:ia ei käytetä missään
    # Liittynee advektioon ja siihen käytetäänkö advektiota kahden ekan
    # kentän luonnissa
    x_values, y_values = np.meshgrid(np.arange(nx_field), np.arange((ny_field)))
    xy_coords = np.stack([x_values, y_values])
    
    ##############################################################################
    # CREATING FIRST TWO PRECIPITATION FIELDS
    
    # Create the first two precipitation fields, the nuber is determined by the order of
    # the ar process, in this example it is two. The second one is a copy of the first
    # one. The first velocity magnitude is zero.
    
    # v_mag[0] = 0.0
    R_ini = []
    
    np.random.seed(seed_random)
    R_ini.append(np.random.normal(0.0, 1.0, size=(ny_field, nx_field)))
    # R_ini.append(np.random.normal(0.0, 1.0, size=(ny_field, nx_field)))
    # Change the type of R to align with pySTEPS
    R_ini = np.concatenate([R_[None, :, :] for R_ in R_ini])
    
    fft = utils.get_method(fft_method, shape=(ny_field, nx_field), n_threads=1)
    init_noise, generate_noise = noise.get_method(noise_method)
    noise_kwargs = dict()
    
    # if constant_betas:
    #     p_pow[0] = scale_break_wn
    #     p_pow[2] = beta1
    #     p_pow[3] = beta2
    # else:
    #     # p_pow[2] = - (a_1 + b_1 * r_mean[0] + c_1 * r_mean[0] ** 2)
    #     # p_pow[3] = - (a_2 + b_2 * r_mean[0] + c_2 * r_mean[0] ** 2)
    #     # p_pow[0] = np.log(nx_field/(a_w0 + b_w0 * r_mean[0] + c_w0 * r_mean[0] ** 2)) #scale break
    #     p_pow[2] = (a_1 + b_1 * r_mean[0] + c_1 * r_mean[0] ** 2) - 0.5 #beta1
    #     p_pow[3] = (a_2 + b_2 * r_mean[0] + c_2 * r_mean[0] ** 2) - 0.5 #beta2
    #     p_pow[0] = scale_break_wn
    
    # p_pow_w0 = np.zeros([n_timesteps, 1])
    # p_pow_b1 = np.zeros([n_timesteps, 1])
    # p_pow_b2 = np.zeros([n_timesteps, 1])
    # p_pow_w0[0] = p_pow[0]
    # p_pow_b1[0] = p_pow[2]
    # p_pow_b2[0] = p_pow[3]
    
    # w0_sim = np.zeros([n_timesteps, 1])
    # w0_sim_km = np.zeros([n_timesteps, 1])
    # beta1_sim = np.zeros([n_timesteps, 1])
    # beta2_sim = np.zeros([n_timesteps, 1])
    
    # pp = init_noise(R_ini, fft_method=fft, **noise_kwargs)
    # pp = Fnp
    pp = Fnp_2048
    
    R = []
    if advection_mode == 2:
        R_0 = np.zeros((ny_field, nx_field))
        R_0[block_ulr:block_ulr+block_ysize,block_ulc:block_ulc+block_xsize] = 1
        R.append(R_0)
        R.append(R_0)
    else:    
        R_0 = generate_noise(pp, randstate=None, seed=seed1, fft_method=fft, domain=domain)
        # Tarvitaanko seuraavaa riviä?
        # Fp = noise.fftgenerators.initialize_param_2d_fft_filter(R_0, scale_break=scale_break_wn)
        R.append(R_0)
        R.append(R_0)
        
    # Generate the first innovation field and append it as the last term in R
    # beta1_in.append(p_pow[2])
    # beta2_in.append(p_pow[3])
    R.append(generate_noise(pp, randstate=None, seed=seed2, fft_method=fft, domain=domain))
    R = np.concatenate([R_[None, :, :] for R_ in R])
    
    # Plot the rainfall field
    #plot_precip_field(R[-1, :, :])
    #plt.show()
    
    # Set missing values with the fill value
    # TEEMU: probably not needed, how can there be nonfinite values in generated
    # fields?
    R[~np.isfinite(R)] = -15.0

    # Nicely print the metadata
    # pprint(metadata)
    
    ##############################################################################
    # SIMULATION LOOP WITH STEPS
    
    # vx = Vx_turn
    # vy = Vy_turn
    # vx = Vx_slower
    # vy = Vy_slower
    # vx = vx / 2
    # vy = vy / 2
    
    nowcast_method = nowcasts.get_method(simulation_method)
    r_mean_test = []
    for i in range(1, n_timesteps):
    # for i in range(n_timesteps):
        # TEEMU: näitten kai kuuluu olla negatiivisia?
        # if constant_betas:
        #     p_pow[0] = scale_break_wn
        #     p_pow[2] = beta1
        #     p_pow[3] = beta2
        # else:
        #     # p_pow[2] = - (a_1 + b_1 * r_mean[i] + c_1 * r_mean[i] ** 2)
        #     # p_pow[3] = - (a_2 + b_2 * r_mean[i] + c_2 * r_mean[i] ** 2)
        #     # p_pow[0] = np.log(nx_field/(a_w0 + b_w0 * r_mean[i] + c_w0 * r_mean[i] ** 2)) #scale break
        #     p_pow[2] = (a_1 + b_1 * r_mean[i] + c_1 * r_mean[i] ** 2) - 0.5 #beta1
        #     p_pow[3] = (a_2 + b_2 * r_mean[i] + c_2 * r_mean[i] ** 2) - 0.5 #beta2
        #     p_pow[0] = scale_break_wn
            
        # pp = init_noise(R_ini, p_pow, fft_method=fft, **noise_kwargs)
        
        # pp = Fnp
        pp = Fnp_2048
        
        # R_prev needs to be saved as R is advected in STEPS loop
        R_prev = R[1].copy()
        R_new = nowcast_method(
            R,
            vx[i-1],
            vy[i-1],
            # vx[i],
            # vy[i],
            ar_par,
            n_cascade_levels=6,
            R_thr=-10.0, #pitäiskö tän olla eri? 8.1830486304816077?
            kmperpixel=kmperpixel,
            timestep=timestep,
            noise_method=noise_method,
            vel_pert_method="bps",
            mask_method="incremental",
            include_AR=AR_mode,
            normalize_field=normalize_field,
        )
    
        # w0_km = nx_field / np.exp(Fp["pars"][0])
        
        R[0] = R_prev
        
        # p_pow_w0[i] = p_pow[0]
        # p_pow_b1[i] = p_pow[2]
        # p_pow_b2[i] = p_pow[3]
        
        R[1] = R_new
        R[2] = generate_noise(pp, randstate=None, fft_method=fft, domain=domain)
        
        # #Ville: added
        # R_new2 = R_new.copy()
        # R_new3 = R_new.copy()
        
        # Fp = noise.fftgenerators.initialize_param_2d_fft_filter(R_new, scale_break=scale_break_wn)
        
        #TODO: Try moving clipping after stat set!!!
        # # Ville: Moved here before set_stats
        R_new, metadata_clip = clip_domain(R_new, metadata, extent)
        
        # #Ville: added
        # Fp1 = noise.fftgenerators.initialize_param_2d_fft_filter(R_new, scale_break=scale_break_wn)
        
        # #Compute the observed and fitted 1D PSD
        # L = np.max(Fp1["input_shape"])
        # if L % 2 == 1:
        #     wn = np.arange(0, int(L / 2) + 1)
        # else:
        #     wn = np.arange(0, int(L / 2))
        # R_fp1_, freq_fp1 = rapsd(R_new, fft_method=np.fft, return_freq=True)
        # f_fp1 = np.exp(Fp1["model"](np.log(wn), *Fp1["pars"]))
        # all_R_fp1_.append(R_fp1_)
        # all_freq_fp1.append(freq_fp1)
        # all_f_fp1.append(f_fp1)
        
        # r_mean = areal_rainfall_ts
        
        stats_kwargs["mean"] = r_mean[i]
        stats_kwargs["std"] = a_v + b_v * r_mean[i] + c_v * r_mean[i] ** 2
        stats_kwargs["war"] = a_war + b_war * r_mean[i] + c_war * r_mean[i] ** 2
        if stats_kwargs["war"] < min_war:
           stats_kwargs["war"] = min_war 
        stats_kwargs["war_thold"] = war_thold
        stats_kwargs["rain_zero_value"] = rain_zero_value
        if set_stats_active == True:
            R_new = set_stats(R_new, stats_kwargs)
        
        # R_new, metadata_clip = clip_domain(R_new, metadata, extent)
        
        # #Ville: added
        # #These parameters must be used!!!
        # Fp2 = noise.fftgenerators.initialize_param_2d_fft_filter(R_new, scale_break=scale_break_wn) #beta1 and beta2 after clipping and setting stats
        # all_Fp.append(Fp2)
        
        # w0_sim[i-1] = Fp2["pars"][0]
        # w0_sim_km[i-1] = R_new.shape[0] / np.exp(w0_sim[i-1])
        # beta1_sim[i-1] = Fp2["pars"][1]
        # beta2_sim[i-1] = Fp2["pars"][2]
        
        # #Betas with only clipping
        # R_new2, metadata_clip2 = clip_domain(R_new2, metadata, extent)
        # Fp_Rnew2 = noise.fftgenerators.initialize_param_2d_fft_filter(
        #     R_new2, scale_break=scale_break_wn)
        # beta1_out_Rnew2.append(Fp_Rnew2["pars"][1])
        # beta2_out_Rnew2.append(Fp_Rnew2["pars"][2])
        
        # #Betas with only setting statistics
        # if set_stats_active == True:
        #     R_new3 = set_stats(R_new3, stats_kwargs)
        # Fp_Rnew3 = noise.fftgenerators.initialize_param_2d_fft_filter(
        #     R_new3, scale_break=scale_break_wn)
        # beta1_out_Rnew3.append(Fp_Rnew3["pars"][1])
        # beta2_out_Rnew3.append(Fp_Rnew3["pars"][2])
        
        # w0_km = nx_field / np.exp(scale_break_wn)
        # beta1_out.append(Fp["pars"][1])
        # beta2_out.append(Fp["pars"][2])
        # scaleb_out.append(w0_km)
        
        # #Ville: added
        # beta1_out1.append(Fp1["pars"][1])
        # beta2_out1.append(Fp1["pars"][2])
        # beta1_out2.append(Fp2["pars"][1])
        # beta2_out2.append(Fp2["pars"][2])
        
        war = len(np.where(R_new > war_thold)[0]) / (R_new.shape[1] * R_new.shape[0])
        
        # #Ville: added
        # war2 = ((np.count_nonzero(R_new[R_new > rain_zero_value])) / np.count_nonzero(~np.isnan(R_new)))
        # war_out2.append(war2) 
        
        r_mean_test.append(r_mean[i])
        
        mean_out.append(R_new.mean())
        std_out.append(R_new.std())
        war_out.append(war)
    
        # R_new, metadata_clip = clip_domain(R_new, metadata, extent)
    
        R_sim.append(R_new)
        mean_in.append(stats_kwargs["mean"])
        std_in.append(stats_kwargs["std"])
        war_in.append(stats_kwargs["war"])
        # beta1_in.append(p_pow[2])
        # beta2_in.append(p_pow[3])
        # scaleb_in.append(scale_break)
        
    # plt.figure()
    # plt.plot(w0_sim_km)
    # plt.plot(scaleb_in)
    
    # f.close()
    R_sim = np.concatenate([R_[None, :, :] for R_ in R_sim])
    # TEEMU: precipfields.py:hyn funktioon plot_precip_field puukotettu yksiköksi dBZ
    #animate(R_sim, savefig=False, path_outputs="../../Local/tmp2")
    
    # ani = animate_interactive(R_sim,grid_on,colorbar_on, predefined_value_range,cmap)
    
    #Clear values over threshold of 45 dBZ
    R_sim[R_sim > 45] = 0.5*(45 + R_sim[R_sim > 45])
    
    ##### To reach 1 simulation to this point takes about 17 mins.
    
    ##############################################################################
    #Plot event as a animation using possibility to jump between timesteps
    if animation_blues:
        #ani = animate_interactive(R_sim,grid_on,colorbar_on, predefined_value_range,cmap) #cmap=None is pysteps colorscale
        ani = pysteps.visualization.animations.animate_interactive(R_sim, False, True, False, "Blues")
        ani2 = pysteps.visualization.animations.animate_interactive(R_sim, False, True, False, None)
    
    if save_animation_blues:
        #Create datetime object
        starttime = "00:00:00"
        title_time = pd.date_range(starttime, periods=len(R_sim), freq="5Min")
        #If needed, add folder where to save the animation
        out_dir2 = os.path.join(out_dir, "Output_animation_blues")
        if not os.path.exists(out_dir2):
            os.makedirs(out_dir2)
        out_dir3 = os.path.join(out_dir, "Output_animation_pysteps")
        if not os.path.exists(out_dir3):
            os.makedirs(out_dir3)
        #Save pngs
        for im in range(0,len(R_sim)):
            #blue colormnap
            plt.figure()
            testi_im = plt.imshow(R_sim[im], cmap="Blues", vmin=0, vmax=round(np.nanmax(R_sim)+0.5))
            plt.colorbar(testi_im, spacing="uniform", extend="max", shrink=0.8, cax=None, label="Precipitation intensity [dBZ]")
            plt.title("Time: %s"% str(title_time[im])[11:16])
            plt.savefig(os.path.join(out_dir2, f"test_{im}.png"))
            plt.close()
            #pysteps colormap
            cmap, norm, clevs, clevs_str= get_colormap_own("intensity","dBZ","pysteps")
            plt.figure()
            testi_im_pysteps = plt.imshow(R_sim[im], cmap=cmap, vmin=0, vmax=round(np.nanmax(R_sim)+0.5)) #"Blues"
            plt.colorbar(testi_im_pysteps, spacing="uniform", extend="max", shrink=0.8, cax=None, label="Precipitation intensity [dBZ]")
            plt.title("Time: %s"% str(title_time[im])[11:16])
            plt.savefig(os.path.join(out_dir3, f"test_pysteps_{im}.png"))
            plt.close()
    
    ##### Animations take about 2.5 mins more.
    
    ##############################################################################
    
    if show_in_out_param_figs:
        x_data = np.arange(0,len(mean_in))
        
        plt.figure()
        plt.plot(x_data,war_in, x_data,war_out)
        # plt.plot(war_ts, label="observed", linestyle="--", color="Red")
        plt.plot(war_ts[1:], label="observed", linestyle="--", color="Red")
        plt.title("WAR")
        plt.legend(['In', 'Out', 'Observed'])
        if save_in_out_param_figs:
            plt.savefig(os.path.join(out_dir, "in-out_WAR.png"))
        
        plt.figure()
        plt.plot(x_data,std_in,x_data, std_out)
        # plt.plot(std_ts, label="observed", linestyle="--", color="Red")
        plt.plot(std_ts[1:], label="observed", linestyle="--", color="Red")
        plt.title("Standard deviation")
        plt.legend(['In', 'Out', 'Observed'])
        if save_in_out_param_figs:
            plt.savefig(os.path.join(out_dir, "in-out_STD.png"))
        
        plt.figure()
        plt.plot(x_data,mean_in,x_data,mean_out)
        # plt.plot(areal_rainfall_ts, label="observed", linestyle="--", color="Red")
        plt.plot(areal_rainfall_ts[1:], label="observed", linestyle="--", color="Red")
        plt.title("Mean")
        plt.legend(['In', 'Out', 'Observed'])
        if save_in_out_param_figs:
            plt.savefig(os.path.join(out_dir, "in-out_MEAN.png"))
        
        # plt.figure()
        # plt.plot(x_data,beta1_in[0:-1], x_data,beta1_out, x_data,beta1_out1, x_data,beta1_out2)
        # plt.title("Beta 1")
        # plt.legend(['In', 'Out', 'Out1', 'Out2'])
        
        # plt.figure()
        # plt.plot(x_data,beta2_in[0:-1], x_data,beta2_out, x_data,beta2_out1, x_data,beta2_out2)
        # plt.title("Beta 2")
        # plt.legend(['In', 'Out', 'Out1', 'Out2'])
        
        # plt.figure()
        # plt.plot(x_data,beta1_in[0:-1], x_data,beta1_out, x_data,beta1_out2, x_data,beta1_out_Rnew2, x_data,beta1_out_Rnew3)
        # plt.title("Beta 1")
        # plt.legend(['In', 'Nothing', 'Both', 'Clip', 'Stats'])
        # if save_in_out_param_figs:
        #     plt.savefig(os.path.join(out_dir, "in-out_beta1.png"))
        
        # plt.figure()
        # plt.plot(x_data,beta2_in[0:-1], x_data,beta2_out , x_data,beta2_out2, x_data,beta2_out_Rnew2, x_data,beta2_out_Rnew3)
        # plt.title("Beta 2")
        # plt.legend(['In', 'Nothing', 'Both', 'Clip', 'Stats'])
        # if save_in_out_param_figs:
        #     plt.savefig(os.path.join(out_dir, "in-out_beta2.png"))
        
        # plt.figure()
        # plt.plot(x_data,beta1_in[0:-1], x_data,beta1_out2)
        # plt.title("Beta 1")
        # plt.legend(['In', 'Out'])
        # if save_in_out_param_figs:
        #     plt.savefig(os.path.join(out_dir, "in-out_beta1_2.png"))
        
        # plt.figure()
        # plt.plot(x_data,beta2_in[0:-1], x_data,beta2_out2)
        # plt.title("Beta 2")
        # plt.legend(['In', 'Out'])
        # if save_in_out_param_figs:
        #     plt.savefig(os.path.join(out_dir, "in-out_beta2_2.png"))
        
        # plt.figure()
        # plt.plot(x_data,mean_in, x_data,beta2_in[0:-1])
        
        # plt.figure()
        # plt.plot(beta1_in[0:-1])
        # plt.plot(beta2_in[0:-1])
        # # plt.plot(beta1_sim[0:-1])
        # # plt.plot(beta2_sim[0:-1])
        # plt.plot(beta1_out2)
        # plt.plot(beta2_out2)
        # plt.legend(['Beta1 In', 'Beta2 In', 'Beta1 Out', 'Beta2 Out'])

    #Event from dbz into mm/h
    R_sim_mmh = R_sim.copy()
    R_sim_mmh = 10**((R_sim_mmh-10*np.log10(a_R))/(10*b_R))
    #Values less than threshold to zero
    R_sim_mmh[R_sim_mmh < 0.1] = 0
    
    #accumulation plot
    if show_cumul_plot:
        R_acc = np.sum(R_sim_mmh,axis=0)
        # cmap, norm, clevs, clevs_str = get_colormap_own("intensity","mm","pysteps")
        plt.figure()
        # plt.imshow(R_acc, cmap ="Blues", alpha = 0.7, interpolation ='bilinear', extent = extent)
        # plt.imshow(R_acc, cmap=cmap, alpha = 0.7, interpolation ='bilinear', extent = extent)
        plt.imshow(R_acc, cmap ="nipy_spectral", alpha = 0.7, interpolation ='bilinear', extent = extent)
        plt.colorbar()
        if save_cumul_plot:
            plt.savefig(os.path.join(out_dir, "accumulation_plot_spectral_observed.png"))
            
            crs = from_epsg(3067) 
            # Coordinates of clipped osapol data: xmin = 214000, xmax = 470000, ymin = 6720500, ymax = 6976500
            transform = from_origin(214000, 6976500, 1000, 1000) #rasterio.transform.from_origin(west, north, xsize, ysize)
            acc_plot = rasterio.open(os.path.join(out_dir, "accumulation_raster_spectral_observed.tif"), "w", driver="GTiff",
                                        height = R_acc.shape[0], width = R_acc.shape[1],
                                        count=1, dtype=str(R_acc.dtype),
                                        crs=crs,
                                        transform=transform)
            acc_plot.write(R_acc, 1)
            acc_plot.close()
            
    ##############################################################################
    # SAVE SIMULATED EVENT IN TIFF-FORMAT
    
    #dbz
    if save_tiff:
        out_dir4 = os.path.join(out_dir, "Event_tiffs_dbz")
        if not os.path.exists(out_dir4):
            os.makedirs(out_dir4)
        crs = from_epsg(3067) 
        # Coordinates of clipped osapol data: xmin = 214000, xmax = 470000, ymin = 6720500, ymax = 6976500
        transform = from_origin(214000, 6976500, 1000, 1000) #rasterio.transform.from_origin(west, north, xsize, ysize)
        for i in range(0,len(R_sim)):
            arr = R_sim[i]
            new_dataset = rasterio.open(os.path.join(out_dir4, f"test_{i}.tif"), "w", driver="GTiff",
                                        height = arr.shape[0], width = arr.shape[1],
                                        count=1, dtype=str(arr.dtype),
                                        crs=crs,
                                        transform=transform)
            new_dataset.write(arr, 1)
            new_dataset.close()
    
    #mm/h
    if save_tiff:
        out_dir_mmh_tiff = os.path.join(out_dir, "Event_tiffs_mmh")
        if not os.path.exists(out_dir_mmh_tiff):
            os.makedirs(out_dir_mmh_tiff)
        crs = from_epsg(3067) 
        # Coordinates of clipped osapol data: xmin = 214000, xmax = 470000, ymin = 6720500, ymax = 6976500
        transform = from_origin(214000, 6976500, 1000, 1000) #rasterio.transform.from_origin(west, north, xsize, ysize)
        for i in range(0,len(R_sim_mmh)):
            arr = R_sim_mmh[i]
            new_dataset = rasterio.open(os.path.join(out_dir_mmh_tiff, f"test_{i}.tif"), "w", driver="GTiff",
                                        height = arr.shape[0], width = arr.shape[1],
                                        count=1, dtype=str(arr.dtype),
                                        crs=crs,
                                        transform=transform)
            new_dataset.write(arr, 1)
            new_dataset.close()
            
    ##### Saving tiffs takes about 1 mins.
    
    ##############################################################################
    # PLOT THE OBSERVED 1D POWER SPECTRUM AND THE MODEL
    
    #The parametric model uses a piece-wise linear function with two spectral slopes (beta1 and beta2) and one breaking point
    #https://pysteps.readthedocs.io/en/latest/auto_examples/plot_noise_generators.html
    
    # for i in range(len(all_Fp)):
    #     #Compute the observed and fitted 1D PSD
    #     L = np.max(all_Fp[i]["input_shape"])
    #     if L % 2 == 1:
    #         wn = np.arange(0, int(L / 2) + 1)
    #     else:
    #         wn = np.arange(0, int(L / 2))
    #     R_, freq = rapsd(R_sim[i], fft_method=np.fft, return_freq=True)
    #     f = np.exp(all_Fp[i]["model"](np.log(wn), *all_Fp[i]["pars"]))
        
    #     all_R_.append(R_)
    #     all_freq.append(freq)
    #     all_f.append(f)
        
    #     #Extract the scaling break in km, beta1 and beta2
    #     w0 = scale_break
    #     b1 = all_Fp[i]["pars"][1]
    #     b2 = all_Fp[i]["pars"][2]
    #     all_b1.append(b1)
    #     all_b2.append(b2)
    
    # # Plot 1d power spectrum
    # fig, ax = plt.subplots()
    # plot_scales = [256, 128, 64, 32, 16, 8, 4, 2]
    # plot_spectrum1d(
    #     all_freq[0],
    #     all_R_[0],
    #     x_units="km",
    #     y_units="dBZ",
    #     color="k",
    #     ax=ax,
    #     label="Observed",
    #     wavelength_ticks=plot_scales,
    # )
    # # # Vile: added for test purposes
    # # plot_spectrum1d(
    # #     all_freq_fp1[0],
    # #     all_R_fp1_[0],
    # #     x_units="km",
    # #     y_units="dBZ",
    # #     color="k",
    # #     ax=ax,
    # #     label="Observed",
    # #     wavelength_ticks=plot_scales,
    # # )
    # # # Vile: added for test purposes
    # # plot_spectrum1d(
    # #     all_freq_fp1[0][:-1],
    # #     all_f_fp1[0],
    # #     x_units="km",
    # #     y_units="dBZ",
    # #     color="r",
    # #     ax=ax,
    # #     label="Fit",
    # #     wavelength_ticks=plot_scales,
    # # )
    # plot_spectrum1d(
    #     all_freq[0][:-1],
    #     all_f[0],
    #     x_units="km",
    #     y_units="dBZ",
    #     color="r",
    #     ax=ax,
    #     label="Fit",
    #     wavelength_ticks=plot_scales,
    # )
    # plt.legend()
    # ax.set_title(
    #     "Radially averaged log-power spectrum of R\n"
    #     r"$\omega_0=%.0f km, \beta_1=%.1f, \beta_2=%.1f$" % (w0, all_b1[0], all_b2[0])
    # )
    # plt.show()
    
    ##############################################################################
    # TIME SERIES FROM 5 DIFFERENT LOCATIONS 

    spots_raster = np.zeros((R_sim.shape[1], R_sim.shape[2]))
    # R_sim[layer, row, column]
    
    # #center
    # spots_raster[int(R_sim.shape[1]/2), int(R_sim.shape[2]/2)] = 1 #512, 512
    # spots_raster[int(R_sim.shape[1]/2), int(R_sim.shape[2]/2)+1] = 1 #512, 513
    # spots_raster[int(R_sim.shape[1]/2)+1, int(R_sim.shape[2]/2)] = 1 #513, 512
    # spots_raster[int(R_sim.shape[1]/2)+1, int(R_sim.shape[2]/2)+1] = 1 #513, 513
    
    # #upper-left
    # spots_raster[int(R_sim.shape[1]/4), int(R_sim.shape[2]/4)] = 2 #256, 256
    # spots_raster[int(R_sim.shape[1]/4), int(R_sim.shape[2]/4)+1] = 2 #256, 257
    # spots_raster[int(R_sim.shape[1]/4)+1, int(R_sim.shape[2]/4)] = 2 #257, 256
    # spots_raster[int(R_sim.shape[1]/4)+1, int(R_sim.shape[2]/4)+1] = 2 #257, 257
    
    # #upper-right
    # spots_raster[int(R_sim.shape[1]/4), int(R_sim.shape[2]/4*3)] = 3 #256, 768
    # spots_raster[int(R_sim.shape[1]/4), int(R_sim.shape[2]/4*3)+1] = 3 #256, 769
    # spots_raster[int(R_sim.shape[1]/4)+1, int(R_sim.shape[2]/4*3)] = 3 #257, 768
    # spots_raster[int(R_sim.shape[1]/4)+1, int(R_sim.shape[2]/4*3)+1] = 3 #257, 769
    
    # #lower-left
    # spots_raster[int(R_sim.shape[1]/4*3), int(R_sim.shape[2]/4)] = 4 #768, 256
    # spots_raster[int(R_sim.shape[1]/4*3), int(R_sim.shape[2]/4)+1] = 4 #768, 257
    # spots_raster[int(R_sim.shape[1]/4*3)+1, int(R_sim.shape[2]/4)] = 4 #769, 256
    # spots_raster[int(R_sim.shape[1]/4*3)+1, int(R_sim.shape[2]/4)+1] = 4 #769, 257
    
    # #lower-right
    # spots_raster[int(R_sim.shape[1]/4*3), int(R_sim.shape[2]/4*3)] = 5 #768, 768
    # spots_raster[int(R_sim.shape[1]/4*3), int(R_sim.shape[2]/4*3)+1] = 5 #768, 769
    # spots_raster[int(R_sim.shape[1]/4*3)+1, int(R_sim.shape[2]/4*3)] = 5 #769, 768
    # spots_raster[int(R_sim.shape[1]/4*3)+1, int(R_sim.shape[2]/4*3)+1] = 5 #769, 769

    # [y-coord, x-coord]
    #first row
    spots_raster[int(R_sim.shape[1]/4), int(R_sim.shape[2]/4)] = 100 #upper-left: 256, 256
    spots_raster[int(R_sim.shape[1]/4), int((R_sim.shape[2]/2 + R_sim.shape[2]/4)/2)] = 100
    spots_raster[int(R_sim.shape[1]/4), int(R_sim.shape[2]/2)] = 100 #upper_middle = 256, 512
    spots_raster[int(R_sim.shape[1]/4), int((R_sim.shape[2]/4*3 + R_sim.shape[2]/2)/2)] = 100
    spots_raster[int(R_sim.shape[1]/4), int(R_sim.shape[2]/4*3)] = 100 #upper-right: 256, 768
    
    #second row
    spots_raster[int((R_sim.shape[1]/2 + R_sim.shape[1]/4)/2), int(R_sim.shape[2]/4)] = 100 #upper-left: 256, 256
    spots_raster[int((R_sim.shape[1]/2 + R_sim.shape[1]/4)/2), int((R_sim.shape[2]/2 + R_sim.shape[2]/4)/2)] = 100
    spots_raster[int((R_sim.shape[1]/2 + R_sim.shape[1]/4)/2), int(R_sim.shape[2]/2)] = 100 #upper_middle = 256, 512
    spots_raster[int((R_sim.shape[1]/2 + R_sim.shape[1]/4)/2), int((R_sim.shape[2]/4*3 + R_sim.shape[2]/2)/2)] = 100
    spots_raster[int((R_sim.shape[1]/2 + R_sim.shape[1]/4)/2), int(R_sim.shape[2]/4*3)] = 100 #upper-right: 256, 768
    
    #third row
    spots_raster[int(R_sim.shape[1]/2), int(R_sim.shape[2]/4)] = 100 #middle-left: 512, 256
    spots_raster[int(R_sim.shape[1]/2), int((R_sim.shape[2]/2 + R_sim.shape[2]/4)/2)] = 100
    spots_raster[int(R_sim.shape[1]/2), int(R_sim.shape[2]/2)] = 100 #center: 512, 512
    spots_raster[int(R_sim.shape[1]/2), int((R_sim.shape[2]/4*3 + R_sim.shape[2]/2)/2)] = 100
    spots_raster[int(R_sim.shape[1]/2), int(R_sim.shape[2]/4*3)] = 100 #middle-right: 512, 768
    
    #forth row
    spots_raster[int((R_sim.shape[1]/4*3 + R_sim.shape[1]/2)/2), int(R_sim.shape[2]/4)] = 100 #middle-left: 512, 256
    spots_raster[int((R_sim.shape[1]/4*3 + R_sim.shape[1]/2)/2), int((R_sim.shape[2]/2 + R_sim.shape[2]/4)/2)] = 100
    spots_raster[int((R_sim.shape[1]/4*3 + R_sim.shape[1]/2)/2), int(R_sim.shape[2]/2)] = 100 #center: 512, 512
    spots_raster[int((R_sim.shape[1]/4*3 + R_sim.shape[1]/2)/2), int((R_sim.shape[2]/4*3 + R_sim.shape[2]/2)/2)] = 100
    spots_raster[int((R_sim.shape[1]/4*3 + R_sim.shape[1]/2)/2), int(R_sim.shape[2]/4*3)] = 100 #middle-right: 512, 768
    
    #fifth row
    spots_raster[int(R_sim.shape[1]/4*3), int(R_sim.shape[2]/4)] = 100 #lower-left: 768, 256
    spots_raster[int(R_sim.shape[1]/4*3), int((R_sim.shape[2]/2 + R_sim.shape[2]/4)/2)] = 100
    spots_raster[int(R_sim.shape[1]/4*3), int(R_sim.shape[2]/2)] = 100 #lower-middle: 768, 512
    spots_raster[int(R_sim.shape[1]/4*3), int((R_sim.shape[2]/4*3 + R_sim.shape[2]/2)/2)] = 100
    spots_raster[int(R_sim.shape[1]/4*3), int(R_sim.shape[2]/4*3)] = 100 #lower-right: 768, 768
    
    #plot locations
    plt.figure()
    plt.imshow(spots_raster)
    if save_figs:
        plt.savefig(os.path.join(out_dir, "point_locations.png"))

    # #Unit convesion
    # event_sim_array = R_sim.copy()
    # # #Clear values over threshold of 45 dBZ
    # # event_sim_array[event_sim_array > 45] = 0.5*(45 + event_sim_array[event_sim_array > 45])
    # # #Z-R relationship: Z = a*R^b (Reflectivity)
    # # a_R=223
    # # b_R=1.53
    # #Event from dbz into mm/h
    # event_sim_array = 10**((event_sim_array-10*np.log10(a_R))/(10*b_R))
    # #Values less than threshold to zero
    # event_sim_array[event_sim_array < 0.1] = 0
    
    event_sim_array = R_sim_mmh.copy()
    
    # #To get point time series for observed event
    # event_sim_array = radar_kiira.copy()
    # event_sim_array = 10**((event_sim_array-10*np.log10(a_R))/(10*b_R))
    # event_sim_array[event_sim_array < 0.1] = 0
    
    # ts_center = np.zeros((4, len(event_sim_array)))
    # ts_up_left = np.zeros((4, len(event_sim_array)))
    # ts_up_right = np.zeros((4, len(event_sim_array)))
    # ts_low_left = np.zeros((4, len(event_sim_array)))
    # ts_low_right = np.zeros((4, len(event_sim_array)))
    
    # ts_up_left = np.zeros((1, len(event_sim_array)))
    # ts_up_mid = np.zeros((1, len(event_sim_array)))
    # ts_up_right = np.zeros((1, len(event_sim_array)))
    # ts_mid_left = np.zeros((1, len(event_sim_array)))
    # ts_mid_mid = np.zeros((1, len(event_sim_array)))
    # ts_mid_right = np.zeros((1, len(event_sim_array)))
    # ts_low_left = np.zeros((1, len(event_sim_array)))
    # ts_low_mid = np.zeros((1, len(event_sim_array)))
    # ts_low_right = np.zeros((1, len(event_sim_array)))
    
    ts_1 = np.zeros((1, len(event_sim_array)))
    ts_2 = np.zeros((1, len(event_sim_array)))
    ts_3 = np.zeros((1, len(event_sim_array)))
    ts_4 = np.zeros((1, len(event_sim_array)))
    ts_5 = np.zeros((1, len(event_sim_array)))
    ts_6 = np.zeros((1, len(event_sim_array)))
    ts_7 = np.zeros((1, len(event_sim_array)))
    ts_8 = np.zeros((1, len(event_sim_array)))
    ts_9 = np.zeros((1, len(event_sim_array)))
    ts_10 = np.zeros((1, len(event_sim_array)))
    ts_11 = np.zeros((1, len(event_sim_array)))
    ts_12 = np.zeros((1, len(event_sim_array)))
    ts_13 = np.zeros((1, len(event_sim_array)))
    ts_14 = np.zeros((1, len(event_sim_array)))
    ts_15 = np.zeros((1, len(event_sim_array)))
    ts_16 = np.zeros((1, len(event_sim_array)))
    ts_17 = np.zeros((1, len(event_sim_array)))
    ts_18 = np.zeros((1, len(event_sim_array)))
    ts_19 = np.zeros((1, len(event_sim_array)))
    ts_20 = np.zeros((1, len(event_sim_array)))
    ts_21 = np.zeros((1, len(event_sim_array)))
    ts_22 = np.zeros((1, len(event_sim_array)))
    ts_23 = np.zeros((1, len(event_sim_array)))
    ts_24 = np.zeros((1, len(event_sim_array)))
    ts_25 = np.zeros((1, len(event_sim_array)))
    
    # for i in range(len(event_sim_array)):
    #     #center
    #     ts_center[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/2)] #512, 512
    #     ts_center[1,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/2)+1] #512, 513
    #     ts_center[2,i] = event_sim_array[i, int(event_sim_array.shape[1]/2)+1, int(event_sim_array.shape[2]/2)] #513, 512
    #     ts_center[3,i] = event_sim_array[i, int(event_sim_array.shape[1]/2)+1, int(event_sim_array.shape[2]/2)+1] #513, 513
    #     #upper-left
    #     ts_up_left[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/4)] #256, 256
    #     ts_up_left[1,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/4)+1] #256, 257
    #     ts_up_left[2,i] = event_sim_array[i, int(event_sim_array.shape[1]/4)+1, int(event_sim_array.shape[2]/4)] #257, 256
    #     ts_up_left[3,i] = event_sim_array[i, int(event_sim_array.shape[1]/4)+1, int(event_sim_array.shape[2]/4)+1] #257, 257
    #     #upper-right
    #     ts_up_right[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/4*3)] #256, 768
    #     ts_up_right[1,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/4*3)+1] #256, 769
    #     ts_up_right[2,i] = event_sim_array[i, int(event_sim_array.shape[1]/4)+1, int(event_sim_array.shape[2]/4*3)] #257, 768
    #     ts_up_right[3,i] = event_sim_array[i, int(event_sim_array.shape[1]/4)+1, int(event_sim_array.shape[2]/4*3)+1] #257, 769
    #     #lower-left
    #     ts_low_left[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/4)] #768, 256
    #     ts_low_left[1,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/4)+1] #768, 257
    #     ts_low_left[2,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3)+1, int(event_sim_array.shape[2]/4)] #769, 256
    #     ts_low_left[3,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3)+1, int(event_sim_array.shape[2]/4)+1] #769, 257
    #     #lower-right
    #     ts_low_right[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/4*3)] #768, 768
    #     ts_low_right[1,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/4*3)+1] #768, 769
    #     ts_low_right[2,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3)+1, int(event_sim_array.shape[2]/4*3)] #769, 768
    #     ts_low_right[3,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3)+1, int(event_sim_array.shape[2]/4*3)+1] #769, 769
    
    for i in range(len(event_sim_array)):
        # ts_up_left[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/4)]
        # ts_up_mid[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/2)]
        # ts_up_right[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/4*3)]
        # ts_mid_left[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/4)]
        # ts_mid_mid[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/2)]
        # ts_mid_right[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/4*3)]
        # ts_low_left[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/4)]
        # ts_low_mid[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/2)]
        # ts_low_right[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/4*3)]
        
        #first row
        ts_1[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/4)]
        ts_2[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int((event_sim_array.shape[2]/2 + event_sim_array.shape[2]/4)/2)]
        ts_3[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/2)]
        ts_4[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int((event_sim_array.shape[2]/4*3 + event_sim_array.shape[2]/2)/2)]
        ts_5[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/4*3)]
        #second row
        ts_6[0,i] = event_sim_array[i, int((event_sim_array.shape[1]/2 + event_sim_array.shape[1]/4)/2), int(event_sim_array.shape[2]/4)]
        ts_7[0,i] = event_sim_array[i, int((event_sim_array.shape[1]/2 + event_sim_array.shape[1]/4)/2), int((event_sim_array.shape[2]/2 + event_sim_array.shape[2]/4)/2)]
        ts_8[0,i] = event_sim_array[i, int((event_sim_array.shape[1]/2 + event_sim_array.shape[1]/4)/2), int(event_sim_array.shape[2]/2)]
        ts_9[0,i] = event_sim_array[i, int((event_sim_array.shape[1]/2 + event_sim_array.shape[1]/4)/2), int((event_sim_array.shape[2]/4*3 + event_sim_array.shape[2]/2)/2)]
        ts_10[0,i] = event_sim_array[i, int((event_sim_array.shape[1]/2 + event_sim_array.shape[1]/4)/2), int(event_sim_array.shape[2]/4*3)]
        #third row
        ts_11[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/4)]
        ts_12[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int((event_sim_array.shape[2]/2 + event_sim_array.shape[2]/4)/2)]
        ts_13[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/2)]
        ts_14[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int((event_sim_array.shape[2]/4*3 + event_sim_array.shape[2]/2)/2)]
        ts_15[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/4*3)]
        #forth row
        ts_16[0,i] = event_sim_array[i, int((event_sim_array.shape[1]/4*3 + event_sim_array.shape[1]/2)/2), int(event_sim_array.shape[2]/4)]
        ts_17[0,i] = event_sim_array[i, int((event_sim_array.shape[1]/4*3 + event_sim_array.shape[1]/2)/2), int((event_sim_array.shape[2]/2 + event_sim_array.shape[2]/4)/2)]
        ts_18[0,i] = event_sim_array[i, int((event_sim_array.shape[1]/4*3 + event_sim_array.shape[1]/2)/2), int(event_sim_array.shape[2]/2)]
        ts_19[0,i] = event_sim_array[i, int((event_sim_array.shape[1]/4*3 + event_sim_array.shape[1]/2)/2), int((event_sim_array.shape[2]/4*3 + event_sim_array.shape[2]/2)/2)]
        ts_20[0,i] = event_sim_array[i, int((event_sim_array.shape[1]/4*3 + event_sim_array.shape[1]/2)/2), int(event_sim_array.shape[2]/4*3)]
        #fifth row
        ts_21[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/4)]
        ts_22[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int((event_sim_array.shape[2]/2 + event_sim_array.shape[2]/4)/2)]
        ts_23[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/2)]
        ts_24[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int((event_sim_array.shape[2]/4*3 + event_sim_array.shape[2]/2)/2)]
        ts_25[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/4*3)]

    # limit_upper = round(np.max((ts_center, ts_up_left, ts_up_right, ts_low_left, ts_low_right))+0.5)
    limit_upper = round(np.max((ts_1, ts_2, ts_3, ts_4, ts_5,
                                ts_6, ts_7, ts_8, ts_9, ts_10,
                                ts_11, ts_12, ts_13, ts_14, ts_15,
                                ts_16, ts_17, ts_18, ts_19, ts_20,
                                ts_21, ts_22, ts_23, ts_24, ts_25))+0.5)
    limit_lower = 0
    
    # plt.figure()
    # for row in range(len(ts_center)):
    #     plt.plot(ts_center[row])
    # plt.title("center")
    # plt.ylim(limit_lower, limit_upper)
    # if save_figs:
    #     plt.savefig(os.path.join(out_dir, "tss_mmh_center.png"))
        
    # plt.figure()
    # for row in range(len(ts_up_left)):
    #     plt.plot(ts_up_left[row])
    # plt.title("upper-left")
    # plt.ylim(limit_lower, limit_upper)
    # if save_figs:
    #     plt.savefig(os.path.join(out_dir, "tss_mmh_ul.png"))
        
    # plt.figure()
    # for row in range(len(ts_up_right)):
    #     plt.plot(ts_up_right[row])
    # plt.title("upper-right")
    # plt.ylim(limit_lower, limit_upper)
    # if save_figs:
    #     plt.savefig(os.path.join(out_dir, "tss_mmh_ur.png"))
        
    # plt.figure()
    # for row in range(len(ts_low_left)):
    #     plt.plot(ts_low_left[row])
    # plt.title("lower-left")
    # plt.ylim(limit_lower, limit_upper)
    # if save_figs:
    #     plt.savefig(os.path.join(out_dir, "tss_mmh_ll.png"))
    
    # plt.figure()
    # for row in range(len(ts_low_right)):
    #     plt.plot(ts_low_right[row])
    # plt.title("lower-right")
    # plt.ylim(limit_lower, limit_upper)
    # if save_figs:
    # #     plt.savefig(os.path.join(out_dir, "tss_mmh_lr.png"))
        
    # ts_averages = np.zeros((5, len(event_sim_array)))
    # for i in range(len(event_sim_array)):
    #     ts_averages[0,i] = np.mean(ts_center[:,i])
    #     ts_averages[1,i] = np.mean(ts_up_left[:,i])
    #     ts_averages[2,i] = np.mean(ts_up_right[:,i])
    #     ts_averages[3,i] = np.mean(ts_low_left[:,i])
    #     ts_averages[4,i] = np.mean(ts_low_right[:,i])
    
    # #Plot 4-pixel-averages for each 5 locations
    # plt.figure()
    # for row in range(len(ts_averages)):
    #     plt.plot(ts_averages[row])
    # plt.title("avereges of 4 pixels")
    # plt.ylim(limit_lower, limit_upper)
    # if save_figs:
    #     plt.savefig(os.path.join(out_dir, "tss_mmh_loc_averages.png"))
    
    # if csv_locations:
    #     np.savetxt(os.path.join(out_dir, "tss_mmh_center.csv"), ts_averages, delimiter = ";")
    #     np.savetxt(os.path.join(out_dir, "tss_mmh_ul.csv"), ts_averages, delimiter = ";")
    #     np.savetxt(os.path.join(out_dir, "tss_mmh_ur.csv"), ts_averages, delimiter = ";")
    #     np.savetxt(os.path.join(out_dir, "tss_mmh_ll.csv"), ts_averages, delimiter = ";")
    #     np.savetxt(os.path.join(out_dir, "tss_mmh_lr.csv"), ts_averages, delimiter = ";")
    #     np.savetxt(os.path.join(out_dir, "tss_mmh_loc_averages.csv"), ts_averages, delimiter = ";")
    
    plt.figure()
    # plt.plot(ts_up_left[0])
    # plt.plot(ts_up_mid[0])
    # plt.plot(ts_up_right[0])
    # plt.plot(ts_mid_left[0])
    # plt.plot(ts_mid_mid[0])
    # plt.plot(ts_mid_right[0])
    # plt.plot(ts_low_left[0])
    # plt.plot(ts_low_mid[0])
    # plt.plot(ts_low_right[0])
    plt.plot(ts_1[0])
    plt.plot(ts_2[0])
    plt.plot(ts_3[0])
    plt.plot(ts_4[0])
    plt.plot(ts_5[0])
    plt.plot(ts_6[0])
    plt.plot(ts_7[0])
    plt.plot(ts_8[0])
    plt.plot(ts_9[0])
    plt.plot(ts_10[0])
    plt.plot(ts_11[0])
    plt.plot(ts_12[0])
    plt.plot(ts_13[0])
    plt.plot(ts_14[0])
    plt.plot(ts_15[0])
    plt.plot(ts_16[0])
    plt.plot(ts_17[0])
    plt.plot(ts_18[0])
    plt.plot(ts_19[0])
    plt.plot(ts_20[0])
    plt.plot(ts_21[0])
    plt.plot(ts_22[0])
    plt.plot(ts_23[0])
    plt.plot(ts_24[0])
    plt.plot(ts_25[0])
    plt.ylim(limit_lower, limit_upper)
    # plt.ylim(limit_lower, 30)
    if save_figs:
        plt.savefig(os.path.join(out_dir, "tss_point_mmh.png"))
        # plt.savefig(os.path.join(out_dir, "tss_point_mmh_cut.png"))
        
    if csv_locations:
        # data_temp = [ts_up_left[0], ts_up_mid[0], ts_up_right[0], ts_mid_left[0], ts_mid_mid[0], ts_mid_right[0], ts_low_left[0], ts_low_mid[0], ts_low_right[0]]
        # mmh_ts = pd.DataFrame(data_temp, index=['ts_up_left', 'ts_up_mid', 'ts_up_right', 'ts_mid_left', 'ts_mid_mid', 'ts_mid_right', 'ts_low_left', 'ts_low_mid', 'ts_low_right'])
        data_temp = [ts_1[0], ts_2[0], ts_3[0], ts_4[0], ts_5[0], ts_6[0], ts_7[0], ts_8[0], ts_9[0],
                     ts_10[0], ts_11[0], ts_12[0], ts_13[0], ts_14[0], ts_15[0], ts_16[0], ts_17[0], ts_18[0],
                     ts_19[0], ts_20[0], ts_21[0], ts_22[0], ts_23[0], ts_24[0], ts_25[0]]
        mmh_ts = pd.DataFrame(data_temp, index=['ts_1', 'ts_2', 'ts_3', 'ts_4', 'ts_5', 'ts_6', 'ts_7', 'ts_8', 'ts_9',
                                                'ts_10', 'ts_11', 'ts_12', 'ts_13', 'ts_14', 'ts_15', 'ts_16', 'ts_17', 'ts_18',
                                                'ts_19', 'ts_20', 'ts_21', 'ts_22', 'ts_23', 'ts_24', 'ts_25'])
        pd.DataFrame(mmh_ts).to_csv(os.path.join(out_dir, "tss_point_mmh.csv"))
    
    ##############################################################################
    
    # #Plot correlations for center
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    # ax1.scatter(ts_averages[0], ts_averages[1])
    # ax2.scatter(ts_averages[0], ts_averages[2])
    # ax3.scatter(ts_averages[0], ts_averages[3])
    # ax4.scatter(ts_averages[0], ts_averages[4])
    # ax1.title.set_text("center-ul")
    # ax2.title.set_text("center-ur")
    # ax3.title.set_text("center-ll")
    # ax4.title.set_text("center-lr")
    # textstr1 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[0], ts_averages[1])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[0], ts_averages[1]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[0], ts_averages[1]).correlation)))
    # ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, verticalalignment='top')
    # textstr2 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[0], ts_averages[2])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[0], ts_averages[2]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[0], ts_averages[2]).correlation)))
    # ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, verticalalignment='top')
    # textstr3 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[0], ts_averages[3])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[0], ts_averages[3]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[0], ts_averages[3]).correlation)))
    # ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, verticalalignment='top')
    # textstr4 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[0], ts_averages[4])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[0], ts_averages[4]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[0], ts_averages[4]).correlation)))
    # ax4.text(0.05, 0.95, textstr4, transform=ax4.transAxes, verticalalignment='top')
    # if save_figs:
    #     plt.savefig(os.path.join(out_dir, "cors_center.png"))
    
    # #Plot correlations for upper-left
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    # ax1.scatter(ts_averages[1], ts_averages[0])
    # ax2.scatter(ts_averages[1], ts_averages[2])
    # ax3.scatter(ts_averages[1], ts_averages[3])
    # ax4.scatter(ts_averages[1], ts_averages[4])
    # ax1.title.set_text("ul-center")
    # ax2.title.set_text("ul-ur")
    # ax3.title.set_text("ul-ll")
    # ax4.title.set_text("ul-lr")
    # textstr1 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[1], ts_averages[0])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[1], ts_averages[0]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[1], ts_averages[0]).correlation)))
    # ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, verticalalignment='top')
    # textstr2 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[1], ts_averages[2])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[1], ts_averages[2]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[1], ts_averages[2]).correlation)))
    # ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, verticalalignment='top')
    # textstr3 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[1], ts_averages[3])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[1], ts_averages[3]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[1], ts_averages[3]).correlation)))
    # ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, verticalalignment='top')
    # textstr4 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[1], ts_averages[4])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[1], ts_averages[4]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[1], ts_averages[4]).correlation)))
    # ax4.text(0.05, 0.95, textstr4, transform=ax4.transAxes, verticalalignment='top')
    # if save_figs:
    #     plt.savefig(os.path.join(out_dir, "cors_ul.png"))
    
    # #Plot correlations for upper-right
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    # ax1.scatter(ts_averages[2], ts_averages[0])
    # ax2.scatter(ts_averages[2], ts_averages[1])
    # ax3.scatter(ts_averages[2], ts_averages[3])
    # ax4.scatter(ts_averages[2], ts_averages[4])
    # ax1.title.set_text("ur-center")
    # ax2.title.set_text("ur-ul")
    # ax3.title.set_text("ur-ll")
    # ax4.title.set_text("ur-lr")
    # textstr1 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[2], ts_averages[0])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[2], ts_averages[0]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[2], ts_averages[0]).correlation)))
    # ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, verticalalignment='top')
    # textstr2 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[2], ts_averages[1])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[2], ts_averages[1]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[2], ts_averages[1]).correlation)))
    # ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, verticalalignment='top')
    # textstr3 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[2], ts_averages[3])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[2], ts_averages[3]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[2], ts_averages[3]).correlation)))
    # ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, verticalalignment='top')
    # textstr4 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[2], ts_averages[4])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[2], ts_averages[4]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[2], ts_averages[4]).correlation)))
    # ax4.text(0.05, 0.95, textstr4, transform=ax4.transAxes, verticalalignment='top')
    # if save_figs:
    #     plt.savefig(os.path.join(out_dir, "cors_ur.png"))

    # #Plot correlations for lower-left
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    # ax1.scatter(ts_averages[3], ts_averages[0])
    # ax2.scatter(ts_averages[3], ts_averages[1])
    # ax3.scatter(ts_averages[3], ts_averages[2])
    # ax4.scatter(ts_averages[3], ts_averages[4])
    # ax1.title.set_text("ll-center")
    # ax2.title.set_text("ll-ul")
    # ax3.title.set_text("ll-ur")
    # ax4.title.set_text("ll-lr")
    # textstr1 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[3], ts_averages[0])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[3], ts_averages[0]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[3], ts_averages[0]).correlation)))
    # ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, verticalalignment='top')
    # textstr2 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[3], ts_averages[1])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[3], ts_averages[1]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[3], ts_averages[1]).correlation)))
    # ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, verticalalignment='top')
    # textstr3 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[3], ts_averages[2])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[3], ts_averages[2]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[3], ts_averages[2]).correlation)))
    # ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, verticalalignment='top')
    # textstr4 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[3], ts_averages[4])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[3], ts_averages[4]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[3], ts_averages[4]).correlation)))
    # ax4.text(0.05, 0.95, textstr4, transform=ax4.transAxes, verticalalignment='top')
    # if save_figs:
    #     plt.savefig(os.path.join(out_dir, "cors_ll.png"))
    
    # #Plot correlations for lower-left
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    # ax1.scatter(ts_averages[4], ts_averages[0])
    # ax2.scatter(ts_averages[4], ts_averages[1])
    # ax3.scatter(ts_averages[4], ts_averages[2])
    # ax4.scatter(ts_averages[4], ts_averages[3])
    # ax1.title.set_text("lr-center")
    # ax2.title.set_text("lr-ul")
    # ax3.title.set_text("lr-ur")
    # ax4.title.set_text("lr-ll")
    # textstr1 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[4], ts_averages[0])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[4], ts_averages[0]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[4], ts_averages[0]).correlation)))
    # ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, verticalalignment='top')
    # textstr2 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[4], ts_averages[1])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[4], ts_averages[1]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[4], ts_averages[1]).correlation)))
    # ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, verticalalignment='top')
    # textstr3 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[4], ts_averages[2])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[4], ts_averages[2]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[4], ts_averages[2]).correlation)))
    # ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, verticalalignment='top')
    # textstr4 = '\n'.join((
    #     r'p=%.3f' % (stats.pearsonr(ts_averages[4], ts_averages[3])[0]),
    #     r's=%.3f' % (stats.spearmanr(ts_averages[4], ts_averages[3]).correlation),
    #     r'k=%.3f' % (stats.kendalltau(ts_averages[4], ts_averages[3]).correlation)))
    # ax4.text(0.05, 0.95, textstr4, transform=ax4.transAxes, verticalalignment='top')
    # if save_figs:
    #     plt.savefig(os.path.join(out_dir, "cors_lr.png"))
    
    ##############################################################################
    # SAVE TIMESERIES
    
    if csv_brokenlines:
        data_temp2 = [r_mean, r_mean_org, vx, vy]
        bl_arrays_to_save = pd.DataFrame(data_temp2, index=['r_mean_scaled', 'r_mean_org', 'vx', 'vy'])
        pd.DataFrame(bl_arrays_to_save).to_csv(os.path.join(out_dir, "sim_brokenlines.csv"))
        
    # if csv_betas:
    #     data_temp3 = [all_b1, all_b2]
    #     beta_arrays_to_save = pd.DataFrame(data_temp3, index=['beta1', 'beta2'])
    #     pd.DataFrame(beta_arrays_to_save).to_csv(os.path.join(out_dir, "sim_betas.csv"))
    
    ##############################################################################
    plt.close("all")
    n_sim += 1
    
##############################################################################

# cmap, norm, clevs, clevs_str= get_colormap_own("intensity","dBZ","pysteps")
# out_input = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
# out_input = os.path.join(out_input, "Input_animation_pysteps")
# if not os.path.exists(out_input):
#     os.makedirs(out_input)

# for im in range(0,len(radar_kiira)):
#     plt.figure()
#     testi_im_pysteps = plt.imshow(radar_kiira[im], cmap=cmap, vmin=0, vmax=round(np.nanmax(R_sim)+0.5)) #"Blues"
#     plt.colorbar(testi_im_pysteps, spacing="uniform", extend="max", shrink=0.8, cax=None, label="Precipitation intensity [dBZ]")
#     plt.title(f"Step: {im}")
#     plt.savefig(os.path.join(out_input, f"input_pysteps_{im}.png"))
#     plt.close()
