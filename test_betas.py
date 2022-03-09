# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:25:07 2022

@author: tkokko
"""

import matplotlib.pyplot as plt
import pysteps
from datetime import datetime
import numpy as np

seed1 = 300
domain = "spatial"  # spatial or spectral


# INPUT DATA
#Read in the event with pySTEPS
date = datetime.strptime("201408071325", "%Y%m%d%H%M") #last radar image of the event
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
fns = pysteps.io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext,
timestep=5, num_prev_files=0)

#Select the importer
importer = pysteps.io.get_method(importer_name, "importer")

#Read the radar composites
R, quality, metadata = pysteps.io.read_timeseries(fns, importer, **importer_kwargs)
del quality #delete quality variable because it is not used

#Add unit to metadata
metadata["unit"] = "mm/h"

#Plot R-field
#plt.figure()
#pysteps.visualization.plot_precip_field(R[0], title=date)

# DATA TRANSFORMATION from mm/h into dbz
#Values less than threshold to zero
#Replace non-finite values with the minimum value
#R2 = R.copy()
#for i in range(R2.shape[0]):
#R2[i, ~np.isfinite(R[i, :])] = np.nanmin(R2[i, :])
#R[R<0.1] = 0

#Replace non-finite values with the minimum value
for i in range(R.shape[0]):
    R[i, ~np.isfinite(R[i, :])] = np.nanmin(R[i, :])


#Information into metadata
metadata["zerovalue"] = 0
metadata["threshold"] = 0.1


#dBZ transformation for mm/h-data (Cannot use dBR transformation.)
#Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with size. J. Meteor., 5, 165–166.
#Z-R relationship: Z = a*R^b (Reflectivity)
#dBZ conversion: dBZ = 10 * log10(Z) (Decibels of Reflectivity)
a_R=223
b_R=1.53
#Leinonen et al. 2012 - A Climatology of Disdrometer Measurements of Rainfall in Finland over Five Years with Implications for Global Radar Observations
#https://doi.org/10.1175/JAMC-D-11-056.1

#R from mm/h into dbz
R, metadata = pysteps.utils.conversion.to_reflectivity(R, metadata, zr_a=a_R, zr_b=b_R)

#Plot R-field
plt.figure()
pysteps.visualization.plot_precip_field(R[0], title=date)

scale_break_wn = np.log(256/18)
Fp = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(
        R[0], scale_break=scale_break_wn)
beta1 = Fp["pars"][1]
beta2 = Fp["pars"][2]
nx = R[0].shape[1]
ny = R[0].shape[0]
mean_obs = R[0].mean()
std_obs = R[0].std()
war_thold = 3.19  # below this value no rain
rain_zero_value = 3.19/2  # no rain pixels assigned with this value
war_obs = len(np.where(R[0] > war_thold)[0]) / (nx * ny)

nx_sim = 8192
ny_sim = 8192
p_pow = np.array([scale_break_wn, 0, -2.0, -2.0]) 
R_ini = np.random.normal(0.0, 1.0, size=(ny_sim, nx_sim))
noise_kwargs = dict()
fft_method = "numpy"
noise_method = "parametric_sim"
fft = pysteps.utils.get_method(fft_method, shape=(ny_sim, nx_sim), n_threads=1)
p_pow[2] = beta1
p_pow[3] = beta2
init_noise, generate_noise = pysteps.noise.get_method(noise_method)
pp = init_noise(R_ini, p_pow, fft_method=fft, **noise_kwargs)
R_sim = generate_noise(
        pp, randstate=None, fft_method=fft, domain=domain
    )
Fp = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(
        R_sim, scale_break=scale_break_wn)
beta1_sim1 = Fp["pars"][1]
beta2_sim1 = Fp["pars"][2]

stats_kwargs = dict()
stats_kwargs["mean"] = mean_obs
stats_kwargs["std"] = std_obs
stats_kwargs["war"] = war_obs
stats_kwargs["war_thold"] = war_thold
stats_kwargs["rain_zero_value"] = rain_zero_value
R_sim = pysteps.postprocessing.probmatching.set_stats(R_sim, stats_kwargs)
plt.figure()
pysteps.visualization.plot_precip_field(R_sim, title=date)
Fp = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(
        R_sim, scale_break=scale_break_wn)
beta1_sim2 = Fp["pars"][1]
beta2_sim2 = Fp["pars"][2]