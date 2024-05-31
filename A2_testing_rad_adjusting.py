# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:30:13 2023

@author: lindgrv1
"""

import wradlib.adjust as adjust
import wradlib.verify as verify
import wradlib.util as util
import numpy as np
import matplotlib.pyplot as pl

###### 1D
##############################################################################
###### Create synthetic data 
##############################################################################

# gage and radar coordinates
obs_coords = np.array([5, 10, 15, 20, 30, 45, 65, 70, 77, 90])
radar_coords = np.arange(0, 101)

# true rainfall
np.random.seed(1319622840)
truth = np.abs(1.5 + np.sin(0.075 * radar_coords)) + np.random.uniform(-0.1, 0.1, len(radar_coords))

# radar error
erroradd = 0.7 * np.sin(0.2 * radar_coords + 10.0)
errormult = 0.75 + 0.015 * radar_coords
noise = np.random.uniform(-0.05, 0.05, len(radar_coords))

# radar observation
radar = errormult * truth + erroradd + noise

# gage observations are assumed to be perfect
obs = truth[obs_coords]

# add a missing value to observations (just for testing)
obs[1] = np.nan

##############################################################################
##### Apply different adjustment methods
##############################################################################

# number of neighbours to be used
nnear_raws = 3

# adjust the radar observation by additive model
add_adjuster = adjust.AdjustAdd(obs_coords, radar_coords, nnear_raws=nnear_raws)
add_adjusted = add_adjuster(obs, radar)

# adjust the radar observation by multiplicative model
mult_adjuster = adjust.AdjustMultiply(obs_coords, radar_coords, nnear_raws=nnear_raws)
mult_adjusted = mult_adjuster(obs, radar)

# adjust the radar observation by AdjustMixed
mixed_adjuster = adjust.AdjustMixed(obs_coords, radar_coords, nnear_raws=nnear_raws)
mixed_adjusted = mixed_adjuster(obs, radar)

# adjust the radar observation by MFB
mfb_adjuster = adjust.AdjustMFB(obs_coords, radar_coords, nnear_raws=nnear_raws, mfb_args=dict(method="median"))
mfb_adjusted = mfb_adjuster(obs, radar)

##############################################################################
##### Plot adjustment results
##############################################################################

# Enlarge all label fonts
font = {"size": 15}
pl.rc("font", **font)

pl.figure(figsize=(10, 5))
pl.plot(
    radar_coords,
    radar,
    "k-",
    linewidth=2.0,
    linestyle="dashed",
    label="Unadjusted radar",
)
pl.plot(
    radar_coords,
    truth,
    "k-",
    linewidth=2.0,
    label="True rainfall",
)
pl.plot(
    obs_coords,
    obs,
    "o",
    markersize=10.0,
    markerfacecolor="grey",
    label="Gage observation",
)
pl.plot(radar_coords, add_adjusted, "-", color="red", label="Additive adjustment")
pl.plot(
    radar_coords, mult_adjusted, "-", color="green", label="Multiplicative adjustment"
)
pl.plot(
    radar_coords, mfb_adjusted, "-", color="orange", label="Mean Field Bias adjustment"
)
pl.plot(
    radar_coords,
    mixed_adjusted,
    "-",
    color="blue",
    label="Mixed (mult./add.) adjustment",
)
pl.xlabel("Distance (km)")
pl.ylabel("Rainfall intensity (mm/h)")
leg = pl.legend(prop={"size": 10})

##############################################################################
##### Verification
##############################################################################

# Verification for this example
rawerror = verify.ErrorMetrics(truth, radar)
mfberror = verify.ErrorMetrics(truth, mfb_adjusted)
adderror = verify.ErrorMetrics(truth, add_adjusted)
multerror = verify.ErrorMetrics(truth, mult_adjusted)
mixerror = verify.ErrorMetrics(truth, mixed_adjusted)

# Helper function for scatter plot
def scatterplot(x, y, title=""):
    """Quick and dirty helper function to produce scatter plots"""
    pl.scatter(x, y)
    pl.plot([0, 1.2 * maxval], [0, 1.2 * maxval], "-", color="grey")
    pl.xlabel("True rainfall (mm)")
    pl.ylabel("Estimated rainfall (mm)")
    pl.xlim(0, maxval + 0.1 * maxval)
    pl.ylim(0, maxval + 0.1 * maxval)
    pl.title(title)
    
# Verification reports
maxval = 4.0
# Enlarge all label fonts
font = {"size": 10}
pl.rc("font", **font)
fig = pl.figure(figsize=(14, 8))
ax = fig.add_subplot(231, aspect=1.0)
scatterplot(rawerror.obs, rawerror.est, title="Unadjusted radar")
ax.text(0.2, maxval, "Nash=%.1f" % rawerror.nash(), fontsize=12)
ax = fig.add_subplot(232, aspect=1.0)
scatterplot(adderror.obs, adderror.est, title="Additive adjustment")
ax.text(0.2, maxval, "Nash=%.1f" % adderror.nash(), fontsize=12)
ax = fig.add_subplot(233, aspect=1.0)
scatterplot(multerror.obs, multerror.est, title="Multiplicative adjustment")
ax.text(0.2, maxval, "Nash=%.1f" % multerror.nash(), fontsize=12)
ax = fig.add_subplot(234, aspect=1.0)
scatterplot(mixerror.obs, mixerror.est, title="Mixed (mult./add.) adjustment")
ax.text(0.2, maxval, "Nash=%.1f" % mixerror.nash(), fontsize=12)
ax = fig.add_subplot(235, aspect=1.0)
scatterplot(mfberror.obs, mfberror.est, title="Mean Field Bias adjustment")
ax.text(0.2, maxval, "Nash=%.1f" % mfberror.nash(), fontsize=12)
pl.tight_layout()

##############################################################################
###### 2D
##############################################################################
###### Create synthetic data 
##############################################################################

# grid axes
xgrid = np.arange(0, 10)
ygrid = np.arange(20, 30)

# number of observations
num_obs = 10

# create grid
gridshape = len(xgrid), len(ygrid)
grid_coords = util.gridaspoints(ygrid, xgrid)

# Synthetic true rainfall
truth = np.abs(10.0 * np.sin(0.1 * grid_coords).sum(axis=1))

# Creating radar data by perturbing truth with multiplicative and additive error
# YOU CAN EXPERIMENT WITH THE ERROR STRUCTURE
np.random.seed(1319622840)
radar = 0.6 * truth + 1.0 * np.random.uniform(low=-1.0, high=1, size=len(truth))
radar[radar < 0.0] = 0.0

# indices for creating obs from raw (random placement of gauges)
obs_ix = np.random.uniform(low=0, high=len(grid_coords), size=num_obs).astype("i4")

# creating obs_coordinates
obs_coords = grid_coords[obs_ix]

# creating gauge observations from truth
obs = truth[obs_ix]

##############################################################################
##### Apply different adjustment methods
##############################################################################

# Mean Field Bias Adjustment
mfbadjuster = adjust.AdjustMFB(obs_coords, grid_coords)
mfbadjusted = mfbadjuster(obs, radar)

# Additive Error Model
addadjuster = adjust.AdjustAdd(obs_coords, grid_coords)
addadjusted = addadjuster(obs, radar)

# Multiplicative Error Model
multadjuster = adjust.AdjustMultiply(obs_coords, grid_coords)
multadjusted = multadjuster(obs, radar)

# VILLE LISÄSI
mixedadjuster = adjust.AdjustMixed(obs_coords, grid_coords)
mixedadjusted = mixedadjuster(obs, radar)

##############################################################################
##### Plot 2-D adjustment results
##############################################################################

# Helper functions for grid plots
def gridplot(data, title):
    """Quick and dirty helper function to produce a grid plot"""
    xplot = np.append(xgrid, xgrid[-1] + 1.0) - 0.5
    yplot = np.append(ygrid, ygrid[-1] + 1.0) - 0.5
    grd = ax.pcolormesh(xplot, yplot, data.reshape(gridshape), vmin=0, vmax=maxval)
    ax.scatter(
        obs_coords[:, 0],
        obs_coords[:, 1],
        c=obs.ravel(),
        marker="s",
        s=50,
        vmin=0,
        vmax=maxval,
    )
    # pl.colorbar(grd, shrink=0.5)
    pl.title(title)
    
# Maximum value (used for normalisation of colorscales)
maxval = np.max(np.concatenate((truth, radar, obs, addadjusted)).ravel())

# open figure
fig = pl.figure(figsize=(10, 6))

# True rainfall
ax = fig.add_subplot(231, aspect="equal")
gridplot(truth, "True rainfall")

# Unadjusted radar rainfall
ax = fig.add_subplot(232, aspect="equal")
gridplot(radar, "Radar rainfall")

# Adjusted radar rainfall (MFB)
ax = fig.add_subplot(234, aspect="equal")
gridplot(mfbadjusted, "Adjusted (MFB)")

# Adjusted radar rainfall (additive)
ax = fig.add_subplot(235, aspect="equal")
gridplot(addadjusted, "Adjusted (Add.)")

# Adjusted radar rainfall (multiplicative)
ax = fig.add_subplot(236, aspect="equal")
gridplot(multadjusted, "Adjusted (Mult.)")

# VILLE LISÄSI
ax = fig.add_subplot(233, aspect="equal")
gridplot(multadjusted, "Adjusted (Mixed)")

pl.tight_layout()

##############################################################################

fig = pl.figure(figsize=(6, 6))

# Scatter plot radar vs. observations
ax = fig.add_subplot(221, aspect="equal")
scatterplot(truth, radar, "Radar vs. Truth (red: Gauges)")
pl.plot(obs, radar[obs_ix], linestyle="None", marker="o", color="red")

# Adjusted (MFB) vs. radar (for control purposes)
ax = fig.add_subplot(222, aspect="equal")
scatterplot(truth, mfbadjusted, "Adjusted (MFB) vs. Truth")

# Adjusted (Add) vs. radar (for control purposes)
ax = fig.add_subplot(223, aspect="equal")
scatterplot(truth, addadjusted, "Adjusted (Add.) vs. Truth")

# Adjusted (Mult.) vs. radar (for control purposes)
ax = fig.add_subplot(224, aspect="equal")
scatterplot(truth, multadjusted, "Adjusted (Mult.) vs. Truth")

pl.tight_layout()

##############################################################################

fig = pl.figure(figsize=(6, 6))

# VILLE LISÄSI
ax = fig.add_subplot(221, aspect="equal")
scatterplot(truth, mixedadjusted, "Adjusted (Mixed) vs. Truth")

# Adjusted (MFB) vs. radar (for control purposes)
ax = fig.add_subplot(222, aspect="equal")
scatterplot(truth, mfbadjusted, "Adjusted (MFB) vs. Truth")

# Adjusted (Add) vs. radar (for control purposes)
ax = fig.add_subplot(223, aspect="equal")
scatterplot(truth, addadjusted, "Adjusted (Add.) vs. Truth")

# Adjusted (Mult.) vs. radar (for control purposes)
ax = fig.add_subplot(224, aspect="equal")
scatterplot(truth, multadjusted, "Adjusted (Mult.) vs. Truth")

pl.tight_layout()

##############################################################################

rawerror2 = verify.ErrorMetrics(truth, radar)
mfberror2 = verify.ErrorMetrics(truth, mfbadjusted)
adderror2 = verify.ErrorMetrics(truth, addadjusted)
multerror2 = verify.ErrorMetrics(truth, multadjusted)
mixerror2 = verify.ErrorMetrics(truth, mixedadjusted)

rawerror2.nash()
mfberror2.nash()
adderror2.nash()
multerror2.nash()
mixerror2.nash()

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

import numpy as np
import pysteps
from datetime import datetime
import matplotlib.pyplot as plt

#Read in the event with pySTEPS
date = datetime.strptime("201306271955", "%Y%m%d%H%M") #last radar image of the event
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
fns = pysteps.io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=141)

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

#dBZ transformation for mm/h-data
#Z-R relationship: Z = a*R^b (Reflectivity)
#dBZ conversion: dBZ = 10 * log10(Z) (Decibels of Reflectivity)
#Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with size. J. Meteor., 5, 165–166.
a_R=223
b_R=1.53
#Leinonen et al. 2012 - A Climatology of Disdrometer Measurements of Rainfall in Finland over Five Years with Implications for Global Radar Observations
#https://doi.org/10.1175/JAMC-D-11-056.1

R, metadata = pysteps.utils.conversion.to_reflectivity(R, metadata, zr_a=a_R, zr_b=b_R)


R_test = R[49,:,:]

plt.figure()
plt.imshow(R_test)

# Gauge locations
gauge1_loc = np.reshape(np.array([11, 208]), (1,2)) #47.9785268753008
gauge2_loc = np.reshape(np.array([29, 186]), (1,2)) #51.24754750578536
gauge3_loc = np.reshape(np.array([117, 125]), (1,2)) #54.13171770806993

gauge4_loc = np.reshape(np.array([55, 50]), (1,2)) #42.5212
gauge5_loc = np.reshape(np.array([230, 50]), (1,2)) #42.6983
gauge6_loc = np.reshape(np.array([200, 100]), (1,2)) #21.6492
gauge7_loc = np.reshape(np.array([150, 175]), (1,2)) #49.3704
gauge8_loc = np.reshape(np.array([160, 230]), (1,2)) #37.8185

gauge1_loc_row_flipped = (R_test.shape[0]-1)-gauge1_loc[0,0]
gauge2_loc_row_flipped = (R_test.shape[0]-1)-gauge2_loc[0,0]
gauge3_loc_row_flipped = (R_test.shape[0]-1)-gauge3_loc[0,0]

gauge1_loc_flipped = gauge1_loc.copy()
gauge1_loc_flipped[0,0] = gauge1_loc_row_flipped
gauge2_loc_flipped = gauge2_loc.copy()
gauge2_loc_flipped[0,0] = gauge2_loc_row_flipped
gauge3_loc_flipped = gauge3_loc.copy()
gauge3_loc_flipped[0,0] = gauge3_loc_row_flipped

gauge_coords = np.zeros([3,2])
gauge_coords[0] = gauge1_loc
gauge_coords[1] = gauge2_loc
gauge_coords[2] = gauge3_loc
gauge_coords_f = np.zeros([3,2])
gauge_coords_f[0] = gauge1_loc_flipped 
gauge_coords_f[1] = gauge2_loc_flipped
gauge_coords_f[2] = gauge3_loc_flipped

plt.figure()
plt.imshow(R_test)
plt.scatter(gauge_coords[0,0], gauge_coords[0,1], s=30, c='red', marker='o')
plt.scatter(gauge_coords[1,0], gauge_coords[1,1], s=30, c='red', marker='o')
plt.scatter(gauge_coords[2,0], gauge_coords[2,1], s=30, c='red', marker='o')

gauge_coords2 = np.zeros([8,2])
gauge_coords2[0] = gauge1_loc
gauge_coords2[1] = gauge2_loc
gauge_coords2[2] = gauge3_loc
gauge_coords2[3] = gauge4_loc
gauge_coords2[4] = gauge5_loc
gauge_coords2[5] = gauge6_loc
gauge_coords2[6] = gauge7_loc
gauge_coords2[7] = gauge8_loc

plt.figure()
plt.imshow(R_test)
plt.scatter(gauge_coords2[0,0], gauge_coords2[0,1], s=30, c='red', marker='o')
plt.scatter(gauge_coords2[1,0], gauge_coords2[1,1], s=30, c='red', marker='o')
plt.scatter(gauge_coords2[2,0], gauge_coords2[2,1], s=30, c='red', marker='o')
plt.scatter(gauge_coords2[3,0], gauge_coords2[3,1], s=30, c='red', marker='o')
plt.scatter(gauge_coords2[4,0], gauge_coords2[4,1], s=30, c='red', marker='o')
plt.scatter(gauge_coords2[5,0], gauge_coords2[5,1], s=30, c='red', marker='o')
plt.scatter(gauge_coords2[6,0], gauge_coords2[6,1], s=30, c='red', marker='o')
plt.scatter(gauge_coords2[7,0], gauge_coords2[7,1], s=30, c='red', marker='o')

# R_test2 = np.flipud(R_test)
R_test3 = R_test.reshape([(R_test.shape[0]*R_test.shape[1]), ])

gauge_obs = np.zeros([3,])
gauge_obs[0] = R_test[int(gauge_coords[0,1]), int(gauge_coords[0,0])]
gauge_obs[1] = R_test[int(gauge_coords[1,1]), int(gauge_coords[1,0])]
gauge_obs[2] = R_test[int(gauge_coords[2,1]), int(gauge_coords[2,0])]

gauge_obs2 = np.zeros([8,])
gauge_obs2[0] = R_test[int(gauge_coords2[0,1]), int(gauge_coords2[0,0])]
gauge_obs2[1] = R_test[int(gauge_coords2[1,1]), int(gauge_coords2[1,0])]
gauge_obs2[2] = R_test[int(gauge_coords2[2,1]), int(gauge_coords2[2,0])]
gauge_obs2[3] = R_test[int(gauge_coords2[3,1]), int(gauge_coords2[3,0])]
gauge_obs2[4] = R_test[int(gauge_coords2[4,1]), int(gauge_coords2[4,0])]
gauge_obs2[5] = R_test[int(gauge_coords2[5,1]), int(gauge_coords2[5,0])]
gauge_obs2[6] = R_test[int(gauge_coords2[6,1]), int(gauge_coords2[6,0])]
gauge_obs2[7] = R_test[int(gauge_coords2[7,1]), int(gauge_coords2[7,0])]

radar_loc_x = np.array(np.arange(R_test.shape[0]))
# radar_loc_x = np.flipud(radar_loc_x)
radar_loc_y = np.array(np.arange(R_test.shape[1]))
# radar_loc_y = np.flipud(radar_loc_y)

radar_coords = util.gridaspoints(radar_loc_y, radar_loc_x)

# Mielivaltainen modifionti testi mielessä
true_obs = gauge_obs.copy()
true_obs[0] = true_obs[0] * 1.05
true_obs[1] = true_obs[1] + 5
true_obs[2] = true_obs[2] * 1.1

true_obs2 = gauge_obs2.copy()
true_obs2[0] = true_obs2[0] * 1.05
true_obs2[1] = true_obs2[1] + 5
true_obs2[2] = true_obs2[2] * 1.1
true_obs2[3] = true_obs2[3] + 3
true_obs2[4] = true_obs2[4] + 4
true_obs2[5] = true_obs2[5] * 1.4
true_obs2[6] = true_obs2[6] + 1
true_obs2[7] = true_obs2[7] + 2

# Mean Field Bias Adjustment
adjuster_mfb_rad = adjust.AdjustMFB(gauge_coords2, radar_coords)
adjusted_mfb_rad = adjuster_mfb_rad(true_obs2, R_test3)
# Additive Error Model
adjuster_add_rad = adjust.AdjustAdd(gauge_coords2, radar_coords)
adjusted_add_rad = adjuster_add_rad(true_obs2, R_test3)
# Multiplicative Error Model
adjuster_mult_rad = adjust.AdjustMultiply(gauge_coords2, radar_coords)
adjusted_mult_rad = adjuster_mult_rad(true_obs2, R_test3)

adjuster_mixed_rad = adjust.AdjustMixed(gauge_coords2, radar_coords) #Ville added
adjusted_mixed_rad = adjuster_mixed_rad(true_obs2, R_test3) #Ville added

((R_test3 == adjusted_mfb_rad) | (np.isnan(R_test3) & np.isnan(adjusted_mfb_rad))).all() #Ville added
((R_test3 == adjusted_add_rad) | (np.isnan(R_test3) & np.isnan(adjusted_add_rad))).all() #Ville added
((R_test3 == adjusted_mult_rad) | (np.isnan(R_test3) & np.isnan(adjusted_mult_rad))).all() #Ville added
((R_test3 == adjusted_mixed_rad) | (np.isnan(R_test3) & np.isnan(adjusted_mixed_rad))).all() #Ville added


radar_shape = len(radar_loc_x), len(radar_loc_y)
maxval_radar = np.nanmax(np.concatenate((true_obs2, R_test3, gauge_obs2, adjusted_add_rad)).ravel())

gauge_coords3 = gauge_coords2.copy()
gauge_coords3[0,1] = (R_test.shape[1]-1)-gauge1_loc[0,1]
gauge_coords3[1,1] = (R_test.shape[1]-1)-gauge2_loc[0,1]
gauge_coords3[2,1] = (R_test.shape[1]-1)-gauge3_loc[0,1]
gauge_coords3[3,1] = (R_test.shape[1]-1)-gauge4_loc[0,1]
gauge_coords3[4,1] = (R_test.shape[1]-1)-gauge5_loc[0,1]
gauge_coords3[5,1] = (R_test.shape[1]-1)-gauge6_loc[0,1]
gauge_coords3[6,1] = (R_test.shape[1]-1)-gauge7_loc[0,1]
gauge_coords3[7,1] = (R_test.shape[1]-1)-gauge8_loc[0,1]

def gridplot_radar(data, title):
    """Quick and dirty helper function to produce a grid plot"""
    xplot = np.append(radar_loc_x, radar_loc_x[-1] + 1.0) - 0.5
    yplot = np.append(radar_loc_y, radar_loc_y[-1] + 1.0) - 0.5
    yplot = np.flipud(yplot) #Ville added
    grd = ax.pcolormesh(xplot, yplot, data.reshape(radar_shape), vmin=0, vmax=maxval_radar)
    ax.scatter(
        gauge_coords3[:, 0],
        gauge_coords3[:, 1],
        c=true_obs2.ravel(),
        marker="s",
        edgecolors="red", #Ville added
        s=20,
        vmin=0,
        vmax=maxval_radar,
    )
    pl.colorbar(grd, shrink=0.5)
    pl.title(title)

# open figure
fig = pl.figure(figsize=(10, 6))

# Unadjusted radar rainfall
ax = fig.add_subplot(232, aspect="equal")
gridplot_radar(R_test3, "Radar rainfall")

# Adjusted radar rainfall (MFB)
ax = fig.add_subplot(234, aspect="equal")
gridplot_radar(adjusted_mfb_rad, "Adjusted (MFB)")

# Adjusted radar rainfall (additive)
ax = fig.add_subplot(235, aspect="equal")
gridplot_radar(adjusted_add_rad, "Adjusted (Add.)")

# Adjusted radar rainfall (multiplicative)
ax = fig.add_subplot(236, aspect="equal")
gridplot_radar(adjusted_mult_rad, "Adjusted (Mult.)")

ax = fig.add_subplot(233, aspect="equal") #Ville added
gridplot_radar(adjusted_mixed_rad, "Adjusted (Mixed)") #Ville added

pl.tight_layout()


# Helper function for scatter plot
def scatterplot_radar(x, y, title=""):
    """Quick and dirty helper function to produce scatter plots"""
    pl.scatter(x, y)
    pl.plot([0, 1.2 * maxval_radar], [0, 1.2 * maxval_radar], "-", color="grey")
    pl.xlabel("True rainfall (mm)")
    pl.ylabel("Estimated rainfall (mm)")
    pl.xlim(0, maxval_radar + 0.1 * maxval_radar)
    pl.ylim(0, maxval_radar + 0.1 * maxval_radar)
    pl.title(title)

#verify
fig = pl.figure(figsize=(6, 6))

ax = fig.add_subplot(221, aspect="equal")
scatterplot_radar(R_test3, adjusted_mixed_rad, "Adjusted (Mixed) vs. Truth")
plt.plot(gauge_obs2, true_obs2, color="red")
# Adjusted (MFB) vs. radar (for control purposes)
ax = fig.add_subplot(222, aspect="equal")
scatterplot_radar(R_test3, adjusted_mfb_rad, "Adjusted (MFB) vs. Truth")
plt.plot(gauge_obs2, true_obs2, color="red")
# Adjusted (Add) vs. radar (for control purposes)
ax = fig.add_subplot(223, aspect="equal")
scatterplot_radar(R_test3, adjusted_add_rad, "Adjusted (Add.) vs. Truth")
plt.plot(gauge_obs2, true_obs2, color="red")
# Adjusted (Mult.) vs. radar (for control purposes)
ax = fig.add_subplot(224, aspect="equal")
scatterplot_radar(R_test3, adjusted_mult_rad, "Adjusted (Mult.) vs. Truth")
plt.plot(gauge_obs2, true_obs2, color="red")

pl.tight_layout()

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

from matplotlib import cm, pyplot as plt

R2 = R.copy()
for i in range(R2.shape[0]):
    R2[i, ~np.isfinite(R2[i, :])] = np.nanmin(R2[i, :])

scale_break = 18  # scale break in km
scale_break_wn = np.log(max(R2.shape[1],R2.shape[2])/scale_break)
# for i in range(0, len(R)):
step=0
Fp = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(R2[step])
Fnp = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2[step])
Fp_ave = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(R2)
Fnp_ave = pysteps.noise.fftgenerators.initialize_nonparam_2d_fft_filter(R2)

seed = 1234
num_realizations = 3

# Generate noise
Np = []
Nnp = []
Np_ave = []
Nnp_ave = []
for k in range(num_realizations):
    Np.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fp, seed=seed + k))
    Nnp.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp, seed=seed + k))
    Np_ave.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fp_ave, seed=seed + k))
    Nnp_ave.append(pysteps.noise.fftgenerators.generate_noise_2d_fft_filter(Fnp_ave, seed=seed + k))

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

# Plot the generated noise fields
fig, ax = plt.subplots(nrows=2, ncols=3)
# parametric noise
ax[0, 0].imshow(Np_ave[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 1].imshow(Np_ave[1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[0, 2].imshow(Np_ave[2], cmap=cm.RdBu_r, vmin=-3, vmax=3)
# nonparametric noise
ax[1, 0].imshow(Nnp_ave[0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 1].imshow(Nnp_ave[1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
ax[1, 2].imshow(Nnp_ave[2], cmap=cm.RdBu_r, vmin=-3, vmax=3)

for i in range(2):
    for j in range(3):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.tight_layout()
plt.show()
    

