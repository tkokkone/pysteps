#!/bin/env python

import os
from numpy import genfromtxt
import numpy as np
import math
import random
import matplotlib.pyplot as plt
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

##############################################################################
# INDIR AND INPUT DATA

in_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simuloinnit/Final_sims/event_6"

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
# SET CONTROL PARAMETERS FOR SIMULATION

#Visualization
grid_on = False #gridlines on or off
colorbar_on = True #colorbar on or off
cmap = 'Blues' #None means pysteps color mapping

predefined_value_range = False #predefined value range in colorbar on or off
#This sets vmin and vmax parameters in imshow and sets normalization True/False

show_in_out_param_figs = True #in and out parameter value plots on/off

show_cumul_plot = True #cumulative plot on/off

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
n_timesteps = 50  # number of timesteps
timestep = 5  # timestep length
seed1 = 1000 #= random.randint(1, 10000) # seed number 1: seed number for generation of the first precipitation field
seed2 = 1000 #= random.randint(1, 10000) # seed number 2: seed number for generation of the first innovation field
seed_bl = 1000 #= random.randint(1, 10000) #seed for generating broken lines
seed_random = 1000 #= random.randint(1, 10000) #seed for generating random fields
nx_field = 512  # number of columns in precip fields
ny_field = 512  # number of rows in precip fields
kmperpixel = 1.0  # grid resolution
domain = "spatial"  # spatial or spectral
metadata["x1"] = 0.0  # x-coordinate of lower left
metadata["y1"] = 0.0  # y-coordinate of lower left
metadata["x2"] = nx_field * kmperpixel  # x-coordinate of upper right
metadata["y2"] = ny_field * kmperpixel  # y-coordinate of upper right
metadata["xpixelsize"] = kmperpixel  # grid resolution
metadata["ypixelsize"] = kmperpixel  # grid resolution
metadata["zerovalue"] = 0.0  # maybe not in use?
metadata["yorigin"] = "lower"  # location of grid origin (y), lower or upper

rain_zero_value = 3.1830486304816077  # no rain pixels assigned with this value
war_thold = rain_zero_value + 5  # below this value no rain

# bounding box coordinates for the extracted middel part of the entire domain
extent = [0, 0, 0, 0]
extent[0] = nx_field / 4 * kmperpixel
extent[1] = 3 * nx_field / 4 * kmperpixel
extent[2] = ny_field / 4 * kmperpixel
extent[3] = 3 * ny_field / 4 * kmperpixel

# Set power filter parameters
# where power filter pareameters given not estimated
noise_method = "parametric_sim"
fft_method = "numpy"

scale_break = 18  # scale break in km
scale_break_wn = np.log(nx_field/scale_break)
constant_betas = False
beta1 = -1.9
beta2 = -3.5

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

p_pow = np.array([scale_break_wn, 0, beta1, beta2]) # initialization

# Initialise AR parameter array, Seed et al. 2014 eqs. 9-11
tlen_a = all_params[9] #lower tlen_a makes result more random, higher tlen_a makes evolution make more sense
tlen_b = all_params[10] #lower tlen_b makes result more random, higher tlen_b makes evolution make more sense
tlen_c = all_params[11] #changing this doesnt change the result much 
ar_par = np.array([tlen_a, tlen_b, tlen_c]) 

# Set std and WAR parameters, Seed et al. eq. 4
a_v = all_params[12]
b_v = all_params[13]
c_v = all_params[14]
a_war = all_params[15]
b_war = all_params[16]
c_war = all_params[17]
min_war = 0.1

# Broken line parameters for field mean
mu_z = all_params[18] #mean of mean areal reflectivity over the simulation period
sigma2_z = all_params[19] #variance of mean areal reflectivity over the simulation period
h_val_z = all_params[20] #structure function exponent
q_val_z = all_params[30] #scale ratio between levels n and n+1 (constant) [-]
a_zero_z = all_params[21] #time series decorrelation time [min]
no_bls = 1 #number of broken lines
var_tol_z = all_params[32] #acceptable tolerance for variance as ratio of input variance [-]
mar_tol_z = all_params[33] #acceptable value for first and last elements of the final broken line as ratio of input mean -> maybe should be less than 1 -> 0.2

# Broken line parameters for velocity magnitude
mu_vmag = all_params[22] #mean of mean areal reflectivity over the simulation period
sigma2_vmag = all_params[23] #variance of mean areal reflectivity over the simulation period
h_val_vmag = all_params[24]  #structure function exponent
q_val_vmag = all_params[30] #scale ratio between levels n and n+1 (constant) [-]
a_zero_vmag = all_params[25]  #time series decorrelation time [min]
no_bls = 1 #number of broken lines
var_tol_vmag = all_params[32] #acceptable tolerance for variance as ratio of input variance [-]
mar_tol_vmag = all_params[33] #acceptable value for first and last elements of the final broken line as ratio of input mean 

# Broken line parameters for velocity direction
mu_vdir = all_params[26] #mean of mean areal reflectivity over the simulation period
sigma2_vdir = all_params[27] #variance of mean areal reflectivity over the simulation period
h_val_vdir = all_params[28]  #structure function exponent
q_val_vdir = all_params[30] #scale ratio between levels n and n+1 (constant) [-]
a_zero_vdir = all_params[29]  #time series decorrelation time [min]
no_bls = 1 #number of broken lines
var_tol_vdir = all_params[32] #acceptable tolerance for variance as ratio of input variance [-]
mar_tol_vdir = all_params[33] #acceptable value for first and last elements of the final broken line as ratio of input mean:

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
beta1_in = []
beta2_in = []
scaleb_in = []
mean_out = []
std_out = []
war_out = []
beta1_out = []
beta2_out = []
scaleb_out = []

#Ville: added
beta1_out1 = []
beta2_out1 = []
beta1_out2 = []
beta2_out2 = []
war_out2 = []
beta1_out_Rnew2 = []
beta2_out_Rnew2 = []
beta1_out_Rnew3 = []
beta2_out_Rnew3 = []
all_Fp = []
all_R_ = []
all_freq = []
all_f = []
all_b1 = []
all_b2 = []

all_R_fp1_ = []
all_freq_fp1 = []
all_f_fp1 = []

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
# GENERATING BROKEN LINE TIME SERIES FOR MEAN R AND ADVECTION

# Create the field mean for the requested number of simulation time steps
np.random.seed(seed_bl)
r_mean = create_broken_lines(mu_z, sigma2_z, h_val_z, q_val_z,
                             a_zero_z, timestep, (n_timesteps-1) * timestep,
                             no_bls, var_tol_z, mar_tol_z)

# Create velocity magnitude for the requested number of simulation time steps
np.random.seed(seed_bl)
v_mag = create_broken_lines(mu_vmag, sigma2_vmag, h_val_vmag, q_val_vmag,
                            a_zero_vmag, timestep, (n_timesteps-1) * timestep,
                            no_bls, var_tol_vmag, mar_tol_vmag)

# Create velocity direction (deg) for the requested number of simulation time steps
np.random.seed(seed_bl)
v_dir = create_broken_lines(mu_vdir, sigma2_vdir, h_val_vdir, q_val_vdir,
                            a_zero_vdir, timestep, (n_timesteps-1) * timestep,
                            no_bls, var_tol_vdir, mar_tol_vdir)

# Compute advection variables in x- and y-directions
vx = np.cos(v_dir / 360 * 2 * np.pi) * v_mag
vy = np.sin(v_dir / 360 * 2 * np.pi) * v_mag

if advection_mode == 0:
    vx[:] = 0
    vy[:] = 0
elif advection_mode == 1 or advection_mode == 2:
    vx[:] = const_v_x
    vy[:] = const_v_y


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

v_mag[0] = 0.0
R_ini = []

np.random.seed(seed_random)
R_ini.append(np.random.normal(0.0, 1.0, size=(ny_field, nx_field)))
#R_ini.append(np.random.normal(0.0, 1.0, size=(ny_field, nx_field)))
# Change the type of R to align with pySTEPS
R_ini = np.concatenate([R_[None, :, :] for R_ in R_ini])

fft = utils.get_method(fft_method, shape=(ny_field, nx_field), n_threads=1)
init_noise, generate_noise = noise.get_method(noise_method)
noise_kwargs = dict()

if constant_betas:
    p_pow[0] = scale_break_wn
    p_pow[2] = beta1
    p_pow[3] = beta2
else:
    # p_pow[2] = - (a_1 + b_1 * r_mean[0] + c_1 * r_mean[0] ** 2)
    # p_pow[3] = - (a_2 + b_2 * r_mean[0] + c_2 * r_mean[0] ** 2)
    p_pow[2] = (a_1 + b_1 * r_mean[0] + c_1 * r_mean[0] ** 2) - 0.5 #beta1
    p_pow[3] = (a_2 + b_2 * r_mean[0] + c_2 * r_mean[0] ** 2) - 0.5 #beta2
    p_pow[0] = scale_break_wn
    # p_pow[0] = np.log(nx_field/(a_w0 + b_w0 * r_mean[0] + c_w0 * r_mean[0] ** 2)) #scale break

p_pow_w0 = np.zeros([n_timesteps, 1])
p_pow_b1 = np.zeros([n_timesteps, 1])
p_pow_b2 = np.zeros([n_timesteps, 1])
p_pow_w0[0] = p_pow[0]
p_pow_b1[0] = p_pow[2]
p_pow_b2[0] = p_pow[3]

w0_sim = np.zeros([n_timesteps, 1])
w0_sim_km = np.zeros([n_timesteps, 1])
beta1_sim = np.zeros([n_timesteps, 1])
beta2_sim = np.zeros([n_timesteps, 1])

pp = init_noise(R_ini, p_pow, fft_method=fft, **noise_kwargs)

R = []
if advection_mode == 2:
    R_0 = np.zeros((ny_field, nx_field))
    R_0[block_ulr:block_ulr+block_ysize,block_ulc:block_ulc+block_xsize] = 1
    R.append(R_0)
    R.append(R_0)
else:    
    R_0 = generate_noise(
        pp, randstate=None, seed=seed1, fft_method=fft, domain=domain
    )
    # Tarvitaanko seuraavaa riviä?
    Fp = noise.fftgenerators.initialize_param_2d_fft_filter(
        R_0, scale_break=scale_break_wn)
    R.append(R_0)
    R.append(R_0)
# Generate the first innovation field and append it as the last term in R
beta1_in.append(p_pow[2])
beta2_in.append(p_pow[3])
R.append(generate_noise(
    pp, randstate=None, seed=seed2, fft_method=fft, domain=domain
))
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

nowcast_method = nowcasts.get_method("steps_sim")

for i in range(1, n_timesteps):
# for i in range(n_timesteps):
    # TEEMU: näitten kai kuuluu olla negatiivisia?
    if constant_betas:
        p_pow[0] = scale_break_wn
        p_pow[2] = beta1
        p_pow[3] = beta2
    else:
        # p_pow[2] = - (a_1 + b_1 * r_mean[i] + c_1 * r_mean[i] ** 2)
        # p_pow[3] = - (a_2 + b_2 * r_mean[i] + c_2 * r_mean[i] ** 2)
        p_pow[2] = (a_1 + b_1 * r_mean[i] + c_1 * r_mean[i] ** 2) - 0.5 #beta1
        p_pow[3] = (a_2 + b_2 * r_mean[i] + c_2 * r_mean[i] ** 2) - 0.5 #beta2
        p_pow[0] = scale_break_wn
        # p_pow[0] = np.log(nx_field/(a_w0 + b_w0 * r_mean[i] + c_w0 * r_mean[i] ** 2)) #scale break

    pp = init_noise(R_ini, p_pow, fft_method=fft, **noise_kwargs)
    # R_prev needs to be saved as R is advected in STEPS loop
    R_prev = R[1].copy()
    R_new = nowcast_method(
        R,
        vx[i],
        vy[i],
        ar_par,
        n_cascade_levels=6,
        R_thr=-10.0,
        kmperpixel=kmperpixel,
        timestep=timestep,
        noise_method="parametric_sim",
        vel_pert_method="bps",
        mask_method="incremental",
        include_AR=AR_mode,
        normalize_field=normalize_field,
    )

    # w0_km = nx_field / np.exp(Fp["pars"][0])
    
    R[0] = R_prev
    
    p_pow_w0[i] = p_pow[0]
    p_pow_b1[i] = p_pow[2]
    p_pow_b2[i] = p_pow[3]
    
    R[1] = R_new
    R[2] = generate_noise(pp, randstate=None, fft_method=fft, domain=domain)
    
    #Ville: added
    R_new2 = R_new.copy()
    R_new3 = R_new.copy()
    Fp = noise.fftgenerators.initialize_param_2d_fft_filter(
        R_new, scale_break=scale_break_wn)
    
    #Ville: Moved here before set_stats
    R_new, metadata_clip = clip_domain(R_new, metadata, extent)
    
    #Ville: added
    Fp1 = noise.fftgenerators.initialize_param_2d_fft_filter(
        R_new, scale_break=scale_break_wn)
    
    #Compute the observed and fitted 1D PSD
    L = np.max(Fp1["input_shape"])
    if L % 2 == 1:
        wn = np.arange(0, int(L / 2) + 1)
    else:
        wn = np.arange(0, int(L / 2))
    R_fp1_, freq_fp1 = rapsd(R_new, fft_method=np.fft, return_freq=True)
    f_fp1 = np.exp(Fp1["model"](np.log(wn), *Fp1["pars"]))
    all_R_fp1_.append(R_fp1_)
    all_freq_fp1.append(freq_fp1)
    all_f_fp1.append(f_fp1)
    
    stats_kwargs["mean"] = r_mean[i]
    stats_kwargs["std"] = a_v + b_v * r_mean[i] + c_v * r_mean[i] ** 2
    stats_kwargs["war"] = a_war + b_war * r_mean[i] + c_war * r_mean[i] ** 2
    if stats_kwargs["war"] < min_war:
       stats_kwargs["war"] = min_war 
    stats_kwargs["war_thold"] = war_thold
    stats_kwargs["rain_zero_value"] = rain_zero_value
    if set_stats_active == True:
        R_new = set_stats(R_new, stats_kwargs)
    
    #Ville: added
    Fp2 = noise.fftgenerators.initialize_param_2d_fft_filter(
        R_new, scale_break=scale_break_wn)
    all_Fp.append(Fp2)
    
    w0_sim[i-1] = Fp2["pars"][0]
    w0_sim_km[i-1] = R_new.shape[0] / np.exp(w0_sim[i-1])
    beta1_sim[i-1] = Fp2["pars"][1]
    beta2_sim[i-1] = Fp2["pars"][2]
    
    #Betas with only clipping
    R_new2, metadata_clip2 = clip_domain(R_new2, metadata, extent)
    Fp_Rnew2 = noise.fftgenerators.initialize_param_2d_fft_filter(
        R_new2, scale_break=scale_break_wn)
    beta1_out_Rnew2.append(Fp_Rnew2["pars"][1])
    beta2_out_Rnew2.append(Fp_Rnew2["pars"][2])
    
    #Betas with only setting statistics
    if set_stats_active == True:
        R_new3 = set_stats(R_new3, stats_kwargs)
    Fp_Rnew3 = noise.fftgenerators.initialize_param_2d_fft_filter(
        R_new3, scale_break=scale_break_wn)
    beta1_out_Rnew3.append(Fp_Rnew3["pars"][1])
    beta2_out_Rnew3.append(Fp_Rnew3["pars"][2])
    
    w0_km = nx_field / np.exp(scale_break_wn)
    beta1_out.append(Fp["pars"][1])
    beta2_out.append(Fp["pars"][2])
    scaleb_out.append(w0_km)
    
    #Ville: added
    beta1_out1.append(Fp1["pars"][1])
    beta2_out1.append(Fp1["pars"][2])
    beta1_out2.append(Fp2["pars"][1])
    beta2_out2.append(Fp2["pars"][2])
    
    war = len(np.where(R_new > war_thold)[0]) / (R_new.shape[1] * R_new.shape[0])
    
    #Ville: added
    war2 = ((np.count_nonzero(R_new[R_new > rain_zero_value])) / np.count_nonzero(~np.isnan(R_new)))
    war_out2.append(war2) 
    
    mean_out.append(R_new.mean())
    std_out.append(R_new.std())
    war_out.append(war)

    # R_new, metadata_clip = clip_domain(R_new, metadata, extent)

    R_sim.append(R_new)
    mean_in.append(stats_kwargs["mean"])
    std_in.append(stats_kwargs["std"])
    war_in.append(stats_kwargs["war"])
    beta1_in.append(p_pow[2])
    beta2_in.append(p_pow[3])
    scaleb_in.append(scale_break)
    
plt.figure()
plt.plot(w0_sim_km)
plt.plot(scaleb_in)

# f.close()
R_sim = np.concatenate([R_[None, :, :] for R_ in R_sim])
# TEEMU: precipfields.py:hyn funktioon plot_precip_field puukotettu yksiköksi dBZ
#animate(R_sim, savefig=False, path_outputs="../../Local/tmp2")
ani = animate_interactive(R_sim,grid_on,colorbar_on, predefined_value_range,cmap)

if show_in_out_param_figs:
    x_data = np.arange(0,len(mean_in))
    plt.figure()
    plt.plot(x_data,war_in, x_data,war_out, x_data,war_out2)
    plt.title("WAR")
    plt.legend(['In', 'Out', 'Out2'])
    plt.figure()
    plt.plot(x_data,std_in,x_data, std_out)
    plt.title("Standard deviation")
    plt.legend(['In', 'Out'])
    plt.figure()
    plt.plot(x_data,mean_in,x_data,mean_out)
    plt.title("Mean")
    plt.legend(['In', 'Out'])
    plt.figure()
    plt.plot(x_data,beta1_in[0:-1], x_data,beta1_out, x_data,beta1_out1, x_data,beta1_out2)
    plt.title("Beta 1")
    plt.legend(['In', 'Out', 'Out1', 'Out2'])
    plt.figure()
    plt.plot(x_data,beta2_in[0:-1], x_data,beta2_out, x_data,beta2_out1, x_data,beta2_out2)
    plt.title("Beta 2")
    plt.legend(['In', 'Out', 'Out1', 'Out2'])
    
    plt.figure()
    plt.plot(x_data,beta1_in[0:-1], x_data,beta1_out, x_data,beta1_out2, x_data,beta1_out_Rnew2, x_data,beta1_out_Rnew3)
    plt.title("Beta 1")
    plt.legend(['In', 'Nothing', 'Both', 'Clip', 'Stats'])
    plt.figure()
    plt.plot(x_data,beta2_in[0:-1], x_data,beta2_out , x_data,beta2_out2, x_data,beta2_out_Rnew2, x_data,beta2_out_Rnew3)
    plt.title("Beta 2")
    plt.legend(['In', 'Nothing', 'Both', 'Clip', 'Stats'])
    
    plt.figure()
    plt.plot(x_data,mean_in, x_data,beta2_in[0:-1])
    
    plt.figure()
    plt.plot(beta1_in[0:-1])
    plt.plot(beta2_in[0:-1])
    # plt.plot(beta1_sim[0:-1])
    # plt.plot(beta2_sim[0:-1])
    plt.plot(beta1_out2)
    plt.plot(beta2_out2)
    plt.legend(['Beta1 In', 'Beta2 In', 'Beta1 Out', 'Beta2 Out'])

if show_cumul_plot:
    R_acc = np.sum(R_sim,axis=0)
    plt.figure()
    plt.imshow(R_acc, cmap ="Blues", alpha = 0.7, interpolation ='bilinear', extent = extent)
    plt.colorbar()

##############################################################################
# PLOT THE OBSERVED 1D POWER SPECTRUM AND THE MODEL

#The parametric model uses a piece-wise linear function with two spectral slopes (beta1 and beta2) and one breaking point
#https://pysteps.readthedocs.io/en/latest/auto_examples/plot_noise_generators.html

for i in range(len(all_Fp)):
    #Compute the observed and fitted 1D PSD
    L = np.max(all_Fp[i]["input_shape"])
    if L % 2 == 1:
        wn = np.arange(0, int(L / 2) + 1)
    else:
        wn = np.arange(0, int(L / 2))
    R_, freq = rapsd(R_sim[i], fft_method=np.fft, return_freq=True)
    f = np.exp(all_Fp[i]["model"](np.log(wn), *all_Fp[i]["pars"]))
    
    all_R_.append(R_)
    all_freq.append(freq)
    all_f.append(f)
    
    #Extract the scaling break in km, beta1 and beta2
    w0 = scale_break
    b1 = all_Fp[i]["pars"][1]
    b2 = all_Fp[i]["pars"][2]
    all_b1.append(b1)
    all_b2.append(b2)

fig, ax = plt.subplots()
plot_scales = [256, 128, 64, 32, 16, 8, 4, 2]
plot_spectrum1d(
    all_freq[0],
    all_R_[0],
    x_units="km",
    y_units="dBZ",
    color="k",
    ax=ax,
    label="Observed",
    wavelength_ticks=plot_scales,
)
# Vile: added for test purposes
plot_spectrum1d(
    all_freq_fp1[0],
    all_R_fp1_[0],
    x_units="km",
    y_units="dBZ",
    color="k",
    ax=ax,
    label="Observed",
    wavelength_ticks=plot_scales,
)
# Vile: added for test purposes
plot_spectrum1d(
    all_freq_fp1[0][:-1],
    all_f_fp1[0],
    x_units="km",
    y_units="dBZ",
    color="r",
    ax=ax,
    label="Fit",
    wavelength_ticks=plot_scales,
)
plot_spectrum1d(
    all_freq[0][:-1],
    all_f[0],
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
    r"$\omega_0=%.0f km, \beta_1=%.1f, \beta_2=%.1f$" % (w0, all_b1[0], all_b2[0])
)
plt.show()