#!/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import math

from pprint import pprint
from pysteps import nowcasts, noise, utils
from pysteps.utils import conversion, dimension, transformation
from pysteps.utils.dimension import clip_domain
from pysteps.visualization import plot_precip_field, animate
from pysteps.postprocessing.probmatching import set_stats

#Initialisation of variables
stats_kwargs = dict()
metadata = dict()

# Set general simulation parameters
n_timesteps = 10 #number of timesteps
timestep = 6 #timestep length
seed1 = 124 #seed number for generation of the first precipitation field
seed2 = 234 #seed number for generation of the first innovation field
nx_field = 1024 #number of columns in precip fields
ny_field = 1024 #number of rows in precip fields
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

# bounding box coordinates for the extracted middel part of the entire domain
extent = [0,0,0,0]
extent[0] = nx_field / 4 * kmperpixel
extent[1] = 3 * nx_field / 4 * kmperpixel
extent[2] = ny_field / 4 * kmperpixel
extent[3] = 3 * ny_field / 4 * kmperpixel


# Set power filter parameters
#TEEMU: Näillä a_1...c_1 ja a_2_c_2 parametreilla ei tule negatiivista kulmakerrointa?
noise_method="parametric_sim" #where power filter pareameters given not estimated 
fft_method="numpy"
scale_break = 18 #scale break in km
scale_break_wn = np.log(nx_field/scale_break)
a_1 = 1.65 #a_1...c_2 see Seed et al. 2014
b_1 = 0.25
c_1 = -0.013
a_2 = 3.6
b_2 = 0.005
c_2 = 0  
p_pow = np.array([scale_break_wn,0,-2.0,-2.0]) #initialization 

# Initialise AR parameter array, Seed et al. 2014 eqs. 9-11 
ar_par = np.array([0.2,1.8,2]) #order: at, bt, ct

# Set std and WAR parameters, Seed et al. eq. 4
a_v = 3.91
b_v = 1.62
c_v = -0.059
a_war = 0.0048
b_war = 0.0404
c_war = -0.0004

# Broken line parameters for field mean
mu_z = 5.39 #mean of mean areal reflectivity over the simulation period
sigma2_z = 11.53 #variance of mean areal reflectivity over the simulation period
h_val_z = 0.94  #structure function exponent
q_val_z = 0.93  #scale ratio between levels n and n+1 (constant) [-]
a_zero_z = 438 #time series decorrelation time [min]
no_bls = 1 #number of broken lines
var_tol_z = 1 #acceptable tolerance for variance as ratio of input variance [-]
mar_tol_z = 1 #acceptable value for first and last elements of the final broken line as ratio of input mean:

# Broken line parameters for velocity magnitude
mu_vmag = 2.39 #mean of mean areal reflectivity over the simulation period
sigma2_vmag = 0.14 #variance of mean areal reflectivity over the simulation period
h_val_vmag = 0.53  #structure function exponent
q_val_vmag = 0.8  #scale ratio between levels n and n+1 (constant) [-]
a_zero_vmag = 60 #time series decorrelation time [min]
no_bls = 1 #number of broken lines
var_tol_vmag = 1 #acceptable tolerance for variance as ratio of input variance [-]
mar_tol_vmag = 1 #acceptable value for first and last elements of the final broken line as ratio of input mean:    

# Broken line parameters for velocity direction
mu_vdir = 97 #mean of mean areal reflectivity over the simulation period
sigma2_vdir = 60 #variance of mean areal reflectivity over the simulation period
h_val_vdir = 0.41  #structure function exponent
q_val_vdir = 0.8  #scale ratio between levels n and n+1 (constant) [-]
a_zero_vdir = 40 #time series decorrelation time [min]
no_bls = 1 #number of broken lines
var_tol_vdir = 1 #acceptable tolerance for variance as ratio of input variance [-]
mar_tol_vdir = 1 #acceptable value for first and last elements of the final broken line as ratio of input mean:    

# Set method for advection    
extrap_method = "semilagrangian_wrap"

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
    
    return blines_final

# Create the field mean for the requested number of simulation time steps
r_mean = create_broken_lines(mu_z, sigma2_z, h_val_z, q_val_z,
                             a_zero_z, timestep, (n_timesteps-1) * timestep,
                             no_bls, var_tol_z, mar_tol_z)


# Create velocity magnitude for the requested number of simulation time steps
v_mag = create_broken_lines(mu_vmag, sigma2_vmag, h_val_vmag, q_val_vmag, 
                            a_zero_vmag, timestep, (n_timesteps-1) * timestep,
                            no_bls, var_tol_vmag, mar_tol_vmag)

# Create velocity direction (deg) for the requested number of simulation time steps
v_dir = create_broken_lines(mu_vdir, sigma2_vdir, h_val_vdir, q_val_vdir, 
                            a_zero_vdir, timestep, (n_timesteps-1) * timestep,
                            no_bls, var_tol_vdir, mar_tol_vdir)

#Compute advection variables in x- and y-directions
vx = np.cos(v_dir / 360 * 2 * np.pi) * v_mag 
vy = np.sin(v_dir / 360 * 2 * np.pi) * v_mag

#Tämä vain kokeilua varten, V joka paikassa 1 tai 0
V = [np.zeros((ny_field, nx_field)),np.zeros((ny_field, nx_field))]
V = np.concatenate([V_[None, :, :] for V_ in V])

x_values, y_values = np.meshgrid(np.arange(nx_field), np.arange((ny_field)))
xy_coords = np.stack([x_values, y_values])

###############################################################################
# Create the first two precipitation fields, the nuber is determined by the order of
# the ar process, in this example it is two. The second one is a copy of the first
# one. The first velocity magnitude is zero.

v_mag[0] = 0.0
R_ini = []
R_ini.append(np.random.normal(0.0, 1.0, size=(ny_field, nx_field)))
#R_ini.append(np.random.normal(0.0, 1.0, size=(ny_field, nx_field)))
# Change the type of R to align with pySTEPS
R_ini = np.concatenate([R_[None, :, :] for R_ in R_ini])

fft = utils.get_method(fft_method, shape=(ny_field, nx_field), n_threads=1)
init_noise, generate_noise = noise.get_method(noise_method)
noise_kwargs=dict()
#p_pow[2] = - (a_1 + b_1 * r_mean[0] + c_1 * r_mean[0] ** 2)
#p_pow[3] = - (a_2 + b_2 * r_mean[0] + c_2 * r_mean[0] ** 2)
p_pow[2] = -2.5
p_pow[3] = -2.5
pp = init_noise(R_ini, p_pow, fft_method=fft, **noise_kwargs) 
R = []
R_0 = generate_noise(
                    pp, randstate=None,seed=seed1,fft_method=fft, domain=domain
                )
Fp = noise.fftgenerators.initialize_param_2d_fft_filter(R_0, scale_break=scale_break_wn)       
R.append(R_0)
R.append(R_0)
# Generate the first innovation field and append it as the last term in R
beta1_in.append(p_pow[2])
beta2_in.append(p_pow[3])
R.append(generate_noise(
                    pp, randstate=None, seed=seed2,fft_method=fft, domain=domain
                ))
R = np.concatenate([R_[None, :, :] for R_ in R])   


# Plot the rainfall field
plot_precip_field(R[-1, :, :])
plt.show()

# Set missing values with the fill value
# TEEMU: probably not needed, how can there be nonfinite values in generated
# fields?
R[~np.isfinite(R)] = -15.0

# Nicely print the metadata
#pprint(metadata)


###############################################################################
# Simulation loop with STEPS

nowcast_method = nowcasts.get_method("steps_sim")

#f = open("../../Local/tmp/mean_std.txt", "a")
for i in range(n_timesteps):
    #TEEMU: näitten kai kuuluu olla negatiivisia?
    p_pow[2] = -2.5 #- (a_1 + b_1 * r_mean[i] + c_1 * r_mean[i] ** 2)
    p_pow[3] = -2.5 #- (a_2 + b_2 * r_mean[i] + c_2 * r_mean[i] ** 2)
    pp = init_noise(R_ini, p_pow, fft_method=fft, **noise_kwargs)
    #R_prev needs to be saved as R is advected in STEPS loop
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
    )

    #f.write("mean: {a: 8.3f} std: {b: 8.3f} \n".format(a=R_new.mean(), b=R_new.std()))
    Fp = noise.fftgenerators.initialize_param_2d_fft_filter(R_new, scale_break=scale_break_wn)
    w0_km = nx_field / np.exp(scale_break_wn)
    beta1_out.append(Fp["pars"][1])
    beta2_out.append(Fp["pars"][2])
    scaleb_out.append(w0_km)
    R[0] = R_prev
    R[1] = R_new
    R[2] = generate_noise(
                    pp, randstate=None,fft_method=fft, domain=domain
                )
    stats_kwargs["mean"] = r_mean[i]
    stats_kwargs["std"] =  a_v + b_v * r_mean[i] + c_v * r_mean[i] ** 2
    stats_kwargs["war"] = a_war + b_war * r_mean[i] + c_war * r_mean[i] ** 2
    R_new = set_stats(R_new,stats_kwargs)
    mean_out.append(R_new.mean())
    std_out.append(R_new.std())
    #R_new, metadata_clip = clip_domain(R_new, metadata, extent)
    #R_new = transformation.dB_transform(R_new, threshold=-10.0, inverse=True)[0]
    R_sim.append(R_new)
    mean_in.append(stats_kwargs["mean"])
    std_in.append(stats_kwargs["std"])
    war_in.append(stats_kwargs["war"])
    beta1_in.append(p_pow[2])
    beta2_in.append(p_pow[3])
    scaleb_in.append(scale_break)
    
#f.close()
R_sim = np.concatenate([R_[None, :, :] for R_ in R_sim])
#TEEMU: precipfields.py:hyn funktioon plot_precip_field puukotettu yksiköksi dBZ
animate(R_sim, savefig=False,path_outputs="../../Local/tmp2")

