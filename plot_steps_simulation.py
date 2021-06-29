#!/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import math

from datetime import datetime
from pprint import pprint
from pysteps import extrapolation
from pysteps import io, nowcasts, rcparams, noise, utils
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import conversion, dimension, transformation
from pysteps.utils.dimension import clip_domain
from pysteps.visualization import plot_precip_field, animate
from pysteps.postprocessing.probmatching import set_stats

stats_kwargs = dict()

# Set simulation parameters
n_timesteps = 100
timestep = 6 #length of timestep between precipitation fields
seed = 24
nx_field = 264 #number of columns in simulated fields
ny_field = 264 #number of rows in simulated fields
domain = "spatial"

# Set noise parameters
noise_method="parametric_sim"
fft_method="numpy"
#a_beta1 = -1.1
#b_beta1 = -0.1077
#c_beta1 = -0.0127
#a_beta2 = -3.262
#b_beta2 = -0.02521
#c_beta2 = -0.07435  
p_pow = np.array([18,1.0,-2.703,-3.665]) #~p0 from fftgenerators.py
ar_par = np.array([0.2,1.8,2])

# Broken line parameters for field mean
mu_z = 0.72 #mean of mean areal reflectivity over the simulation period
sigma2_z = 0.19 #variance of mean areal reflectivity over the simulation period
h_val_z = 0.94  #structure function exponent
q_val_z = 0.8  #scale ratio between levels n and n+1 (constant) [-]
a_zero_z = 80 #time series decorrelation time [min]
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

extrap_method = "semilagrangian_wrap"   
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
V = [np.ones((ny_field, nx_field)),np.ones((ny_field, nx_field))]
V = np.concatenate([V_[None, :, :] for V_ in V])

x_values, y_values = np.meshgrid(np.arange(nx_field), np.arange((ny_field)))
xy_coords = np.stack([x_values, y_values])

###############################################################################
# Create the first precipitation fields, the nuber is determined by the order of
# the ar process, in this example it is two. Maybe later the second one should
# be AR(1) of the first one, but is ignored for now
R_ini = []
R_ini.append(np.random.normal(0.0, 1.0, size=(ny_field, nx_field)))
R_ini.append(np.random.normal(0.0, 1.0, size=(ny_field, nx_field)))
# Change the type of R to align with pySTEPS
R_ini = np.concatenate([R_[None, :, :] for R_ in R_ini])

fft = utils.get_method(fft_method, shape=(ny_field, nx_field), n_threads=1)
init_noise, generate_noise = noise.get_method(noise_method)
noise_kwargs=dict()
pp = init_noise(R_ini, p_pow, fft_method=fft, **noise_kwargs) 
R = []        
R.append(generate_noise(
                    pp, randstate=None,seed=110,fft_method=fft, domain=domain
                ))
extrapolator_method = extrapolation.get_method(extrap_method)
extrap_kwargs = dict()
extrap_kwargs["xy_coords"] = xy_coords
extrap_kwargs["allow_nonfinite_values"] = True
R.append(extrapolator_method(R[0], V, 1, "min", **extrap_kwargs)[-1])
R.append(generate_noise(
                    pp, randstate=None, seed=2345,fft_method=fft, domain=domain
                ))
R = np.concatenate([R_[None, :, :] for R_ in R])   


# Plot the rainfall field
plot_precip_field(R[-1, :, :])
plt.show()

# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
# set the fill value to -15 dBR
# TEEMU: Missä yksikössä simuloimme? Kun luomme kohinan, niin onko mm vai dbZ?
#R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)

# Set missing values with the fill value
R[~np.isfinite(R)] = -15.0

# Nicely print the metadata
#pprint(metadata)


###############################################################################
# Stochastic nowcast with STEPS
# -----------------------------
#
# The S-PROG approach is extended to include a stochastic term which represents
# the variance associated to the unpredictable development of precipitation. This
# approach is known as STEPS (short-term ensemble prediction system).

# The STEPS nowcast
# TEEMU: STEPS simulation. Mitä argumentteja tarvitaan?
nowcast_method = nowcasts.get_method("steps_sim")

R_sim = []
f = open("../../Local/tmp/mean_std.txt", "a")
for i in range(n_timesteps):
    R_prev = R[1].copy()
    R_new = nowcast_method(
                R,
                r_mean,
                vx,
                vy,
                ar_par,
                n_cascade_levels=6,
                R_thr=-10.0,
                kmperpixel=1.0,
                timestep=timestep,
                noise_method="parametric_sim",
                vel_pert_method="bps",
                mask_method="incremental",
                seed=seed,
    )

    #f.write("mean: {a: 8.3f} std: {b: 8.3f} \n".format(a=R_new.mean(), b=R_new.std()))
    R[0] = R_prev
    R[1] = R_new
    R[2] = generate_noise(
                    pp, randstate=None,fft_method=fft, domain=domain
                )
    stats_kwargs["mean"] = 13
    stats_kwargs["std"] = 2
    stats_kwargs["war"] = 0.17
    R_new = set_stats(R_new,stats_kwargs)
    R_new = clip_domain(R_new, metadata)
    R_sim.append(R_new)
    
f.close()
R_sim = np.concatenate([R_[None, :, :] for R_ in R_sim])
animate(R_sim, savefig=False,path_outputs="../../Local/tmp2")
# Back-transform to rain rates
R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]


# Plot the ensemble mean
R_f_mean = np.mean(R_f[:, -1, :, :], axis=0)
plot_precip_field(
    R_f_mean,
    geodata=metadata,
    title="Ensemble mean (+ %i min)" % (n_leadtimes * timestep),
)
plt.show()

###############################################################################
# The mean of the ensemble displays similar properties as the S-PROG
# forecast seen above, although the degree of smoothing also depends on
# the ensemble size. In this sense, the S-PROG forecast can be seen as
# the mean of an ensemble of infinite size.

# Plot some of the realizations
fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(221 + i)
    ax = plot_precip_field(
        R_f[i, -1, :, :], geodata=metadata, colorbar=False, axis="off"
    )
    ax.set_title("Member %02d" % i)
plt.tight_layout()
plt.show()

###############################################################################
# As we can see from these two members of the ensemble, the stochastic forecast
# mantains the same variance as in the observed rainfall field.
# STEPS also includes a stochatic perturbation of the motion field in order
# to quantify the its uncertainty.

###############################################################################
# Finally, it is possible to derive probabilities from our ensemble forecast.

# Compute exceedence probabilities for a 0.5 mm/h threshold
P = excprob(R_f[:, -1, :, :], 0.5)

# Plot the field of probabilities
plot_precip_field(
    P,
    geodata=metadata,
    ptype="prob",
    units="mm/h",
    probthr=0.5,
    title="Exceedence probability (+ %i min)" % (n_leadtimes * timestep),
)
plt.show()

# sphinx_gallery_thumbnail_number = 5



