# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:06:45 2021

@author: lindgrv1
"""

import numpy as np
import matplotlib.pyplot as plt

import pysteps
from pprint import pprint
from pysteps import nowcasts, noise, utils
from pysteps.utils import conversion, dimension, transformation
from pysteps.utils.dimension import clip_domain
from pysteps.visualization import plot_precip_field, animate
from pysteps.postprocessing.probmatching import set_stats
from pysteps import extrapolation
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.timeseries import autoregression


# random kenttä
R_ini = []
np.random.seed(4321)
R_ini.append(np.random.normal(0.0, 1.0, size=(264, 264)))
R_ini = np.concatenate([R_[None, :, :] for R_ in R_ini])

plt.figure()
plt.imshow(R_ini[0])

# syötetään fft parametrit
L = 264
# p_pow = np.array([np.log(L/18), 0.0,-1.5,-3.5])
p_pow = np.array([np.log(L/12.5714), 0.0,-2.0761,-3.43499])

# np.log(L/28.9152)
# L / np.exp(2.6855773452501515)

# luodaan filtteri ja kenttä
noise_method="parametric_sim" #where power filter pareameters given not estimated 
fft_method="numpy"
fft = utils.get_method(fft_method, shape=(264, 264), n_threads=1)
init_noise, generate_noise = noise.get_method(noise_method)
noise_kwargs=dict()
pp = init_noise(R_ini, p_pow, fft_method=fft, **noise_kwargs)
 
domain = "spatial" #spatial or spectral
R = []
R_0 = generate_noise(pp, randstate=None,seed=1234, fft_method=fft, domain=domain)
R.append(R_0)

plt.figure()
plt.imshow(R_0)

np.mean(R_0)
np.std(R_0)

# estimoidaan siitä parametrit
Fp = pysteps.noise.fftgenerators.initialize_param_2d_fft_filter(R_0)
w0s = L / np.exp(Fp["pars"][0])



