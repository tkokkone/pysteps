# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 16:03:17 2022

@author: tkokko
"""

from pysteps.extrapolation.semilagrangian import extrapolate
import pytest
import numpy as np

def test_ascending_timesteps():
    precip = np.ones((8, 8))
    v = np.ones((8, 8))
    velocity = np.stack([v, v])

    not_ascending_timesteps = [1,2,3,5,4,6,7]
    with pytest.raises(ValueError):
        test = extrapolate(precip, velocity, not_ascending_timesteps)
        
test_ascending_timesteps()