# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:25:07 2022

@author: tkokko
"""

import matplotlib.pyplot as plt
import pysteps
from datetime import datetime

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
plt.figure()
pysteps.visualization.plot_precip_field(R[0], title=date)

# DATA TRANSFORMATION from mm/h into dbz
#Values less than threshold to zero
R[R<0.1] = 0

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

