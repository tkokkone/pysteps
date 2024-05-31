# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:03:32 2024

@author: lindgrv1
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import pandas as pd
import seaborn as sns

##############################################################################

# DIRECTORIES

dir_sim = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations"

file_list = os.listdir(dir_sim)
file_list_normal = [x for x in file_list if "normal" in x]
file_list_slow = [y for y in file_list if "slow" in y]
file_list_turned = [v for v in file_list if "turned" in v]
file_list_long = [w for w in file_list if "long" in w]
file_list_noadvection = [z for z in file_list if "no-advection" in z]
file_list_noadvection_long = [u for u in file_list if "no-advection_long" in u]

for i in range(len(file_list_noadvection_long)):
    file_list_long.remove(file_list_noadvection_long[i])
    file_list_noadvection.remove(file_list_noadvection_long[i])

dir_obs = r"W:\lindgrv1\Kiira_whole_day\kiira_14-22_dbz"

file_list_obs = os.listdir(dir_obs)
file_list_obs = [x for x in file_list_obs if ".tif" in x]
file_list_obs = [f for f in file_list_obs if f.endswith(".tif")]
file_list_obs = file_list_obs[8:-1]

out_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Accumulations"
save_figs = False

##############################################################################

# OBSERVED KIIRA

radar_kiira = []
for i in range(len(file_list_obs)):
    src = rasterio.open(os.path.join(dir_obs, file_list_obs[i]))
    array = src.read(1)
    radar_kiira.append(array)
    
radar_kiira = np.concatenate([radar_kiira_[None, :, :] for radar_kiira_ in radar_kiira])

#This have to be added for new event data: Remove last column from each layer
radar_kiira = radar_kiira[:,:,:-1]
#The following data is available for Finnish radar composite: radar reflectivity (dbz), conversion: Z[dBZ] = 0.5 * pixel value - 32
radar_kiira = (radar_kiira * 0.5) - 32
#Values less than threshold to wanted value, which represents no-rain
radar_kiira[radar_kiira < 10] = 3.1830486304816077

#Clear values over threshold of 45 dBZ -> This is not done for observed event, but just for simulated events
# radar_kiira[radar_kiira > 45] = 0.5*(45 + radar_kiira[radar_kiira > 45])

#Z-R relationship: Z = a*R^b (Reflectivity)
#dBZ conversion: dBZ = 10 * log10(Z) (Decibels of Reflectivity)
#Marshall, J. S., and W. McK. Palmer, 1948: The distribution of raindrops with size. J. Meteor., 5, 165–166.
a_R=223
b_R=1.53
#Leinonen et al. 2012 - A Climatology of Disdrometer Measurements of Rainfall in Finland over Five Years with Implications for Global Radar Observations

#Event from dbz into mm/h
radar_kiira_mmh = radar_kiira.copy()
radar_kiira_mmh = 10**((radar_kiira_mmh-10*np.log10(a_R))/(10*b_R))
#Values less than threshold to zero
radar_kiira_mmh[radar_kiira_mmh < 0.1] = 0

event_sim_array = radar_kiira_mmh.copy()

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

for i in range(len(event_sim_array)):
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

accus_obs = []
accus_obs.append(ts_1)
accus_obs.append(ts_2)
accus_obs.append(ts_3)
accus_obs.append(ts_4)
accus_obs.append(ts_5)
accus_obs.append(ts_6)
accus_obs.append(ts_7)
accus_obs.append(ts_8)
accus_obs.append(ts_9)
accus_obs.append(ts_10)
accus_obs.append(ts_11)
accus_obs.append(ts_12)
accus_obs.append(ts_13)
accus_obs.append(ts_14)
accus_obs.append(ts_15)
accus_obs.append(ts_16)
accus_obs.append(ts_17)
accus_obs.append(ts_18)
accus_obs.append(ts_19)
accus_obs.append(ts_20)
accus_obs.append(ts_21)
accus_obs.append(ts_22)
accus_obs.append(ts_23)
accus_obs.append(ts_24)
accus_obs.append(ts_25)

accus_obs = np.concatenate([accus_obs_[None, :] for accus_obs_ in accus_obs])
accus_obs_reshape = np.reshape(accus_obs, (accus_obs.shape[0]*accus_obs.shape[1], accus_obs.shape[2]))

accus_obs_reshape = accus_obs_reshape * (5/60)

accus_obs_cum = np.zeros((accus_obs_reshape.shape[0], accus_obs_reshape.shape[1]))
for j in range(accus_obs_cum.shape[1]):
    if j == 0:
        accus_obs_cum[:,j] = accus_obs_reshape[:,j]
    else:
        accus_obs_cum[:,j] = accus_obs_cum[:,j-1] + accus_obs_reshape[:,j]

accus_obs_cum = accus_obs_cum[:,:-1]

accus_obs_cum_mean = np.zeros((1, accus_obs_cum.shape[1]))
for k in range(accus_obs_cum_mean.shape[1]):
    accus_obs_cum_mean[0,k] = np.mean(accus_obs_cum[:,k])

#Plot
plt.figure()
for im in range(len(accus_obs_cum)):
    if im == 0:
        plt.plot(accus_obs_cum[im], color="gray", alpha=0.3, label="observed")
    else:
        plt.plot(accus_obs_cum[im], color="gray", alpha=0.3)
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean ({accus_obs_cum_mean[0,-1]})")
plt.legend()
plt.ylim((0,70))
plt.title("Observed")
if save_figs:
    plt.savefig(os.path.join(out_dir, "00_observed.png"))

##############################################################################

#Function to find closest value in array to a given value

def find_closest(arr, val):
       idx = np.abs(arr - val).argmin()
       return arr[idx]

##############################################################################

# NORMAL SIMULATIONS

accus_cum_normal = []
for i in range(len(file_list_normal)):
    temp_csv = os.path.join(dir_sim, file_list_normal[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    
    temp_mm_cum = np.zeros((temp_mm.shape[0], temp_mm.shape[1]))
    for j in range(temp_mm.shape[1]):
        if j == 0:
            temp_mm_cum[:,j] = temp_mm[:,j]
        else:
            temp_mm_cum[:,j] = temp_mm_cum[:,j-1] + temp_mm[:,j]
    
    accus_cum_normal.append(temp_mm_cum)

accus_cum_normal_arr = np.concatenate([accus_cum_normal_[None, :] for accus_cum_normal_ in accus_cum_normal])
accus_cum_normal_arr_reshape = np.reshape(accus_cum_normal_arr, (accus_cum_normal_arr.shape[0]*accus_cum_normal_arr.shape[1], accus_cum_normal_arr.shape[2]))

accus_cum_normal_arr_reshape_mean = np.zeros((1,accus_cum_normal_arr_reshape.shape[1]))
for k in range(accus_cum_normal_arr_reshape.shape[1]):
    accus_cum_normal_arr_reshape_mean[0,k] = np.mean(accus_cum_normal_arr_reshape[:,k])
    
accus_cum_normal_arr_reshape_median = np.zeros((1,accus_cum_normal_arr_reshape.shape[1]))
for k in range(accus_cum_normal_arr_reshape.shape[1]):
    accus_cum_normal_arr_reshape_median[0,k] = np.median(accus_cum_normal_arr_reshape[:,k])

#Plotting
plt.figure()
for im in range(len(accus_cum_normal_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_normal_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_normal_arr_reshape[im], color="gray", alpha=0.1)
# for im in range(len(accus_obs_cum)):
#     if im == 0:
#         plt.plot(accus_obs_cum[im], color="blue", alpha=0.4, label="observed")
#     else:
#         plt.plot(accus_obs_cum[im], color="blue", alpha=0.4) 
plt.plot(accus_cum_normal_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_normal_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_normal_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_normal_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_normal_arr_reshape[np.argmax(accus_cum_normal_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_normal_arr_reshape[int(np.where(accus_cum_normal_arr_reshape[:,-1] == find_closest(accus_cum_normal_arr_reshape[:,-1], np.median(accus_cum_normal_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_normal_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_normal_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_normal_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_normal_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_normal_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_normal_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_normal_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,70))
plt.title("Normal")

##############################################################################

# TURNED SIMULATIONS

accus_cum_turned = []
for i in range(len(file_list_turned)):
    temp_csv = os.path.join(dir_sim, file_list_turned[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    
    temp_mm_cum = np.zeros((temp_mm.shape[0], temp_mm.shape[1]))
    for j in range(temp_mm.shape[1]):
        if j == 0:
            temp_mm_cum[:,j] = temp_mm[:,j]
        else:
            temp_mm_cum[:,j] = temp_mm_cum[:,j-1] + temp_mm[:,j]
    
    accus_cum_turned.append(temp_mm_cum)

accus_cum_turned_arr = np.concatenate([accus_cum_turned_[None, :] for accus_cum_turned_ in accus_cum_turned])
accus_cum_turned_arr_reshape = np.reshape(accus_cum_turned_arr, (accus_cum_turned_arr.shape[0]*accus_cum_turned_arr.shape[1], accus_cum_turned_arr.shape[2]))

accus_cum_turned_arr_reshape_mean = np.zeros((1,accus_cum_turned_arr_reshape.shape[1]))
for k in range(accus_cum_turned_arr_reshape.shape[1]):
    accus_cum_turned_arr_reshape_mean[0,k] = np.mean(accus_cum_turned_arr_reshape[:,k])
    
accus_cum_turned_arr_reshape_median = np.zeros((1,accus_cum_turned_arr_reshape.shape[1]))
for k in range(accus_cum_turned_arr_reshape.shape[1]):
    accus_cum_turned_arr_reshape_median[0,k] = np.median(accus_cum_turned_arr_reshape[:,k])

#Plotting
plt.figure()
for im in range(len(accus_cum_turned_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_turned_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_turned_arr_reshape[im], color="gray", alpha=0.1)
# for im in range(len(accus_obs_cum)):
#     if im == 0:
#         plt.plot(accus_obs_cum[im], color="blue", alpha=0.4, label="observed")
#     else:
#         plt.plot(accus_obs_cum[im], color="blue", alpha=0.4)
plt.plot(accus_cum_turned_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_turned_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_turned_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_turned_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_turned_arr_reshape[np.argmax(accus_cum_turned_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_turned_arr_reshape[int(np.where(accus_cum_turned_arr_reshape[:,-1] == find_closest(accus_cum_turned_arr_reshape[:,-1], np.median(accus_cum_turned_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_turned_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_turned_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_turned_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_turned_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_turned_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_turned_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_turned_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,70))
plt.title("Turned")

##############################################################################

# SLOW SIMULATIONS

accus_cum_slow = []
for i in range(len(file_list_slow)):
    temp_csv = os.path.join(dir_sim, file_list_slow[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    
    temp_mm_cum = np.zeros((temp_mm.shape[0], temp_mm.shape[1]))
    for j in range(temp_mm.shape[1]):
        if j == 0:
            temp_mm_cum[:,j] = temp_mm[:,j]
        else:
            temp_mm_cum[:,j] = temp_mm_cum[:,j-1] + temp_mm[:,j]
    
    accus_cum_slow.append(temp_mm_cum)

accus_cum_slow_arr = np.concatenate([accus_cum_slow_[None, :] for accus_cum_slow_ in accus_cum_slow])
accus_cum_slow_arr_reshape = np.reshape(accus_cum_slow_arr, (accus_cum_slow_arr.shape[0]*accus_cum_slow_arr.shape[1], accus_cum_slow_arr.shape[2]))

accus_cum_slow_arr_reshape_mean = np.zeros((1,accus_cum_slow_arr_reshape.shape[1]))
for k in range(accus_cum_slow_arr_reshape.shape[1]):
    accus_cum_slow_arr_reshape_mean[0,k] = np.mean(accus_cum_slow_arr_reshape[:,k])
    
accus_cum_slow_arr_reshape_median = np.zeros((1,accus_cum_slow_arr_reshape.shape[1]))
for k in range(accus_cum_slow_arr_reshape.shape[1]):
    accus_cum_slow_arr_reshape_median[0,k] = np.median(accus_cum_slow_arr_reshape[:,k])

#Plotting
plt.figure()
for im in range(len(accus_cum_slow_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_slow_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_slow_arr_reshape[im], color="gray", alpha=0.1)
# for im in range(len(accus_obs_cum)):
#     if im == 0:
#         plt.plot(accus_obs_cum[im], color="blue", alpha=0.4, label="observed")
#     else:
#         plt.plot(accus_obs_cum[im], color="blue", alpha=0.4) 
plt.plot(accus_cum_slow_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_slow_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_slow_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_slow_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_slow_arr_reshape[np.argmax(accus_cum_slow_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_slow_arr_reshape[int(np.where(accus_cum_slow_arr_reshape[:,-1] == find_closest(accus_cum_slow_arr_reshape[:,-1], np.median(accus_cum_slow_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_slow_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_slow_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_slow_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_slow_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_slow_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_slow_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_slow_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,70))
plt.title("Slow")

##############################################################################

# LONG SIMULATIONS

accus_cum_long = []
for i in range(len(file_list_long)):
    temp_csv = os.path.join(dir_sim, file_list_long[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    
    temp_mm_cum = np.zeros((temp_mm.shape[0], temp_mm.shape[1]))
    for j in range(temp_mm.shape[1]):
        if j == 0:
            temp_mm_cum[:,j] = temp_mm[:,j]
        else:
            temp_mm_cum[:,j] = temp_mm_cum[:,j-1] + temp_mm[:,j]
    
    accus_cum_long.append(temp_mm_cum)

accus_cum_long_arr = np.concatenate([accus_cum_long_[None, :] for accus_cum_long_ in accus_cum_long])
accus_cum_long_arr_reshape = np.reshape(accus_cum_long_arr, (accus_cum_long_arr.shape[0]*accus_cum_long_arr.shape[1], accus_cum_long_arr.shape[2]))

accus_cum_long_arr_reshape_mean = np.zeros((1,accus_cum_long_arr_reshape.shape[1]))
for k in range(accus_cum_long_arr_reshape.shape[1]):
    accus_cum_long_arr_reshape_mean[0,k] = np.mean(accus_cum_long_arr_reshape[:,k])
    
accus_cum_long_arr_reshape_median = np.zeros((1,accus_cum_long_arr_reshape.shape[1]))
for k in range(accus_cum_long_arr_reshape.shape[1]):
    accus_cum_long_arr_reshape_median[0,k] = np.median(accus_cum_long_arr_reshape[:,k])

#Plotting
plt.figure()
for im in range(len(accus_cum_long_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_long_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_long_arr_reshape[im], color="gray", alpha=0.1)
plt.plot(accus_cum_long_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_long_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_long_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_long_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_long_arr_reshape[np.argmax(accus_cum_long_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_long_arr_reshape[int(np.where(accus_cum_long_arr_reshape[:,-1] == find_closest(accus_cum_long_arr_reshape[:,-1], np.median(accus_cum_long_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_long_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_long_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_long_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_long_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_long_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_long_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_long_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,70))
plt.title("Long")

##############################################################################

#NORMAL MM
mm_normal = []
for i in range(len(file_list_normal)):
    temp_csv = os.path.join(dir_sim, file_list_normal[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    mm_normal.append(temp_mm)
mm_normal_arr = np.concatenate([mm_normal_[None, :] for mm_normal_ in mm_normal])
mm_normal_arr_reshape = np.reshape(mm_normal_arr, (mm_normal_arr.shape[0]*mm_normal_arr.shape[1], mm_normal_arr.shape[2]))
   
#TURNED MM
mm_turned = []
for i in range(len(file_list_turned)):
    temp_csv = os.path.join(dir_sim, file_list_turned[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    mm_turned.append(temp_mm)
mm_turned_arr = np.concatenate([mm_turned_[None, :] for mm_turned_ in mm_turned])
mm_turned_arr_reshape = np.reshape(mm_turned_arr, (mm_turned_arr.shape[0]*mm_turned_arr.shape[1], mm_turned_arr.shape[2]))

#SLOW MM
mm_slow = []
for i in range(len(file_list_slow)):
    temp_csv = os.path.join(dir_sim, file_list_slow[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    mm_slow.append(temp_mm)
mm_slow_arr = np.concatenate([mm_slow_[None, :] for mm_slow_ in mm_slow])
mm_slow_arr_reshape = np.reshape(mm_slow_arr, (mm_slow_arr.shape[0]*mm_slow_arr.shape[1], mm_slow_arr.shape[2]))

#LONG MM
mm_long = []
for i in range(len(file_list_long)):
    temp_csv = os.path.join(dir_sim, file_list_long[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    mm_long.append(temp_mm)
mm_long_arr = np.concatenate([mm_long_[None, :] for mm_long_ in mm_long])
mm_long_arr_reshape = np.reshape(mm_long_arr, (mm_long_arr.shape[0]*mm_long_arr.shape[1], mm_long_arr.shape[2]))

# #SAVE CSVS
# csv_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
# mm_normal_arr_reshape_df = pd.DataFrame(mm_normal_arr_reshape)
# mm_normal_arr_reshape_df.to_csv(os.path.join(csv_dir, "mm_normal.csv"), header=False, index=False)
# mm_turned_arr_reshape_df = pd.DataFrame(mm_turned_arr_reshape)
# mm_turned_arr_reshape_df.to_csv(os.path.join(csv_dir, "mm_turned.csv"), header=False, index=False)
# mm_slow_arr_reshape_df = pd.DataFrame(mm_slow_arr_reshape)
# mm_slow_arr_reshape_df.to_csv(os.path.join(csv_dir, "mm_slow.csv"), header=False, index=False)
# mm_long_arr_reshape_df = pd.DataFrame(mm_long_arr_reshape)
# mm_long_arr_reshape_df.to_csv(os.path.join(csv_dir, "mm_long.csv"), header=False, index=False)

##############################################################################

# NO-ADVECTION SIMULATIONS

accus_cum_noadvection = []
for i in range(len(file_list_noadvection)):
    temp_csv = os.path.join(dir_sim, file_list_noadvection[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    
    temp_mm_cum = np.zeros((temp_mm.shape[0], temp_mm.shape[1]))
    for j in range(temp_mm.shape[1]):
        if j == 0:
            temp_mm_cum[:,j] = temp_mm[:,j]
        else:
            temp_mm_cum[:,j] = temp_mm_cum[:,j-1] + temp_mm[:,j]
    
    accus_cum_noadvection.append(temp_mm_cum)

accus_cum_noadvection_arr = np.concatenate([accus_cum_noadvection_[None, :] for accus_cum_noadvection_ in accus_cum_noadvection])
accus_cum_noadvection_arr_reshape = np.reshape(accus_cum_noadvection_arr, (accus_cum_noadvection_arr.shape[0]*accus_cum_noadvection_arr.shape[1], accus_cum_noadvection_arr.shape[2]))

accus_cum_noadvection_arr_reshape_mean = np.zeros((1,accus_cum_noadvection_arr_reshape.shape[1]))
for k in range(accus_cum_noadvection_arr_reshape.shape[1]):
    accus_cum_noadvection_arr_reshape_mean[0,k] = np.mean(accus_cum_noadvection_arr_reshape[:,k])
    
accus_cum_noadvection_arr_reshape_median = np.zeros((1,accus_cum_noadvection_arr_reshape.shape[1]))
for k in range(accus_cum_noadvection_arr_reshape.shape[1]):
    accus_cum_noadvection_arr_reshape_median[0,k] = np.median(accus_cum_noadvection_arr_reshape[:,k])

#Plotting
plt.figure()
for im in range(len(accus_cum_noadvection_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_noadvection_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_noadvection_arr_reshape[im], color="gray", alpha=0.1)
plt.plot(accus_cum_noadvection_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_noadvection_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_noadvection_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_noadvection_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_noadvection_arr_reshape[np.argmax(accus_cum_noadvection_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_noadvection_arr_reshape[int(np.where(accus_cum_noadvection_arr_reshape[:,-1] == find_closest(accus_cum_noadvection_arr_reshape[:,-1], np.median(accus_cum_noadvection_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_noadvection_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,230))
plt.title("No advection")

##############################################################################

# LONG NO-ADVECTION SIMULATIONS

accus_cum_noadvection_long = []
for i in range(len(file_list_noadvection_long)):
    temp_csv = os.path.join(dir_sim, file_list_noadvection_long[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    
    temp_mm_cum = np.zeros((temp_mm.shape[0], temp_mm.shape[1]))
    for j in range(temp_mm.shape[1]):
        if j == 0:
            temp_mm_cum[:,j] = temp_mm[:,j]
        else:
            temp_mm_cum[:,j] = temp_mm_cum[:,j-1] + temp_mm[:,j]
    
    accus_cum_noadvection_long.append(temp_mm_cum)

accus_cum_noadvection_long_arr = np.concatenate([accus_cum_noadvection_long_[None, :] for accus_cum_noadvection_long_ in accus_cum_noadvection_long])
accus_cum_noadvection_long_arr_reshape = np.reshape(accus_cum_noadvection_long_arr, (accus_cum_noadvection_long_arr.shape[0]*accus_cum_noadvection_long_arr.shape[1], accus_cum_noadvection_long_arr.shape[2]))

accus_cum_noadvection_long_arr_reshape_mean = np.zeros((1,accus_cum_noadvection_long_arr_reshape.shape[1]))
for k in range(accus_cum_noadvection_long_arr_reshape.shape[1]):
    accus_cum_noadvection_long_arr_reshape_mean[0,k] = np.mean(accus_cum_noadvection_long_arr_reshape[:,k])
    
accus_cum_noadvection_long_arr_reshape_median = np.zeros((1,accus_cum_noadvection_long_arr_reshape.shape[1]))
for k in range(accus_cum_noadvection_long_arr_reshape.shape[1]):
    accus_cum_noadvection_long_arr_reshape_median[0,k] = np.median(accus_cum_noadvection_long_arr_reshape[:,k])

#Plotting
plt.figure()
for im in range(len(accus_cum_noadvection_long_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_noadvection_long_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_noadvection_long_arr_reshape[im], color="gray", alpha=0.1)
plt.plot(accus_cum_noadvection_long_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_noadvection_long_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_noadvection_long_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_noadvection_long_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_noadvection_long_arr_reshape[np.argmax(accus_cum_noadvection_long_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_noadvection_long_arr_reshape[int(np.where(accus_cum_noadvection_long_arr_reshape[:,-1] == find_closest(accus_cum_noadvection_long_arr_reshape[:,-1], np.median(accus_cum_noadvection_long_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_noadvection_long_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,460))
plt.title("No advection and Long")

##############################################################################

# HISTOGRAMS

plt.figure()
plt.hist(accus_cum_normal_arr_reshape[:,-1], bins=20, color='skyblue', edgecolor='black', density=True)
plt.title("Normal")
plt.ylim((0,0.125))
plt.xlim((0,70))
if save_figs:
    plt.savefig(os.path.join(out_dir, "01_hist_normal.png"))

# np.histogram(accus_cum_normal_arr_reshape[:,-1], bins=20, range=None, density=True, weights=None)

plt.figure()
plt.hist(accus_cum_turned_arr_reshape[:,-1], bins=20, color='skyblue', edgecolor='black', density=True)
plt.title("Turned")
plt.ylim((0,0.125))
plt.xlim((0,70))
if save_figs:
    plt.savefig(os.path.join(out_dir, "02_hist_turned.png"))

plt.figure()
plt.hist(accus_cum_slow_arr_reshape[:,-1], bins=20, color='skyblue', edgecolor='black', density=True)
plt.title("Slow")
plt.ylim((0,0.125))
plt.xlim((0,70))
if save_figs:
    plt.savefig(os.path.join(out_dir, "03_hist_slow.png"))

plt.figure()
plt.hist(accus_cum_long_arr_reshape[:,-1], bins=20, color='skyblue', edgecolor='black', density=True)
plt.title("Long")
plt.ylim((0,0.125))
plt.xlim((0,70))
if save_figs:
    plt.savefig(os.path.join(out_dir, "04_hist_long.png"))
    
plt.figure()
plt.hist([accus_cum_normal_arr_reshape[:,-1], accus_cum_turned_arr_reshape[:,-1], accus_cum_slow_arr_reshape[:,-1], accus_cum_long_arr_reshape[:,-1]], 
         bins=20, stacked=False, edgecolor='black', density=True)
plt.legend(['Normal', 'Turned', 'Slow', 'Long'])
plt.title("All")
if save_figs:
    plt.savefig(os.path.join(out_dir, "05_hist_all_bars.png"))

plt.figure()
plt.hist([accus_cum_normal_arr_reshape[:,-1], accus_cum_turned_arr_reshape[:,-1], accus_cum_slow_arr_reshape[:,-1], accus_cum_long_arr_reshape[:,-1]], 
         bins=20, stacked=False, density=True, histtype='step')
plt.title("All")

plt.figure()
sns.histplot(data=[accus_cum_normal_arr_reshape[:,-1], accus_cum_turned_arr_reshape[:,-1], accus_cum_slow_arr_reshape[:,-1], accus_cum_long_arr_reshape[:,-1]], 
             bins=20, kde=True, stat="percent")
plt.title("All")
if save_figs:
    plt.savefig(os.path.join(out_dir, "06_hist_all_bars-lines.png"))

plt.figure()
sns.kdeplot(data=[accus_cum_normal_arr_reshape[:,-1], accus_cum_turned_arr_reshape[:,-1], accus_cum_slow_arr_reshape[:,-1], accus_cum_long_arr_reshape[:,-1]], cut=0)
plt.title("All")
if save_figs:
    plt.savefig(os.path.join(out_dir, "07_hist_all_lines.png"))
    
##############################################################################

plt.figure()
plt.hist(accus_cum_noadvection_arr_reshape[:,-1], bins=20, color='skyblue', edgecolor='black', density=True)
plt.title("No advection")
plt.ylim((0,0.08))
plt.xlim((0,230))
if save_figs:
    plt.savefig(os.path.join(out_dir, "15_hist_noadvection.png"))

plt.figure()
sns.histplot(data=[accus_cum_noadvection_arr_reshape[:,-1]], bins=20, kde=True, stat="percent")
plt.title("No advection")
if save_figs:
    plt.savefig(os.path.join(out_dir, "16_hist_noadvection_bars-lines.png"))

plt.figure()
sns.kdeplot(data=[accus_cum_noadvection_arr_reshape[:,-1]], cut=0)
plt.title("No advection")
if save_figs:
    plt.savefig(os.path.join(out_dir, "17_hist_noadvection_lines.png"))

##############################################################################

plt.figure()
plt.hist(accus_cum_noadvection_long_arr_reshape[:,-1], bins=20, color='skyblue', edgecolor='black', density=True)
plt.title("No advection and Long")
plt.ylim((0,0.04))
plt.xlim((0,460))
if save_figs:
    plt.savefig(os.path.join(out_dir, "23_hist_noadvection_long.png"))

plt.figure()
sns.histplot(data=[accus_cum_noadvection_long_arr_reshape[:,-1]], bins=20, kde=True, stat="percent")
plt.title("No advection and Long")
if save_figs:
    plt.savefig(os.path.join(out_dir, "24_hist_noadvection_long_bars-lines.png"))

plt.figure()
sns.kdeplot(data=[accus_cum_noadvection_long_arr_reshape[:,-1]], cut=0)
plt.title("No advection and Long")
if save_figs:
    plt.savefig(os.path.join(out_dir, "25_hist_noadvection_long_lines.png"))

##############################################################################

plt.figure()
sns.kdeplot(data=[accus_cum_normal_arr_reshape[:,-1], accus_cum_turned_arr_reshape[:,-1], accus_cum_slow_arr_reshape[:,-1], accus_cum_long_arr_reshape[:,-1], accus_cum_noadvection_arr_reshape[:,-1], accus_cum_noadvection_long_arr_reshape[:,-1]], cut=0)
plt.title("All")

##############################################################################

# TOP 100

accus_cum_normal_arr_reshape_top100 = accus_cum_normal_arr_reshape[(np.where(accus_cum_normal_arr_reshape[:,-1] > np.percentile(accus_cum_normal_arr_reshape[:,-1], 90)))]
accus_cum_turned_arr_reshape_top100 = accus_cum_turned_arr_reshape[(np.where(accus_cum_turned_arr_reshape[:,-1] > np.percentile(accus_cum_turned_arr_reshape[:,-1], 90)))]
accus_cum_slow_arr_reshape_top100 = accus_cum_slow_arr_reshape[(np.where(accus_cum_slow_arr_reshape[:,-1] > np.percentile(accus_cum_slow_arr_reshape[:,-1], 90)))]
accus_cum_long_arr_reshape_top100 = accus_cum_long_arr_reshape[(np.where(accus_cum_long_arr_reshape[:,-1] > np.percentile(accus_cum_long_arr_reshape[:,-1], 90)))]

plt.figure()
plt.hist([accus_cum_normal_arr_reshape_top100[:,-1], accus_cum_turned_arr_reshape_top100[:,-1], accus_cum_slow_arr_reshape_top100[:,-1], accus_cum_long_arr_reshape_top100[:,-1]], 
         bins=20, stacked=False, edgecolor='black', density=True)
plt.legend(['Normal', 'Turned', 'Slow', 'Long'])
plt.title("Top 100")
if save_figs:
    plt.savefig(os.path.join(out_dir, "08_hist_top100_bars.png"))

# np.histogram(accus_cum_normal_arr_reshape_top100[:,-1], bins=20, range=None, density=True, weights=None)

plt.figure()
sns.histplot(data=[accus_cum_normal_arr_reshape_top100[:,-1], accus_cum_turned_arr_reshape_top100[:,-1], accus_cum_slow_arr_reshape_top100[:,-1], accus_cum_long_arr_reshape_top100[:,-1]], 
             bins=20, kde=True, stat="percent")
plt.title("Top 100")
if save_figs:
    plt.savefig(os.path.join(out_dir, "09_hist_top100_bars-lines.png"))

plt.figure()
sns.kdeplot(data=[accus_cum_normal_arr_reshape_top100[:,-1], accus_cum_turned_arr_reshape_top100[:,-1], accus_cum_slow_arr_reshape_top100[:,-1], accus_cum_long_arr_reshape_top100[:,-1]], cut=0)
plt.title("Top 100")
if save_figs:
    plt.savefig(os.path.join(out_dir, "10_hist_top100_lines.png"))

plt.figure()
for im in range(len(accus_cum_normal_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_normal_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_normal_arr_reshape[im], color="gray", alpha=0.1)
for im in range(len(accus_cum_normal_arr_reshape_top100)):
    if im == 0:
        plt.plot(accus_cum_normal_arr_reshape_top100[im], color="teal", alpha=0.1, label="top 100")
    else:
        plt.plot(accus_cum_normal_arr_reshape_top100[im], color="teal", alpha=0.1) 
plt.plot(accus_cum_normal_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_normal_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_normal_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_normal_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_normal_arr_reshape[np.argmax(accus_cum_normal_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_normal_arr_reshape[int(np.where(accus_cum_normal_arr_reshape[:,-1] == find_closest(accus_cum_normal_arr_reshape[:,-1], np.median(accus_cum_normal_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_normal_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_normal_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_normal_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_normal_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_normal_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_normal_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_normal_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,70))
plt.title("Normal")
if save_figs:
    plt.savefig(os.path.join(out_dir, "11_ts_normal.png"))

plt.figure()
for im in range(len(accus_cum_turned_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_turned_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_turned_arr_reshape[im], color="gray", alpha=0.1)
for im in range(len(accus_cum_turned_arr_reshape_top100)):
    if im == 0:
        plt.plot(accus_cum_turned_arr_reshape_top100[im], color="teal", alpha=0.1, label="top 100")
    else:
        plt.plot(accus_cum_turned_arr_reshape_top100[im], color="teal", alpha=0.1)
plt.plot(accus_cum_turned_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_turned_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_turned_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_turned_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_turned_arr_reshape[np.argmax(accus_cum_turned_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_turned_arr_reshape[int(np.where(accus_cum_turned_arr_reshape[:,-1] == find_closest(accus_cum_turned_arr_reshape[:,-1], np.median(accus_cum_turned_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_turned_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_turned_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_turned_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_turned_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_turned_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_turned_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_turned_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,70))
plt.title("Turned")
if save_figs:
    plt.savefig(os.path.join(out_dir, "12_ts_turned.png"))

plt.figure()
for im in range(len(accus_cum_slow_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_slow_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_slow_arr_reshape[im], color="gray", alpha=0.1)
for im in range(len(accus_cum_slow_arr_reshape_top100)):
    if im == 0:
        plt.plot(accus_cum_slow_arr_reshape_top100[im], color="teal", alpha=0.1, label="top 100")
    else:
        plt.plot(accus_cum_slow_arr_reshape_top100[im], color="teal", alpha=0.1)
plt.plot(accus_cum_slow_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_slow_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_slow_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_slow_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_slow_arr_reshape[np.argmax(accus_cum_slow_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_slow_arr_reshape[int(np.where(accus_cum_slow_arr_reshape[:,-1] == find_closest(accus_cum_slow_arr_reshape[:,-1], np.median(accus_cum_slow_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_slow_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_slow_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_slow_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_slow_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_slow_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_slow_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_slow_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,70))
plt.title("Slow")
if save_figs:
    plt.savefig(os.path.join(out_dir, "13_ts_slow.png"))

plt.figure()
for im in range(len(accus_cum_long_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_long_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_long_arr_reshape[im], color="gray", alpha=0.1)
for im in range(len(accus_cum_long_arr_reshape_top100)):
    if im == 0:
        plt.plot(accus_cum_long_arr_reshape_top100[im], color="teal", alpha=0.1, label="top 100")
    else:
        plt.plot(accus_cum_long_arr_reshape_top100[im], color="teal", alpha=0.1)
plt.plot(accus_cum_long_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_long_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_long_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_long_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_long_arr_reshape[np.argmax(accus_cum_long_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_long_arr_reshape[int(np.where(accus_cum_long_arr_reshape[:,-1] == find_closest(accus_cum_long_arr_reshape[:,-1], np.median(accus_cum_long_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_long_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_long_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_long_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_long_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_long_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_long_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_long_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,70))
plt.title("Long")
if save_figs:
    plt.savefig(os.path.join(out_dir, "14_ts_long.png"))

accus_cum_long_arr_reshape_half = accus_cum_long_arr_reshape[:,0:accus_cum_normal_arr_reshape.shape[1]]

plt.figure()
for im in range(len(accus_cum_long_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_long_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_long_arr_reshape[im], color="gray", alpha=0.1)
for im in range(len(accus_cum_long_arr_reshape_half)):
    if im == 0:
        plt.plot(accus_cum_long_arr_reshape_half[im], color="teal", alpha=0.1, label="top 100")
    else:
        plt.plot(accus_cum_long_arr_reshape_half[im], color="teal", alpha=0.1)
plt.plot(accus_cum_long_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_long_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_long_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_long_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_long_arr_reshape[np.argmax(accus_cum_long_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_long_arr_reshape[int(np.where(accus_cum_long_arr_reshape[:,-1] == find_closest(accus_cum_long_arr_reshape[:,-1], np.median(accus_cum_long_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_long_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_long_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_long_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_long_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_long_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_long_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_long_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,70))
plt.title("Long")

##############################################################################

accus_cum_noadvection_arr_reshape_top100 = accus_cum_noadvection_arr_reshape[(np.where(accus_cum_noadvection_arr_reshape[:,-1] > np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 90)))]

plt.figure()
plt.hist(accus_cum_noadvection_arr_reshape_top100[:,-1], bins=20, color='skyblue', edgecolor='black', density=True)
plt.title("No advection - Top 100")
plt.ylim((0,0.08))
plt.xlim((0,230))
if save_figs:
    plt.savefig(os.path.join(out_dir, "18_hist_top100_noadvection.png"))
    
plt.figure()
plt.hist(accus_cum_noadvection_arr_reshape_top100[:,-1], bins=20, color='skyblue', edgecolor='black', density=True)
plt.title("No advection - Top 100")
plt.ylim((0,0.035))
plt.xlim((0,230))
if save_figs:
    plt.savefig(os.path.join(out_dir, "19_hist_top100_noadvection_scaled.png"))

plt.figure()
sns.histplot(data=[accus_cum_noadvection_arr_reshape_top100[:,-1]], bins=20, kde=True, stat="percent")
plt.title("No advection - Top 100")
if save_figs:
    plt.savefig(os.path.join(out_dir, "20_hist_top100_noadvection_bars-lines.png"))

plt.figure()
sns.kdeplot(data=[accus_cum_noadvection_arr_reshape_top100[:,-1]], cut=0)
plt.title("No advection - Top 100")
if save_figs:
    plt.savefig(os.path.join(out_dir, "21_hist_top100_noadvection_lines.png"))
    
plt.figure()
for im in range(len(accus_cum_noadvection_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_noadvection_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_noadvection_arr_reshape[im], color="gray", alpha=0.1)
for im in range(len(accus_cum_noadvection_arr_reshape_top100)):
    if im == 0:
        plt.plot(accus_cum_noadvection_arr_reshape_top100[im], color="teal", alpha=0.1, label="top 100")
    else:
        plt.plot(accus_cum_noadvection_arr_reshape_top100[im], color="teal", alpha=0.1) 
plt.plot(accus_cum_noadvection_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_noadvection_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_noadvection_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_noadvection_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_noadvection_arr_reshape[np.argmax(accus_cum_noadvection_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_noadvection_arr_reshape[int(np.where(accus_cum_noadvection_arr_reshape[:,-1] == find_closest(accus_cum_noadvection_arr_reshape[:,-1], np.median(accus_cum_noadvection_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_noadvection_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_noadvection_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,230))
plt.title("No advection")
if save_figs:
    plt.savefig(os.path.join(out_dir, "22_ts_noadvection.png"))
    
##############################################################################

accus_cum_noadvection_long_arr_reshape_top100 = accus_cum_noadvection_long_arr_reshape[(np.where(accus_cum_noadvection_long_arr_reshape[:,-1] > np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 90)))]

plt.figure()
plt.hist(accus_cum_noadvection_long_arr_reshape_top100[:,-1], bins=20, color='skyblue', edgecolor='black', density=True)
plt.title("No advection and Long - Top 100")
plt.ylim((0,0.04))
plt.xlim((0,460))
if save_figs:
    plt.savefig(os.path.join(out_dir, "26_hist_top100_noadvection_long.png"))
    
plt.figure()
plt.hist(accus_cum_noadvection_long_arr_reshape_top100[:,-1], bins=20, color='skyblue', edgecolor='black', density=True)
plt.title("No advection and Long - Top 100")
plt.ylim((0,0.02))
plt.xlim((0,460))
if save_figs:
    plt.savefig(os.path.join(out_dir, "27_hist_top100_noadvection_long_scaled.png"))

plt.figure()
sns.histplot(data=[accus_cum_noadvection_long_arr_reshape_top100[:,-1]], bins=20, kde=True, stat="percent")
plt.title("No advection and Long - Top 100")
if save_figs:
    plt.savefig(os.path.join(out_dir, "28_hist_top100_noadvection_long_bars-lines.png"))

plt.figure()
sns.kdeplot(data=[accus_cum_noadvection_long_arr_reshape_top100[:,-1]], cut=0)
plt.title("No advection and Long - Top 100")
if save_figs:
    plt.savefig(os.path.join(out_dir, "29_hist_top100_noadvection_long_lines.png"))

plt.figure()
for im in range(len(accus_cum_noadvection_long_arr_reshape)):
    if im == 0:
        plt.plot(accus_cum_noadvection_long_arr_reshape[im], color="gray", alpha=0.1, label="member")
    else:
        plt.plot(accus_cum_noadvection_long_arr_reshape[im], color="gray", alpha=0.1)
for im in range(len(accus_cum_noadvection_long_arr_reshape_top100)):
    if im == 0:
        plt.plot(accus_cum_noadvection_long_arr_reshape_top100[im], color="teal", alpha=0.1, label="top 100")
    else:
        plt.plot(accus_cum_noadvection_long_arr_reshape_top100[im], color="teal", alpha=0.1) 
plt.plot(accus_cum_noadvection_long_arr_reshape_mean[0], color="red", label=f"mean ({accus_cum_noadvection_long_arr_reshape_mean[0,-1]})")
plt.plot(accus_cum_noadvection_long_arr_reshape_median[0], color="blue", label=f"median ({accus_cum_noadvection_long_arr_reshape_median[0,-1]})")
plt.plot(accus_obs_cum_mean[0], color="cyan", label=f"mean obs ({accus_obs_cum_mean[0,-1]})")
plt.plot((accus_cum_noadvection_long_arr_reshape[np.argmax(accus_cum_noadvection_long_arr_reshape[:,-1], axis=0)]), color="orange", label="max")
plt.plot((accus_cum_noadvection_long_arr_reshape[int(np.where(accus_cum_noadvection_long_arr_reshape[:,-1] == find_closest(accus_cum_noadvection_long_arr_reshape[:,-1], np.median(accus_cum_noadvection_long_arr_reshape[:,-1])))[0])]), color="purple", label=f"median ({np.median(accus_cum_noadvection_long_arr_reshape[:,-1])})")
plt.axhline(y = np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 90), color = 'black', linestyle = ':', label=f"90% ({np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 90)})")
plt.axhline(y = np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 75), color = 'black', linestyle = ':', label=f"75% ({np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 75)})")
plt.axhline(y = np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 25), color = 'black', linestyle = ':', label=f"25% ({np.percentile(accus_cum_noadvection_long_arr_reshape[:,-1], 25)})")
plt.legend()
plt.ylim((0,460))
plt.title("No advection and Long")
if save_figs:
    plt.savefig(os.path.join(out_dir, "30_ts_noadvection_long.png"))
    
##############################################################################

plt.figure()
sns.kdeplot(data=[accus_cum_normal_arr_reshape_top100[:,-1], accus_cum_turned_arr_reshape_top100[:,-1], accus_cum_slow_arr_reshape_top100[:,-1], accus_cum_long_arr_reshape_top100[:,-1], accus_cum_noadvection_arr_reshape_top100[:,-1], accus_cum_noadvection_long_arr_reshape_top100[:,-1]], cut=0)
plt.title("Top 100")

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

# MODEL PERFORMANCE

from numpy import genfromtxt
from pysteps.utils import rapsd
from pysteps.visualization import plot_spectrum1d

# 1D POWER SPECTRUMS - SIMULATIONS

R_powers_means = []
R_freqs_means = []

for i in range(len(file_list_normal)):
    R_powers = []
    R_freqs = []
    R_powers_mean = []
    R_freqs_mean = []
    
    chosen_realization = file_list_normal[i]
    data_dir2 = os.path.join(dir_sim, chosen_realization, "Event_tiffs_dbz")
    files = os.listdir(data_dir2)
    
    event_sim_array_sim  = []
    for i in range(len(files)):
        temp_raster = rasterio.open(os.path.join(data_dir2, f"test_{i}.tif"))
        temp_array = temp_raster.read(1)
        event_sim_array_sim.append(temp_array)
        if i == 0:
            event_affine = temp_raster.transform  
    event_sim_arra_simy = np.concatenate([event_sim_array_sim_[None, :, :] for event_sim_array_sim_ in event_sim_array_sim])
    
    for k in range(len(event_sim_array_sim)):
        R_, freq = rapsd(event_sim_array_sim[k], fft_method=np.fft, return_freq=True)
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

##############################################################################

# 1D POWER SPECTRUMS - OBSERVED KIIRA

R_powers_kiira = []
R_freqs_kiira = []
R_powers_kiira_mean = []
R_freqs_kiira_mean = []

R = radar_kiira.copy()
R = R[:-1,:,:]

#Replace non-finite values with the minimum value
for i in range(R.shape[0]):
    R[i, ~np.isfinite(R[i, :])] = np.nanmin(R[i, :])

for k in range(len(R)):
    R_, freq = rapsd(R[k], fft_method=np.fft, return_freq=True)
    R_powers_kiira.append(R_)
    R_freqs_kiira.append(freq)
    
R_powers_kiira = np.vstack(R_powers_kiira)
R_freqs_kiira = np.vstack(R_freqs_kiira)

for i in range(R_powers_kiira.shape[1]):
    R_powers_kiira_mean.append(np.mean(R_powers_kiira[:,i]))
    R_freqs_kiira_mean.append(np.mean(R_freqs_kiira[:,i]))
    
##############################################################################

#SAVE CSVs

R_powers_means_csv = np.vstack(R_powers_means)
R_powers_means_csv = np.hstack(R_powers_means_csv)
R_freqs_means_csv = np.vstack(R_freqs_means)
R_freqs_means_csv = np.hstack(R_freqs_means_csv)
R_powers_means_mean_csv = np.vstack(R_powers_means_mean)
R_freqs_means_mean_csv = np.vstack(R_freqs_means_mean)

R_powers_kiira_csv = np.hstack(R_powers_kiira)
R_freqs_kiira_csv = np.hstack(R_freqs_kiira)
R_powers_kiira_mean_csv = np.vstack(R_powers_kiira_mean)
R_freqs_kiira_mean_csv = np.vstack(R_freqs_kiira_mean)

outdir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Model_performance"

# data_temp_powers = [R_powers_means_csv, R_freqs_means_csv]
# data_temp_save = pd.DataFrame(data_temp_powers, index=['pow', 'freq'])
# pd.DataFrame(data_temp_save).to_csv(os.path.join(outdir, "fig_a_pows_freqs.csv"))

# data_temp_powers_mean = [R_powers_means_mean_csv[:,0], R_freqs_means_mean_csv[:,0]]
# data_temp_save = pd.DataFrame(data_temp_powers_mean, index=['pow', 'freq'])
# pd.DataFrame(data_temp_save).to_csv(os.path.join(outdir, "fig_a_pows_freqs_mean.csv"))

# data_temp_kiira_powers = [R_powers_kiira_csv, R_freqs_kiira_csv]
# data_temp_save = pd.DataFrame(data_temp_kiira_powers, index=['pow', 'freq'])
# pd.DataFrame(data_temp_save).to_csv(os.path.join(outdir, "fig_a_kiira_pows_freqs.csv"))

# data_temp_kiira_powers_mean = [R_powers_kiira_mean_csv[:,0], R_freqs_kiira_mean_csv[:,0]]
# data_temp_save = pd.DataFrame(data_temp_kiira_powers_mean, index=['pow', 'freq'])
# pd.DataFrame(data_temp_save).to_csv(os.path.join(outdir, "fig_a_kiira_pows_freqs_mean.csv"))

# READ IN ABOVE CSVs
 
pows_freqs = genfromtxt(fname=os.path.join(outdir, "fig_a_pows_freqs.csv"), delimiter=',', skip_header=1)

p_pows_mean = pows_freqs[0,:]
p_pows_mean = np.delete(p_pows_mean, 0)
p_pows_mean = np.vstack(p_pows_mean)
p_pows_mean = np.transpose(p_pows_mean)
p_pows_mean = np.split(p_pows_mean, 40, axis=1)
p_pows_mean = np.vstack(p_pows_mean)
R_powers_means = p_pows_mean

p_freqs_mean = pows_freqs[1,:]
p_freqs_mean= np.delete(p_freqs_mean, 0)
p_freqs_mean = np.vstack(p_freqs_mean)
p_freqs_mean = np.transpose(p_freqs_mean)
p_freqs_mean = np.split(p_freqs_mean, 40, axis=1)
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
plot_scales = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
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
if save_figs:
    plt.savefig(os.path.join(outdir, "fig_a1.png"))

#Kiira
fig, ax = plt.subplots()
plot_scales = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
for l in range(len(R_freqs_kiira)):
    plot_spectrum1d(
        R_freqs_kiira[l],
        R_powers_kiira[l],
        x_units="km",
        y_units="dBZ",
        color="k",
        ax=ax,
        label="",
        wavelength_ticks=plot_scales,
    )
plot_spectrum1d(
    R_freqs_kiira_mean,
    R_powers_kiira_mean,
    x_units="km",
    y_units="dBZ",
    color="b",
    ax=ax,
    label="",
    wavelength_ticks=plot_scales,
)
# plt.legend()
plt.show()
ax.set_title("Kiira")
if save_figs:
    plt.savefig(os.path.join(outdir, "fig_a2.png"))

#Realizations vs. Osapol
fig, ax = plt.subplots()
plot_scales = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
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
    R_freqs_kiira_mean,
    R_powers_kiira_mean,
    x_units="km",
    y_units="dBZ",
    color="b",
    ax=ax,
    label="osapol",
    wavelength_ticks=plot_scales,
)
plt.legend()
# ax.set_ylim(-1, 61)
# ax.set_title("Means")
plt.show()
if save_figs:
    plt.savefig(os.path.join(outdir, "fig_a3.png"))

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

# SAVING DAT-FILES

max_accus_normal = np.where(accus_cum_normal_arr_reshape[:,-1] > np.percentile(accus_cum_normal_arr_reshape[:,-1], 90))
max_accus_normal = max_accus_normal[0]

#TOP 100
tss_normal = []
for i in range(len(file_list_normal)):
    temp_csv = os.path.join(dir_sim, file_list_normal[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    tss_normal.append(temp_mmh)
tss_normal_arr = np.concatenate([tss_normal_[None, :] for tss_normal_ in tss_normal])
tss_normal_arr_reshape = np.reshape(tss_normal_arr, (tss_normal_arr.shape[0]*tss_normal_arr.shape[1], tss_normal_arr.shape[2]))

tss_normal_top100 = []
for i in range(len(max_accus_normal)):
    row = int(max_accus_normal[i])
    temp_tss = tss_normal_arr_reshape[row]
    tss_normal_top100.append(temp_tss)
tss_normal_top100_arr = np.concatenate([tss_normal_top100_[None, :] for tss_normal_top100_ in tss_normal_top100])

#BOTTOM 900
indices_accus_normal = []
for i in range(0,1000):
    if i not in max_accus_normal:
        indices_accus_normal.append(i)
indices_accus_normal = np.asarray(indices_accus_normal)

tss_normal_bottom900 = []
for i in range(len(indices_accus_normal)):
    row = int(indices_accus_normal[i])
    temp_tss = tss_normal_arr_reshape[row]
    tss_normal_bottom900.append(temp_tss)
tss_normal_bottom900_arr = np.concatenate([tss_normal_bottom900_[None, :] for tss_normal_bottom900_ in tss_normal_bottom900])

##############################################################################

max_accus_turned = np.where(accus_cum_turned_arr_reshape[:,-1] > np.percentile(accus_cum_turned_arr_reshape[:,-1], 90))
max_accus_turned = max_accus_turned[0]

tss_turned = []
for i in range(len(file_list_turned)):
    temp_csv = os.path.join(dir_sim, file_list_turned[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    tss_turned.append(temp_mmh)
tss_turned_arr = np.concatenate([tss_turned_[None, :] for tss_turned_ in tss_turned])
tss_turned_arr_reshape = np.reshape(tss_turned_arr, (tss_turned_arr.shape[0]*tss_turned_arr.shape[1], tss_turned_arr.shape[2]))

tss_turned_top100 = []
for i in range(len(max_accus_turned)):
    row = int(max_accus_turned[i])
    temp_tss = tss_turned_arr_reshape[row]
    tss_turned_top100.append(temp_tss)
tss_turned_top100_arr = np.concatenate([tss_turned_top100_[None, :] for tss_turned_top100_ in tss_turned_top100])

##############################################################################

max_accus_slow = np.where(accus_cum_slow_arr_reshape[:,-1] > np.percentile(accus_cum_slow_arr_reshape[:,-1], 90))
max_accus_slow = max_accus_slow[0]

tss_slow = []
for i in range(len(file_list_slow)):
    temp_csv = os.path.join(dir_sim, file_list_slow[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    tss_slow.append(temp_mmh)
tss_slow_arr = np.concatenate([tss_slow_[None, :] for tss_slow_ in tss_slow])
tss_slow_arr_reshape = np.reshape(tss_slow_arr, (tss_slow_arr.shape[0]*tss_slow_arr.shape[1], tss_slow_arr.shape[2]))

tss_slow_top100 = []
for i in range(len(max_accus_slow)):
    row = int(max_accus_slow[i])
    temp_tss = tss_slow_arr_reshape[row]
    tss_slow_top100.append(temp_tss)
tss_slow_top100_arr = np.concatenate([tss_slow_top100_[None, :] for tss_slow_top100_ in tss_slow_top100])

##############################################################################

max_accus_long = np.where(accus_cum_long_arr_reshape[:,-1] > np.percentile(accus_cum_long_arr_reshape[:,-1], 90))
max_accus_long = max_accus_long[0]

tss_long = []
for i in range(len(file_list_long)):
    temp_csv = os.path.join(dir_sim, file_list_long[i], "tss_point_mmh.csv")
    temp_mmh = pd.read_csv(temp_csv, delimiter=(","))
    temp_mmh = temp_mmh.to_numpy()
    temp_mmh = temp_mmh[:, 1:]
    temp_mmh = temp_mmh.astype(float)
    temp_mm = temp_mmh * (5/60)
    tss_long.append(temp_mmh)
tss_long_arr = np.concatenate([tss_long_[None, :] for tss_long_ in tss_long])
tss_long_arr_reshape = np.reshape(tss_long_arr, (tss_long_arr.shape[0]*tss_long_arr.shape[1], tss_long_arr.shape[2]))

tss_long_top100 = []
for i in range(len(max_accus_long)):
    row = int(max_accus_long[i])
    temp_tss = tss_long_arr_reshape[row]
    tss_long_top100.append(temp_tss)
tss_long_top100_arr = np.concatenate([tss_long_top100_[None, :] for tss_long_top100_ in tss_long_top100])

##############################################################################

event_name = np.repeat("KIIRA", tss_normal_top100_arr.shape[1])
year = np.repeat("2017", tss_normal_top100_arr.shape[1])
month = np.repeat("08", tss_normal_top100_arr.shape[1])
day = np.repeat("12", tss_normal_top100_arr.shape[1])

event_name_long = np.repeat("KIIRA", tss_long_top100_arr.shape[1])
year_long = np.repeat("2017", tss_long_top100_arr.shape[1])
month_long = np.repeat("08", tss_long_top100_arr.shape[1])
day_long = np.repeat("12", tss_long_top100_arr.shape[1])

hour = np.zeros(tss_normal_top100_arr.shape[1])
hour[0:3] = 14
hour[3:15] = 15
hour[15:27] = 16
hour[27:39] = 17
hour[39:51] = 18
hour[51:63] = 19
hour[63:75] = 20
hour[75:87] = 21
hour = hour.astype(int)

hour_long = np.zeros(tss_long_top100_arr.shape[1])
hour_long[162:174] = 21
hour_long[150:162] = 20
hour_long[138:150] = 19
hour_long[126:138] = 18
hour_long[114:126] = 17
hour_long[102:114] = 16
hour_long[90:102] = 15
hour_long[78:90] = 14
hour_long[66:78] = 13
hour_long[54:66] = 12
hour_long[42:54] = 11
hour_long[30:42] = 10
hour_long[18:30] = 9
hour_long[6:18] = 8
hour_long[0:6] = 7
hour_long = hour_long.astype(int)

minutes = np.arange(0, 60, 5)
minute = np.zeros(tss_normal_top100_arr.shape[1])
minute[0] = 45
minute[1] = 50
minute[2] = 55
for i in range(0,12):
    minute[i+3] = minutes[i]
for i in range(0,12):
    minute[i+15] = minutes[i]
for i in range(0,12):
    minute[i+27] = minutes[i]
for i in range(0,12):
    minute[i+39] = minutes[i]
for i in range(0,12):
    minute[i+51] = minutes[i]
for i in range(0,12):
    minute[i+63] = minutes[i]
for i in range(0,12):
    minute[i+75] = minutes[i]
minute = minute.astype(int)

minute_long = np.zeros(tss_long_top100_arr.shape[1])
minute_long[0] = 30
minute_long[1] = 35
minute_long[2] = 40
minute_long[3] = 45
minute_long[4] = 50
minute_long[5] = 55
for i in range(0,12):
    minute_long[i+6] = minutes[i]
for i in range(0,12):
    minute_long[i+18] = minutes[i]
for i in range(0,12):
    minute_long[i+30] = minutes[i]
for i in range(0,12):
    minute_long[i+42] = minutes[i]
for i in range(0,12):
    minute_long[i+54] = minutes[i]
for i in range(0,12):
    minute_long[i+66] = minutes[i]
for i in range(0,12):
    minute_long[i+78] = minutes[i]
for i in range(0,12):
    minute_long[i+90] = minutes[i]
for i in range(0,12):
    minute_long[i+102] = minutes[i]
for i in range(0,12):
    minute_long[i+114] = minutes[i]
for i in range(0,12):
    minute_long[i+126] = minutes[i]
for i in range(0,12):
    minute_long[i+138] = minutes[i]
for i in range(0,12):
    minute_long[i+150] = minutes[i]
for i in range(0,12):
    minute_long[i+162] = minutes[i]
minute_long = minute_long.astype(int)

##############################################################################

df = pd.DataFrame()
df["event"] = event_name
df["year"] = year
df["month"] = month
df["day"] = day
df["hour"] = hour
df["hour"] = df["hour"].astype(str)
df["minute"] = minute
df["minute"] = df["minute"].astype(str)
df["minute"] = df["minute"].replace("0", "00")
df["minute"] = df["minute"].replace("5", "05")

##############################################################################

df_long = pd.DataFrame()
df_long["event"] = event_name_long
df_long["year"] = year_long
df_long["month"] = month_long
df_long["day"] = day_long
df_long["hour"] = hour_long
df_long["hour"] = df_long["hour"].astype(str)
df_long["hour"] = df_long["hour"].replace("7", "07")
df_long["hour"] = df_long["hour"].replace("8", "08")
df_long["hour"] = df_long["hour"].replace("9", "09")
df_long["minute"] = minute_long
df_long["minute"] = df_long["minute"].astype(str)
df_long["minute"] = df_long["minute"].replace("0", "00")
df_long["minute"] = df_long["minute"].replace("5", "05")

##############################################################################

out_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
out_dir = os.path.join(out_dir, "SWMM_input_normal")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
for i in range(len(tss_normal_top100_arr)):
    df["rain"] = tss_normal_top100_arr[i]
    df.to_csv(os.path.join(out_dir, f"input_rain_normal_{i}.dat"), sep = "\t", header=False, index=False)

##############################################################################

out_dir_900 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
out_dir_900 = os.path.join(out_dir, "SWMM_input_normal_900")
if not os.path.exists(out_dir_900):
    os.makedirs(out_dir_900)
    
for i in range(len(tss_normal_bottom900_arr)):
    df["rain"] = tss_normal_bottom900_arr[i]
    df.to_csv(os.path.join(out_dir_900, f"input_rain_normal_{i}.dat"), sep = "\t", header=False, index=False)

##############################################################################

out_dir2 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
out_dir2 = os.path.join(out_dir2, "SWMM_input_turned")
if not os.path.exists(out_dir2):
    os.makedirs(out_dir2)
    
for i in range(len(tss_turned_top100_arr)):
    df["rain"] = tss_turned_top100_arr[i]
    df.to_csv(os.path.join(out_dir2, f"input_rain_turned_{i}.dat"), sep = "\t", header=False, index=False)

##############################################################################

out_dir3 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
out_dir3 = os.path.join(out_dir3, "SWMM_input_slow")
if not os.path.exists(out_dir3):
    os.makedirs(out_dir3)
    
for i in range(len(tss_slow_top100_arr)):
    df["rain"] = tss_slow_top100_arr[i]
    df.to_csv(os.path.join(out_dir3, f"input_rain_slow_{i}.dat"), sep = "\t", header=False, index=False)

##############################################################################

out_dir4 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
out_dir4 = os.path.join(out_dir4, "SWMM_input_long")
if not os.path.exists(out_dir4):
    os.makedirs(out_dir4)
    
for i in range(len(tss_long_top100_arr)):
    df_long["rain"] = tss_long_top100_arr[i]
    df_long.to_csv(os.path.join(out_dir4, f"input_rain_long_{i}.dat"), sep = "\t", header=False, index=False)
