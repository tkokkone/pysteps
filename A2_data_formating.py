# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:36:56 2023

@author: lindgrv1
"""

# Rainfall data for SWMM has to be tab delimitated .DAT file. 
# Columns: event name, year, month, day, hour, minute, mm/h value

import numpy as np
import pandas as pd
import os
from numpy import genfromtxt

##############################################################################

in_sim_10 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_54248_17249_37999_22960"
event = genfromtxt(fname=os.path.join(in_sim_10, "tss_point_mmh.csv"), delimiter=',')
event = event[1:,1:]

event_name = np.repeat("KIIRA", event.shape[1])
year = np.repeat("2017", event.shape[1])
month = np.repeat("09", event.shape[1])
day = np.repeat("12", event.shape[1])

hour = np.zeros(event.shape[1])
hour[0:3] = 14
hour[3:15] = 15
hour[15:27] = 16
hour[27:39] = 17
hour[39:51] = 18
hour[51:63] = 19
hour[63:75] = 20
hour[75:87] = 21
hour = hour.astype(int)

minutes = np.arange(0, 60, 5)
minute = np.zeros(event.shape[1])
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

out_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
out_dir = os.path.join(out_dir, "Rain_for_SWMM")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
for i in range(len(event)):
    df["rain"] = event[i]
    df.to_csv(os.path.join(out_dir, f"input_rain_{i}.dat"), sep = "\t", header=False, index=False)





