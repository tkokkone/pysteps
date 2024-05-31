# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:09:14 2024

@author: lindgrv1
"""

import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import rasterio
import pandas as pd
import seaborn as sns
from shapely.geometry import box
from shapely.geometry import Point
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar

##############################################################################

#Function to find closest value in array to a given value

def find_closest(arr, val):
       idx = np.abs(arr - val).argmin()
       return arr[idx]

##############################################################################

# Urban Water Journal - Taylor and Francis
# https://www.tandfonline.com/journals/nurw20

# DIRECTORIES

dir_normal = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\SWMM_output_normal"
dir_slow = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\SWMM_output_slow"
dir_turned = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\SWMM_output_turned"
dir_long = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\SWMM_output_long"

dir_normal_900 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\SWMM_output_normal_900"

dir_obs_radar = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\SWMM_output_obs_radar"
dir_obs_gauge = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\SWMM_output_obs_gauge"

out_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Analyses_SWMM"
save_figs = False
save_csvs = False

##############################################################################

#Nodes missing: 384 and 387

#Normal simulations: 12.9.2017 14:46 - 12.9.2017 23:59
normal_flow = genfromtxt(fname=os.path.join(dir_normal, "SWMM_output_normal_flow.csv"), delimiter=',', skip_header=1)
normal_critical = genfromtxt(fname=os.path.join(dir_normal, "SWMM_output_normal_nodes_CRITICAL.csv"), delimiter=',', skip_header=1)
normal_flooding = genfromtxt(fname=os.path.join(dir_normal, "SWMM_output_normal_nodes_FLOODING.csv"), delimiter=',', skip_header=1)

#Slow siulations: 12.9.2017 14:46 - 12.9.2017 23:59
slow_flow = genfromtxt(fname=os.path.join(dir_slow, "SWMM_output_slow_flow.csv"), delimiter=',', skip_header=1)
slow_critical = genfromtxt(fname=os.path.join(dir_slow, "SWMM_output_slow_nodes_CRITICAL.csv"), delimiter=',', skip_header=1)
slow_flooding = genfromtxt(fname=os.path.join(dir_slow, "SWMM_output_slow_nodes_FLOODING.csv"), delimiter=',', skip_header=1)

#Turned simulations: 12.9.2017 14:46 - 12.9.2017 23:59
turned_flow = genfromtxt(fname=os.path.join(dir_turned, "SWMM_output_turned_flow.csv"), delimiter=',', skip_header=1)
turned_critical = genfromtxt(fname=os.path.join(dir_turned, "SWMM_output_turned_nodes_CRITICAL.csv"), delimiter=',', skip_header=1)
turned_flooding = genfromtxt(fname=os.path.join(dir_turned, "SWMM_output_turned_nodes_FLOODING.csv"), delimiter=',', skip_header=1)

#Long simulations: 9.12.2017 7:30 - 9.12.2017 23:59
long_flow = genfromtxt(fname=os.path.join(dir_long, "SWMM_output_long_flow.csv"), delimiter=',', skip_header=1)
long_critical = genfromtxt(fname=os.path.join(dir_long, "SWMM_output_long_nodes_CRITICAL.csv"), delimiter=',', skip_header=1)
long_flooding = genfromtxt(fname=os.path.join(dir_long, "SWMM_output_long_nodes_FLOODING.csv"), delimiter=',', skip_header=1)

# Keep only columns containing 100 simulations
normal_flow = normal_flow[:,1:]
normal_critical = normal_critical[:,1:]
normal_flooding = normal_flooding[:,1:]
slow_flow = slow_flow[:,1:]
slow_critical = slow_critical[:,1:]
slow_flooding = slow_flooding[:,1:]
turned_flow = turned_flow[:,1:]
turned_critical = turned_critical[:,1:]
turned_flooding = turned_flooding[:,1:]
long_flow = long_flow[:,1:]
long_critical = long_critical[:,1:]
long_flooding = long_flooding[:,1:]

#Normal simulations - lowes 900: 12.9.2017 14:46 - 12.9.2017 23:59
normal_900_flow = genfromtxt(fname=os.path.join(dir_normal_900, "SWMM_output_normal_900_flow_final.csv"), delimiter=',', skip_header=1)
normal_900_critical = genfromtxt(fname=os.path.join(dir_normal_900, "SWMM_output_normal_900_nodes_CRITICAL.csv"), delimiter=',', skip_header=1)
normal_900_flooding = genfromtxt(fname=os.path.join(dir_normal_900, "SWMM_output_normal_900_nodes_FLOODING.csv"), delimiter=',', skip_header=1)
# Keep only columns containing 100 simulations
normal_900_flow = normal_900_flow[:,1:]
normal_900_critical = normal_900_critical[:,1:]
normal_900_flooding = normal_900_flooding[:,1:]

#Radar observations
obs_radar_flow = genfromtxt(fname=os.path.join(dir_obs_radar, "SWMM_output_obs_flow.csv"), delimiter=',', skip_header=1)
obs_radar_flow = obs_radar_flow[2:556,1:]
obs_radar_critical = genfromtxt(fname=os.path.join(dir_obs_radar, "SWMM_output_obs_nodes_CRITICAL.csv"), delimiter=',', skip_header=1)
obs_radar_critical = obs_radar_critical[:,1:]
obs_radar_flooding = genfromtxt(fname=os.path.join(dir_obs_radar, "SWMM_output_obs_nodes_FLOODING.csv"), delimiter=',', skip_header=1)
obs_radar_flooding = obs_radar_flooding[:,1:]

#Gauge observations
obs_gauge_flow = genfromtxt(fname=os.path.join(dir_obs_gauge, "SWMM_output_measured_flow.csv"), delimiter=',', skip_header=1)
obs_gauge_flow = obs_gauge_flow[2:556,1:]
obs_gauge_critical = genfromtxt(fname=os.path.join(dir_obs_gauge, "SWMM_output_measured_nodes_CRITICAL.csv"), delimiter=',', skip_header=1)
obs_gauge_critical = obs_gauge_critical[:,1:]
obs_gauge_flooding = genfromtxt(fname=os.path.join(dir_obs_gauge, "SWMM_output_measured_nodes_FLOODING.csv"), delimiter=',', skip_header=1)
obs_gauge_flooding = obs_gauge_flooding[:,1:]

##############################################################################

# Number of nodes with water level over the critical level (0.5 meters from ground level) and number of nodes flooding

#Normal
normal_critical_sum = np.zeros((1, normal_critical.shape[1]))
for i in range(normal_critical_sum.shape[1]):
    normal_critical_sum[0,i] = int(sum(normal_critical[:,i]))
normal_flooding_sum = np.zeros((1, normal_flooding.shape[1]))
for i in range(normal_flooding_sum.shape[1]):
    normal_flooding_sum[0,i] = int(sum(normal_flooding[:,i]))
#Simple line plot    
plt.figure()
plt.plot(normal_critical_sum[0], marker='o', label='Critical')
plt.plot(normal_flooding_sum[0], marker='o', label='Flooding')
plt.legend()
plt.title('Normal: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "01_nodes_normal_lines.png"))

#Slow
slow_critical_sum = np.zeros((1, slow_critical.shape[1]))
for i in range(slow_critical_sum.shape[1]):
    slow_critical_sum[0,i] = int(sum(slow_critical[:,i]))
slow_flooding_sum = np.zeros((1, slow_flooding.shape[1]))
for i in range(slow_flooding_sum.shape[1]):
    slow_flooding_sum[0,i] = int(sum(slow_flooding[:,i]))
#Simple line plot    
plt.figure()
plt.plot(slow_critical_sum[0], marker='o', label='Critical')
plt.plot(slow_flooding_sum[0], marker='o', label='Flooding')
plt.legend()
plt.title('Slow: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "02_nodes_slow_lines.png"))

#Turned
turned_critical_sum = np.zeros((1, turned_critical.shape[1]))
for i in range(turned_critical_sum.shape[1]):
    turned_critical_sum[0,i] = int(sum(turned_critical[:,i]))
turned_flooding_sum = np.zeros((1, turned_flooding.shape[1]))
for i in range(turned_flooding_sum.shape[1]):
    turned_flooding_sum[0,i] = int(sum(turned_flooding[:,i]))
#Simple line plot    
plt.figure()
plt.plot(turned_critical_sum[0], marker='o', label='Critical')
plt.plot(turned_flooding_sum[0], marker='o', label='Flooding')
plt.legend()
plt.title('Turned: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "03_nodes_turned_lines.png"))

#Long
long_critical_sum = np.zeros((1, long_critical.shape[1]))
for i in range(long_critical_sum.shape[1]):
    long_critical_sum[0,i] = int(sum(long_critical[:,i]))
long_flooding_sum = np.zeros((1, long_flooding.shape[1]))
for i in range(long_flooding_sum.shape[1]):
    long_flooding_sum[0,i] = int(sum(long_flooding[:,i]))
#Simple line plot    
plt.figure()
plt.plot(long_critical_sum[0], marker='o', label='Critical')
plt.plot(long_flooding_sum[0], marker='o', label='Flooding')
plt.legend()
plt.title('Long: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "04_nodes_long_lines.png"))

###################################

#Number of flooding and critical nodes
normal_900_critical_sum = np.zeros((1, normal_900_critical.shape[1]))
for i in range(normal_900_critical_sum.shape[1]):
    normal_900_critical_sum[0,i] = int(sum(normal_900_critical[:,i]))
normal_900_flooding_sum = np.zeros((1, normal_900_flooding.shape[1]))
for i in range(normal_900_flooding_sum.shape[1]):
    normal_900_flooding_sum[0,i] = int(sum(normal_900_flooding[:,i]))
#Simple line plot    
plt.figure()
plt.plot(normal_900_critical_sum[0], marker='o', label='Critical')
plt.plot(normal_900_flooding_sum[0], marker='o', label='Flooding')
plt.legend()
plt.title('Normal - lowest 900: Nodes over critical level / flooding')

#Sorting from low to high
normal_900_critical_sum_ordered = normal_900_critical_sum[0,:].copy()
normal_900_critical_sum_ordered = np.sort(normal_900_critical_sum_ordered)
normal_900_flooding_sum_ordered = normal_900_flooding_sum[0,:].copy()
normal_900_flooding_sum_ordered = np.sort(normal_900_flooding_sum_ordered)

#Sorted line plots - Critical
plt.figure()
plt.plot(normal_900_critical_sum_ordered, label='Normal - lowest 900')
plt.ylim(0, 473)
plt.legend()
plt.title('Nodes over critical level')
#Sorted line plots - Flooding
plt.figure()
plt.plot(normal_900_flooding_sum_ordered, label='Normal - lowest 900')
plt.ylim(0, 473)
plt.legend()
plt.title('Nodes flooding')

#Combine and re-order flooding+critical
normal_900_flooding_critical = np.zeros((2, normal_900_flooding_sum.shape[1]))
normal_900_flooding_critical[0,:] = normal_900_flooding_sum[0,:].copy()
normal_900_flooding_critical[1,:] = normal_900_critical_sum[0,:].copy()
indices_normal_900 = np.argsort(normal_900_flooding_critical[0,:])
normal_900_flooding_critical = normal_900_flooding_critical[:,indices_normal_900]
#Simple line plot  
plt.figure()
plt.plot(normal_900_flooding_critical[1], label='Critical', marker='o')
plt.plot(normal_900_flooding_critical[0], label='Flooding', marker='o')
plt.legend()
plt.ylim(0, 473)
plt.title('Normal - lowest 900: Nodes over critical level / flooding')

##############################################################################

# normal_flooding_potential = np.zeros((normal_flooding_sum.shape[0], normal_flooding_sum.shape[1]))
# for i in range(normal_flooding_potential.shape[1]):
#     normal_flooding_potential[0,i] = normal_flooding_sum[0,i] / normal_critical_sum[0,i] *100
# np.min(normal_flooding_potential)
# np.max(normal_flooding_potential)

##############################################################################

#Sorting from low to high

#Normal
normal_critical_sum_ordered = normal_critical_sum[0,:].copy()
normal_critical_sum_ordered = np.sort(normal_critical_sum_ordered)
normal_flooding_sum_ordered = normal_flooding_sum[0,:].copy()
normal_flooding_sum_ordered = np.sort(normal_flooding_sum_ordered)

#Slow
slow_critical_sum_ordered = slow_critical_sum[0,:].copy()
slow_critical_sum_ordered = np.sort(slow_critical_sum_ordered)
slow_flooding_sum_ordered = slow_flooding_sum[0,:].copy()
slow_flooding_sum_ordered = np.sort(slow_flooding_sum_ordered)

#Turned
turned_critical_sum_ordered = turned_critical_sum[0,:].copy()
turned_critical_sum_ordered = np.sort(turned_critical_sum_ordered)
turned_flooding_sum_ordered = turned_flooding_sum[0,:].copy()
turned_flooding_sum_ordered = np.sort(turned_flooding_sum_ordered)

#Long
long_critical_sum_ordered = long_critical_sum[0,:].copy()
long_critical_sum_ordered = np.sort(long_critical_sum_ordered)
long_flooding_sum_ordered = long_flooding_sum[0,:].copy()
long_flooding_sum_ordered = np.sort(long_flooding_sum_ordered)

#Sorted line plots
plt.figure()
plt.plot(normal_critical_sum_ordered, label='Normal')
plt.plot(turned_critical_sum_ordered, label='Turned')
plt.plot(slow_critical_sum_ordered, label='Slow')
plt.plot(long_critical_sum_ordered, label='Long')
plt.ylim(0, 473)
plt.legend()
plt.title('Nodes over critical level')
if save_figs:
    plt.savefig(os.path.join(out_dir, "05_nodes_critical_lines.png"))

plt.figure()
plt.plot(normal_flooding_sum_ordered, label='Normal')
plt.plot(turned_flooding_sum_ordered, label='Turned')
plt.plot(slow_flooding_sum_ordered, label='Slow')
plt.plot(long_flooding_sum_ordered, label='Long')
plt.ylim(0, 473)
plt.legend()
plt.title('Nodes flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "06_nodes_flooding_lines.png"))

##############################################################################

# Combine and re-order flooding+critical

#Normal
normal_flooding_critical = np.zeros((2, normal_flooding_sum.shape[1]))
normal_flooding_critical[0,:] = normal_flooding_sum[0,:].copy()
normal_flooding_critical[1,:] = normal_critical_sum[0,:].copy()
indices_normal = np.argsort(normal_flooding_critical[0,:])
normal_flooding_critical = normal_flooding_critical[:,indices_normal]
#Simple line plot  
plt.figure()
plt.plot(normal_flooding_critical[1], label='Critical', marker='o')
plt.plot(normal_flooding_critical[0], label='Flooding', marker='o')
plt.legend()
plt.ylim(0, 473)
plt.title('Normal: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "07_nodes_normal_lines_sorted.png"))
#Simple bar plot
plt.figure()
plt.bar(x=np.arange(100), height=normal_flooding_critical[1], label='Critical')
plt.bar(x=np.arange(100), height=normal_flooding_critical[0], label='Flooding')
plt.legend()
plt.ylim(0, 473)
plt.title('Normal: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "08_nodes_normal_bars_sorted.png"))

#Slow
slow_flooding_critical = np.zeros((2, slow_flooding_sum.shape[1]))
slow_flooding_critical[0,:] = slow_flooding_sum[0,:].copy()
slow_flooding_critical[1,:] = slow_critical_sum[0,:].copy()
indices_slow = np.argsort(slow_flooding_critical[0,:])
slow_flooding_critical = slow_flooding_critical[:,indices_slow]
#Simple line plot  
plt.figure()
plt.plot(slow_flooding_critical[1], label='Critical', marker='o')
plt.plot(slow_flooding_critical[0], label='Flooding', marker='o')
plt.legend()
plt.ylim(0, 473)
plt.title('Slow: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "09_nodes_slow_lines_sorted.png"))
#Simple bar plot
plt.figure()
plt.bar(x=np.arange(100), height=slow_flooding_critical[1], label='Critical')
plt.bar(x=np.arange(100), height=slow_flooding_critical[0], label='Flooding')
plt.legend()
plt.ylim(0, 473)
plt.title('Slow: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "10_nodes_slow_bars_sorted.png"))

#Turned
turned_flooding_critical = np.zeros((2, turned_flooding_sum.shape[1]))
turned_flooding_critical[0,:] = turned_flooding_sum[0,:].copy()
turned_flooding_critical[1,:] = turned_critical_sum[0,:].copy()
indices_turned = np.argsort(turned_flooding_critical[0,:])
turned_flooding_critical = turned_flooding_critical[:,indices_turned]
#Simple line plot  
plt.figure()
plt.plot(turned_flooding_critical[1], label='Critical', marker='o')
plt.plot(turned_flooding_critical[0], label='Flooding', marker='o')
plt.legend()
plt.ylim(0, 473)
plt.title('Turned: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "11_nodes_turned_lines_sorted.png"))
#Simple bar plot
plt.figure()
plt.bar(x=np.arange(100), height=turned_flooding_critical[1], label='Critical')
plt.bar(x=np.arange(100), height=turned_flooding_critical[0], label='Flooding')
plt.legend()
plt.ylim(0, 473)
plt.title('Turned: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "12_nodes_turned_bars_sorted.png"))

#Long
long_flooding_critical = np.zeros((2, long_flooding_sum.shape[1]))
long_flooding_critical[0,:] = long_flooding_sum[0,:].copy()
long_flooding_critical[1,:] = long_critical_sum[0,:].copy()
indices_long = np.argsort(long_flooding_critical[0,:])
long_flooding_critical = long_flooding_critical[:,indices_long]
#Simple line plot  
plt.figure()
plt.plot(long_flooding_critical[1], label='Critical', marker='o')
plt.plot(long_flooding_critical[0], label='Flooding', marker='o')
plt.legend()
plt.ylim(0, 473)
plt.title('Long: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "13_nodes_long_lines_sorted.png"))
#Simple bar plot
plt.figure()
plt.bar(x=np.arange(100), height=long_flooding_critical[1], label='Critical')
plt.bar(x=np.arange(100), height=long_flooding_critical[0], label='Flooding')
plt.legend()
plt.ylim(0, 473)
plt.title('Long: Nodes over critical level / flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "14_nodes_long_bars_sorted.png"))

#Subplots: lines
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10), constrained_layout = True)
ax1.plot(normal_flooding_critical[1], label='Critical', marker='o')
ax1.plot(normal_flooding_critical[0], label='Flooding', marker='o')
ax1.set_ylim(0, 473)
ax1.title.set_text("Normal")
ax2.plot(slow_flooding_critical[1], label='Critical', marker='o')
ax2.plot(slow_flooding_critical[0], label='Flooding', marker='o')
ax2.set_ylim(0, 473)
ax2.title.set_text("Slow")
ax3.plot(turned_flooding_critical[1], label='Critical', marker='o')
ax3.plot(turned_flooding_critical[0], label='Flooding', marker='o')
ax3.set_ylim(0, 473)
ax3.title.set_text("Turned")
ax4.plot(long_flooding_critical[1], label='Critical', marker='o')
ax4.plot(long_flooding_critical[0], label='Flooding', marker='o')
ax4.set_ylim(0, 473)
ax4.title.set_text("Long")
fig.suptitle("Nodes over critical level / flooding", fontsize=16)
if save_figs:
    plt.savefig(os.path.join(out_dir, "15_nodes_lines_sorted.png"))

#Subplots: bars
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10), constrained_layout = True)
ax1.bar(x=np.arange(100), height=normal_flooding_critical[1], label='Critical')
ax1.bar(x=np.arange(100), height=normal_flooding_critical[0], label='Flooding')
ax1.set_ylim(0, 473)
ax1.title.set_text("Normal")
ax2.bar(x=np.arange(100), height=slow_flooding_critical[1], label='Critical')
ax2.bar(x=np.arange(100), height=slow_flooding_critical[0], label='Flooding')
ax2.set_ylim(0, 473)
ax2.title.set_text("Slow")
ax3.bar(x=np.arange(100), height=turned_flooding_critical[1], label='Critical')
ax3.bar(x=np.arange(100), height=turned_flooding_critical[0], label='Flooding')
ax3.set_ylim(0, 473)
ax3.title.set_text("Turned")
ax4.bar(x=np.arange(100), height=long_flooding_critical[1], label='Critical')
ax4.bar(x=np.arange(100), height=long_flooding_critical[0], label='Flooding')
ax4.set_ylim(0, 473)
ax4.title.set_text("Long")
fig.suptitle("Nodes over critical level / flooding", fontsize=16)
if save_figs:
    plt.savefig(os.path.join(out_dir, "16_nodes_bars_sorted.png"))

##############################################################################

# Peak runoff [l/s]

#Normal
normal_peak_flow = np.zeros((2, normal_flow.shape[1]))
for i in range(normal_peak_flow.shape[1]):
    normal_peak_flow[0,i] = np.max(normal_flow[:,i])
    temp_row = np.where(normal_flow[:,i] == np.max(normal_flow[:,i]))
    normal_peak_flow[1,i] = int(temp_row[0])

#Slow
slow_peak_flow = np.zeros((2, slow_flow.shape[1]))
for i in range(slow_peak_flow.shape[1]):
    slow_peak_flow[0,i] = np.max(slow_flow[:,i])
    temp_row = np.where(slow_flow[:,i] == np.max(slow_flow[:,i]))
    slow_peak_flow[1,i] = int(temp_row[0])

#Turned
turned_peak_flow = np.zeros((2, turned_flow.shape[1]))
for i in range(turned_peak_flow.shape[1]):
    turned_peak_flow[0,i] = np.max(turned_flow[:,i])
    temp_row = np.where(turned_flow[:,i] == np.max(turned_flow[:,i]))
    turned_peak_flow[1,i] = int(temp_row[0])

#Long
long_peak_flow = np.zeros((2, long_flow.shape[1]))
for i in range(long_peak_flow.shape[1]):
    long_peak_flow[0,i] = np.max(long_flow[:,i])
    temp_row = np.where(long_flow[:,i] == np.max(long_flow[:,i]))
    long_peak_flow[1,i] = int(temp_row[0])

#Simple line plots
plt.figure()
plt.plot(normal_peak_flow[0], label='Normal')
plt.plot(turned_peak_flow[0], label='Turned')
plt.plot(slow_peak_flow[0], label='Slow')
plt.plot(long_peak_flow[0], label='Long')
plt.legend()
plt.title('Peak runoff [l/s]')
if save_figs:
    plt.savefig(os.path.join(out_dir, "17_peak_runoff.png"))

#Sorting from low to high
normal_peak_flow_ordered = normal_peak_flow[0,:].copy()
normal_peak_flow_ordered = np.sort(normal_peak_flow_ordered)
slow_peak_flow_ordered = slow_peak_flow[0,:].copy()
slow_peak_flow_ordered = np.sort(slow_peak_flow_ordered)
turned_peak_flow_ordered = turned_peak_flow[0,:].copy()
turned_peak_flow_ordered = np.sort(turned_peak_flow_ordered)
long_peak_flow_ordered = long_peak_flow[0,:].copy()
long_peak_flow_ordered = np.sort(long_peak_flow_ordered)

#Sorted line plots
plt.figure()
plt.plot(normal_peak_flow_ordered, label='Normal')
plt.plot(turned_peak_flow_ordered, label='Turned')
plt.plot(slow_peak_flow_ordered, label='Slow')
plt.plot(long_peak_flow_ordered, label='Long')
plt.legend()
plt.title('Peak runoff [l/s]')
if save_figs:
    plt.savefig(os.path.join(out_dir, "18_peak_runoff_sorted.png"))
    
#Plot histograms
plt.figure()
sns.histplot(data=[normal_peak_flow_ordered, turned_peak_flow_ordered, slow_peak_flow_ordered, long_peak_flow_ordered], 
             bins=20, kde=True, stat="percent")
if save_figs:
    plt.savefig(os.path.join(out_dir, "29_hist_bars-lines_peakflow.png"))

plt.figure()
sns.kdeplot(data=[normal_peak_flow_ordered, turned_peak_flow_ordered, slow_peak_flow_ordered, long_peak_flow_ordered], cut=0)
if save_figs:
    plt.savefig(os.path.join(out_dir, "30_hist_lines_peakflow.png"))
    
##############################################################################

obs_radar_peak_flow = np.zeros((2, obs_radar_flow.shape[1]))
for i in range(obs_radar_peak_flow.shape[1]):
    obs_radar_peak_flow[0,i] = np.max(obs_radar_flow[:,i])
    temp_row = np.where(obs_radar_flow[:,i] == np.max(obs_radar_flow[:,i]))
    obs_radar_peak_flow[1,i] = int(temp_row[0])
    
obs_radar_peak_flow_ordered = obs_radar_peak_flow[0,:].copy()
obs_radar_peak_flow_ordered = np.sort(obs_radar_peak_flow_ordered)

plt.figure()
plt.plot(normal_peak_flow_ordered, label='Normal')
plt.plot(turned_peak_flow_ordered, label='Turned')
plt.plot(slow_peak_flow_ordered, label='Slow')
plt.plot(long_peak_flow_ordered, label='Long')
plt.plot(obs_radar_peak_flow_ordered, label='Radar')
plt.legend()
plt.title('Peak runoff [l/s]')

##############################################################################

# Runoff accumulations [l/s]

#Normal
normal_cumulative_flow = np.zeros((normal_flow.shape[0], normal_flow.shape[1]))
normal_cumulative_flow[0] = normal_flow[0]
for i in range(1,len(normal_flow)):
    normal_cumulative_flow[i] = normal_flow[i] + normal_cumulative_flow[i-1]
#Means
normal_cumulative_flow_mean = np.zeros((normal_cumulative_flow.shape[0], 1))
for i in range(len(normal_cumulative_flow_mean)):
    normal_cumulative_flow_mean[i] = np.mean(normal_cumulative_flow[i,:])

#Slow
slow_cumulative_flow = np.zeros((slow_flow.shape[0], slow_flow.shape[1]))
slow_cumulative_flow[0] = slow_flow[0]
for i in range(1,len(slow_flow)):
    slow_cumulative_flow[i] = slow_flow[i] + slow_cumulative_flow[i-1]
#Means
slow_cumulative_flow_mean = np.zeros((slow_cumulative_flow.shape[0], 1))
for i in range(len(slow_cumulative_flow_mean)):
    slow_cumulative_flow_mean[i] = np.mean(slow_cumulative_flow[i,:])

#Turned
turned_cumulative_flow = np.zeros((turned_flow.shape[0], turned_flow.shape[1]))
turned_cumulative_flow[0] = turned_flow[0]
for i in range(1,len(turned_flow)):
    turned_cumulative_flow[i] = turned_flow[i] + turned_cumulative_flow[i-1]
#Means
turned_cumulative_flow_mean = np.zeros((turned_cumulative_flow.shape[0], 1))
for i in range(len(turned_cumulative_flow_mean)):
    turned_cumulative_flow_mean[i] = np.mean(turned_cumulative_flow[i,:])

#Long
long_cumulative_flow = np.zeros((long_flow.shape[0], long_flow.shape[1]))
long_cumulative_flow[0] = long_flow[0]
for i in range(1,len(long_flow)):
    long_cumulative_flow[i] = long_flow[i] + long_cumulative_flow[i-1]
#Means
long_cumulative_flow_mean = np.zeros((long_cumulative_flow.shape[0], 1))
for i in range(len(long_cumulative_flow_mean)):
    long_cumulative_flow_mean[i] = np.mean(long_cumulative_flow[i,:])  

###############

#Radar
obs_radar_cumulative_flow = np.zeros((obs_radar_flow.shape[0], obs_radar_flow.shape[1]))
obs_radar_cumulative_flow[0] = obs_radar_flow[0]
for i in range(1,len(obs_radar_flow)):
    obs_radar_cumulative_flow[i] = obs_radar_flow[i] + obs_radar_cumulative_flow[i-1]
#Means
obs_radar_cumulative_flow_mean = np.zeros((obs_radar_cumulative_flow.shape[0], 1))
for i in range(len(obs_radar_cumulative_flow_mean)):
    obs_radar_cumulative_flow_mean[i] = np.mean(obs_radar_cumulative_flow[i,:])
    
#Plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), constrained_layout = True)
for j in range(obs_radar_cumulative_flow.shape[1]):
    ax.plot(obs_radar_cumulative_flow[:,j], color='blue', alpha=0.2)
ax.plot(obs_radar_cumulative_flow_mean, color='black')
ax.set_ylim(0, 180000)
ax.set_xlim(0, 1000)
ax.title.set_text("Radar")

###############

#Normal - lowest 900
normal_900_cumulative_flow = np.zeros((normal_900_flow.shape[0], normal_900_flow.shape[1]))
normal_900_cumulative_flow[0] = normal_900_flow[0]
for i in range(1,len(normal_900_flow)):
    normal_900_cumulative_flow[i] = normal_900_flow[i] + normal_900_cumulative_flow[i-1]
#Means
normal_900_cumulative_flow_mean = np.zeros((normal_900_cumulative_flow.shape[0], 1))
for i in range(len(normal_900_cumulative_flow_mean)):
    normal_900_cumulative_flow_mean[i] = np.mean(normal_900_cumulative_flow[i,:])

#All 1000 normal simulations
normal_all_cumulative_flow_900 = normal_900_cumulative_flow.copy()
normal_all_cumulative_flow_100 = normal_cumulative_flow.copy()
normal_all_cumulative_flow = np.concatenate((normal_all_cumulative_flow_900, normal_all_cumulative_flow_100), axis=1)
#mean
normal_all_cumulative_flow_mean = np.zeros((normal_all_cumulative_flow.shape[0], 1))
for i in range(len(normal_all_cumulative_flow_mean)):
    normal_all_cumulative_flow_mean[i] = np.mean(normal_all_cumulative_flow[i,:])

#Plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), constrained_layout = True)
for j in range(normal_900_cumulative_flow.shape[1]):
    ax.plot(normal_900_cumulative_flow[:,j], color='blue', alpha=0.2)
ax.plot(normal_900_cumulative_flow_mean, color='black')
ax.set_ylim(0, 180000)
ax.set_xlim(0, 1000)
ax.title.set_text("Normal - lowest 900")

###############

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), constrained_layout = True)
#lowest 900
for j in range(normal_900_cumulative_flow.shape[1]):
    if j == 0:
        ax.plot(normal_900_cumulative_flow[:,j], color='orange', alpha=0.2, label= "bottom 900 simulations")
    else:
        ax.plot(normal_900_cumulative_flow[:,j], color='orange', alpha=0.2)
#top 100
for j in range(normal_cumulative_flow.shape[1]):
    if j == 0:
        ax.plot(normal_cumulative_flow[:,j], color='blue', alpha=0.2, label= "top 100 simulations")
    else:
        ax.plot(normal_cumulative_flow[:,j], color='blue', alpha=0.2)
#observed_radar
for j in range(obs_radar_cumulative_flow.shape[1]):
    if j == 0:
        ax.plot(obs_radar_cumulative_flow[:,j], color='teal', label= "radar obs")
    else:
        ax.plot(obs_radar_cumulative_flow[:,j], color='teal')
#means
ax.plot(normal_900_cumulative_flow_mean, color='purple', label="mean of bottom 900 simulations")
ax.plot(normal_cumulative_flow_mean, color='black', label="mean of top 100 simulations")
ax.plot(obs_radar_cumulative_flow_mean, color='cyan', label="mean of 25 radar obs")
#mean of 1000 simulations
ax.plot(normal_all_cumulative_flow_mean, color='red', label="mean of 1000 simulations")
plt.legend()

###############

#Line plots
plt.figure()
for j in range(normal_cumulative_flow.shape[1]):
    plt.plot(normal_cumulative_flow[:,j], color='blue', alpha=0.2)
for j in range(turned_cumulative_flow.shape[1]):
    plt.plot(turned_cumulative_flow[:,j], color='orange', alpha=0.2)
for j in range(slow_cumulative_flow.shape[1]):
    plt.plot(slow_cumulative_flow[:,j], color='green', alpha=0.2)
for j in range(long_cumulative_flow.shape[1]):
    plt.plot(long_cumulative_flow[:,j], color='red', alpha=0.2)
if save_figs:
    plt.savefig(os.path.join(out_dir, "19_runoff_accumulations.png"))

#Subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10), constrained_layout = True)
for j in range(normal_cumulative_flow.shape[1]):
    ax1.plot(normal_cumulative_flow[:,j], color='blue', alpha=0.2)
ax1.plot(normal_cumulative_flow_mean, color='black')
ax1.set_ylim(0, 180000)
ax1.set_xlim(0, 1000)
ax1.title.set_text("Normal")
for j in range(turned_cumulative_flow.shape[1]):
    ax2.plot(turned_cumulative_flow[:,j], color='orange', alpha=0.2)
ax2.plot(turned_cumulative_flow_mean, color='black')
ax2.set_ylim(0, 180000)
ax2.set_xlim(0, 1000)
ax2.title.set_text("Turned")
for j in range(slow_cumulative_flow.shape[1]):
    ax3.plot(slow_cumulative_flow[:,j], color='green', alpha=0.2)
ax3.plot(slow_cumulative_flow_mean, color='black')
ax3.set_ylim(0, 180000)
ax3.set_xlim(0, 1000)
ax3.title.set_text("Slow")
for j in range(long_cumulative_flow.shape[1]):
    ax4.plot(long_cumulative_flow[:,j], color='red', alpha=0.2)
ax4.plot(long_cumulative_flow_mean, color='black')
ax4.set_ylim(0, 180000)
ax4.set_xlim(0, 1000)
ax4.title.set_text("Long")
fig.suptitle("Runoff accumulations [l/s]", fontsize=16)
if save_figs:
    plt.savefig(os.path.join(out_dir, "21_runoff_accumulations.png"))
    
#l/s to mm
normal_cumulative_flow_mm = normal_cumulative_flow.copy()
slow_cumulative_flow_mm = slow_cumulative_flow.copy()
turned_cumulative_flow_mm = turned_cumulative_flow.copy()
long_cumulative_flow_mm = long_cumulative_flow.copy()
# x * 1000000 / (85*10000000000) * 60
normal_cumulative_flow_mm = normal_cumulative_flow_mm * 1000000 / (85*10000000000) * 60
slow_cumulative_flow_mm = slow_cumulative_flow_mm * 1000000 / (85*10000000000) * 60
turned_cumulative_flow_mm = turned_cumulative_flow * 1000000 / (85*10000000000) * 60
long_cumulative_flow_mm = long_cumulative_flow_mm * 1000000 / (85*10000000000) * 60
#means
normal_cumulative_flow_mm_mean = np.zeros((normal_cumulative_flow_mm.shape[0], 1))
slow_cumulative_flow_mm_mean = np.zeros((slow_cumulative_flow_mm.shape[0], 1))
turned_cumulative_flow_mm_mean = np.zeros((turned_cumulative_flow_mm.shape[0], 1))
long_cumulative_flow_mm_mean = np.zeros((long_cumulative_flow_mm.shape[0], 1))
for i in range(len(normal_cumulative_flow_mm_mean)):
    normal_cumulative_flow_mm_mean[i] = np.mean(normal_cumulative_flow_mm[i,:])
for i in range(len(slow_cumulative_flow_mm_mean)):
    slow_cumulative_flow_mm_mean[i] = np.mean(slow_cumulative_flow_mm[i,:])
for i in range(len(turned_cumulative_flow_mm_mean)):
    turned_cumulative_flow_mm_mean[i] = np.mean(turned_cumulative_flow_mm[i,:])
for i in range(len(long_cumulative_flow_mm_mean)):
    long_cumulative_flow_mm_mean[i] = np.mean(long_cumulative_flow_mm[i,:]) 

#Runoff of measured event
obs_runoff_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
pakila_runoff = genfromtxt(fname=os.path.join(obs_runoff_dir, "pakila_outflow_1min_12082017.csv"), delimiter=';')
pakila_runoff = pakila_runoff[1:,:]
pakila_runoff = pakila_runoff[:,6]

pakila_runoff_nans = np.argwhere(np.isnan(pakila_runoff))
pakila_runoff_uusi = pakila_runoff.copy()

#Interpolate missing values
x_puuttuva = 1
x_mittaukset = [0, 2]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 3
x_mittaukset = [2, 4]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 42
x_mittaukset = [41, 43]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = [56, 57]
x_mittaukset = [55, 58]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva[0]] = y_puuttuva[0]
pakila_runoff_uusi[x_puuttuva[1]] = y_puuttuva[1]
#####
x_puuttuva = 59
x_mittaukset = [58, 60]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 61
x_mittaukset = [60, 62]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 67
x_mittaukset = [66, 68]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 84
x_mittaukset = [83, 85]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = [88, 89]
x_mittaukset = [87, 90]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva[0]] = y_puuttuva[0]
pakila_runoff_uusi[x_puuttuva[1]] = y_puuttuva[1]
#####
x_puuttuva = 93
x_mittaukset = [92, 94]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 96
x_mittaukset = [95, 97]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 102
x_mittaukset = [101, 103]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 107
x_mittaukset = [106, 108]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 115
x_mittaukset = [114, 116]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 166
x_mittaukset = [165, 167]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = [174, 175, 176, 177]
x_mittaukset = [173, 178]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva[0]] = y_puuttuva[0]
pakila_runoff_uusi[x_puuttuva[1]] = y_puuttuva[1]
pakila_runoff_uusi[x_puuttuva[2]] = y_puuttuva[2]
pakila_runoff_uusi[x_puuttuva[3]] = y_puuttuva[3]
#####
x_puuttuva = [185, 186]
x_mittaukset = [184, 187]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva[0]] = y_puuttuva[0]
pakila_runoff_uusi[x_puuttuva[1]] = y_puuttuva[1]
#####
x_puuttuva = [188, 189, 190]
x_mittaukset = [187, 191]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva[0]] = y_puuttuva[0]
pakila_runoff_uusi[x_puuttuva[1]] = y_puuttuva[1]
pakila_runoff_uusi[x_puuttuva[2]] = y_puuttuva[2]
#####
x_puuttuva = 229
x_mittaukset = [228, 230]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva
#####
x_puuttuva = 261
x_mittaukset = [260, 262]
y_mittaukset = [pakila_runoff_uusi[x_mittaukset[0]], pakila_runoff_uusi[x_mittaukset[1]]]
y_puuttuva = np.interp(x_puuttuva, x_mittaukset, y_mittaukset)
pakila_runoff_uusi[x_puuttuva] = y_puuttuva

pakila_runoff_uusi = pakila_runoff_uusi[1:]

#Plot runoff ts [l/s]
plt.figure()
plt.plot(pakila_runoff_uusi)

#Calculate cumulative runoff ts [l/s]
pakila_runoff_cumulative = np.zeros((len(pakila_runoff_uusi), 1))
pakila_runoff_cumulative[0] = pakila_runoff_uusi[0]
for i in range(1,len(pakila_runoff_uusi)):
    pakila_runoff_cumulative[i] = pakila_runoff_uusi[i] + pakila_runoff_cumulative[i-1]
#l/s to mm
pakila_runoff_cumulative_mm = pakila_runoff_cumulative.copy()
pakila_runoff_cumulative_mm = pakila_runoff_cumulative_mm * 1000000 / (85*10000000000) * 60

#Subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10), constrained_layout = True)
for j in range(normal_cumulative_flow_mm.shape[1]):
    ax1.plot(normal_cumulative_flow_mm[:,j], color='blue', alpha=0.2)
ax1.plot(normal_cumulative_flow_mm_mean, color='black')
ax1.plot(pakila_runoff_cumulative_mm, color='red')
ax1.set_ylim(0, 13)
ax1.set_xlim(0, 1000)
ax1.title.set_text("Normal")
for j in range(turned_cumulative_flow_mm.shape[1]):
    ax2.plot(turned_cumulative_flow_mm[:,j], color='orange', alpha=0.2)
ax2.plot(turned_cumulative_flow_mm_mean, color='black')
ax2.plot(pakila_runoff_cumulative_mm, color='red')
ax2.set_ylim(0, 13)
ax2.set_xlim(0, 1000)
ax2.title.set_text("Turned")
for j in range(slow_cumulative_flow_mm.shape[1]):
    ax3.plot(slow_cumulative_flow_mm[:,j], color='green', alpha=0.2)
ax3.plot(slow_cumulative_flow_mm_mean, color='black')
ax3.plot(pakila_runoff_cumulative_mm, color='red')
ax3.set_ylim(0, 13)
ax3.set_xlim(0, 1000)
ax3.title.set_text("Slow")
for j in range(long_cumulative_flow_mm.shape[1]):
    ax4.plot(long_cumulative_flow_mm[:,j], color='red', alpha=0.2)
ax4.plot(long_cumulative_flow_mm_mean, color='black')
ax4.set_ylim(0, 13)
ax4.set_xlim(0, 1000)
ax4.title.set_text("Long")
fig.suptitle("Runoff accumulations [mm]", fontsize=16)
if save_figs:
    plt.savefig(os.path.join(out_dir, "22_runoff_accumulations_mm.png"))

#Separated figs
plt.figure()
for im in range(normal_cumulative_flow_mm.shape[1]):
    if im == 0:
        plt.plot(normal_cumulative_flow_mm[:,im], color='blue', alpha=0.2, label="ensemble member")
    else:
        plt.plot(normal_cumulative_flow_mm[:,im], color='blue', alpha=0.2)
plt.plot(normal_cumulative_flow_mm_mean, color='black', label=f"mean ({normal_cumulative_flow_mm_mean[-1,0]})")
plt.plot(pakila_runoff_cumulative_mm, color='magenta', label=f"obs pakila gauge ({pakila_runoff_cumulative_mm[-1,0]})")
plt.ylim(0, 13)
# plt.xlim(0, 1000)
plt.title("Normal")
plt.legend()
if save_figs:
    plt.savefig(os.path.join(out_dir, "45_runoff_accu_normal.png"))
    
plt.figure()
for im in range(slow_cumulative_flow_mm.shape[1]):
    if im == 0:
        plt.plot(slow_cumulative_flow_mm[:,im], color='green', alpha=0.2, label="ensemble member")
    else:
        plt.plot(slow_cumulative_flow_mm[:,im], color='green', alpha=0.2)
plt.plot(slow_cumulative_flow_mm_mean, color='black', label=f"mean ({slow_cumulative_flow_mm_mean[-1,0]})")
plt.plot(pakila_runoff_cumulative_mm, color='magenta', label=f"obs pakila gauge ({pakila_runoff_cumulative_mm[-1,0]})")
plt.ylim(0, 13)
# plt.xlim(0, 1000)
plt.title("Slow")
plt.legend()
if save_figs:
    plt.savefig(os.path.join(out_dir, "46_runoff_accu_slow.png"))

plt.figure()
for im in range(turned_cumulative_flow_mm.shape[1]):
    if im == 0:
        plt.plot(turned_cumulative_flow_mm[:,im], color='orange', alpha=0.2, label="ensemble member")
    else:
        plt.plot(turned_cumulative_flow_mm[:,im], color='orange', alpha=0.2)
plt.plot(turned_cumulative_flow_mm_mean, color='black', label=f"mean ({turned_cumulative_flow_mm_mean[-1,0]})")
plt.plot(pakila_runoff_cumulative_mm, color='magenta', label=f"obs pakila gauge ({pakila_runoff_cumulative_mm[-1,0]})")
plt.ylim(0, 13)
# plt.xlim(0, 1000)
plt.title("Turned")
plt.legend()
if save_figs:
    plt.savefig(os.path.join(out_dir, "47_runoff_accu_turned.png"))
    
plt.figure()
for im in range(long_cumulative_flow_mm.shape[1]):
    if im == 0:
        plt.plot(long_cumulative_flow_mm[:,im], color='red', alpha=0.2, label="ensemble member")
    else:
        plt.plot(long_cumulative_flow_mm[:,im], color='red', alpha=0.2)
plt.plot(long_cumulative_flow_mm_mean, color='black', label=f"mean ({long_cumulative_flow_mm_mean[-1,0]})")
plt.plot(pakila_runoff_cumulative_mm, color='magenta', label=f"obs pakila gauge ({pakila_runoff_cumulative_mm[-1,0]})")
plt.ylim(0, 13)
# plt.xlim(0, 1000)
plt.title("Long")
plt.legend()
if save_figs:
    plt.savefig(os.path.join(out_dir, "48_runoff_accu_long.png"))
 
###############################
#Plot histograms
plt.figure()
sns.histplot(data=[normal_cumulative_flow_mm[-1,:], turned_cumulative_flow_mm[-1,:], slow_cumulative_flow_mm[-1,:], long_cumulative_flow_mm[-1,:]], 
             bins=20, kde=True, stat="percent")
if save_figs:
    plt.savefig(os.path.join(out_dir, "23_hist_bars-lines.png"))

plt.figure()
sns.kdeplot(data=[normal_cumulative_flow_mm[-1,:], turned_cumulative_flow_mm[-1,:], slow_cumulative_flow_mm[-1,:], long_cumulative_flow_mm[-1,:]], cut=0)
if save_figs:
    plt.savefig(os.path.join(out_dir, "24_hist_lines.png"))

##############################################################################
##############################################################################

# # normal_cumulative_flow to mm
# normal_cumulative_flow_mm = normal_cumulative_flow.copy()
# normal_cumulative_flow_mm = normal_cumulative_flow_mm * 1000000 / (85*10000000000) * 60
# normal_cumulative_flow_mm_mean = np.zeros((normal_cumulative_flow_mm.shape[0], 1))
# for i in range(len(normal_cumulative_flow_mm_mean)):
#     normal_cumulative_flow_mm_mean[i] = np.mean(normal_cumulative_flow_mm[i,:])

# normal_900_cumulative_flow to mm
normal_900_cumulative_flow_mm = normal_900_cumulative_flow.copy()
normal_900_cumulative_flow_mm = normal_900_cumulative_flow_mm * 1000000 / (85*10000000000) * 60
normal_900_cumulative_flow_mm_mean = np.zeros((normal_900_cumulative_flow_mm.shape[0], 1))
for i in range(len(normal_900_cumulative_flow_mm_mean)):
    normal_900_cumulative_flow_mm_mean[i] = np.mean(normal_900_cumulative_flow_mm[i,:])

# obs_radar_cumulative_flow to mm
obs_radar_cumulative_flow_mm = obs_radar_cumulative_flow.copy()
obs_radar_cumulative_flow_mm = obs_radar_cumulative_flow * 1000000 / (85*10000000000) * 60
obs_radar_cumulative_flow_mm_mean = np.zeros((obs_radar_cumulative_flow_mm.shape[0], 1))
for i in range(len(obs_radar_cumulative_flow_mm_mean)):
    obs_radar_cumulative_flow_mm_mean[i] = np.mean(obs_radar_cumulative_flow_mm[i,:])

#All 1000 normal simulations
normal_all_cumulative_flow_mm_900 = normal_900_cumulative_flow_mm.copy()
normal_all_cumulative_flow_mm_100 = normal_cumulative_flow_mm.copy()
normal_all_cumulative_flow_mm = np.concatenate((normal_all_cumulative_flow_mm_900, normal_all_cumulative_flow_mm_100), axis=1)
#mean
normal_all_cumulative_flow_mm_mean = np.zeros((normal_all_cumulative_flow_mm.shape[0], 1))
for i in range(len(normal_all_cumulative_flow_mm_mean)):
    normal_all_cumulative_flow_mm_mean[i] = np.mean(normal_all_cumulative_flow_mm[i,:])
#median    
normal_all_cumulative_flow_mm_median = np.zeros((normal_all_cumulative_flow_mm.shape[0], 1))
for i in range(len(normal_all_cumulative_flow_mm_median)):
    normal_all_cumulative_flow_mm_median[i] = np.median(normal_all_cumulative_flow_mm[i,:])

#Plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5), constrained_layout = True)
#lowest 900
for j in range(normal_900_cumulative_flow_mm.shape[1]):
    if j == 0:
        ax.plot(normal_900_cumulative_flow_mm[:,j], color='gray', alpha=0.2, label= "bottom 900 simulations")
    else:
        ax.plot(normal_900_cumulative_flow_mm[:,j], color='gray', alpha=0.2)
#top 100
for j in range(normal_cumulative_flow_mm.shape[1]):
    if j == 0:
        ax.plot(normal_cumulative_flow_mm[:,j], color='teal', alpha=0.2, label= "top 100 simulations")
    else:
        ax.plot(normal_cumulative_flow_mm[:,j], color='teal', alpha=0.2)
# #observed_radar
# for j in range(obs_radar_cumulative_flow_mm.shape[1]):
#     if j == 0:
#         ax.plot(obs_radar_cumulative_flow_mm[:,j], color='blue', label= "obs radar")
#     else:
#         ax.plot(obs_radar_cumulative_flow_mm[:,j], color='blue')
#means
# ax.plot(normal_900_cumulative_flow_mm_mean, color='purple', label="mean of bottom 900 simulations")
# ax.plot(normal_cumulative_flow_mm_mean, color='black', label="mean of top 100 simulations")
#mean of 1000 simulations
ax.plot(normal_all_cumulative_flow_mm_mean, color='red', label=f"mean of 1000 simulations ({normal_all_cumulative_flow_mm_mean[-1,0]})")
ax.plot(normal_all_cumulative_flow_mm_median, color='blue', label=f"median of 1000 simulations ({normal_all_cumulative_flow_mm_median[-1,0]})")
ax.plot((normal_all_cumulative_flow_mm[:,int(np.where(normal_all_cumulative_flow_mm[-1,:] == find_closest(normal_all_cumulative_flow_mm[-1,:], np.median(normal_all_cumulative_flow_mm[-1,:])))[0])]), color="purple", label="median of 1000 simulations")
ax.plot((normal_all_cumulative_flow_mm[:,np.argmax(normal_all_cumulative_flow_mm[-1,:], axis=0)]), color="orange", label="max")
ax.plot(obs_radar_cumulative_flow_mm_mean, color='cyan', label=f"mean of 25 radar obs ({obs_radar_cumulative_flow_mm_mean[-1,0]})")
ax.plot(pakila_runoff_cumulative_mm, color='magenta', label=f"obs pakila gauge ({pakila_runoff_cumulative_mm[-1,0]})")
plt.legend()
ax.title.set_text("Runoff [mm]")
if save_figs:
    plt.savefig(os.path.join(out_dir, "44_runoff-normal_vs_obs.png"))

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

# Intensitetti aikasarjat sadalle isoimmalle simuloinnille



# Max intensiteetit

##############################################################################

# Aikasarjat Ottarille SWMM-simulointeja varten
# - Havaitun tutkatapahtuman intensitettiaikasarjat (25 kpl)
# - Havaitun tutkatapahtuman kenttäkeskiarvoaikasarja intensiteetille -> Niin pieniä arvoja, ettei järkeä
# - Mitattu gauge-aikasarja Pakila

dir_obs = r"W:\lindgrv1\Kiira_whole_day\kiira_14-22_dbz"
file_list_obs = os.listdir(dir_obs)
file_list_obs = [x for x in file_list_obs if ".tif" in x]
file_list_obs = [f for f in file_list_obs if f.endswith(".tif")]
file_list_obs = file_list_obs[8:-1]

# Start is 14:45 and 21:55
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

#Event from dbz into mm/h
a_R=223
b_R=1.53
radar_kiira_mmh = radar_kiira.copy()
radar_kiira_mmh = 10**((radar_kiira_mmh-10*np.log10(a_R))/(10*b_R))
#Values less than threshold to zero
radar_kiira_mmh[radar_kiira_mmh < 0.1] = 0

#Areal mean rainfall in mm/h
areal_rainfall_mmh = np.zeros(len(radar_kiira_mmh))
for i in range (len(radar_kiira_mmh)):
    areal_rainfall_mmh[i] = np.nanmean(radar_kiira_mmh[i])
#Plot areal mean rainfall ts
plt.figure()
plt.plot(areal_rainfall_mmh)
plt.title("Areal mean rainfall (mm/h)")

#Corrected areal mean rainfall in mm/h
corrected_areal_rainfall_mmh = areal_rainfall_mmh.copy()
x_obs_17 = [16, 18]
y_mar_obs_17 = [corrected_areal_rainfall_mmh[16], corrected_areal_rainfall_mmh[18]]
x_new_17 = 17
y_mar_new_17 = np.interp(x_new_17, x_obs_17, y_mar_obs_17)
corrected_areal_rainfall_mmh[17] = y_mar_new_17
x_obs_36 = [34, 37]
y_mar_obs_36 = [corrected_areal_rainfall_mmh[34], corrected_areal_rainfall_mmh[37]]
x_new_36 = [35, 36]
y_mar_new_36 = np.interp(x_new_36, x_obs_36, y_mar_obs_36)
corrected_areal_rainfall_mmh[35] = y_mar_new_36[0]
corrected_areal_rainfall_mmh[36] = y_mar_new_36[1]
x_obs_46 = [45, 50]
y_mar_obs_46 = [corrected_areal_rainfall_mmh[45], corrected_areal_rainfall_mmh[50]]
x_new_46 = [46, 47, 48, 49]
y_mar_new_46 = np.interp(x_new_46, x_obs_46, y_mar_obs_46)
corrected_areal_rainfall_mmh[46] = y_mar_new_46[0]
corrected_areal_rainfall_mmh[47] = y_mar_new_46[1]
corrected_areal_rainfall_mmh[48] = y_mar_new_46[2]
corrected_areal_rainfall_mmh[49] = y_mar_new_46[3]
#Plot corrected areal mean rainfall ts
plt.figure()
plt.plot(corrected_areal_rainfall_mmh)
plt.title("Areal mean rainfall (mm/h)")

np.mean(corrected_areal_rainfall_mmh)
np.max(corrected_areal_rainfall_mmh)

# #Corrected areal mean rainfall in dbz
# corrected_mar_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
# corrected_mar = genfromtxt(fname=os.path.join(corrected_mar_dir, "data_tss_corrected.csv"), delimiter=',', skip_header=1)
# corrected_mar = np.delete(corrected_mar, 0, axis=1)
# plt.figure()
# plt.plot(corrected_mar[0])
# plt.title("Corrected - areal mean rainfall (dBZ)")

# corrected_mar_mmh = 10**((corrected_mar[0]-10*np.log10(a_R))/(10*b_R))
# plt.figure()
# plt.plot(corrected_mar_mmh)
# plt.title("Corrected - areal mean rainfall (mm/h)")

#Gauge time series
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
    
plt.figure()
plt.plot(ts_1[0])
plt.plot(ts_2[0])
plt.plot(ts_3[0])
plt.plot(ts_4[0])
plt.plot(ts_5[0])
plt.plot(ts_6[0])
plt.plot(ts_7[0])
plt.plot(ts_8[0])
plt.plot(ts_9[0])
plt.plot(ts_10[0])
plt.plot(ts_11[0])
plt.plot(ts_12[0])
plt.plot(ts_13[0])
plt.plot(ts_14[0])
plt.plot(ts_15[0])
plt.plot(ts_16[0])
plt.plot(ts_17[0])
plt.plot(ts_18[0])
plt.plot(ts_19[0])
plt.plot(ts_20[0])
plt.plot(ts_21[0])
plt.plot(ts_22[0])
plt.plot(ts_23[0])
plt.plot(ts_24[0])
plt.plot(ts_25[0])

if save_csvs:
    data_temp = [ts_1[0], ts_2[0], ts_3[0], ts_4[0], ts_5[0], ts_6[0], ts_7[0], ts_8[0], ts_9[0],
                 ts_10[0], ts_11[0], ts_12[0], ts_13[0], ts_14[0], ts_15[0], ts_16[0], ts_17[0], ts_18[0],
                 ts_19[0], ts_20[0], ts_21[0], ts_22[0], ts_23[0], ts_24[0], ts_25[0]]
    mmh_ts = pd.DataFrame(data_temp, index=['ts_1', 'ts_2', 'ts_3', 'ts_4', 'ts_5', 'ts_6', 'ts_7', 'ts_8', 'ts_9',
                                            'ts_10', 'ts_11', 'ts_12', 'ts_13', 'ts_14', 'ts_15', 'ts_16', 'ts_17', 'ts_18',
                                            'ts_19', 'ts_20', 'ts_21', 'ts_22', 'ts_23', 'ts_24', 'ts_25'])
    pd.DataFrame(mmh_ts).to_csv(os.path.join(out_dir, "tss_point_mmh_observed.csv"))
 
##############################################################################

event = genfromtxt(fname=os.path.join(out_dir, "tss_point_mmh_observed.csv"), delimiter=',')
event = event[1:,1:]
event = event[:,:-1]

event_name = np.repeat("KIIRA_OBS", event.shape[1])
year = np.repeat("2017", event.shape[1])
month = np.repeat("08", event.shape[1])
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

obs_ts_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
obs_ts_dir = os.path.join(obs_ts_dir, "SWMM_input_obs")
if not os.path.exists(obs_ts_dir):
    os.makedirs(obs_ts_dir)

if save_csvs:
    for i in range(len(event)):
        df["rain"] = event[i]
        df.to_csv(os.path.join(obs_ts_dir, f"input_rain_obs_{i}.dat"), sep = "\t", header=False, index=False)

##############################################################################

#Pakila gauges (Tero)
# Start is 14:45 and 21:59

pakila_dir = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
#in mm
pakila_obs = genfromtxt(fname=os.path.join(pakila_dir, "pakila_gauge_obs.csv"), delimiter=';')
pakila_obs = pakila_obs[1:436,:]
pakila_obs = pakila_obs[:,1:5]
#in mm/h
pakila_obs_mmh = genfromtxt(fname=os.path.join(pakila_dir, "pakila_gauge_obs_mmh.csv"), delimiter=';')
pakila_obs_mmh = pakila_obs_mmh[1:436,:]
pakila_obs_mmh = pakila_obs_mmh[:,1:5]

for i in range(len(pakila_obs)):
    pakila_obs[i,3] = np.mean(pakila_obs[i,0:3])
    
for i in range(len(pakila_obs_mmh )):
    pakila_obs_mmh [i,3] = np.mean(pakila_obs_mmh [i,0:3])
    
pakila_name = np.repeat("PAKILA_OBS", pakila_obs.shape[0])
pakila_year = np.repeat("2017", pakila_obs.shape[0])
pakila_month = np.repeat("08", pakila_obs.shape[0])
pakila_day = np.repeat("12", pakila_obs.shape[0])

pakila_hour = np.zeros(pakila_obs.shape[0])
pakila_hour[0:15] = 14
pakila_hour[15:75] = 15
pakila_hour[75:135] = 16
pakila_hour[135:195] = 17
pakila_hour[195:255] = 18
pakila_hour[255:315] = 19
pakila_hour[315:375] = 20
pakila_hour[375:435] = 21
pakila_hour = pakila_hour.astype(int)

pakila_minutes = np.arange(0, 60, 1)
pakila_minute = np.zeros(pakila_obs.shape[0])
pakila_minute[0] = 45
pakila_minute[1] = 46
pakila_minute[2] = 47
pakila_minute[3] = 48
pakila_minute[4] = 49
pakila_minute[5] = 50
pakila_minute[6] = 51
pakila_minute[7] = 52
pakila_minute[8] = 53
pakila_minute[9] = 54
pakila_minute[10] = 55
pakila_minute[11] = 56
pakila_minute[12] = 57
pakila_minute[13] = 58
pakila_minute[14] = 59
for i in range(0,60):
    pakila_minute[i+15] = pakila_minutes[i]
for i in range(0,60):
    pakila_minute[i+75] = pakila_minutes[i]
for i in range(0,60):
    pakila_minute[i+135] = pakila_minutes[i]
for i in range(0,60):
    pakila_minute[i+195] = pakila_minutes[i]
for i in range(0,60):
    pakila_minute[i+255] = pakila_minutes[i]
for i in range(0,60):
    pakila_minute[i+315] = pakila_minutes[i]
    
for i in range(0,60):
    pakila_minute[i+375] = pakila_minutes[i]
pakila_minute = pakila_minute.astype(int)

df_pakila = pd.DataFrame()
df_pakila["event"] = pakila_name
df_pakila["year"] = pakila_year
df_pakila["month"] = pakila_month
df_pakila["day"] = pakila_day
df_pakila["hour"] = pakila_hour
df_pakila["hour"] = df_pakila["hour"].astype(str)
df_pakila["minute"] = pakila_minute
df_pakila["minute"] = df_pakila["minute"].astype(str)
df_pakila["minute"] = df_pakila["minute"].replace("0", "00")
df_pakila["minute"] = df_pakila["minute"].replace("1", "01")
df_pakila["minute"] = df_pakila["minute"].replace("2", "02")
df_pakila["minute"] = df_pakila["minute"].replace("3", "03")
df_pakila["minute"] = df_pakila["minute"].replace("4", "04")
df_pakila["minute"] = df_pakila["minute"].replace("5", "05")
df_pakila["minute"] = df_pakila["minute"].replace("6", "06")
df_pakila["minute"] = df_pakila["minute"].replace("7", "07")
df_pakila["minute"] = df_pakila["minute"].replace("8", "08")
df_pakila["minute"] = df_pakila["minute"].replace("9", "09")
df_pakila["rain"] = pakila_obs[:,3]

if save_csvs:
    df_pakila.to_csv(os.path.join(pakila_dir, "SWMM_input_pakila_obs.dat"), sep = "\t", header=False, index=False)

df_pakila_mmh = pd.DataFrame()
df_pakila_mmh["event"] = pakila_name
df_pakila_mmh["year"] = pakila_year
df_pakila_mmh["month"] = pakila_month
df_pakila_mmh["day"] = pakila_day
df_pakila_mmh["hour"] = pakila_hour
df_pakila_mmh["hour"] = df_pakila["hour"].astype(str)
df_pakila_mmh["minute"] = pakila_minute
df_pakila_mmh["minute"] = df_pakila["minute"].astype(str)
df_pakila_mmh["minute"] = df_pakila["minute"].replace("0", "00")
df_pakila_mmh["minute"] = df_pakila["minute"].replace("1", "01")
df_pakila_mmh["minute"] = df_pakila["minute"].replace("2", "02")
df_pakila_mmh["minute"] = df_pakila["minute"].replace("3", "03")
df_pakila_mmh["minute"] = df_pakila["minute"].replace("4", "04")
df_pakila_mmh["minute"] = df_pakila["minute"].replace("5", "05")
df_pakila_mmh["minute"] = df_pakila["minute"].replace("6", "06")
df_pakila_mmh["minute"] = df_pakila["minute"].replace("7", "07")
df_pakila_mmh["minute"] = df_pakila["minute"].replace("8", "08")
df_pakila_mmh["minute"] = df_pakila["minute"].replace("9", "09")
df_pakila_mmh["rain"] = pakila_obs_mmh[:,3]

if save_csvs:
    df_pakila_mmh.to_csv(os.path.join(pakila_dir, "SWMM_input_pakila_obs_mmh.dat"), sep = "\t", header=False, index=False)

##############################################################################

#Map of simulation area

map_dir = r"W:\lindgrv1\Kiira_whole_day\kiira_14-22_dbz"
file_list_map = os.listdir(map_dir)
file_list_map = [x for x in file_list_map if ".tif" in x]
file_list_map = [f for f in file_list_map if f.endswith(".tif")]

temp_raster_map = rasterio.open(os.path.join(map_dir, file_list_map[0])) #import one rain field
print(temp_raster_map.crs) #coordinate reference system of the raster
temp_bounds_map = temp_raster_map.bounds #raster corner coordinates
bbox_map = box(*temp_bounds_map) #raster to GeoDataFrame
# print(bbox_map.wkt)
bbox_map_df = gpd.GeoDataFrame({"geometry":[bbox_map]}, crs=temp_raster_map.crs)

gis_dir = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/GIS-aineistot"
#Suomi
suomi = gpd.read_file(os.path.join(gis_dir, "mml-hallintorajat_10k", "2021", "SuomenValtakunta_2021_10k.shp"))
suomi_maa = gpd.read_file(os.path.join(gis_dir, "mml-shapet", "maakunnat2021_4500000_eimeri.shp"))
#World
#http://www.naturalearthdata.com/downloads/10m-cultural-vectors/
countries = gpd.read_file(os.path.join(gis_dir, "ne_10m_admin_0_countries", "ne_10m_admin_0_countries.shp"))
countries.crs
countries_finland = countries[countries["SOVEREIGNT"]=="Finland"]
countries_sweden = countries[countries["SOVEREIGNT"]=="Sweden"]
countries_norway = countries[countries["SOVEREIGNT"]=="Norway"]
countries_estonia = countries[countries["SOVEREIGNT"]=="Estonia"]
countries_russia = countries[countries["SOVEREIGNT"]=="Russia"]

suomi = suomi.to_crs(countries.crs)
suomi_maa = suomi_maa.to_crs(countries.crs)
bbox_map_df = bbox_map_df.to_crs(countries.crs)

# 60.2410802, 24.9271415 (WGS84 = EPSG:4326)
# P: 6680631 I: 2549602 (EPSG:3067)
pakila_pt = Point(24.9271415,60.2410802)
pakila_pt_df = gpd.GeoDataFrame({"geometry":[pakila_pt]}, crs="EPSG:4326")
pakila_pt_df  = pakila_pt_df .to_crs(countries.crs)

#Finnish georef-system
suomi = suomi.to_crs(temp_raster_map.crs)
suomi_maa = suomi_maa.to_crs(temp_raster_map.crs)
pakila_pt_df  = pakila_pt_df .to_crs(temp_raster_map.crs)
countries_finland = countries_finland.to_crs(temp_raster_map.crs)
countries_sweden = countries_sweden.to_crs(temp_raster_map.crs)
countries_norway = countries_norway.to_crs(temp_raster_map.crs)
countries_estonia = countries_estonia.to_crs(temp_raster_map.crs)
countries_russia = countries_russia.to_crs(temp_raster_map.crs)

#Plot map
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7,10), constrained_layout = True)
# ax1.set_xlim(16, 33)
# ax1.set_ylim(58.5, 71)
ax1.set_xlim(0, 800000)
ax1.set_ylim(6500000, 7850000)
countries_sweden.plot(ax = ax1, color="gainsboro")
countries_norway.plot(ax = ax1, color="gainsboro")
countries_estonia.plot(ax = ax1, color="gainsboro")
countries_russia.plot(ax = ax1, color="gainsboro")
suomi_maa.plot(ax = ax1, fc="green", ec="green", linewidth=1)
suomi.plot(ax = ax1, fc="none", ec="black", linewidth=2)
bbox_map_df.plot(ax = ax1, fc="none", ec="black", linewidth=2)
pakila_pt_df.plot(ax = ax1, marker="o", fc="red", ec="black", linewidth=2, markersize=200)
x, y, arrow_length = 0.1, 0.95, 0.1
ax1.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black'),
            ha='center', va='center',
            xycoords=ax1.transAxes)
scalebar1 = ScaleBar(1, "m", length_fraction=0.4, location="lower right", scale_loc="top")
ax1.add_artist(scalebar1)
if save_figs:
    plt.savefig(os.path.join(pakila_dir, "map_rain_simulations.png"))

##############################################################################

# Return periods / probabilities of events

##############################################################################

# Nonparametric tests to compare two unpaired groups of data
# - Wilcox 
# - Mann-Whitney (Wilcox Rank Sum Test)
# - Kolmogorov-Smirnov

##############################################################################

# Scatterplotit
# - peak flow vs. flooded nodes (critical levels)
# - runoff accumulations vs. flooded nodes (critical levels)
# - rainfall max intensities vs. flooded nodes (critical levels)
# - rainfall accumultaions vs. flooded nodes (critical levels)

#peak flow vs. flooded nodes (critical levels)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10), constrained_layout = True)
#normal
ax1.scatter(normal_peak_flow[0], normal_flooding_sum)
ax1.scatter(normal_peak_flow[0], normal_critical_sum)
ax1.title.set_text("Normal")
ax1.set_ylim(0, 480)
ax1.set_xlim(400, 2500)
#turned
ax2.scatter(turned_peak_flow[0], turned_flooding_sum)
ax2.scatter(turned_peak_flow[0], turned_critical_sum)
ax2.title.set_text("Turned")
ax2.set_ylim(0, 480)
ax2.set_xlim(400, 2500)
#slow
ax3.scatter(slow_peak_flow[0], slow_flooding_sum)
ax3.scatter(slow_peak_flow[0], slow_critical_sum)
ax3.title.set_text("Slow")
ax3.set_ylim(0, 480)
ax3.set_xlim(400, 2500)
#long
ax4.scatter(long_peak_flow[0], long_flooding_sum)
ax4.scatter(long_peak_flow[0], long_critical_sum)
ax4.title.set_text("Long")
ax4.set_ylim(0, 480)
ax4.set_xlim(400, 2500)
fig.suptitle("Peak flow [l/s] vs. flooded nodes (critical levels)", fontsize=16)
if save_figs:
    plt.savefig(os.path.join(out_dir, "31_scat_peakflow-vs-nodes.png"))

plt.figure()
plt.scatter(normal_peak_flow[0], normal_flooding_sum, label="Normal")
plt.scatter(turned_peak_flow[0], turned_flooding_sum, label="Turned")
plt.scatter(slow_peak_flow[0], slow_flooding_sum, label="Slow")
plt.scatter(long_peak_flow[0], long_flooding_sum, label="Long")
plt.legend()
plt.title("Peak flow [l/s] vs. flooded nodes")
if save_figs:
    plt.savefig(os.path.join(out_dir, "32_scat_peakflow-vs-flooding.png"))

plt.figure()
plt.scatter(normal_peak_flow[0], normal_critical_sum, label="Normal")
plt.scatter(turned_peak_flow[0], turned_critical_sum, label="Turned")
plt.scatter(slow_peak_flow[0], slow_critical_sum, label="Slow")
plt.scatter(long_peak_flow[0], long_critical_sum, label="Long")
plt.legend()
plt.title("Peak flow [l/s] vs. critical nodes")
if save_figs:
    plt.savefig(os.path.join(out_dir, "33_scat_peakflow-vs-critical.png"))

##########

#runoff accumulations vs. flooded nodes (critical levels)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10), constrained_layout = True)
#normal
ax1.scatter(normal_cumulative_flow[-1], normal_flooding_sum)
ax1.scatter(normal_cumulative_flow[-1], normal_critical_sum)
ax1.title.set_text("Normal")
ax1.set_ylim(0, 480)
ax1.set_xlim(35000, 180000)
#turned
ax2.scatter(turned_cumulative_flow[-1], turned_flooding_sum)
ax2.scatter(turned_cumulative_flow[-1], turned_critical_sum)
ax2.title.set_text("Turned")
ax2.set_ylim(0, 480)
ax2.set_xlim(35000, 180000)
#slow
ax3.scatter(slow_cumulative_flow[-1], slow_flooding_sum)
ax3.scatter(slow_cumulative_flow[-1], slow_critical_sum)
ax3.title.set_text("Slow")
ax3.set_ylim(0, 480)
ax3.set_xlim(35000, 180000)
#long
ax4.scatter(long_cumulative_flow[-1], long_flooding_sum)
ax4.scatter(long_cumulative_flow[-1], long_critical_sum)
ax4.title.set_text("Long")
ax4.set_ylim(0, 480)
ax4.set_xlim(35000, 180000)
fig.suptitle("Cumulative flow [l/s] vs. flooded nodes (critical levels)", fontsize=16)
if save_figs:
    plt.savefig(os.path.join(out_dir, "34_scat_cumflow-vs-nodes.png"))

plt.figure()
plt.scatter(normal_cumulative_flow[-1], normal_flooding_sum, label="Normal")
plt.scatter(turned_cumulative_flow[-1], turned_flooding_sum, label="Turned")
plt.scatter(slow_cumulative_flow[-1], slow_flooding_sum, label="Slow")
plt.scatter(long_cumulative_flow[-1], long_flooding_sum, label="Long")
plt.legend()
plt.title("Cumulative flow [l/s] vs. flooded nodes")
if save_figs:
    plt.savefig(os.path.join(out_dir, "35_scat_cumflow-vs-flooding.png"))

plt.figure()
plt.scatter(normal_cumulative_flow[-1], normal_critical_sum, label="Normal")
plt.scatter(turned_cumulative_flow[-1], turned_critical_sum, label="Turned")
plt.scatter(slow_cumulative_flow[-1], slow_critical_sum, label="Slow")
plt.scatter(long_cumulative_flow[-1], long_critical_sum, label="Long")
plt.legend()
plt.title("Cumulative flow [l/s] vs. critical nodes")
if save_figs:
    plt.savefig(os.path.join(out_dir, "36_scat_cumflow-vs-critical.png"))

##############################################################################

# Jokaiselle nodelle simulointien lukumäärä, jolloin tulvii (kriittinen taso ylittyy) - csv-tiedostojen rivisummat
# Histogrammi jossa x=node ja y=lukumäärä

#normal
normal_nodes_critical_sum = np.zeros((1, normal_critical.shape[0]))
for i in range(normal_nodes_critical_sum.shape[1]):
    normal_nodes_critical_sum[0,i] = int(sum(normal_critical[i,:]))
normal_nodes_flooding_sum = np.zeros((1, normal_flooding.shape[0]))
for i in range(normal_nodes_flooding_sum.shape[1]):
    normal_nodes_flooding_sum[0,i] = int(sum(normal_flooding[i,:]))
    
#slow
slow_nodes_critical_sum = np.zeros((1, slow_critical.shape[0]))
for i in range(slow_nodes_critical_sum.shape[1]):
    slow_nodes_critical_sum[0,i] = int(sum(slow_critical[i,:]))
slow_nodes_flooding_sum = np.zeros((1, slow_flooding.shape[0]))
for i in range(slow_nodes_flooding_sum.shape[1]):
    slow_nodes_flooding_sum[0,i] = int(sum(slow_flooding[i,:]))
    
#turned
turned_nodes_critical_sum = np.zeros((1, turned_critical.shape[0]))
for i in range(turned_nodes_critical_sum.shape[1]):
    turned_nodes_critical_sum[0,i] = int(sum(turned_critical[i,:]))
turned_nodes_flooding_sum = np.zeros((1, turned_flooding.shape[0]))
for i in range(turned_nodes_flooding_sum.shape[1]):
    turned_nodes_flooding_sum[0,i] = int(sum(turned_flooding[i,:]))
    
#long
long_nodes_critical_sum = np.zeros((1, long_critical.shape[0]))
for i in range(long_nodes_critical_sum.shape[1]):
    long_nodes_critical_sum[0,i] = int(sum(long_critical[i,:]))
long_nodes_flooding_sum = np.zeros((1, long_flooding.shape[0]))
for i in range(long_nodes_flooding_sum.shape[1]):
    long_nodes_flooding_sum[0,i] = int(sum(long_flooding[i,:]))

plt.figure()
plt.bar(x=np.arange(normal_nodes_critical_sum.shape[1]), height=normal_nodes_critical_sum[0], label='Critical')
plt.bar(x=np.arange(normal_nodes_flooding_sum.shape[1]), height=normal_nodes_flooding_sum[0], label='Flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "37_flooding-critical_times_for_nodes_normal.png"))

plt.figure()
plt.bar(x=np.arange(turned_nodes_critical_sum.shape[1]), height=turned_nodes_critical_sum[0], label='Critical')
plt.bar(x=np.arange(turned_nodes_flooding_sum.shape[1]), height=turned_nodes_flooding_sum[0], label='Flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "38_flooding-critical_times_for_nodes_turned.png"))

plt.figure()
plt.bar(x=np.arange(slow_nodes_critical_sum.shape[1]), height=slow_nodes_critical_sum[0], label='Critical')
plt.bar(x=np.arange(slow_nodes_flooding_sum.shape[1]), height=slow_nodes_flooding_sum[0], label='Flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "39_flooding-critical_times_for_nodes_slow.png"))

plt.figure()
plt.bar(x=np.arange(long_nodes_critical_sum.shape[1]), height=long_nodes_critical_sum[0], label='Critical')
plt.bar(x=np.arange(long_nodes_flooding_sum.shape[1]), height=long_nodes_flooding_sum[0], label='Flooding')
if save_figs:
    plt.savefig(os.path.join(out_dir, "40_flooding-critical_times_for_nodes_long.png"))

#combined critical arrays
combined_nodes_critical_sum = np.zeros((4, normal_critical.shape[0]))
combined_nodes_critical_sum[0] = normal_nodes_critical_sum[0]
combined_nodes_critical_sum[1] = slow_nodes_critical_sum[0]
combined_nodes_critical_sum[2] = turned_nodes_critical_sum[0]
combined_nodes_critical_sum[3] = long_nodes_critical_sum[0]
#combined flooding arrays
combined_nodes_flooding_sum = np.zeros((4, normal_flooding.shape[0]))
combined_nodes_flooding_sum[0] = normal_nodes_flooding_sum[0]
combined_nodes_flooding_sum[1] = slow_nodes_flooding_sum[0]
combined_nodes_flooding_sum[2] = turned_nodes_flooding_sum[0]
combined_nodes_flooding_sum[3] = long_nodes_flooding_sum[0]

plt.figure()
plt.plot(combined_nodes_flooding_sum[0], label="Normal")
plt.plot(combined_nodes_flooding_sum[2], label="Turned")
plt.plot(combined_nodes_flooding_sum[1], label="Slow")
plt.plot(combined_nodes_flooding_sum[3], label="Long")
plt.legend()
if save_figs:
    plt.savefig(os.path.join(out_dir, "41_flooding_times_for_nodes_long.png"))

plt.figure()
plt.plot(combined_nodes_critical_sum[0], label="Normal")
plt.plot(combined_nodes_critical_sum[2], label="Turned")
plt.plot(combined_nodes_critical_sum[1], label="Slow")
plt.plot(combined_nodes_critical_sum[3], label="Long")
plt.legend()
if save_figs:
    plt.savefig(os.path.join(out_dir, "42_critical_times_for_nodes_long.png"))
