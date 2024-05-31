# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 22:13:22 2023

@author: lindgrv1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##############################################################################
#DIRECTORY

# data_dir_location = r"//home.org.aalto.fi/lindgrv1/data/Desktop/Vaitoskirjaprojekti/Tutkimus/Simulations_pysteps/Event3_new/Accumulations_500"
data_dir_location = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Event3_new/Accumulations_500"

out_figs = r"W:/lindgrv1/Simuloinnit/Simulations_pysteps/Figures"
# plt.savefig(os.path.join(out_figs,"figure_1.pdf"), format="pdf", bbox_inches="tight")

##############################################################################
#ACCUMULATIONS

#GRID
accus_grid_pd = pd.read_csv(os.path.join(data_dir_location, "accu_whole_grid.csv"), delimiter=(","))
accus_grid = accus_grid_pd.to_numpy()
accus_grid = accus_grid[:, 1:]

#BASIN
accus_basin_pd = pd.read_csv(os.path.join(data_dir_location, "accu_whole_basin_test.csv"), delimiter=(","))
accus_basin = accus_basin_pd.to_numpy()
accus_basin = accus_basin[:, 1:]

#NORTH
accus_north_1_pd = pd.read_csv(os.path.join(data_dir_location, "north_accu_1_2smallest.csv"), delimiter=(","))
accus_north_1 = accus_north_1_pd.to_numpy()
accus_north_1 = accus_north_1[:, 1:]

accus_north_2_pd = pd.read_csv(os.path.join(data_dir_location, "north_accu_2_2smallest.csv"), delimiter=(","))
accus_north_2 = accus_north_2_pd.to_numpy()
accus_north_2 = accus_north_2[:, 1:]

accus_north_3_pd = pd.read_csv(os.path.join(data_dir_location, "north_accu_3_test_2smallest.csv"), delimiter=(","))
accus_north_3 = accus_north_3_pd.to_numpy()
accus_north_3 = accus_north_3[:, 1:]

#WEST
accus_west_1_pd = pd.read_csv(os.path.join(data_dir_location, "west_accu_1_largest.csv"), delimiter=(","))
accus_west_1 = accus_west_1_pd.to_numpy()
accus_west_1 = accus_west_1[:, 1:]

accus_west_2_pd = pd.read_csv(os.path.join(data_dir_location, "west_accu_2_largest.csv"), delimiter=(","))
accus_west_2 = accus_west_2_pd.to_numpy()
accus_west_2 = accus_west_2[:, 1:]

accus_west_3_pd = pd.read_csv(os.path.join(data_dir_location, "west_accu_3_test_largest.csv"), delimiter=(","))
accus_west_3 = accus_west_3_pd.to_numpy()
accus_west_3 = accus_west_3[:, 1:]

#SOUTH
accus_south_1_pd = pd.read_csv(os.path.join(data_dir_location, "south_accu_1_2smallest.csv"), delimiter=(","))
accus_south_1 = accus_south_1_pd.to_numpy()
accus_south_1 = accus_south_1[:, 1:]

accus_south_2_pd = pd.read_csv(os.path.join(data_dir_location, "south_accu_2_2smallest.csv"), delimiter=(","))
accus_south_2 = accus_south_2_pd.to_numpy()
accus_south_2 = accus_south_2[:, 1:]

accus_south_3_pd = pd.read_csv(os.path.join(data_dir_location, "south_accu_3_test_2smallest.csv"), delimiter=(","))
accus_south_3 = accus_south_3_pd.to_numpy()
accus_south_3 = accus_south_3[:, 1:]

##############################################################################
#CUMULATIVE RAINFALLS

#GRID
accus_cum_grid = np.zeros((len(accus_grid), len(accus_grid[0])))
accus_cum_grid[:,0] = accus_grid[:,0]
for column in range(1, len(accus_grid[0])):
    accus_cum_grid[:,column] = accus_cum_grid[:,column-1] + accus_grid[:,column]
accus_cum_grid = accus_cum_grid * (5/60)

#BASIN
accus_cum_basin = np.zeros((len(accus_basin), len(accus_basin[0])))
accus_cum_basin[:,0] = accus_basin[:,0]
for column in range(1, len(accus_basin[0])):
    accus_cum_basin[:,column] = accus_cum_basin[:,column-1] + accus_basin[:,column]
accus_cum_basin = accus_cum_basin * (5/60)

#NORTH
accus_cum_north_1 = np.zeros((len(accus_north_1), len(accus_north_1[0])))
accus_cum_north_1[:,0] = accus_north_1[:,0]
for column in range(1, len(accus_north_1[0])):
    accus_cum_north_1[:,column] = accus_cum_north_1[:,column-1] + accus_north_1[:,column]
accus_cum_north_1 = accus_cum_north_1 * (5/60)

accus_cum_north_2 = np.zeros((len(accus_north_2), len(accus_north_2[0])))
accus_cum_north_2[:,0] = accus_north_2[:,0]
for column in range(1, len(accus_north_2[0])):
    accus_cum_north_2[:,column] = accus_cum_north_2[:,column-1] + accus_north_2[:,column]
accus_cum_north_2 = accus_cum_north_2 * (5/60)

accus_cum_north_3 = np.zeros((len(accus_north_3), len(accus_north_3[0])))
accus_cum_north_3[:,0] = accus_north_3[:,0]
for column in range(1, len(accus_north_3[0])):
    accus_cum_north_3[:,column] = accus_cum_north_3[:,column-1] + accus_north_3[:,column]
accus_cum_north_3 = accus_cum_north_3 * (5/60)

#WEST
accus_cum_west_1 = np.zeros((len(accus_west_1), len(accus_west_1[0])))
accus_cum_west_1[:,0] = accus_west_1[:,0]
for column in range(1, len(accus_west_1[0])):
    accus_cum_west_1[:,column] = accus_cum_west_1[:,column-1] + accus_west_1[:,column]
accus_cum_west_1 = accus_cum_west_1 * (5/60)

accus_cum_west_2 = np.zeros((len(accus_west_2), len(accus_west_2[0])))
accus_cum_west_2[:,0] = accus_west_2[:,0]
for column in range(1, len(accus_west_2[0])):
    accus_cum_west_2[:,column] = accus_cum_west_2[:,column-1] + accus_west_2[:,column]
accus_cum_west_2 = accus_cum_west_2 * (5/60)

accus_cum_west_3 = np.zeros((len(accus_west_3), len(accus_west_3[0])))
accus_cum_west_3[:,0] = accus_west_3[:,0]
for column in range(1, len(accus_west_3[0])):
    accus_cum_west_3[:,column] = accus_cum_west_3[:,column-1] + accus_west_3[:,column]
accus_cum_west_3 = accus_cum_west_3 * (5/60)

#SOUTH
accus_cum_south_1 = np.zeros((len(accus_south_1), len(accus_south_1[0])))
accus_cum_south_1[:,0] = accus_south_1[:,0]
for column in range(1, len(accus_south_1[0])):
    accus_cum_south_1[:,column] = accus_cum_south_1[:,column-1] + accus_south_1[:,column]
accus_cum_south_1 = accus_cum_south_1 * (5/60)

accus_cum_south_2 = np.zeros((len(accus_south_2), len(accus_south_2[0])))
accus_cum_south_2[:,0] = accus_south_2[:,0]
for column in range(1, len(accus_south_2[0])):
    accus_cum_south_2[:,column] = accus_cum_south_2[:,column-1] + accus_south_2[:,column]
accus_cum_south_2 = accus_cum_south_2 * (5/60)

accus_cum_south_3 = np.zeros((len(accus_south_3), len(accus_south_3[0])))
accus_cum_south_3[:,0] = accus_south_3[:,0]
for column in range(1, len(accus_south_3[0])):
    accus_cum_south_3[:,column] = accus_cum_south_3[:,column-1] + accus_south_3[:,column]
accus_cum_south_3 = accus_cum_south_3 * (5/60)

##############################################################################
#STATS OF LAST COLUMNS

#GRID
np.max(accus_cum_grid[:,-1])
np.min(accus_cum_grid[:,-1])
np.mean(accus_cum_grid[:,-1])
np.median(accus_cum_grid[:,-1])
np.std(accus_cum_grid[:,-1])
np.std(accus_cum_grid[:,-1]) / np.mean(accus_cum_grid[:,-1])

#BASIN
np.max(accus_cum_basin[:,-1])
np.min(accus_cum_basin[:,-1])
np.mean(accus_cum_basin[:,-1])
np.median(accus_cum_basin[:,-1])
np.std(accus_cum_basin[:,-1])
np.std(accus_cum_basin[:,-1]) / np.mean(accus_cum_basin[:,-1])

#NORTH
np.max(accus_cum_north_1[:,-1])
np.min(accus_cum_north_1[:,-1])
np.mean(accus_cum_north_1[:,-1])
np.median(accus_cum_north_1[:,-1])
np.std(accus_cum_north_1[:,-1])
np.std(accus_cum_north_1[:,-1]) / np.mean(accus_cum_north_1[:,-1])

np.max(accus_cum_north_2[:,-1])
np.min(accus_cum_north_2[:,-1])
np.mean(accus_cum_north_2[:,-1])
np.median(accus_cum_north_2[:,-1])
np.std(accus_cum_north_2[:,-1])
np.std(accus_cum_north_2[:,-1]) / np.mean(accus_cum_north_2[:,-1])

np.max(accus_cum_north_3[:,-1])
np.min(accus_cum_north_3[:,-1])
np.mean(accus_cum_north_3[:,-1])
np.median(accus_cum_north_3[:,-1])
np.std(accus_cum_north_3[:,-1])
np.std(accus_cum_north_3[:,-1]) / np.mean(accus_cum_north_3[:,-1])

#WEST
np.max(accus_cum_west_1[:,-1])
np.min(accus_cum_west_1[:,-1])
np.mean(accus_cum_west_1[:,-1])
np.median(accus_cum_west_1[:,-1])
np.std(accus_cum_west_1[:,-1])
np.std(accus_cum_west_1[:,-1]) / np.mean(accus_cum_west_1[:,-1])

np.max(accus_cum_west_2[:,-1])
np.min(accus_cum_west_2[:,-1])
np.mean(accus_cum_west_2[:,-1])
np.median(accus_cum_west_2[:,-1])
np.std(accus_cum_west_2[:,-1])
np.std(accus_cum_west_2[:,-1]) / np.mean(accus_cum_west_2[:,-1])

np.max(accus_cum_west_3[:,-1])
np.min(accus_cum_west_3[:,-1])
np.mean(accus_cum_west_3[:,-1])
np.median(accus_cum_west_3[:,-1])
np.std(accus_cum_west_3[:,-1])
np.std(accus_cum_west_3[:,-1]) / np.mean(accus_cum_west_3[:,-1])

#SOUTH
np.max(accus_cum_south_1[:,-1])
np.min(accus_cum_south_1[:,-1])
np.mean(accus_cum_south_1[:,-1])
np.median(accus_cum_south_1[:,-1])
np.std(accus_cum_south_1[:,-1])
np.std(accus_cum_south_1[:,-1]) / np.mean(accus_cum_south_1[:,-1])

np.max(accus_cum_south_2[:,-1])
np.min(accus_cum_south_2[:,-1])
np.mean(accus_cum_south_2[:,-1])
np.median(accus_cum_south_2[:,-1])
np.std(accus_cum_south_2[:,-1])
np.std(accus_cum_south_2[:,-1]) / np.mean(accus_cum_south_2[:,-1])

np.max(accus_cum_south_3[:,-1])
np.min(accus_cum_south_3[:,-1])
np.mean(accus_cum_south_3[:,-1])
np.median(accus_cum_south_3[:,-1])
np.std(accus_cum_south_3[:,-1])
np.std(accus_cum_south_3[:,-1]) / np.mean(accus_cum_south_3[:,-1])

##############################################################################
# ACCU PLOTS
# timeseries_size = 141 #event1
timeseries_size = 115 #event3
timestep = 5

accu_cum_1_2smallest_mean = np.zeros((1,timeseries_size))
accu_cum_2_2smallest_mean = np.zeros((1,timeseries_size))
accu_cum_3_test_2smallest_mean = np.zeros((1,timeseries_size))
for k in range(accus_cum_south_1.shape[1]):
    accu_cum_1_2smallest_mean[0,k] = np.mean(accus_cum_south_1[:,k])
    accu_cum_2_2smallest_mean[0,k] = np.mean(accus_cum_south_2[:,k])
    accu_cum_3_test_2smallest_mean[0,k] = np.mean(accus_cum_south_3[:,k])

accu_cum_1_2smallest_median = np.zeros((1,timeseries_size))
accu_cum_2_2smallest_median = np.zeros((1,timeseries_size))
accu_cum_3_test_2smallest_median = np.zeros((1,timeseries_size))
for k in range(accus_cum_south_1.shape[1]):
    accu_cum_1_2smallest_median[0,k] = np.median(accus_cum_south_1[:,k])
    accu_cum_2_2smallest_median[0,k] = np.median(accus_cum_south_2[:,k])
    accu_cum_3_test_2smallest_median[0,k] = np.median(accus_cum_south_3[:,k])

#Function to find closest value in array to a given value
def find_closest(arr, val):
       idx = np.abs(arr - val).argmin()
       return arr[idx]

#WHOLE GRID
#accus_cum_grid
accu_cum_whole_grid_mean = np.zeros((1,timeseries_size))
for k in range(accus_cum_grid.shape[1]):
    accu_cum_whole_grid_mean[0,k] = np.mean(accus_cum_grid[:,k])
#plot
plt.figure()
for l in range(accus_cum_grid.shape[0]):
    if l == 0:
        plt.plot(accus_cum_grid[l], color="grey", alpha=0.1, label="ensemble members")
    else:
        plt.plot(accus_cum_grid[l], color="grey", alpha=0.1)
plt.plot(accu_cum_whole_grid_mean[0], color="red", label="ensemble mean")
plt.title("Whole radar grid")
plt.xlabel("Time [min]")
plt.ylabel("Cumulative rainfall [mm]")
plt.ylim((0,90))
plt.yticks(np.arange(0,91,10))
# plt.ylim((0,130))
# plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
plt.legend()
plt.savefig(os.path.join(data_dir_location, "Final_fig_0_grid_alpha2.png"))

plt.savefig(os.path.join(out_figs,"figure_3b.pdf"), format="pdf", bbox_inches="tight")

#########################################
#WHOLE BASIN
#accus_cum_basin
accu_cum_whole_basin_mean = np.zeros((1,timeseries_size))
for k in range(accus_cum_basin.shape[1]):
    accu_cum_whole_basin_mean[0,k] = np.mean(accus_cum_basin[:,k])

#plot
plt.figure()
for m in range(accus_cum_basin.shape[0]):
    if m == 0:
        plt.plot(accus_cum_basin[m], color="grey", alpha=0.1, label="ensemble members")
    else:
        plt.plot(accus_cum_basin[m], color="grey", alpha=0.1)
plt.plot(accu_cum_whole_basin_mean[0], color="red", label="ensemble mean")
plt.title("Kokemaki basin")
plt.xlabel("Time [min]")
plt.ylabel("Cumulative rainfall [mm]")
plt.ylim((0,90))
plt.yticks(np.arange(0,91,10))
# plt.ylim((0,130))
# plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
plt.legend()
plt.savefig(os.path.join(data_dir_location, "Final_fig_1_basin_alpha2.png"))

plt.savefig(os.path.join(out_figs,"figure_3d.pdf"), format="pdf", bbox_inches="tight")

#########################################
#1ST LVL
#accus_cum_south_1
plt.figure()
for t in range(accus_cum_south_1.shape[0]):
    if t == 0:
        plt.plot((accus_cum_south_1[t]), color="grey", alpha=0.1, label="ensemble members")
    else:
        plt.plot((accus_cum_south_1[t]), color="grey", alpha=0.1)
plt.plot(accu_cum_1_2smallest_mean[0], color="red", label="ensemble mean")
plt.plot((accus_cum_south_1[np.argmax(accus_cum_south_1[:,-1], axis=0)]), color="blue", label="ensemble max")
plt.plot((accus_cum_south_1[int(np.where(accus_cum_south_1[:,-1] == find_closest(accus_cum_south_1[:,-1], accu_cum_1_2smallest_median[0,-1]))[0])]), color="orange", label="ensemble median")
plt.title("1st delineation level")
plt.xlabel("Time [min]")
plt.ylabel("Cumulative rainfall [mm]")
plt.ylim((0,90))
plt.yticks(np.arange(0,91,10))
# plt.ylim((0,130))
# plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
plt.legend()
plt.savefig(os.path.join(data_dir_location, "Final_fig_2_lvl1_alpha2.png"))

plt.savefig(os.path.join(out_figs,"figure_3f.pdf"), format="pdf", bbox_inches="tight")

#########################################
#2ND LVL
#accus_cum_south_2
plt.figure()
for t in range(accus_cum_south_2.shape[0]):
    if t == 0:
        plt.plot((accus_cum_south_2[t]), color="grey", alpha=0.1, label="ensemble members")
    else:
        plt.plot((accus_cum_south_2[t]), color="grey", alpha=0.1)
plt.plot(accu_cum_2_2smallest_mean[0], color="red", label="ensemble mean")
plt.plot((accus_cum_south_2[np.argmax(accus_cum_south_2[:,-1], axis=0)]), color="blue", label="ensemble max")
plt.plot((accus_cum_south_2[int(np.where(accus_cum_south_2[:,-1] == find_closest(accus_cum_south_2[:,-1], accu_cum_2_2smallest_median[0,-1]))[0])]), color="orange", label="ensemble median")
plt.title("2nd delineation level")
plt.xlabel("Time [min]")
plt.ylabel("Cumulative rainfall [mm]")
plt.ylim((0,90))
plt.yticks(np.arange(0,91,10))
# plt.ylim((0,130))
# plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
plt.legend()
plt.savefig(os.path.join(data_dir_location, "Final_fig_2_lvl2_alpha2.png"))

plt.savefig(os.path.join(out_figs,"figure_3h.pdf"), format="pdf", bbox_inches="tight")

#########################################
#3RD LVL
#accus_cum_south_3
plt.figure()
for t in range(accus_cum_south_3.shape[0]):
    if t == 0:
        plt.plot((accus_cum_south_3[t]), color="grey", alpha=0.1, label="ensemble members")
    else:
        plt.plot((accus_cum_south_3[t]), color="grey", alpha=0.1)
plt.plot(accu_cum_3_test_2smallest_mean[0], color="red", label="ensemble mean")
plt.plot((accus_cum_south_3[np.argmax(accus_cum_south_3[:,-1], axis=0)]), color="blue", label="ensemble max")
plt.plot((accus_cum_south_3[int(np.where(accus_cum_south_3[:,-1] == find_closest(accus_cum_south_3[:,-1], accu_cum_3_test_2smallest_median[0,-1]))[0])]), color="orange", label="ensemble median")
plt.title("3rd delineation level")
plt.xlabel("Time [min]")
plt.ylabel("Cumulative rainfall [mm]")
plt.ylim((0,90))
plt.yticks(np.arange(0,91,10))
# plt.ylim((0,130))
# plt.yticks(np.arange(0,131,10))
plt.grid(True, which='major', axis='y')
plt.legend()
plt.savefig(os.path.join(data_dir_location, "Final_fig_2_lvl3_alpha2.png"))

plt.savefig(os.path.join(out_figs,"figure_3j.pdf"), format="pdf", bbox_inches="tight")
