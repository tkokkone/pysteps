# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:09:11 2023

@author: lindgrv1
"""

import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import rasterio
import pandas as pd

##############################################################################

in_obs = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"
in_sim_01 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_72842_77480_40556_40267"
in_sim_02 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_99237_22095_19605_30922"
in_sim_03 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_59217_75605_13981_41521"
in_sim_04 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_49488_2468_79787_88992"
in_sim_05 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_62574_1193_50312_52850"
in_sim_06 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_80919_91708_30850_5673"
in_sim_07 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_87050_71574_14228_59240"
in_sim_08 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_85948_70692_4841_85604"
in_sim_09 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_55056_1737_87845_85735"
in_sim_10 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_54248_17249_37999_22960"

out_dir = os.path.join(in_obs, "Comparing_accumulations")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

##############################################################################

accu_obs = []
src_accu_obs = rasterio.open(os.path.join(in_obs, "accumulation_raster_spectral_observed.tif"))
array_accu_obs = src_accu_obs.read(1)
accu_obs.append(array_accu_obs) 
accu_obs = np.concatenate([accu_obs_[None, :, :] for accu_obs_ in accu_obs])

# plt.figure()
# plt.imshow(accu_obs[0])

accu_sim_01 = []
src_accu_sim_01 = rasterio.open(os.path.join(in_sim_01, "accumulation_raster_spectral.tif"))
array_accu_sim_01 = src_accu_sim_01.read(1)
accu_sim_01.append(array_accu_sim_01) 
accu_sim_01 = np.concatenate([accu_sim_01_[None, :, :] for accu_sim_01_ in accu_sim_01])

accu_sim_02 = []
src_accu_sim_02 = rasterio.open(os.path.join(in_sim_02, "accumulation_raster_spectral.tif"))
array_accu_sim_02 = src_accu_sim_02.read(1)
accu_sim_02.append(array_accu_sim_02) 
accu_sim_02 = np.concatenate([accu_sim_02_[None, :, :] for accu_sim_02_ in accu_sim_02])

accu_sim_03 = []
src_accu_sim_03 = rasterio.open(os.path.join(in_sim_03, "accumulation_raster_spectral.tif"))
array_accu_sim_03 = src_accu_sim_03.read(1)
accu_sim_03.append(array_accu_sim_03) 
accu_sim_03 = np.concatenate([accu_sim_03_[None, :, :] for accu_sim_03_ in accu_sim_03])

accu_sim_04 = []
src_accu_sim_04 = rasterio.open(os.path.join(in_sim_04, "accumulation_raster_spectral.tif"))
array_accu_sim_04 = src_accu_sim_04.read(1)
accu_sim_04.append(array_accu_sim_04) 
accu_sim_04 = np.concatenate([accu_sim_04_[None, :, :] for accu_sim_04_ in accu_sim_04])

accu_sim_05 = []
src_accu_sim_05 = rasterio.open(os.path.join(in_sim_05, "accumulation_raster_spectral.tif"))
array_accu_sim_05 = src_accu_sim_05.read(1)
accu_sim_05.append(array_accu_sim_05) 
accu_sim_05 = np.concatenate([accu_sim_05_[None, :, :] for accu_sim_05_ in accu_sim_05])

accu_sim_06 = []
src_accu_sim_06 = rasterio.open(os.path.join(in_sim_06, "accumulation_raster_spectral.tif"))
array_accu_sim_06 = src_accu_sim_06.read(1)
accu_sim_06.append(array_accu_sim_06) 
accu_sim_06 = np.concatenate([accu_sim_06_[None, :, :] for accu_sim_06_ in accu_sim_06])

accu_sim_07 = []
src_accu_sim_07 = rasterio.open(os.path.join(in_sim_07, "accumulation_raster_spectral.tif"))
array_accu_sim_07 = src_accu_sim_07.read(1)
accu_sim_07.append(array_accu_sim_07) 
accu_sim_07 = np.concatenate([accu_sim_07_[None, :, :] for accu_sim_07_ in accu_sim_07])

accu_sim_08 = []
src_accu_sim_08 = rasterio.open(os.path.join(in_sim_08, "accumulation_raster_spectral.tif"))
array_accu_sim_08 = src_accu_sim_08.read(1)
accu_sim_08.append(array_accu_sim_08) 
accu_sim_08 = np.concatenate([accu_sim_08_[None, :, :] for accu_sim_08_ in accu_sim_08])

accu_sim_09 = []
src_accu_sim_09 = rasterio.open(os.path.join(in_sim_09, "accumulation_raster_spectral.tif"))
array_accu_sim_09 = src_accu_sim_09.read(1)
accu_sim_09.append(array_accu_sim_09) 
accu_sim_09 = np.concatenate([accu_sim_09_[None, :, :] for accu_sim_09_ in accu_sim_09])

accu_sim_10 = []
src_accu_sim_10 = rasterio.open(os.path.join(in_sim_10, "accumulation_raster_spectral.tif"))
array_accu_sim_10 = src_accu_sim_10.read(1)
accu_sim_10.append(array_accu_sim_10) 
accu_sim_10 = np.concatenate([accu_sim_10_[None, :, :] for accu_sim_10_ in accu_sim_10])

##############################################################################

scale_max = max(np.max(accu_obs[0]),np.max(accu_sim_01[0]),np.max(accu_sim_02[0]),np.max(accu_sim_03[0]),np.max(accu_sim_04[0]),np.max(accu_sim_05[0]),np.max(accu_sim_06[0]),np.max(accu_sim_07[0]),np.max(accu_sim_08[0]),np.max(accu_sim_09[0]),np.max(accu_sim_10[0]))
scale_max = round(scale_max+7)
scale_min = 0

plt.figure()
plt.imshow(accu_obs[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_obs.png"))

plt.figure()
plt.imshow(accu_sim_01[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_sim_01.png"))

plt.figure()
plt.imshow(accu_sim_02[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_sim_02.png"))

plt.figure()
plt.imshow(accu_sim_03[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_sim_03.png"))

plt.figure()
plt.imshow(accu_sim_04[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_sim_04.png"))

plt.figure()
plt.imshow(accu_sim_05[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_sim_05.png"))

plt.figure()
plt.imshow(accu_sim_06[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_sim_06.png"))

plt.figure()
plt.imshow(accu_sim_07[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_sim_07.png"))

plt.figure()
plt.imshow(accu_sim_08[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_sim_08.png"))

plt.figure()
plt.imshow(accu_sim_09[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_sim_09.png"))

plt.figure()
plt.imshow(accu_sim_10[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "1060_accu_sim_10.png"))

##############################################################################

scale_max = 850

plt.figure()
plt.imshow(accu_obs[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_obs.png"))

plt.figure()
plt.imshow(accu_sim_01[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_sim_01.png"))

plt.figure()
plt.imshow(accu_sim_02[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_sim_02.png"))

plt.figure()
plt.imshow(accu_sim_03[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_sim_03.png"))

plt.figure()
plt.imshow(accu_sim_04[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_sim_04.png"))

plt.figure()
plt.imshow(accu_sim_05[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_sim_05.png"))

plt.figure()
plt.imshow(accu_sim_06[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_sim_06.png"))

plt.figure()
plt.imshow(accu_sim_07[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_sim_07.png"))

plt.figure()
plt.imshow(accu_sim_08[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_sim_08.png"))

plt.figure()
plt.imshow(accu_sim_09[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_sim_09.png"))

plt.figure()
plt.imshow(accu_sim_10[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "850_accu_sim_10.png"))

##############################################################################

scale_max = 650

plt.figure()
plt.imshow(accu_obs[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_obs.png"))

plt.figure()
plt.imshow(accu_sim_01[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_sim_01.png"))

plt.figure()
plt.imshow(accu_sim_02[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_sim_02.png"))

plt.figure()
plt.imshow(accu_sim_03[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_sim_03.png"))

plt.figure()
plt.imshow(accu_sim_04[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_sim_04.png"))

plt.figure()
plt.imshow(accu_sim_05[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_sim_05.png"))

plt.figure()
plt.imshow(accu_sim_06[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_sim_06.png"))

plt.figure()
plt.imshow(accu_sim_07[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_sim_07.png"))

plt.figure()
plt.imshow(accu_sim_08[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_sim_08.png"))

plt.figure()
plt.imshow(accu_sim_09[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_sim_09.png"))

plt.figure()
plt.imshow(accu_sim_10[0], cmap ="nipy_spectral", vmin=scale_min, vmax=scale_max)
plt.colorbar()
plt.savefig(os.path.join(out_dir, "650_accu_sim_10.png"))

##############################################################################

in_cum_accus = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole"

cum_accucs_df = pd.read_csv(os.path.join(in_cum_accus, "comparing_point_tss.csv"), delimiter=(";"), header=None)
cum_accucs_df.drop(columns=cum_accucs_df.columns[0], axis=1, inplace=True)

cum_accucs = cum_accucs_df.to_numpy()

plt.figure()
for i in range(9,cum_accucs.shape[0]):
    plt.plot((cum_accucs[i]*(5/60)), color="grey", alpha=0.2)
for j in range(0,9):
    plt.plot((cum_accucs[j]*(5/60)))

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

dir_sim_turn = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_55056_1737_87845_85735_turn\Event_tiffs_mmh"
dir_sim_slow = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_55056_1737_87845_85735_slow\Event_tiffs_mmh"
dir_sim_slow2 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_55056_1737_87845_85735_slow2\Event_tiffs_mmh"

out_dir_sim_turn = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_55056_1737_87845_85735_turn"
out_dir_sim_slow = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_55056_1737_87845_85735_slow"
out_dir_sim_slow2 = r"W:\lindgrv1\Simuloinnit\Simulations_kiira_whole\Simulations\Simulation_55056_1737_87845_85735_slow2"

save_figs = False
csv_locations = False

in_dir = dir_sim_slow2

#Import simulated event in raster-format and convert it into array
event_sim_array  = []
for i in range(0, 87):
    temp_raster = rasterio.open(os.path.join(in_dir, f"test_{i}.tif"))
    temp_array = temp_raster.read(1)
    event_sim_array.append(temp_array)
    if i == 0:
        event_affine = temp_raster.transform  
event_sim_array = np.concatenate([event_sim_array_[None, :, :] for event_sim_array_ in event_sim_array])

# R_sim = event_sim_turn
# spots_raster = np.zeros((R_sim.shape[1], R_sim.shape[2]))
# spots_raster[int(R_sim.shape[1]/4), int(R_sim.shape[2]/4)] = 1 #upper-left: 256, 256
# spots_raster[int(R_sim.shape[1]/4), int(R_sim.shape[2]/2)] = 2 #upper_middle = 256, 512
# spots_raster[int(R_sim.shape[1]/4), int(R_sim.shape[2]/4*3)] = 3 #upper-right: 256, 768
# spots_raster[int(R_sim.shape[1]/2), int(R_sim.shape[2]/4)] = 4 #middle-left: 512, 256
# spots_raster[int(R_sim.shape[1]/2), int(R_sim.shape[2]/2)] = 5 #center: 512, 512
# spots_raster[int(R_sim.shape[1]/2), int(R_sim.shape[2]/4*3)] = 6 #middle-right: 512, 768
# spots_raster[int(R_sim.shape[1]/4*3), int(R_sim.shape[2]/4)] = 7 #lower-left: 768, 256
# spots_raster[int(R_sim.shape[1]/4*3), int(R_sim.shape[2]/2)] = 8 #lower-middle: 768, 512
# spots_raster[int(R_sim.shape[1]/4*3), int(R_sim.shape[2]/4*3)] = 9 #lower-right: 768, 768

ts_up_left = np.zeros((1, len(event_sim_array)))
ts_up_mid = np.zeros((1, len(event_sim_array)))
ts_up_right = np.zeros((1, len(event_sim_array)))
ts_mid_left = np.zeros((1, len(event_sim_array)))
ts_mid_mid = np.zeros((1, len(event_sim_array)))
ts_mid_right = np.zeros((1, len(event_sim_array)))
ts_low_left = np.zeros((1, len(event_sim_array)))
ts_low_mid = np.zeros((1, len(event_sim_array)))
ts_low_right = np.zeros((1, len(event_sim_array)))

for i in range(len(event_sim_array)):
    ts_up_left[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/4)]
    ts_up_mid[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/2)]
    ts_up_right[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4), int(event_sim_array.shape[2]/4*3)]
    ts_mid_left[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/4)]
    ts_mid_mid[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/2)]
    ts_mid_right[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/2), int(event_sim_array.shape[2]/4*3)]
    ts_low_left[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/4)]
    ts_low_mid[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/2)]
    ts_low_right[0,i] = event_sim_array[i, int(event_sim_array.shape[1]/4*3), int(event_sim_array.shape[2]/4*3)]

limit_upper = round(np.max((ts_up_left, ts_up_mid, ts_up_right, 
                            ts_mid_left, ts_mid_mid, ts_mid_right,
                            ts_low_left, ts_low_mid, ts_low_right))+0.5)
limit_lower = 0

out_dir = out_dir_sim_slow2

plt.figure()
plt.plot(ts_up_left[0])
plt.plot(ts_up_mid[0])
plt.plot(ts_up_right[0])
plt.plot(ts_mid_left[0])
plt.plot(ts_mid_mid[0])
plt.plot(ts_mid_right[0])
plt.plot(ts_low_left[0])
plt.plot(ts_low_mid[0])
plt.plot(ts_low_right[0])
plt.ylim(limit_lower, limit_upper)
if save_figs:
    plt.savefig(os.path.join(out_dir, "tss_point_mmh.png"))

if csv_locations:
    data_temp = [ts_up_left[0], ts_up_mid[0], ts_up_right[0], ts_mid_left[0], ts_mid_mid[0], ts_mid_right[0], ts_low_left[0], ts_low_mid[0], ts_low_right[0]]
    mmh_ts= pd.DataFrame(data_temp, index=['ts_up_left', 'ts_up_mid', 'ts_up_right', 'ts_mid_left', 'ts_mid_mid', 'ts_mid_right', 'ts_low_left', 'ts_low_mid', 'ts_low_right'])
    pd.DataFrame(mmh_ts).to_csv(os.path.join(out_dir, "tss_point_mmh.csv"))