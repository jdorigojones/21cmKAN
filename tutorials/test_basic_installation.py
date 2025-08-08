#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import h5py
import torch
import numpy as np
from matplotlib import pyplot as plt
import Global21cmKAN as Global21cmKAN
from Global21cmKAN.emulate_21cmGEM import error

# create and load 21cmKAN emulator instance already trained on the 21cmGEM data set
emulator_21cmGEM = Global21cmKAN.emulate_21cmGEM.Emulate()
emulator_21cmGEM.load_model()

# print the accuracy metrics when evaluated on the 21cmGEM test set
# the mean and max relative RMSE values match those reported at the end of Section 3.3.1 of DJ+25,
# if you haven't yet saved over the trained models included in the GitHub repository
test_rel_RMSE_values_21cmGEM = emulator_21cmGEM.test_error()
test_mean_rel_RMSE_21cmGEM = np.mean(test_rel_RMSE_values_21cmGEM)
test_max_rel_RMSE_21cmGEM = np.max(test_rel_RMSE_values_21cmGEM)
print(f"Mean relative RMSE of 21cmKAN trained and tested on 21cmGEM (%): {test_mean_rel_RMSE_21cmGEM}")
print(f"Max relative RMSE of 21cmKAN trained and tested on 21cmGEM (%): {test_max_rel_RMSE_21cmGEM}")

# choose an example global 21 cm signal from the 21cmGEM test set to emulate using 21cmKAN
params_21cmGEM_test = emulator_21cmGEM.par_test.copy()
signals_21cmGEM_test = emulator_21cmGEM.signal_test.copy()
n=1000
params_21cmGEM = params_21cmGEM_test[n]
signal_21cmGEM = signals_21cmGEM_test[n]
signal_21cmKAN = emulator_21cmGEM.predict(params_21cmGEM)
print('physical parameter values of example 21cmGEM signal:', params_21cmGEM)
print('relative rms error between "true" 21cmGEM and emulated 21cmKAN example signal:', error(signal_21cmGEM, signal_21cmKAN), '%')
print('absolute rms error between "true" 21cmGEM and emulated 21cmKAN example signal:', error(signal_21cmGEM, signal_21cmKAN, relative=False), 'mK')

vr = 1420.405751
def freq(zs):
    return vr/(zs+1)

def redshift(v):
    return (vr/v)-1

z_list = emulator_21cmGEM.redshifts
nu_list = freq(z_list)

# plot the true and emulated signals together
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(nu_list, signal_21cmKAN, color='blue', linewidth=8, alpha=0.4, label='emulated signal')
ax.plot(nu_list, signal_21cmGEM, color='k', linestyle='dotted', linewidth=3, label='true 21cmGEM signal')
ax.set_xlabel(r'$\nu$ (MHz)', fontsize=32)
ax.set_ylabel(r'$\delta T_b$ (mK)', fontsize=32)
ax.set_xlim(27.85, 236.74)
ax.set_ylim(-300, 50)
ax.set_yticks([50, 0,-50,-100,-150,-200,-250,-300])
ax.set_yticklabels(['50', '0','-50','-100','-150','-200','-250','-300'], fontsize=30)
ax.set_xticks([40, 60, 80, 100, 120, 140, 160, 180, 200, 220])
ax.set_xticklabels(['40', '60', '80', '100', '120', '140', '160', '180', '200', '220'], fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30, width=2, length=10)
ax.minorticks_on()
ax.tick_params(axis='both', which='minor', width=2, length=5)
secax = ax.secondary_xaxis('top', functions=(redshift, freq))
secax.set_xlabel(r'$z$', fontsize=32)
secax.set_xticks([5, 10, 15, 20, 30, 50])
secax.set_xticklabels(['5', '10', '15', '20', '30', '50'], fontsize=30)
secax.tick_params(which='major', direction='out', width=2, length=10, labelsize=30)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('true_and_emulated_signals_21cmKAN_21cmGEM.jpg', dpi=300, bbox_inches='tight', facecolor='w')
plt.show()

### now do the same for 21cmKAN trained and tested on the ARES data set !

emulator_ARES = Global21cmKAN.emulate_ARES.Emulate()
emulator_ARES.load_model()
test_rel_RMSE_values_ARES = emulator_ARES.test_error()
test_mean_rel_RMSE_ARES = np.mean(test_rel_RMSE_values_ARES)
test_max_rel_RMSE_ARES = np.max(test_rel_RMSE_values_ARES)
print(f"Mean relative RMSE of 21cmKAN trained and tested on ARES (%): {test_mean_rel_RMSE_ARES}")
print(f"Max relative RMSE of 21cmKAN trained and tested on ARES (%): {test_max_rel_RMSE_ARES}")

params_ARES_test = emulator_ARES.par_test.copy()
signals_ARES_test = emulator_ARES.signal_test.copy()
params_ARES = params_ARES_test[n]
signal_ARES = signals_ARES_test[n]
signal_21cmKAN = emulator_ARES.predict(params_ARES)
print('physical parameter values of example ARES signal:', params_ARES)
print('relative rms error between "true" ARES and emulated 21cmKAN example signal:', error(signal_ARES, signal_21cmKAN), '%')
print('absolute rms error between "true" ARES and emulated 21cmKAN example signal:', error(signal_ARES, signal_21cmKAN, relative=False), 'mK')

z_list = emulator_ARES.redshifts
nu_list = freq(z_list)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(nu_list, signal_21cmKAN, color='blue', linewidth=8, alpha=0.4, label='emulated signal')
ax.plot(nu_list, signal_ARES, color='k', linestyle='dotted', linewidth=3, label='true ARES signal')
ax.set_xlabel(r'$\nu$ (MHz)', fontsize=32)
ax.set_ylabel(r'$\delta T_b$ (mK)', fontsize=32)
ax.set_xlim(27.85, 236.74)
ax.set_ylim(-300, 50)
ax.set_yticks([50, 0,-50,-100,-150,-200,-250,-300])
ax.set_yticklabels(['50', '0','-50','-100','-150','-200','-250','-300'], fontsize=30)
ax.set_xticks([40, 60, 80, 100, 120, 140, 160, 180, 200, 220])
ax.set_xticklabels(['40', '60', '80', '100', '120', '140', '160', '180', '200', '220'], fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30, width=2, length=10)
ax.minorticks_on()
ax.tick_params(axis='both', which='minor', width=2, length=5)
secax = ax.secondary_xaxis('top', functions=(redshift, freq))
secax.set_xlabel(r'$z$', fontsize=32)
secax.set_xticks([5, 10, 15, 20, 30, 50])
secax.set_xticklabels(['5', '10', '15', '20', '30', '50'], fontsize=30)
secax.tick_params(which='major', direction='out', width=2, length=10, labelsize=30)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('true_and_emulated_signals_21cmKAN_ARES.jpg', dpi=300, bbox_inches='tight', facecolor='w')
plt.show()

print('your 21cmKAN basic installation works!')
print('look at the downloaded plots to see example signal emulations of the two popular physical models, 21cmGEM and ARES')

