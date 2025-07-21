#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import interpolate
import globalemu
from globalemu.preprocess import process
from globalemu.network import nn
from globalemu.eval import evaluate

rc('figure', figsize=(6.5, 5.0))
plt.rcParams['mathtext.fontset'] = 'cm'

redshifts = np.linspace(5, 50, 451)
print('number of points in each curve:', len(redshifts))
vr = 1420.405751 # rest-frame frequency (MHz) of the 21-cm hydrogen line
frequencies = np.array([vr/(z+1) for z in redshifts])

data_dir_testing = 'downloaded_data_21cmGEM/' # define directory containing the training and testing data to be processed
base_dir_testing = 'results_21cmGEM_4/' # define directory where the preprocessed data and later the trained models will be placed.
# This should be thought of as the working directory as it will be needed when training a model and making evaluations of trained models

num = 'full' # number of models in the training set that will be used to train globalemu
log_list = [0,1,2]
# logs=[]: The indices corresponding to the astrophysical parameters in “train_data.txt” that need to be logged. 
# logging the following 3 parameters in 21cmGEM: f_*, V_c, f_X

# first test with AFB and resampling OFF
process(num, redshifts, base_dir=base_dir_testing, data_location=data_dir_testing, AFB=True, std_division=True, resampling=True, logs=log_list)

# train globalemu; NOTE THAT GLOBALEMU STILL IMPROPERLY USES THE TEST SET AS THE VALIDATION SET, WHICH LEADS TO BETTER TEST SET ACCURACY
size=len(redshifts)
nn(batch_size=size, epochs=100, input_shape=8, layer_sizes=[32,32,32], base_dir=base_dir_testing, early_stop=True)

# import test data used to evaluate the trained network
test_data = np.loadtxt(data_dir_testing + 'test_data.txt')
test_labels = np.loadtxt(data_dir_testing + 'test_labels.txt')

# comapre all test signals and their emulations
input_params = test_data[:, :]
true_signal = test_labels[:, :]

# evaluate() is used to make an evaluation of a trained instance of globalemu
predictor_testing = evaluate(base_dir=base_dir_testing, logs=log_list, z=redshifts)
# base_dir: The base_dir is where the trained model is saved.
# logs: The indices corresponding to the parameters that were logged during training.

# Once the class has been initialised you can then make evaluations of the emulator by passing the parameters like so:
signal_testing, z_testing = predictor_testing(input_params)

def freq(z):
    return vr/(z+1)

def redshift(v):
    return (vr/v)-1

fig, ax = plt.subplots(constrained_layout=True)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction = 'out', width = 2, length = 10, labelsize=20)
ax.tick_params(axis='both', which='minor', direction = 'out', width = 2, length = 5, labelsize=20)
ax.set_yticks([50, 0,-50,-100,-150,-200,-250,-300])
ax.set_yticklabels(['50', '0','-50','-100','-150','-200','-250','-300'])
ax.set_xticks([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
ax.set_xticklabels(['20', '40', '60', '80', '100', '120', '140', '160', '180', '200'])
ax.set_ylabel(r'$\delta T_b$ (mK)', fontsize=20, fontname= 'Baskerville')
ax.set_xlabel(r'$\nu$ (MHz)', fontsize=20, fontname= 'Baskerville')
ax.set_ylim(-300,50)
ax.set_xlim(27.85,236.74)

secax = ax.secondary_xaxis('top', functions=(redshift, freq))
secax.set_xlabel(r'$z$', fontsize=20, fontname= 'Baskerville')
secax.set_xticks([5, 10, 15, 20, 30, 50])
secax.set_xticklabels(['5', '10', '15', '20', '30', '50'], fontsize=20, fontname= 'Baskerville')
secax.tick_params(which='major', direction = 'out', width = 2, length = 10, labelsize=20)
i=0
for i in range(len(true_signal)):
    if i==0:
        ax.plot(freq(z_testing), true_signal[i, :], c='k', ls='--', alpha=0.1, label="true signal")
        ax.plot(freq(z_testing), signal_testing[i, :], c='r', alpha=0.1, label='emulated signal')
    else:
        ax.plot(freq(z_testing), true_signal[i, :], c='k', ls='--', alpha=0.1)
        ax.plot(freq(z_testing), signal_testing[i, :], c='r', alpha=0.1)
#ax.legend(fontsize=15, loc='lower left')
plt.savefig('globalemu_training_testing_21cmGEM_4.png', dpi = 300, bbox_inches='tight', facecolor='w')
plt.show()

fig, ax = plt.subplots(constrained_layout=True)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction = 'out', width = 2, length = 10, labelsize=20)
ax.tick_params(axis='both', which='minor', direction = 'out', width = 2, length = 5, labelsize=20)
ax.set_xticks([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
ax.set_xticklabels(['20', '40', '60', '80', '100', '120', '140', '160', '180', '200'])
ax.set_ylabel(r'|$\delta T_{\rm true}$ - $\delta T_{\rm emulated}$| [mK]', fontsize=20, fontname= 'Baskerville')
ax.set_xlabel(r'$\nu$ [MHz]', fontsize=20, fontname= 'Baskerville')
ax.set_xlim(25,203)
secax = ax.secondary_xaxis('top', functions=(redshift, freq))
secax.set_xlabel(r'$z$', fontsize=20, fontname= 'Baskerville')
secax.set_xticks([7, 10, 12, 15, 20, 30, 40, 55])
secax.set_xticklabels(['7', '10', '12', '15', '20', '30', '40', '55'], fontsize=20, fontname= 'Baskerville')
secax.tick_params(which='major', direction = 'out', width = 2, length = 10, labelsize=20)
i=0
residuals_list = []
rmse_list = []
for i in range(len(true_signal)):
    #rmse = np.sqrt(np.sum((true_signal_8paramlarge_no_constraints_combined[i, :] - signal_ares_8paramlarge_no_constraints_combined[i, :])**2)/len(true_signal_8paramlarge_no_constraints_combined))    
    residual = np.abs(true_signal[i, :] - signal_testing[i, :])
    rmse = np.sqrt(np.mean(residual**2))  
    residuals_list.append(residual)
    rmse_list.append(rmse)
    ax.plot(freq(z_testing), residual, c='k')
#ax.legend(fontsize=15, loc='lower right')
plt.savefig('globalemu_training_testing_residuals_21cmGEM_4.png', dpi = 300, bbox_inches='tight', facecolor='w')
plt.show()

err = np.sqrt(np.mean((signal_testing - true_signal)**2, axis=1))
err /= np.max(np.abs(true_signal), axis=1)
err *= 100
err_abs = (err/100)*np.max(np.abs(true_signal), axis=1)
rmse_list_relative = rmse_list/np.max(np.abs(true_signal), axis=1)
rmse_list_relative *= 100

print('median residual:', np.median(residuals_list))
print('mean residual:', np.mean(residuals_list))
print('standard deviation of the residuals:', np.std(residuals_list))
print()
print('median rmse (GLOBAL):', np.median(err_abs))
print('mean rmse (GLOBAL):', np.mean(err_abs))
print('max rmse (GLOBAL):', max(err_abs))
print('standard deviation of the rmse (GLOBAL):', np.std(err_abs))
print()
print()
print('median rmse (GLOBAL):', np.median(err))
print('mean rmse (GLOBAL):', np.mean(err))
print('max rmse (GLOBAL):', max(err))
print('standard deviation of the rmse (GLOBAL):', np.std(err))
print()

###############

err_abs_21cmGEM = np.sqrt(np.mean((signal_testing - true_signal)**2, axis=1)) # absolute error in milliKelvins (mK)
err_rel_21cmGEM = (err_abs_21cmGEM/np.max(np.abs(true_signal), axis=1))*100 # relative error in per cent (%)

mean_abs_err_21cmGEM = np.mean(err_abs_21cmGEM)
median_abs_err_21cmGEM = np.median(err_abs_21cmGEM)
max_abs_err_21cmGEM = np.max(err_abs_21cmGEM)
mean_rel_err_21cmGEM = np.mean(err_rel_21cmGEM)
median_rel_err_21cmGEM = np.median(err_rel_21cmGEM)
max_rel_err_21cmGEM = np.max(err_rel_21cmGEM)
print('Mean absolute rms error for globalemu trained and tested on 21cmGEM:', mean_abs_err_21cmGEM, 'mK')
print('Median absolute rms error for globalemu trained and tested on 21cmGEM:', median_abs_err_21cmGEM, 'mK')
print('Max absolute rms error for globalemu trained and tested on 21cmGEM:', max_abs_err_21cmGEM, 'mK')
print()
print('Mean relative rms error for globalemu trained and tested on 21cmGEM:', mean_rel_err_21cmGEM, '%')
print('Median relative rms error for globalemu trained and tested on 21cmGEM:', median_rel_err_21cmGEM, '%')
print('Max relative rms error for globalemu trained and tested on 21cmGEM:', max_rel_err_21cmGEM, '%')
print()

