#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import gc
import os
from efficient_kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_default_dtype(torch.float64)

# class to evaluate 21cmKAN trained on 21cmGEM
class evaluate_on_21cmGEM(): 
    def __init__(self, **kwargs):
        for key, values in kwargs.items():
            if key not in set(['model_path', 'model']):
                raise KeyError("Unexpected keyword argument in evaluate()")

        # Default model path
        default_model_path = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/models/emulator_21cmGEM.pth"
        model_path = kwargs.pop('model_path', default_model_path)

        # Load normalization data from the same directory as the model
        model_dir = os.path.dirname(model_path) + '/'
        self.train_mins = np.load(model_dir + 'train_mins_21cmGEM.npy')
        self.train_maxs = np.load(model_dir + 'train_maxs_21cmGEM.npy')

        self.model = kwargs.pop('model', None)
        if self.model is None:
            self.model = torch.load(model_path, weights_only=False)
            self.model.to(device)
            
    def __call__(self, parameters):
        if len(np.shape(parameters)) == 1:
            parameters = np.expand_dims(parameters, axis=0) # if doing one signal at a time
            
        unproc_f_s = parameters[:,0].copy() # f_*, star formation efficiency, # preprocess input physical parameters
        unproc_V_c = parameters[:,1].copy() # V_c, minimum circular velocity of star-forming halos 
        unproc_f_X = parameters[:,2].copy() # f_X, X-ray efficiency of sources
        unproc_f_s = np.log10(unproc_f_s)
        unproc_V_c = np.log10(unproc_V_c)
        unproc_f_X[unproc_f_X == 0] = 1e-6 # for f_X, set zero values to 1e-6 before taking log_10
        unproc_f_X = np.log10(unproc_f_X)
        parameters_log = np.empty(parameters.shape)
        parameters_log[:,0] = unproc_f_s
        parameters_log[:,1] = unproc_V_c
        parameters_log[:,2] = unproc_f_X
        parameters_log[:,3:] = parameters[:,3:].copy()
        N_proc = np.shape(parameters_log)[0] # number of signals (i.e., parameter sets) to process
        p = np.shape(parameters)[1] # number of input parameters (# of physical params)
        proc_params = np.zeros((N_proc,p))
        
        for i in range(p):
            x = parameters_log[:,i]
            proc_params[:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])
        proc_params_test = torch.from_numpy(proc_params)
        proc_params = 0
        proc_params_test = proc_params_test.to(device)
        
        self.model.eval()
        with torch.no_grad():
            result = self.model(proc_params_test) # evaluate trained instance of 21cmKAN with processed parameters
        result = result.cpu().detach().numpy()
        unproc_signals = result.copy()
        unproc_signals = (result*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1] # unpreprocess (i.e., denormalize) signals
        unproc_signals = unproc_signals[:,::-1] # flip signals to be from high-z to low-z
        if unproc_signals.shape[0] == 1:
            return unproc_signals[0,:]
        else:
            return unproc_signals

# class to evaluate 21cmKAN trained on ARES
class evaluate_on_ARES(): 
    def __init__(self, **kwargs):
        for key, values in kwargs.items():
            if key not in set(['model_path', 'model']):
                raise KeyError("Unexpected keyword argument in evaluate()")

        # Default model path
        default_model_path = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/models/emulator_ARES.pth"
        model_path = kwargs.pop('model_path', default_model_path)
                
        # Load normalization data from the same directory as the model
        model_dir = os.path.dirname(model_path) + '/'
        self.train_mins = np.load(model_dir + 'train_mins_ARES.npy')
        self.train_maxs = np.load(model_dir + 'train_maxs_ARES.npy')

        self.model = kwargs.pop('model', None)
        if self.model is None:
            self.model = torch.load(model_path, weights_only=False)
            self.model.to(device)
    def __call__(self, parameters):
        if len(np.shape(parameters)) == 1:
            parameters = np.expand_dims(parameters, axis=0) # if doing one signal at a time

        unproc_c_X = parameters[:,0].copy() # c_X, normalization of X-ray luminosity-SFR relation
        unproc_T_min = parameters[:,2].copy() # T_min, minimum temperature of star-forming halos
        unproc_f_s = parameters[:,4].copy() # f_*,0, peak star formation efficiency 
        unproc_M_p = parameters[:,5].copy() # M_p, dark matter halo mass at f_*,0
        unproc_c_X = np.log10(unproc_c_X)
        unproc_T_min = np.log10(unproc_T_min)
        unproc_f_s = np.log10(unproc_f_s)
        unproc_M_p = np.log10(unproc_M_p)
        parameters_log = np.empty(parameters.shape)
        parameters_log[:,0] = unproc_c_X
        parameters_log[:,1] = parameters[:,1].copy()
        parameters_log[:,2] = unproc_T_min
        parameters_log[:,3] = parameters[:,3].copy()
        parameters_log[:,4] = unproc_f_s
        parameters_log[:,5] = unproc_M_p
        parameters_log[:,6] = parameters[:,6].copy()
        parameters_log[:,7] = parameters[:,7].copy()
        N_proc = np.shape(parameters_log)[0] # number of signals (i.e., parameter sets) to process
        p = np.shape(parameters)[1] # number of input parameters (# of physical params)
        proc_params = np.zeros((N_proc,p))
        
        for i in range(p):
            x = parameters_log[:,i]
            proc_params[:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])
        proc_params_test = torch.from_numpy(proc_params)
        proc_params = 0
        proc_params_test = proc_params_test.to(device)

        self.model.eval()
        with torch.no_grad():
            result = self.model(proc_params_test) # evaluate trained instance of 21cmKAN with processed parameters
        result = result.cpu().detach().numpy()
        unproc_signals = result.copy()
        unproc_signals = (result*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1] # unpreprocess (i.e., denormalize) signals
        unproc_signals = unproc_signals[:,::-1] # flip signals to be from high-z to low-z
        if unproc_signals.shape[0] == 1:
            return unproc_signals[0,:]
        else:
            return unproc_signals
