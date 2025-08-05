import numpy as np
import torch
from torch.utils.data import Dataset

class NumPyArray2TensorDataset(Dataset):
    def __init__(self, features_npy, targets_npy):
        # Handle both numpy arrays and tensors
        if isinstance(features_npy, torch.Tensor):
            self.features = features_npy
        else:
            self.features = torch.from_numpy(features_npy)
            
        if isinstance(targets_npy, torch.Tensor):
            self.targets = targets_npy
        else:
            self.targets = torch.from_numpy(targets_npy)
        
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.targets)
    
    def __getitem__(self, idx):
        # Return tensor samples directly (no conversion needed)
        features_tensor = self.features[idx]
        targets_tensor = self.targets[idx]
        return features_tensor, targets_tensor

class NumPy2TensorDataset(Dataset):
    def __init__(self, features_npy_file, targets_npy_file):
        # Load full numpy array into memory. TODO: could put all data in 1 npy file 
        self.features = np.load(features_npy_file)
        self.targets = np.load(targets_npy_file)  
        
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.targets)
    
    def __getitem__(self, idx):

        # Convert feature sample to a PyTorch tensor
        features_tensor = torch.from_numpy(self.features[idx])

        # Convert targets sample to a PyTorch tensor
        targets_tensor = torch.from_numpy(self.targets[idx])

        return features_tensor, targets_tensor

def eval_test_set_21cmGEM(model, data_path, device):

    torch.set_default_dtype(torch.float64)

    proc_params_test_21cmGEM_np = np.load(data_path + 'X_test_21cmGEM.npy')
    train_maxs_21cmGEM = np.load(data_path + 'train_maxs_21cmGEM.npy')
    train_mins_21cmGEM = np.load(data_path + 'train_mins_21cmGEM.npy')
    signals_21cmGEM_true = np.load(data_path + 'signals_21cmGEM_true.npy')

    proc_params_test_21cmGEM = torch.from_numpy(proc_params_test_21cmGEM_np)
    proc_params_test_21cmGEM_np = 0 
    proc_params_test_21cmGEM = proc_params_test_21cmGEM.to(device)
    
    model.eval()
    with torch.no_grad():
        result_21cmGEM = model(proc_params_test_21cmGEM)
    result_21cmGEM = result_21cmGEM.cpu().detach().numpy()

    evaluation_test_21cmGEM = result_21cmGEM #.T[0].T # not sure why this transpose is here 
    unproc_signals_test_21cmGEM = evaluation_test_21cmGEM.copy()
    unproc_signals_test_21cmGEM = (evaluation_test_21cmGEM*(train_maxs_21cmGEM[-1]-train_mins_21cmGEM[-1]))+train_mins_21cmGEM[-1] # unpreprocess (i.e., denormalize) signals
    unproc_signals_test_21cmGEM = unproc_signals_test_21cmGEM[:,::-1] # flip signals to be from high-z to low-z
    signals_21cmGEM_emulated = unproc_signals_test_21cmGEM.copy()

    # calculate relative error between emulated and true signals in test set
    err_21cmGEM = np.sqrt(np.mean((signals_21cmGEM_emulated - signals_21cmGEM_true)**2, axis=1)) # absolute error in milliKelvins (mK)
    err_21cmGEM /= np.max(np.abs(signals_21cmGEM_true), axis=1)
    err_21cmGEM *= 100 # convert to relative error in per cent (%)

    mean_rel_err = np.mean(err_21cmGEM)
    median_rel_err = np.median(err_21cmGEM)
    max_rel_err = np.max(err_21cmGEM)

    return mean_rel_err, median_rel_err, max_rel_err

def eval_test_set_ARES(model, data_path, device):

    torch.set_default_dtype(torch.float64)

    proc_params_test_ARES_np = np.load(data_path + 'X_test_ARES.npy')
    train_maxs_ARES = np.load(data_path + 'train_maxs_ARES.npy')
    train_mins_ARES = np.load(data_path + 'train_mins_ARES.npy')
    signals_ARES_true = np.load(data_path + 'signals_ARES_true.npy')

    proc_params_test_ARES = torch.from_numpy(proc_params_test_ARES_np)
    proc_params_test_ARES_np = 0
    proc_params_test_ARES = proc_params_test_ARES.to(device)

    model.eval()
    with torch.no_grad():
        result_ARES = model(proc_params_test_ARES)
    result_ARES = result_ARES.cpu().detach().numpy()

    evaluation_test_ARES = result_ARES #.T[0].T # not sure why this transpose is here 
    unproc_signals_test_ARES = evaluation_test_ARES.copy()
    unproc_signals_test_ARES = (evaluation_test_ARES*(train_maxs_ARES[-1]-train_mins_ARES[-1]))+train_mins_ARES[-1] # unpreprocess (i.e., denormalize) signals
    unproc_signals_test_ARES = unproc_signals_test_ARES[:,::-1] # flip signals to be from high-z to low-z
    signals_ARES_emulated = unproc_signals_test_ARES.copy()

    # calculate relative error between emulated and true signals in test set
    err_ARES = np.sqrt(np.mean((signals_ARES_emulated - signals_ARES_true)**2, axis=1)) # absolute error in milliKelvins (mK)
    err_ARES /= np.max(np.abs(signals_ARES_true), axis=1)
    err_ARES *= 100 # convert to relative error in per cent (%)

    mean_rel_err = np.mean(err_ARES)
    median_rel_err = np.median(err_ARES)
    max_rel_err = np.max(err_ARES)

    return mean_rel_err, median_rel_err, max_rel_err

def create_file_name(config, test_max_rel_err):

    file_name = ""
    len_config = len(config)
    for key, value in config.items():
        if key == "data_path":
            file_name += "test_max_rel_error_" + str(test_max_rel_err) + ".pth"
        elif key == "model_save_dir":
            pass
        else:
            file_name += key + "_" + str(value) + "_"

    return file_name
