import numpy as np
import torch
from torch.utils.data import Dataset

class NumPyArray2TensorDataset(Dataset):
    def __init__(self, features_npy, targets_npy):
        self.features = features_npy
        self.targets = targets_npy
        
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.targets)
    
    def __getitem__(self, idx):

        # Convert feature sample to a PyTorch tensor
        features_tensor = torch.from_numpy(self.features[idx])

        # Convert targets sample to a PyTorch tensor
        targets_tensor = torch.from_numpy(self.targets[idx])

        return features_tensor, targets_tensor

class NumPy2TensorDataset(Dataset):
    def __init__(self, features_npy_file, targets_npy_file):
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
