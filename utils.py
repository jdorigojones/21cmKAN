import numpy as np
import torch
from torch.utils.data import Dataset

class NumPy2TensorDataset(Dataset):
    def __init__(self, features_npy_file, targets_npy_file):
        # Load full numpy array into memory. TODO: could put all data in 1 npy file 
        self.features = np.load(features_npy_file)  
        self.features = self.features[:, 0, :] # drop the frequency channels for the input. TODO: modify data so this is not necessary 
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