from efficient_kan import KAN
from torch import autograd
from tqdm import tqdm
import torch 
import torch.optim as optim
import numpy as np 
# from kan import LBFGS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Set default torch type to float64
torch.set_default_dtype(torch.float64)

steps = 10  # number of steps for training 
history_size = 10  # history size for LBFGS optimizer 
layers_hidden = [7, 44, 44, 71, 451]  # specify KAN architecture 

model_save_path = './21cmkan_model_simple_train.pth'
data_path = './data/'

# TODO: in the future replace the loading of data with a PyTorch Dataset/DataLoader process 
# Load training data 
X_train_21cmGEM_np = np.load(data_path + 'X_train_21cmGEM.npy') 
y_train_21cmGEM_np = np.load(data_path + 'y_train_21cmGEM.npy')

# Load validation data 
X_val_21cmGEM_np = np.load(data_path + 'X_val_21cmGEM.npy')
y_val_21cmGEM_np = np.load(data_path + 'y_val_21cmGEM.npy')

# Convert to tensors 
X_train_21cmGEM = torch.from_numpy(X_train_21cmGEM_np).to(device)
y_train_21cmGEM = torch.from_numpy(y_train_21cmGEM_np).to(device)
X_val_21cmGEM = torch.from_numpy(X_val_21cmGEM_np).to(device) 
y_val_21cmGEM = torch.from_numpy(y_val_21cmGEM_np).to(device)

# free numpy array memory 
X_train_21cmGEM_np = 0
y_train_21cmGEM_np = 0
X_val_21cmGEM_np = 0
y_val_21cmGEM_np = 0

# Initialize model 
model = KAN(layers_hidden=layers_hidden, grid_size=7, spline_order=3)
model.to(device)

# Specifies if the log is visible during training 
log = 1 

# Calculate the absolute maximum for each batch's frequency channel value
max_abs = torch.abs(y_val_21cmGEM).max(dim=1)[0]

def train():

    # TODO: replace simplistic training with batch training 
    # Use PyTorch's LBFGS
    optimizer = optim.LBFGS(model.parameters(), lr=1, tolerance_grad=1e-32, tolerance_change=1e-32, history_size=history_size, line_search_fn='strong_wolfe')

    # Use Pykan version 
    # optimizer = LBFGS(model.parameters(), lr=1, history_size=history_size, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

    pbar = tqdm(range(steps), desc='description', ncols=100)

    for _ in pbar:
        def closure():
            global loss
            optimizer.zero_grad()

            pred = model(X_train_21cmGEM)
            loss = torch.mean((pred-y_train_21cmGEM)**2)
            loss.backward()
            return loss

        optimizer.step(closure)
        val_pred = model(X_val_21cmGEM)
        sqrt_MSE = torch.sqrt(torch.mean((val_pred-y_val_21cmGEM)**2, dim=1))
        error = (sqrt_MSE/max_abs)*100
        mean_error = torch.mean(error)
        max_error = error.max()

        if _ % log == 0:
            pbar.set_description("loss: %.2e | mean_error: %.2e | max_error: %.2e" % (loss.cpu().detach().numpy(), mean_error.cpu().detach().numpy(), max_error.cpu().detach().numpy()))

train()

torch.save(model, model_save_path)
