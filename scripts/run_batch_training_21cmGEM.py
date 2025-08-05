from efficient_kan import KAN
import torch 
import torch.optim as optim
import numpy as np 
import os
import h5py
from utils import NumPy2TensorDataset
from utils import NumPyArray2TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Set default torch type to float64
torch.set_default_dtype(torch.float64)

# specify KAN model architecture and configs for training on the 21cmGEM set; see Sec 2.3 of Dorigo Jones et al. 2025
layer_nodes = [7, 44, 44, 71, 451]
grid_size = 7
spline_order = 3
num_epochs = 400
batch_size = 100
print(f"nodes in each layer: {layer_nodes}")
print(f"number of grid intervals in B-splines: {grid_size}")
print(f"order of splines in B-splines: {spline_order}")
print(f"number of training epochs: {num_epochs}")
print(f"training batch size: {batch_size}")

# define paths of the saved trained 21cmKAN networks and min/max training set values used for preprocessing
PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"
model_save_path = PATH+"models/emulator_21cmGEM.pth"
train_mins_21cmGEM = np.load(PATH+"models/train_mins_21cmGEM.npy")
train_maxs_21cmGEM = np.load(PATH+"models/train_maxs_21cmGEM.npy")

z_list = np.linspace(5, 50, 451) # list of redshifts for 21cmGEM signals; equiv to np.arange(5, 50.1, 0.1)
vr = 1420.4057517667  # rest frequency of 21 cm line in MHz
# Load in unnormalized training, validation, and test data
with h5py.File(PATH + 'dataset_21cmGEM.h5', "r") as f:
    print("Keys: %s" % f.keys())
    par_train = np.asarray(f['par_train'])[()]
    par_val = np.asarray(f['par_val'])[()]
    X_test_21cmGEM_true = np.asarray(f['par_test'])[()]
    signal_train = np.asarray(f['signal_train'])[()]
    signal_val = np.asarray(f['signal_val'])[()]
    y_test_21cmGEM_true = np.asarray(f['signal_test'])[()]
f.close()

# preprocess/normalize input physical parameters values of training and validation sets
unproc_f_s_train = par_train[:,0].copy() # f_*, star formation efficiency
unproc_V_c_train = par_train[:,1].copy() # V_c, minimum circular velocity of star-forming halos 
unproc_f_X_train = par_train[:,2].copy() # f_X, X-ray efficiency of sources
unproc_f_s_train = np.log10(unproc_f_s_train)
unproc_V_c_train = np.log10(unproc_V_c_train)
unproc_f_X_train[unproc_f_X_train == 0] = 1e-6 # for f_X, set zero values to 1e-6 before taking log_10
unproc_f_X_train = np.log10(unproc_f_X_train)
parameters_log_train = np.empty(par_train.shape)
parameters_log_train[:,0] = unproc_f_s_train
parameters_log_train[:,1] = unproc_V_c_train
parameters_log_train[:,2] = unproc_f_X_train
parameters_log_train[:,3:] = par_train[:,3:].copy()

unproc_f_s_val = par_val[:,0].copy() # f_*, star formation efficiency, # preprocess input physical parameters
unproc_V_c_val = par_val[:,1].copy() # V_c, minimum circular velocity of star-forming halos 
unproc_f_X_val = par_val[:,2].copy() # f_X, X-ray efficiency of sources
unproc_f_s_val = np.log10(unproc_f_s_val)
unproc_V_c_val = np.log10(unproc_V_c_val)
unproc_f_X_val[unproc_f_X_val == 0] = 1e-6 # for f_X, set zero values to 1e-6 before taking log_10
unproc_f_X_val = np.log10(unproc_f_X_val)
parameters_log_val = np.empty(par_val.shape)
parameters_log_val[:,0] = unproc_f_s_val
parameters_log_val[:,1] = unproc_V_c_val
parameters_log_val[:,2] = unproc_f_X_val
parameters_log_val[:,3:] = par_val[:,3:].copy()

N_proc_train = np.shape(parameters_log_train)[0] # number of signals (i.e., parameter sets) to process
p_train = np.shape(par_train)[1] # number of input parameters (# of physical params)
proc_params_train = np.zeros((N_proc_train,p_train))

N_proc_val = np.shape(parameters_log_val)[0] # number of signals (i.e., parameter sets) to process
p_val = np.shape(par_val)[1] # number of input parameters (# of physical params)
proc_params_val = np.zeros((N_proc_val,p_val))
        
for i in range(p_train):
    x_train = parameters_log_train[:,i]
    proc_params_train[:,i] = (x_train-train_mins_21cmGEM[i])/(train_maxs_21cmGEM[i]-train_mins_21cmGEM[i])

for i in range(p_val):
    x_val = parameters_log_val[:,i]
    proc_params_val[:,i] = (x_val-train_mins_21cmGEM[i])/(train_maxs_21cmGEM[i]-train_mins_21cmGEM[i])

#X_train_21cmGEM = torch.from_numpy(proc_params_train)
X_train_21cmGEM = proc_params_train.copy()
proc_params_train = 0
par_train = 0
#X_train_21cmGEM = X_train_21cmGEM.to(device)

X_val_21cmGEM = torch.from_numpy(proc_params_val)
proc_params_val = 0
par_val = 0
X_val_21cmGEM = X_val_21cmGEM.to(device)

# preprocess/normalize signals (dT_b) in training and validaton sets
proc_signals_train = signal_train.copy()
proc_signals_train = (signal_train - train_mins_21cmGEM[-1])/(train_maxs_21cmGEM[-1]-train_mins_21cmGEM[-1])  # global Min-Max normalization
proc_signals_train = proc_signals_train[:,::-1] # flip signals to be from high-z to low-z
#y_train_21cmGEM = torch.from_numpy(np.copy(proc_signals_train))
y_train_21cmGEM = proc_signals_train.copy()
proc_signals_train = 0
signal_train = 0
#y_train_21cmGEM = y_train_21cmGEM.to(device)

proc_signals_val = signal_val.copy()
proc_signals_val = (signal_val - train_mins_21cmGEM[-1])/(train_maxs_21cmGEM[-1]-train_mins_21cmGEM[-1])  # global Min-Max normalization
proc_signals_val = proc_signals_val[:,::-1] # flip signals to be from high-z to low-z
y_val_21cmGEM = torch.from_numpy(proc_signals_val).copy())
proc_signals_val = 0
signal_val = 0
y_val_21cmGEM = y_val_21cmGEM.to(device)

# Calculate the absolute minimum value of each normalized validation set signal, used to compute relative error
min_abs = torch.abs(y_val_21cmGEM).min(dim=1)[0]

# Create normalized training Dataset and DataLoader
train_dataset = NumPyArray2TensorDataset(features_npy=X_train_21cmGEM, targets_npy=y_train_21cmGEM)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# initialize model 
model = KAN(layers_hidden=layer_nodes, grid_size=grid_size, spline_order=spline_order)
model.to(device)

# Define optimizer 
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# code to define learning rate scheduler, if desired
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

for epoch in range(num_epochs):
  model.train()
  for batch_idx, (inputs, targets) in enumerate(train_dataloader):
    inputs = inputs.to(device)
    targets = targets.to(device)
    def closure():
      global loss
      optimizer.zero_grad()
      pred = model(inputs)
      loss = torch.mean((pred-targets)**2)
      loss.backward()
      return loss 
    optimizer.step(closure)
  
  model.eval()
  with torch.no_grad():
    val_pred = model(X_val_21cmGEM)
    val_loss = torch.mean((val_pred-y_val_21cmGEM)**2)
    # code to compute the normalized validation set relative RMSE, if desired to print along with loss
    #RMSE = torch.sqrt(val_loss)
    #rel_error = (RMSE/min_abs)*100
    #mean_rel_error = torch.mean(rel_error)
    #max_rel_error = rel_error.max()
  
  # code to update the learning rate, if desired: scheduler.step() ; scheduler.step(max_error)
  print(f"Epoch: {epoch + 1}, Validation Loss: {val_loss}, Training Loss: {loss}")

torch.save(model, model_save_path)
