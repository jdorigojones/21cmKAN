from efficient_kan import KAN
import torch 
import numpy as np 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Set default torch type to float64
torch.set_default_dtype(torch.float64)

data_path = './data/'

# Load in training data and data needed to transform the output of the model
proc_params_test_21cmGEM_np = np.load(data_path + 'X_test_21cmGEM.npy')
train_maxs_21cmGEM = np.load(data_path + 'train_maxs_21cmGEM.npy')
train_mins_21cmGEM = np.load(data_path + 'train_mins_21cmGEM.npy')
signals_21cmGEM_true = np.load(data_path + 'signals_21cmGEM_true.npy')

# Convert inputs to a tensor and zero out the numpy version
proc_params_test_21cmGEM = torch.from_numpy(proc_params_test_21cmGEM_np[:, 0, :]) # drop the frequency channels for the input 
proc_params_test_21cmGEM_np = 0 
proc_params_test_21cmGEM = proc_params_test_21cmGEM.to(device)

# Specify path to model we want to evaluate 
model_file_path = "./21cmkan_model.pth"

# Load in model and put it on the device 
model = torch.load(model_file_path, weights_only=False)
model.to(device)

# Perform inference on the provided test inputs 
model.eval()
with torch.no_grad():
    result_21cmGEM = model(proc_params_test_21cmGEM)
result_21cmGEM = result_21cmGEM.cpu().detach().numpy()

# Transform prediction, so we can compare against known values
evaluation_test_21cmGEM = result_21cmGEM #.T[0].T # not sure why this transpose is here in the Tensorflow code
unproc_signals_test_21cmGEM = evaluation_test_21cmGEM.copy()
unproc_signals_test_21cmGEM = (evaluation_test_21cmGEM*(train_maxs_21cmGEM[-1]-train_mins_21cmGEM[-1]))+train_mins_21cmGEM[-1] # unpreprocess (i.e., denormalize) signals
unproc_signals_test_21cmGEM = unproc_signals_test_21cmGEM[:,::-1] # flip signals to be from high-z to low-z
signals_21cmGEM_emulated = unproc_signals_test_21cmGEM.copy()

# calculate relative error between emulated and true signals in test set
err_21cmGEM = np.sqrt(np.mean((signals_21cmGEM_emulated - signals_21cmGEM_true)**2, axis=1)) # absolute error in milliKelvins (mK)
err_21cmGEM /= np.max(np.abs(signals_21cmGEM_true), axis=1)
err_21cmGEM *= 100 # convert to relative error in per cent (%)

np.set_printoptions(threshold=np.inf)

mean_rel_err_21cmGEM = np.mean(err_21cmGEM)
median_rel_err_21cmGEM = np.median(err_21cmGEM)
max_rel_err_21cmGEM = np.max(err_21cmGEM)

print('Mean relative rms error for 21cmLSTM pre-trained and tested on 21cmGEM:', mean_rel_err_21cmGEM, '%')
print('Median relative rms error for 21cmLSTM pre-trained and tested on 21cmGEM:', median_rel_err_21cmGEM, '%')
print('Max relative rms error for 21cmLSTM pre-trained and tested on 21cmGEM:', max_rel_err_21cmGEM, '%')
print(" ")
