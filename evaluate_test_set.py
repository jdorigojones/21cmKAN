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
proc_params_test_21cmGEM = torch.from_numpy(proc_params_test_21cmGEM_np)
proc_params_test_21cmGEM_np = 0 
proc_params_test_21cmGEM = proc_params_test_21cmGEM.to(device)

# Specify path to model we want to evaluate 
model_file_path = "./models/21cmkan_model.pth"

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

print('Mean relative rms error for 21cmKAN pre-trained and tested on 21cmGEM:', mean_rel_err_21cmGEM, '%')
print('Median relative rms error for 21cmKAN pre-trained and tested on 21cmGEM:', median_rel_err_21cmGEM, '%')
print('Max relative rms error for 21cmKAN pre-trained and tested on 21cmGEM:', max_rel_err_21cmGEM, '%')
print(" ")

z_list = np.linspace(5, 50, 451) # list of redshifts for 21cmGEM signals
vr = 1420.405751
nu_list = [vr/(z+1) for z in z_list]

def freq(zs):
    return vr/(zs+1)

def redshift(v):
    return (vr/v)-1

fig, ax = plt.subplots(constrained_layout=True)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction = 'out', width = 2, length = 10, labelsize=20)
ax.tick_params(axis='both', which='minor', direction = 'out', width = 2, length = 5, labelsize=20)
ax.set_yticks([50, 0, -50,-100,-150,-200,-250,-300])
ax.set_yticklabels(['50', '0','-50','-100','-150','-200','-250','-300'], fontsize=20)
ax.set_xticks([40, 60, 80, 100, 120, 140, 160, 180, 200, 220])
ax.set_xticklabels(['40', '60', '80', '100', '120', '140', '160', '180', '200', '220'], fontsize=20)
ax.set_ylabel(r'$\delta T_b$ (mK)', fontsize=20)
ax.set_xlabel(r'$\nu$ (MHz)', fontsize=20)
secax = ax.secondary_xaxis('top', functions=(redshift, freq))
secax.tick_params(which='major', direction = 'out', width = 2, length = 10, labelsize=20)
secax.tick_params(which='minor', direction = 'out', width = 1, length = 5, labelsize=20)
secax.set_xlabel(r'$z$', fontsize=20)
secax.set_xticks([5, 10, 15, 20, 30, 50])
secax.set_xticklabels(['5', '10', '15', '20', '30', '50'], fontsize=15)
i=0
for i in range(np.shape(proc_params_test_21cmGEM)[0]):
    ax.plot(nu_list[::-1], signals_21cmGEM_emulated[i][::-1], color='r', alpha=0.1)
    ax.plot(nu_list[::-1], signals_21cmGEM_true[i][::-1], linestyle='dashed', color='k', alpha=0.1)
ax.set_ylim(-300,50)
ax.set_xlim(27.85,236.74)
plt.savefig('21cmKAN_21cmGEM_test_realizations_A100_trialx.png', dpi = 300, bbox_inches='tight', facecolor='w')
plt.show()

plt.cla()
plt.clf()
