### this script performs a Ray Tune Bayesian automatic hyperparameter search by training 21cmKAN with 1,000 different architecture hyperparameters
### on the 21cmGEM training set. See Section 2.3 of Dorigo Jones et al. 2025 for further details

import os 
import h5py
import torch 
import torch.optim as optim
import numpy as np
from efficient_kan import KAN
from utils import *
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.train import RunConfig
import Global21cmKAN as Global21cmKAN

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set default torch type to float64
    torch.set_default_dtype(torch.float64)
    
    # define paths of the saved trained 21cmKAN networks and min/max training set values used for preprocessing
    PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"
    train_mins_21cmGEM = np.load(PATH+"models/train_mins_21cmGEM.npy")
    train_maxs_21cmGEM = np.load(PATH+"models/train_maxs_21cmGEM.npy")

    # create and load 21cmKAN emulator instance already trained on the 21cmGEM data set
    emulator_21cmGEM = Global21cmKAN.emulate_21cmGEM.Emulate()
    
    z_list = emulator_21cmGEM.redshifts # list of redshifts for 21cmGEM signals; equiv to np.arange(5, 50.1, 0.1)
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
    unproc_f_s_val = par_val[:,0].copy()
    unproc_V_c_val = par_val[:,1].copy()
    unproc_f_X_val = par_val[:,2].copy()
    unproc_f_s_val = np.log10(unproc_f_s_val)
    unproc_V_c_val = np.log10(unproc_V_c_val)
    unproc_f_X_val[unproc_f_X_val == 0] = 1e-6
    unproc_f_X_val = np.log10(unproc_f_X_val)
    parameters_log_val = np.empty(par_val.shape)
    parameters_log_val[:,0] = unproc_f_s_val
    parameters_log_val[:,1] = unproc_V_c_val
    parameters_log_val[:,2] = unproc_f_X_val
    parameters_log_val[:,3:] = par_val[:,3:].copy()

    N_proc_train = np.shape(parameters_log_train)[0] # number of signals (i.e., parameter sets) to process
    N_proc_val = np.shape(parameters_log_val)[0]
    p_train = np.shape(par_train)[1] # number of input parameters (# of physical params)
    p_val = np.shape(par_val)[1]
    proc_params_train = np.zeros((N_proc_train,p_train))
    proc_params_val = np.zeros((N_proc_val,p_val))

    for i in range(p_train):
        x_train = parameters_log_train[:,i]
        proc_params_train[:,i] = (x_train-train_mins_21cmGEM[i])/(train_maxs_21cmGEM[i]-train_mins_21cmGEM[i])
    
    for i in range(p_val):
        x_val = parameters_log_val[:,i]
        proc_params_val[:,i] = (x_val-train_mins_21cmGEM[i])/(train_maxs_21cmGEM[i]-train_mins_21cmGEM[i])

    X_train_21cmGEM = proc_params_train.copy()
    X_val_21cmGEM = torch.from_numpy(proc_params_val)
    X_val_21cmGEM = X_val_21cmGEM.to(device)
    proc_params_train = 0
    par_train = 0
    proc_params_val = 0
    par_val = 0

    # preprocess/normalize signals (dT_b) in training and validaton sets
    proc_signals_train = signal_train.copy()
    proc_signals_train = (signal_train - train_mins_21cmGEM[-1])/(train_maxs_21cmGEM[-1]-train_mins_21cmGEM[-1])  # global Min-Max normalization
    proc_signals_train = proc_signals_train[:,::-1] # flip signals to be from high-z to low-z
    y_train_21cmGEM = proc_signals_train.copy()
    proc_signals_val = signal_val.copy()
    proc_signals_val = (signal_val - train_mins_21cmGEM[-1])/(train_maxs_21cmGEM[-1]-train_mins_21cmGEM[-1])
    proc_signals_val = proc_signals_val[:,::-1]
    y_val_21cmGEM = torch.from_numpy(proc_signals_val.copy())
    y_val_21cmGEM = y_val_21cmGEM.to(device)
    proc_signals_train = 0
    signal_train = 0
    proc_signals_val = 0
    signal_val = 0

    # Calculate the absolute minimum value of each normalized validation set signal, used to compute relative error
    min_abs = torch.abs(y_val_21cmGEM).min(dim=1)[0]

    # Create normalized training Dataset and DataLoader
    train_dataset = NumPyArray2TensorDataset(features_npy=X_train_21cmGEM, targets_npy=y_train_21cmGEM)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize model with given configurations 
    model = KAN(layers_hidden=[7, config["layer1"], config["layer2"], config["layer3"], 451], grid_size=config["grid_size"], spline_order=config["spline_order"])
    model.to(device)
    # Define optimizer 
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(config["epochs"]):

        try: 

            # Train model using batches 
            model.train()
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                def closure():
                    global train_loss
                    optimizer.zero_grad()
                    pred = model(inputs)
                    train_loss = torch.mean((pred-targets)**2)
                    train_loss.backward()
                    return train_loss 
                optimizer.step(closure)

            # Calculate max error of validation data set 
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_21cmGEM)
                sqrt_MSE = torch.sqrt(torch.mean((val_pred-y_val_21cmGEM)**2, dim=1))
                error = (sqrt_MSE/min_abs)*100
                max_error = error.max()
                max_error = max_error.cpu().detach().numpy()

        except Exception as e:
            print(f"Error in trial: {e}")
            max_error = 99999

    file_name = create_file_name(config, test_max_rel_err)
    torch.save(model, config["model_save_dir"] + file_name)
    emulator_21cmGEM.load_model(model_path=config["model_save_dir"] + file_name)
    test_rel_RMSE_values_21cmGEM = emulator_21cmGEM.test_error()
    test_mean_rel_err = np.mean(test_rel_RMSE_values_21cmGEM)
    test_median_rel_err = np.median(test_rel_RMSE_values_21cmGEM)
    test_max_rel_err = np.max(test_rel_RMSE_values_21cmGEM)
    tune.report({"max_error": max_error, "test_mean_rel_err": test_mean_rel_err, "test_median_rel_err": test_median_rel_err, "test_max_rel_err": test_max_rel_err}, checkpoint=None)

# Define directories to store results produced by Ray Tune and models produced
results_dir = os.path.abspath("./21cmKAN_ray_tune_results")
model_save_dir = results_dir + "/saved_models/"

# Define the name of the trial directory for Ray Tune 
trial_directory = "test_experiment"

# Create directory that will store models, if it does not exist 
os.makedirs(model_save_dir, exist_ok=True)

# Define hyperparameters to search
config = {
    "batch_size": tune.qrandint(10, 1000),
    "layer1": tune.qrandint(10, 100), 
    "layer2": tune.qrandint(10, 100), 
    "layer3": tune.qrandint(10, 100), 
    "grid_size": tune.qrandint(3, 15),
    "spline_order": tune.qrandint(3, 7),
    "epochs": 400,
    "model_save_dir": model_save_dir}

# Use Optuna to perform Bayesian hyperparameter optimization using max_error produced by the validation set  
optuna_search = OptunaSearch(metric="max_error", mode="min")

# Split up the A100 GPU to run multiple trials at once
trainable_with_cpu_gpu = tune.with_resources(train_model, {"cpu": 4, "gpu": 0.20})

# Define the number of samples to take from search space 
num_samples = 1000

# Define the maximum number of trials that can take place in parallel 
max_concurrent_trials = 5 # if doing tenth of the GPU, have 10 run parallel

# Specify Tuner 
tuner = tune.Tuner(
    trainable_with_cpu_gpu,
    param_space=config,
    tune_config=tune.TuneConfig(
        search_alg=optuna_search,
        num_samples=num_samples, 
        max_concurrent_trials=max_concurrent_trials),
    run_config=RunConfig(storage_path=results_dir, name=trial_directory))

# code to resume a previous tune experiment
#tuner = tune.Tuner.restore(os.path.expanduser("/path/to/21cmKAN_ray_tune_results/test_experiment"),
#        trainable=trainable_with_cpu_gpu,
#        resume_errored=True)

# Perform hyperparameter search 
tuner.fit()
