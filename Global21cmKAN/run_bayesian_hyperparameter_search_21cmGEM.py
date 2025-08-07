from efficient_kan import KAN
import torch 
import torch.optim as optim
from utils import *
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.train import RunConfig
import os 

def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set default torch type to float64
    torch.set_default_dtype(torch.float64)

    # Get validation Dataset 
    val_dataset = NumPy2TensorDataset(features_npy_file=config["data_path"] + 'X_val_21cmGEM.npy', 
                                    targets_npy_file=config["data_path"] + 'y_val_21cmGEM.npy')    

    # This does not make sense, should not use dataset, but is ok for now
    X_val_21cmGEM = torch.from_numpy(val_dataset.features)
    y_val_21cmGEM = torch.from_numpy(val_dataset.targets)
    X_val_21cmGEM = X_val_21cmGEM.to(device)
    y_val_21cmGEM = y_val_21cmGEM.to(device)

    # Calculate the absolute maximum for each batch's frequency channel value
    max_abs = torch.abs(y_val_21cmGEM).max(dim=1)[0]

    # Grab training Dataset
    train_dataset = NumPy2TensorDataset(features_npy_file=config["data_path"] + 'X_train_21cmGEM.npy', 
                                        targets_npy_file=config["data_path"] + 'y_train_21cmGEM.npy')
                                
    # Create training DataLoader
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
                    global loss
                    optimizer.zero_grad()
                    pred = model(inputs)
                    loss = torch.mean((pred-targets)**2)
                    loss.backward()
                    return loss 
                optimizer.step(closure)

            # Calculate max error of validation data set 
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_21cmGEM)
                sqrt_MSE = torch.sqrt(torch.mean((val_pred-y_val_21cmGEM)**2, dim=1))
                error = (sqrt_MSE/max_abs)*100
                max_error = error.max()
                max_error = max_error.cpu().detach().numpy()

        except Exception as e:
            print(f"Error in trial: {e}")
            max_error = 99999

    test_mean_rel_err, test_median_rel_err, test_max_rel_err = eval_test_set_21cmGEM(model, data_path, device)

    file_name = create_file_name(config, test_max_rel_err)

    torch.save(model, config["model_save_dir"] + file_name)
    tune.report({"max_error": max_error, "test_mean_rel_err": test_mean_rel_err, "test_median_rel_err": test_median_rel_err, "test_max_rel_err": test_max_rel_err}, checkpoint=None)

# Define directories to store results produced by Ray Tune and models produced
results_dir = "/scratch/alpine/jodo2960/21cm_ray_tune_KAN_results"
model_save_dir = results_dir + "/saved_models/"

# Define path to training, validation, and test data 
data_path = "/projects/jodo2960/KAN/data/"

# Define the name of the trial directory for Ray Tune 
trial_directory = "test_experiment_3"

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
    "model_save_dir": model_save_dir, 
    "data_path": data_path
}

# Use Optuna to perform Bayesian hyperparameter optimization using 
# max_error produced by the validation set  
optuna_search = OptunaSearch(metric="max_error", mode="min")

# Split up the GH200 so that we can run multiple trials at once 
trainable_with_cpu_gpu = tune.with_resources(train_model, {"cpu": 4, "gpu": 0.20}) # this is for the A100
# this was for the GH200; might not allocate enough with "cpu": 2, "gpu": 0.03; use cpu : 7, and gpu : 0.10 instead

# Define the number of samples to take from search space 
num_samples = 1000 # 800

# Define the maximum number of trials that can take place in parallel 
max_concurrent_trials = 5 #36  # if doing tenth of the GPU, have 10 run parallel # 10 for 6 cpu, 0.1 gpu

# Specify Tuner 
#tuner = tune.Tuner(
#    trainable_with_cpu_gpu,
#    param_space=config,
#    tune_config=tune.TuneConfig(
#        search_alg=optuna_search,
#        num_samples=num_samples, 
#        max_concurrent_trials=max_concurrent_trials
#    ),
#    run_config=RunConfig(storage_path=results_dir, name=trial_directory)
#)

# resume previous tune experiment that was cancelled due to time
tuner = tune.Tuner.restore(os.path.expanduser("/scratch/alpine/jodo2960/21cm_ray_tune_KAN_results/test_experiment_3"),
        trainable=trainable_with_cpu_gpu,
        resume_errored=True)

# Perform hyperparameter search 
tuner.fit()
