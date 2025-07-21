from efficient_kan import KAN
import torch 
import torch.optim as optim
import numpy as np 
from utils import NumPy2TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Set default torch type to float64
torch.set_default_dtype(torch.float64)

num_epochs = 400
history_size = 10  # history size for LBFGS optimizer 
layers_hidden = [7, 44, 44, 71, 451]  # specify KAN architecture
batch_size = 100
print(f"layers_hidden: {layers_hidden}")
print(f"batch_size: {batch_size}")
print(f"num_epochs: {num_epochs}")

model_save_path = '/projects/jodo2960/KAN/21cmKAN/models/21cmKAN_model_21cmGEM_default2_20.pth'
data_path = '/projects/jodo2960/KAN/21cmKAN/data/'

# Create training and validation Datasets 
train_dataset = NumPy2TensorDataset(features_npy_file=data_path + 'X_train_21cmGEM.npy', 
                                    targets_npy_file=data_path + 'y_train_21cmGEM.npy')

val_dataset = NumPy2TensorDataset(features_npy_file=data_path + 'X_val_21cmGEM.npy', 
                                  targets_npy_file=data_path + 'y_val_21cmGEM.npy')                               

# Create training DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Grab validation data and put it on the GPU 
# TODO: This does not make sense, we should not use Dataset for this, but is ok for now
X_val_21cmGEM = torch.from_numpy(val_dataset.features)
y_val_21cmGEM = torch.from_numpy(val_dataset.targets)
X_val_21cmGEM = X_val_21cmGEM.to(device)
y_val_21cmGEM = y_val_21cmGEM.to(device)

# initialize model 
model = KAN(layers_hidden=layers_hidden, grid_size=7, spline_order=3)
model.to(device)

# Calculate the absolute maximum for each signal's frequency channel value
max_abs = torch.abs(y_val_21cmGEM).min(dim=1)[0]

# Define optimizer 
# optimizer = optim.LBFGS(model.parameters(), lr=1, tolerance_grad=1e-32, tolerance_change=1e-32, history_size=history_size, line_search_fn='strong_wolfe')
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Define learning rate scheduler TODO: look into using this in the future 
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
        #sqrt_MSE = torch.sqrt(val_loss)
        #error = (sqrt_MSE/max_abs)*100
        #mean_error = torch.mean(error)
        #max_error = error.max()

    # Update learning rate 
    # scheduler.step()
    # scheduler.step(max_error)
    print(f"Epoch: {epoch + 1}, Val. loss: {val_loss}, Train loss: {loss}")

torch.save(model, model_save_path)
