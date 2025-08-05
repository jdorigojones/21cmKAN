import h5py
import numpy as np
import os
import torch 
import torch.optim as optim
from efficient_kan import KAN
from utils import NumPy2TensorDataset
from utils import NumPyArray2TensorDataset
from Global21cmKAN import __path__

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Set default torch type to float64
torch.set_default_dtype(torch.float64)

# specify KAN model architecture and configs for training on your model
input_nodes = 10
hidden_layer1_nodes = 10
hidden_layer2_nodes = 10
hidden_layer3_nodes = 10
output_nodes = 100
layer_nodes = [input_nodes, hidden_layer1_nodes, hidden_layer2_nodes, hidden_layer3_nodes, output_nodes]
grid_size = 7
spline_order = 3
num_epochs = 400
batch_size = 100
#history_size = 10  # history size for LBFGS optimizer 
print(f"nodes in each layer: {layer_nodes}")
print(f"number of grid intervals in B-splines: {grid_size}")
print(f"order of splines in B-splines: {spline_order}")
print(f"number of training epochs: {num_epochs}")
print(f"training batch size: {batch_size}")

# define path containing saved trained 21cmKAN networks and min/max training set values used for preprocessing
PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"
model_save_path = PATH+"models/emulator_yourmodel.pth"
train_mins_yourmodel = np.load(PATH+"models/train_mins_yourmodel.npy")
train_maxs_yourmodel = np.load(PATH+"models/train_maxs_yourmodel.npy")

z_list = np.linspace(1, 100, 100) # list of redshifts for signals in your model
vr = 1420.4057517667  # rest frequency of 21 cm line in MHz
# Load in unnormalized training, validation, and test data made by your model as numpy arrays; UNCOMMENT
with h5py.File(PATH + 'dataset_yourmodel.h5', "r") as f:
    print("Keys: %s" % f.keys())
    par_train = np.asarray(f['par_train'])[()]
    par_val = np.asarray(f['par_val'])[()]
    X_test_yourmodel_true = np.asarray(f['par_test'])[()]
    signal_train = np.asarray(f['signal_train'])[()]
    signal_val = np.asarray(f['signal_val'])[()]
    y_test_yourmodel_true = np.asarray(f['signal_test'])[()]
f.close()

# now preprocess/normalize the input physical parameters values in your training and validation sets
# see the other emulator scripts for examples of data preprocessing
# includes taking the log10 of certain parameters and then performing global MinMax normalizations
# train_mins_yourmodel[i] is the minimum value of parameter i in the training set, used for normalization
# train_maxs_yourmodel[i] is the maximum value of parameter i in the training set, used for normalization
X_train_yourmodel = np.array([]) # define your normalized training set data/parameters
X_val_yourmodel = np.array([]) # define your normalized validation set data/parameters

# now preprocess/normalize the signals (dT_b values) in your training and validaton sets
# see the other emulator scripts for examples of data preprocessing
# includes performing global MinMax normalizations
# train_mins_yourmodel[-1] is the absolute minimum dT_b value in the training set, used for normalization
# train_maxs_yourmodel[-1] is the absolute maximum dT_b value in the training set, used for normalization
y_train_yourmodel = np.array([]) # define your normalized training set signals
y_val_yourmodel = np.array([]) # define your normalized validation set signals

# Calculate the absolute minimum value of each normalized validation set signal, used to compute relative error
min_abs = torch.abs(y_val_yourmodel).min(dim=1)[0]

# Create normalized training Dataset and DataLoader
train_dataset = NumPyArray2TensorDataset(features_npy=X_train_yourmodel, targets_npy=y_train_yourmodel)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def model(layers_hidden=layer_nodes, grid_size=grid_size, spline_order=spline_order, name=None):
    """
    Generate a 21cmKAN model

    Parameters
    ----------
    layers_hidden : np.ndarray
        array containing the number of nodes in each network layer
        first value is the number of input layer nodes
        next three values are the number of nodes in each of the three hidden layers
        last value is the number of output layer nodes
    grid_size : int
        number of grid intervals in parameterized B-spline activation functions. Default: 7
    spline_order : int
        order of individual splines in parameterized B-spline activation functions. Default: 3
    name : str or None
        Name of the model. Default : None

    Returns
    -------
    model : tf.keras.Model
        The generated model
    """

    model = KAN(layers_hidden, grid_size, spline_order)
    return model

def frequency(z):
    """
    Convert redshift to frequency

    Parameters
    ----------
    z : float or np.ndarray
        The redshift or array of redshifts to convert

    Returns
    -------
    nu : float or np.ndarray
        The corresponding frequency or array of frequencies in MHz
    """
    nu = vr/(z+1)
    return nu

def redshift(nu):
    """
    Convert frequency to redshift

    Parameters
    ----------
    nu : float or np.ndarray
        The frequency or array of frequencies in MHz to convert

    Returns
    -------
    z : float or np.ndarray
        The corresponding redshift or array of redshifts
    """
    z = (vr/nu)-1
    return z

def error(true_signal, emulated_signal, relative=True, nu=None, nu_low=None, nu_high=None):
    """
    Compute the relative rms error (Eq. 3 in DJ+25) between the true and emulated signal(s)

    Parameters
    ----------
    true_signal : np.ndarray
        The true signal(s) created by the model on which the emulator is trained
        An array of brightness temperatures for different redshifts or frequencies
    emulated_signal : np.ndarray
        The emulated signal(s). Must be same shape as true_signal
    relative : bool
        True to compute the rms error in relative (%) units. False for absolute (mK) units. Default : True
    nu : np.ndarray or None
        The array of frequencies corresponding to each signal
        Needed for computing the error in different frequency bands. Default : None.
    nu_low : float or None
        The lower bound of the frequency band to compute the error in
        Cannot be set without nu. Default : None
    nu_high : float or None
        The upper bound of the frequency bnd to compute the error in
        Cannot be set without nu. Default : None

    Returns
    -------
    err : float or np.ndarray
        The computed rms error for each input signal

    Raises
    ------
    ValueError : If nu is None and nu_low or nu_high are not None
    """
    if (nu_low or nu_high) and nu is None:
        raise ValueError("Cannot compute error because no frequency array is given.")
    if len(emulated_signal.shape) == 1:
        emulated_signal = np.expand_dims(emulated_signal, axis=0)
        true_signal = np.expand_dims(true_signal, axis=0)

    if nu_low and nu_high:
        nu_where = np.argwhere((nu >= nu_low) & (nu <= nu_high))[:, 0]
    elif nu_low:
        nu_where = np.argwhere(nu >= nu_low)
    elif nu_high:
        nu_where = np.argwhere(nu <= nu_high)

    if nu_low or nu_high:
        emulated_signal = emulated_signal[:, nu_where]
        true_signal = true_signal[:, nu_where]

    err = np.sqrt(np.mean((emulated_signal - true_signal)**2, axis=1))
    if relative:  # return the rms error as a fraction of the signal amplitude in the chosen frequency band
        err /= np.max(np.abs(true_signal), axis=1)
        err *= 100 # convert to per cent (%)
    return err

class Emulate:
    def __init__(
        self,
        par_train=X_train_yourmodel,
        par_val=X_val_yourmodel,
        par_test=X_test_yourmodel_true,
        signal_train=y_train_yourmodel,
        signal_val=y_val_yourmodel,
        signal_test=y_test_yourmodel_true,
        redshifts=z_list,
        frequencies=None):
        """
        The emulator class for building, training, and using 21cmKAN to emulate signals in your model
        The default parameters for training/testing 21cmKAN on the 21cmGEM and ARES sets are described in Section 2.2 of DJ+25

        Parameters
        ----------
        par_train : np.ndarray
            Parameters in training set (normalized, used in train() via train_dataloader)
        par_val : np.ndarray
            Parameters in validation set (normalized, used in train())
        par_test : np.ndarray
            Parameters in test set (unnormalized, used in test_error())
        signal_train : np.ndarray
            Signals in training set (normalized, used in train() via train_dataloader)
        signal_val : np.ndarray
            Signals in validation set (normalized, used in train())
        signal_test : np.ndarray
            Signals in test set (unnormalized, used in test_error())
        redshifts : np.ndarray or None
            Array of redshifts corresponding to each signal
        frequencies : np.ndarray or None
            Array of frequencies corresponding to each signal

        Attributes
        ----------
        par_train : np.ndarray
            Parameters in training set
        par_val : np.ndarray
            Parameters in validation set
        par_test : np.ndarray
            Parameters in test set
        signal_train : np.ndarray
            Signals in training set
        signal_val : np.ndarray
            Signals in validation set
        signal_test : np.ndarray
            Signals in test set
        par_labels : list of str
            Names of the physical parameters
        emulator : tf.keras.Model
            The emulator model
        redshifts : np.ndarray or None
            Array of redshifts corresponding to each signal
        frequencies : np.ndarray or None
            Array of frequencies corresponding to each signal

        Methods
        -------
        load_model : load an existing instance of 21cmKAN trained on your data
        train : train the emulator on your data
        predict : use the emulator to predict global 21 cm signal(s) from input physical parameters
        test_error : compute the rms error of the emulator evaluated on your test set
        save : save the model class instance with all attributes
        """
        self.par_train = par_train
        self.par_val = par_val
        self.par_test = par_test
        self.signal_train = signal_train
        self.signal_val = signal_val
        self.signal_test = signal_test

        self.par_labels = [r''] # fill with your parameter labels

        self.emulator = model(layer_nodes, grid_size, spline_order, name="emulator_yourmodel")

        self.train_mins = train_mins_yourmodel
        self.train_maxs = train_maxs_yourmodel

        if frequencies is None:
            if redshifts is not None:
                frequencies = frequency(redshifts)
        elif redshifts is None:
            redshifts = redshift(frequencies)
        self.redshifts = redshifts
        self.frequencies = frequencies

    def load_model(self, model_path=model_save_path):
        """
        Load a saved model instance of 21cmKAN trained on your data.

        Parameters
        ----------
        model_path : str
            The path to the saved model instance

        Raises
        ------
        IOError : if model_path does not point to a valid model instance
        """
        self.emulator = torch.load(model_path, weights_only=False)
        self.emulator.to(device)

    def train(self, num_epochs=num_epochs, batch_size=batch_size, callbacks=[], verbose=2, shuffle='True'):
        """
        Train an instance of 21cmKAN on your data set

        Parameters
        ----------
        epochs : int
            Number of epochs to train the network for the given batch_size
            Default : 400
        batch_size : int
            Number of signals whose loss values are averaged at a time to update the network weights during each epoch
            Default : 100
        callbacks : list of tf.keras.callbacks.Callback
            Callbacks to pass to the training loop. Default : []
        verbose : 0, 1, 2
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default : 2

        Returns
        -------
        train_losses : list of floats
           Training set loss values for each epoch
        val_losses : list of floats
           Validation set loss values for each epoch
        """
        optimizer = optim.AdamW(self.emulator.parameters(), lr=1e-3, weight_decay=1e-4) # Define optimizer
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            self.emulator.train()
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                def closure():
                    global train_loss
                    optimizer.zero_grad()
                    pred = self.emulator(inputs)
                    train_loss = torch.mean((pred-targets)**2)
                    train_loss.backward()
                    train_losses.append(train_loss)
                    return train_loss
                optimizer.step(closure)

            self.emulator.eval()
            with torch.no_grad():
                val_pred = self.emulator(self.par_val)
                val_loss = torch.mean((val_pred-self.signal_val)**2)
                val_losses.append(val_loss)
                # code to compute the normalized validation set relative RMSE, if desired to print along with loss
                #RMSE = torch.sqrt(val_loss)
                #rel_error = (RMSE/min_abs)*100
                #mean_rel_error = torch.mean(rel_error)
                #max_rel_error = rel_error.max()
            
            # code to update the learning rate, if desired: scheduler.step() ; scheduler.step(max_error)
            print(f"Epoch: {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

        # save the trained network; overwrites the saved network included in the repository; update model_save_path if this is not desired
        torch.save(self.emulator, model_save_path)
        return (train_losses, val_losses)

    def predict(self, params):
        """
        Predict global 21 cm signal(s) from input physical parameters using trained instance of 21cmKAN

        Parameters
        ----------
        params : np.ndarray
            The values of the physical parameters in the order of par_labels. Input 2D array to predict a set of signals

        Returns
        -------
        emulated_signals : np.ndarray
           The predicted global 21 cm signal(s)
        """
        if len(np.shape(params)) == 1:
            params = np.expand_dims(params, axis=0) # if doing one signal at a time

        # include the same parameter preprocessing code performed earlier to transform the input
        # physical parameters to normalized parameters that can be used to train 21cmKAN
        proc_params = np.array([]) # your processed parameters numpy array
        proc_params_test = torch.from_numpy(proc_params)
        proc_params_test = proc_params_test.to(device)

        self.emulator.eval()
        with torch.no_grad():
            result = self.emulator(proc_params_test) # evaluate trained instance of 21cmKAN with processed parameters

        result = result.cpu().detach().numpy()
        unproc_signals = result.copy()
        unproc_signals = (result*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1] # denormalize signals
        unproc_signals = unproc_signals[:,::-1] # flip signals to be from high-z to low-z
        if unproc_signals.shape[0] == 1:
            return unproc_signals[0,:]
        else:
            return unproc_signals

    def test_error(self, relative=True, nu_low=None, nu_high=None):
        """
        Compute the rms error for the loaded trained instance of 21cmKAN evaluated on your test set

        Parameters
        ----------
        relative : bool
            True to compute the rms error in relative (%) units. False for absolute (mK) units. Default : True
        nu_low : float or None
            The lower bound of the frequency band to compute the error in
            Default : None
        nu_high : float or None
            The upper bound of the frequency band to compute the error in
            Default : None

        Returns
        -------
        err : np.ndarray
            The computed rms errors
        """
        err = error(
            self.signal_test,
            self.predict(self.par_test),
            relative=relative,
            nu=self.frequencies,
            nu_low=nu_low,
            nu_high=nu_high)
        return err

