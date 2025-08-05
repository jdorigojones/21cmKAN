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

# specify KAN model architecture and configs for training on the ARES set; see Sec 2.3 of Dorigo Jones et al. 2025
layer_nodes = [8, 44, 44, 71, 449]
grid_size = 7
spline_order = 3
num_epochs = 800
batch_size = 100
#history_size = 10  # history size for LBFGS optimizer 
print(f"nodes in each layer: {layer_nodes}")
print(f"number of grid intervals in B-splines: {grid_size}")
print(f"order of splines in B-splines: {spline_order}")
print(f"number of training epochs: {num_epochs}")
print(f"training batch size: {batch_size}")

# define paths of the saved trained 21cmKAN networks and min/max training set values used for preprocessing
PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"
model_save_path = PATH+"models/emulator_ARES.pth"
train_mins_ARES = np.load(PATH+"models/train_mins_ARES.npy")
train_maxs_ARES = np.load(PATH+"models/train_maxs_ARES.npy")

z_list = np.linspace(5.1, 49.9, 449) # list of redshifts for ARES signals; equiv to np.arange(5.1, 50, 0.1)
vr = 1420.4057517667  # rest frequency of 21 cm line in MHz
# Load in unnormalized training, validation, and test data
with h5py.File(PATH + 'dataset_ARES.h5', "r") as f:
    print("Keys: %s" % f.keys())
    par_train = np.asarray(f['train_data'])[()]
    par_val = np.asarray(f['val_data'])[()]
    X_test_ARES_true = np.asarray(f['test_data'])[()]
    signal_train = np.asarray(f['train_labels'])[()]
    signal_val = np.asarray(f['val_labels'])[()]
    y_test_ARES_true = np.asarray(f['test_labels'])[()]
f.close()

# preprocess/normalize input physical parameters values of training and validation sets
unproc_c_X_train = par_train[:,0].copy() # c_X, normalization of X-ray luminosity-SFR relation
unproc_T_min_train = par_train[:,2].copy() # T_min, minimum temperature of star-forming halos
unproc_f_s_train = par_train[:,4].copy() # f_*,0, peak star formation efficiency 
unproc_M_p_train = par_train[:,5].copy() # M_p, dark matter halo mass at f_*,0
unproc_c_X_train = np.log10(unproc_c_X_train)
unproc_T_min_train = np.log10(unproc_T_min_train)
unproc_f_s_train = np.log10(unproc_f_s_train)
unproc_M_p_train = np.log10(unproc_M_p_train)
parameters_log_train = np.empty(par_train.shape)
parameters_log_train[:,0] = unproc_c_X_train
parameters_log_train[:,1] = par_train[:,1].copy()
parameters_log_train[:,2] = unproc_T_min_train
parameters_log_train[:,3] = par_train[:,3].copy()
parameters_log_train[:,4] = unproc_f_s_train
parameters_log_train[:,5] = unproc_M_p_train
parameters_log_train[:,6] = par_train[:,6].copy()
parameters_log_train[:,7] = par_train[:,7].copy()

unproc_c_X_val = par_val[:,0].copy() # c_X, normalization of X-ray luminosity-SFR relation
unproc_T_min_val = par_val[:,2].copy() # T_min, minimum temperature of star-forming halos
unproc_f_s_val = par_val[:,4].copy() # f_*,0, peak star formation efficiency 
unproc_M_p_val = par_val[:,5].copy() # M_p, dark matter halo mass at f_*,0
unproc_c_X_val = np.log10(unproc_c_X_val)
unproc_T_min_val = np.log10(unproc_T_min_val)
unproc_f_s_val = np.log10(unproc_f_s_val)
unproc_M_p_val = np.log10(unproc_M_p_val)
parameters_log_val = np.empty(par_val.shape)
parameters_log_val[:,0] = unproc_c_X_val
parameters_log_val[:,1] = par_val[:,1].copy()
parameters_log_val[:,2] = unproc_T_min_val
parameters_log_val[:,3] = par_val[:,3].copy()
parameters_log_val[:,4] = unproc_f_s_val
parameters_log_val[:,5] = unproc_M_p_val
parameters_log_val[:,6] = par_val[:,6].copy()
parameters_log_val[:,7] = par_val[:,7].copy()

N_proc_train = np.shape(parameters_log_train)[0] # number of signals (i.e., parameter sets) to process
p_train = np.shape(par_train)[1] # number of input parameters (# of physical params)
proc_params_train = np.zeros((N_proc_train,p_train))

N_proc_val = np.shape(parameters_log_val)[0] # number of signals (i.e., parameter sets) to process
p_val = np.shape(par_val)[1] # number of input parameters (# of physical params)
proc_params_val = np.zeros((N_proc_val,p_val))

for i in range(p_train):
    x_train = parameters_log_train[:,i]
    proc_params_train[:,i] = (x_train-train_mins_ARES[i])/(train_maxs_ARES[i]-train_mins_ARES[i])

for i in range(p_val):
    x_val = parameters_log_val[:,i]
    proc_params_val[:,i] = (x_val-train_mins_ARES[i])/(train_maxs_ARES[i]-train_mins_ARES[i])

#X_train_ARES = torch.from_numpy(proc_params_train)
X_train_ARES = proc_params_train.copy()
proc_params_train = 0
par_train = 0
#X_train_ARES = X_train_ARES.to(device)

X_val_ARES = torch.from_numpy(proc_params_val)
proc_params_val = 0
par_val = 0
X_val_ARES = X_val_ARES.to(device)

# preprocess/normalize signals (dT_b) in training and validaton sets
proc_signals_train = signal_train.copy()
proc_signals_train = (signal_train - train_mins_ARES[-1])/(train_maxs_ARES[-1]-train_mins_ARES[-1])  # global Min-Max normalization
proc_signals_train = proc_signals_train[:,::-1] # flip signals to be from high-z to low-z
#y_train_ARES = torch.from_numpy(proc_signals_train)
y_train_ARES = proc_signals_train.copy()
proc_signals_train = 0
signal_train = 0
#y_train_ARES = y_train_ARES.to(device)

proc_signals_val = signal_val.copy()
proc_signals_val = (signal_val - train_mins_ARES[-1])/(train_maxs_ARES[-1]-train_mins_ARES[-1])  # global Min-Max normalization
proc_signals_val = proc_signals_val[:,::-1] # flip signals to be from high-z to low-z
y_val_ARES = torch.from_numpy(proc_signals_val.copy())
proc_signals_val = 0
signal_val = 0
y_val_ARES = y_val_ARES.to(device)

# Calculate the absolute minimum value of each normalized validation set signal, used to compute relative error
min_abs = torch.abs(y_val_ARES).min(dim=1)[0]

# Create normalized training Dataset and DataLoader
train_dataset = NumPyArray2TensorDataset(features_npy=X_train_ARES, targets_npy=y_train_ARES)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def model(layers_hidden=layer_nodes, grid_size=grid_size, spline_order=spline_order, name=None):
    """
    Generate a 21cmKAN model

    Parameters
    ----------
    layers_hidden : np.ndarray
        array containing the number of nodes in each network layer
        first value is the number of input layer nodes. Default: 8, for 8 physical parameters in the ARES set
        next three values are the number of nodes in each hidden layer. Default: 44, 44, 71
        last value is the number of output layer nodes. Default: 449, for 449 frequencies/redshifts in the ARES set
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
        par_train=X_train_ARES,
        par_val=X_val_ARES,
        par_test=X_test_ARES_true,
        signal_train=y_train_ARES,
        signal_val=y_val_ARES,
        signal_test=y_test_ARES_true,
        redshifts=z_list,
        frequencies=None):
        """
        The emulator class for building, training, and using 21cmKAN to emulate ARES signals 
        The default parameters are for training/testing 21cmKAN on the ARES set described in Section 2.2 of DJ+25

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
        load_model : load an existing instance of 21cmKAN trained on ARES data
        train : train the emulator on ARES data
        predict : use the emulator to predict global signal(s) from input physical parameters
        test_error : compute the rms error of the emulator evaluated on the test set
        save : save the model class instance with all attributes
        """
        self.par_train = par_train
        self.par_val = par_val
        self.par_test = par_test
        self.signal_train = signal_train
        self.signal_val = signal_val
        self.signal_test = signal_test

        self.par_labels = [r'$c_X$', r'$f_{\rm esc}$', r'${T_{\rm min}}$', r'$\log_{10}{N_{H I}}$',
                           r'$f_{\star, 0}$',r'$M_p$',r'$\gamma_{lo}$', r'$\gamma_{hi}$']

        self.emulator = model(layer_nodes, grid_size, spline_order, name="emulator_ARES")

        self.train_mins = train_mins_ARES
        self.train_maxs = train_maxs_ARES

        if frequencies is None:
            if redshifts is not None:
                frequencies = frequency(redshifts)
        elif redshifts is None:
            redshifts = redshift(frequencies)
        self.redshifts = redshifts
        self.frequencies = frequencies

    def load_model(self, model_path=model_save_path):
        """
        Load a saved model instance of 21cmKAN trained on ARES data.
        The instance of 21cmKAN trained on ARES included in this repository is
        the same one used to perform nested sampling analyses in DJ+25 (Section 3.3)

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
        Train an instance of 21cmKAN on the ARES data set

        Parameters
        ----------
        epochs : int
            Number of epochs to train the network for the given batch_size
            Default : 800
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
                    return train_loss
                optimizer.step(closure)

            self.emulator.eval()
            with torch.no_grad():
                val_pred = self.emulator(self.par_val)
                val_loss = torch.mean((val_pred-self.signal_val)**2)
                # code to compute the normalized validation set relative RMSE, if desired to print along with loss
                #RMSE = torch.sqrt(val_loss)
                #rel_error = (RMSE/min_abs)*100
                #mean_rel_error = torch.mean(rel_error)
                #max_rel_error = rel_error.max()
            
            # code to update the learning rate, if desired: scheduler.step() ; scheduler.step(max_error)
            print(f"Epoch: {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # save the trained network; overwrites the saved network included in the repository; update model_save_path if this is not desired
        torch.save(self.emulator, model_save_path)
        return (train_losses, val_losses)

    def predict(self, params):
        """
        Predict global 21 cm signal(s) from input ARES physical parameters using trained instance of 21cmKAN

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
            
        unproc_c_X = params[:,0].copy() # c_X, normalization of X-ray luminosity-SFR relation
        unproc_T_min = params[:,2].copy() # T_min, minimum temperature of star-forming halos
        unproc_f_s = params[:,4].copy() # f_*,0, peak star formation efficiency 
        unproc_M_p = params[:,5].copy() # M_p, dark matter halo mass at f_*,0
        unproc_c_X = np.log10(unproc_c_X)
        unproc_T_min = np.log10(unproc_T_min)
        unproc_f_s = np.log10(unproc_f_s)
        unproc_M_p = np.log10(unproc_M_p)
        parameters_log = np.empty(params.shape)
        parameters_log[:,0] = unproc_c_X
        parameters_log[:,1] = params[:,1].copy()
        parameters_log[:,2] = unproc_T_min
        parameters_log[:,3] = params[:,3].copy()
        parameters_log[:,4] = unproc_f_s
        parameters_log[:,5] = unproc_M_p
        parameters_log[:,6] = params[:,6].copy()
        parameters_log[:,7] = params[:,7].copy()
        
        N_proc = np.shape(parameters_log)[0] # number of signals (i.e., parameter sets) to process
        p = np.shape(params)[1] # number of input parameters (# of physical params)
        proc_params = np.zeros((N_proc,p))
        
        for i in range(p):
            x = parameters_log[:,i]
            proc_params[:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])

        proc_params_test = torch.from_numpy(proc_params)
        proc_params = 0
        params = 0
        parameters_log = 0
        proc_params_test = proc_params_test.to(device)

        self.emulator.eval()
        with torch.no_grad():
            result = self.emulator(proc_params_test) # evaluate trained instance of 21cmKAN with processed parameters

        result = result.cpu().detach().numpy()
        unproc_signals = result.copy()
        unproc_signals = (result*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1] # unpreprocess (i.e., denormalize) signals
        unproc_signals = unproc_signals[:,::-1] # flip signals to be from high-z to low-z
        if unproc_signals.shape[0] == 1:
            return unproc_signals[0,:]
        else:
            return unproc_signals

    def test_error(self, relative=True, nu_low=None, nu_high=None):
        """
        Compute the rms error for the loaded trained instance of 21cmKAN evaluated on the 1,704-signal ARES test set

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


