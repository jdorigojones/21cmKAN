import h5py
import numpy as np
import os
import torch 
import torch.optim as optim
from efficient_kan import KAN
from utils import NumPy2TensorDataset
from Global21cmKAN import __path__

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

#PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/Global21cmKAN/"
PATH = "/projects/jodo2960/KAN/21cmKAN/"
model_save_path = PATH+"models/emulator_21cmGEM.pth"
data_path = PATH+"data/"

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

X_train_21cmGEM = torch.from_numpy(train_dataset.features)
y_train_21cmGEM = torch.from_numpy(train_dataset.targets)
X_train_21cmGEM = X_train_21cmGEM.to(device)
y_train_21cmGEM = y_train_21cmGEM.to(device)

# Load in unnormalized test data and data needed to transform the output of the model
train_maxs_21cmGEM = np.load(data_path + 'train_maxs_21cmGEM.npy')
train_mins_21cmGEM = np.load(data_path + 'train_mins_21cmGEM.npy')
X_test_21cmGEM_true = np.load(data_path + 'X_test_true_21cmGEM.npy')
y_test_21cmGEM_true = np.load(data_path + 'signals_21cmGEM_true.npy')

# Calculate the absolute maximum for each signal's frequency channel value
max_abs = torch.abs(y_val_21cmGEM).min(dim=1)[0]


#PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmLSTM/"
z_list = np.linspace(5, 50, 451) # list of redshifts for 21cmGEM signals; equiv to np.arange(5, 50.1, 0.1)
vr = 1420.4057517667  # rest frequency of 21 cm line in MHz
#with h5py.File(PATH + 'dataset_21cmGEM.h5', "r") as f:
#    print("Keys: %s" % f.keys())
#    par_train = np.asarray(f['par_train'])[()]
#    par_val = np.asarray(f['par_val'])[()]
#    par_test = np.asarray(f['par_test'])[()]
#    signal_train = np.asarray(f['signal_train'])[()]
#    signal_val = np.asarray(f['signal_val'])[()]
#    signal_test = np.asarray(f['signal_test'])[()]
#f.close()

def model(layers_hidden, name=None):
    """
    Generate a 21cmKAN model

    Parameters
    ----------
    num_params : int
        Number of physical parameters plus one for the frequency step (e.g., 8 for 21cmGEM set; 9 for ARES set)
    dim_output : int
        Dimensionality of the fully-connected output layer of the model
    name : str or None
        Name of the model. Default : None

    Returns
    -------
    model : tf.keras.Model
        The generated model
    """

    model = KAN(layers_hidden=layers_hidden, grid_size=7, spline_order=3) #model.to(device)
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
    Compute the relative rms error (Eq. 3 in DJ+24) between the true and emulated signal(s)

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
        par_train=X_train_21cmGEM,
        par_val=X_val_21cmGEM,
        par_test=X_test_21cmGEM_true,
        signal_train=y_train_21cmGEM,
        signal_val=y_val_21cmGEM,
        signal_test=y_test_21cmGEM_true,
        redshifts=z_list,
        frequencies=None):
        """
        The emulator class for building, training, and using 21cmLSTM to emulate 21cmGEM signals 
        The default parameters are for training/testing 21cmLSTM on the 21cmGEM set described in Section 2.2 of DJ+24

        Parameters
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
        load_model : load an existing instance of 21cmKAN trained on 21cmGEM data
        train : train the emulator on 21cmGEM data
        predict : use the emulator to predict global signals from input physical parameters
        test_error : compute the rms error of the emulator evaluated on the test set
        save : save the model class instance with all attributes
        """
        self.par_train = par_train
        self.par_val = par_val
        self.par_test = par_test
        self.signal_train = signal_train
        self.signal_val = signal_val
        self.signal_test = signal_test

        self.par_labels = [r'$f_*$', r'$V_c$', r'$f_X$', r'$\tau$',
                           r'$\alpha$', r'$\nu_{\rm min}$', r'$R_{\rm mfp}$']

        self.emulator = model(layers_hidden, name="emulator_21cmGEM")

        self.train_mins = train_mins_21cmGEM
        self.train_maxs = train_maxs_21cmGEM

        if frequencies is None:
            if redshifts is not None:
                frequencies = frequency(redshifts)
        elif redshifts is None:
            redshifts = redshift(frequencies)
        self.redshifts = redshifts
        self.frequencies = frequencies

    def load_model(self, model_path=model_save_path):
        """
        Load a saved model instance of 21cmKAN trained on the 21cmGEM data set.
        The instance of 21cmKAN trained on 21cmGEM included in this repository is the one
        used to perform nested sampling analyses in DJ+25

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
        Train an instance of 21cmKAN on the 21cmGEM data set

        Parameters
        ----------
        epochs : int
            Number of epochs to train for the given batch_size
        batch_size : int
            Number of signals in each minibatch trained on
        callbacks : list of tf.keras.callbacks.Callback
            Callbacks to pass to the training loop. Default : []
        verbose : 0, 1, 2
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default : 2

        Returns
        -------
        train_loss : list of floats
           Training set loss values for each epoch
        val_loss : list of floats
           Validation set loss values for each epoch
        """

        optimizer = optim.AdamW(self.emulator.parameters(), lr=1e-3, weight_decay=1e-4) # Define optimizer
        for epoch in range(num_epochs):
            self.emulator.train()
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                def closure():
                    global loss
                    optimizer.zero_grad()
                    pred = self.emulator(inputs)
                    loss = torch.mean((pred-targets)**2)
                    loss.backward()
                    return loss
                optimizer.step(closure)

            self.emulator.eval()
            with torch.no_grad():
                val_pred = self.emulator(X_val_21cmGEM)
                sqrt_MSE = torch.sqrt(torch.mean((val_pred-y_val_21cmGEM)**2, dim=1))
                error = (sqrt_MSE/max_abs)*100
                mean_error = torch.mean(error)
                max_error = error.max()

            # Update learning rate 
            # scheduler.step()
            # scheduler.step(max_error)
            print(f"Epoch: {epoch + 1}, Mean Error: {mean_error}, Max Error: {max_error}, Loss: {loss}")

        torch.save(self.emulator, model_save_path)

        return loss

    def predict(self, params):
        """
        Predict global 21 cm signal(s) from input 21cmGEM physical parameters using trained instance of 21cmKAN

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
            
        unproc_f_s = params[:,0].copy() # f_*, star formation efficiency, # preprocess input physical parameters
        unproc_V_c = params[:,1].copy() # V_c, minimum circular velocity of star-forming halos 
        unproc_f_X = params[:,2].copy() # f_X, X-ray efficiency of sources
        unproc_f_s = np.log10(unproc_f_s)
        unproc_V_c = np.log10(unproc_V_c)
        unproc_f_X[unproc_f_X == 0] = 1e-6 # for f_X, set zero values to 1e-6 before taking log_10
        unproc_f_X = np.log10(unproc_f_X)
        parameters_log = np.empty(params.shape)
        parameters_log[:,0] = unproc_f_s
        parameters_log[:,1] = unproc_V_c
        parameters_log[:,2] = unproc_f_X
        parameters_log[:,3:] = params[:,3:].copy()
        
        N_proc = np.shape(parameters_log)[0] # number of signals (i.e., parameter sets) to process
        p = np.shape(params)[1] # number of input parameters (# of physical params)
        proc_params = np.zeros((N_proc,p))
        
        for i in range(p):
            x = parameters_log[:,i]
            proc_params[:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])

        proc_params_test = torch.from_numpy(proc_params)
        proc_params = 0
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
        Compute the rms error for the loaded trained instance of 21cmKAN evaluated on the 1,704-signal 21cmGEM test set

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


