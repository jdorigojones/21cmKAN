import h5py
import numpy as np
import os
import torch 
import torch.optim as optim
from efficient_kan import KAN
from utils import NumPy2TensorDataset, NumPyArray2TensorDataset
from Global21cmKAN import __path__

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Set default torch type to float64
torch.set_default_dtype(torch.float64)

# specify KAN model architecture and configs for training on beam-weighted foreground spectra
input_nodes = 18
hidden_layer1_nodes = 44
hidden_layer2_nodes = 44
hidden_layer3_nodes = 71
output_nodes = 176
layer_nodes = [input_nodes, hidden_layer1_nodes, hidden_layer2_nodes, hidden_layer3_nodes, output_nodes]
grid_size = 7
spline_order = 3
num_epochs = 20 #400
batch_size = 100
#history_size = 10  # history size for LBFGS optimizer
print(f"nodes in each layer: {layer_nodes}")
print(f"number of grid intervals in B-splines: {grid_size}")
print(f"order of splines in B-splines: {spline_order}")
print(f"number of training epochs: {num_epochs}")
print(f"training batch size: {batch_size}")

# define path containing saved trained 21cmKAN networks and min/max training set values used for preprocessing
#PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"
PATH = '/projects/jodo2960/beam_weighted_foreground/'
model_save_path = PATH+"models/emulator_foreground_beam_meansub.pth"
train_mins_foreground_beam = np.load(PATH+"models/train_mins_foreground_beam_meansub.npy")
train_maxs_foreground_beam = np.load(PATH+"models/train_maxs_foreground_beam_meansub.npy")

frequencies = np.linspace(6,50,176)
vr = 1420.4057517667  # rest frequency of 21 cm line in MHz
#z_list = np.array([(vr/x)-1 for x in frequencies]) # list of redshifts for spectra in your model
z_list = np.linspace(27.40811504, 235.73429196, 176)
# Load in unnormalized training, validation, and test data made by your model as numpy arrays; UNCOMMENT
with h5py.File(PATH + 'bw_training_set_500k_split_meansub.h5', "r") as f:
    print("Keys: %s" % f.keys())
    par_train = np.asarray(f['par_train'])[()]
    par_val = np.asarray(f['par_val'])[()]
    X_test_foreground_beam_true = np.asarray(f['par_test'])[()]
    spectra_train = np.asarray(f['spectra_train'])[()]
    spectra_val = np.asarray(f['spectra_val'])[()]
    y_test_foreground_beam_true = np.asarray(f['spectra_test'])[()]
f.close()

# now preprocess/normalize the input physical parameters values in your training and validation sets
# see the other emulator scripts for examples of data preprocessing
# includes taking the log10 of certain parameters and then performing global MinMax normalizations
# train_mins_foreground_beam[i] is the minimum value of parameter i in the training set, used for normalization
# train_maxs_foreground_beam[i] is the maximum value of parameter i in the training set, used for normalization
N_proc_train = np.shape(par_train)[0] # number of spectra (i.e., parameter sets) to process
N_proc_val = np.shape(par_val)[0]
p_train = np.shape(par_train)[1] # number of input parameters (# of physical params)
p_val = np.shape(par_val)[1]
proc_params_train = np.zeros((N_proc_train,p_train))
proc_params_val = np.zeros((N_proc_val,p_val))

for i in range(p_train):
    x_train = par_train[:,i]
    proc_params_train[:,i] = (x_train-train_mins_foreground_beam[i])/(train_maxs_foreground_beam[i]-train_mins_foreground_beam[i])

for i in range(p_val):
    x_val = par_val[:,i]
    proc_params_val[:,i] = (x_val-train_mins_foreground_beam[i])/(train_maxs_foreground_beam[i]-train_mins_foreground_beam[i])

X_train_foreground_beam = proc_params_train.copy() # define your normalized training set data/parameters
X_val_foreground_beam = torch.from_numpy(proc_params_val) # define your normalized validation set data/parameters
X_val_foreground_beam = X_val_foreground_beam.to(device)
proc_params_train = 0
par_train = 0
proc_params_val = 0
par_val = 0

# now preprocess/normalize the spectra (dT_b values) in your training and validation sets
# see the other emulator scripts for examples of data preprocessing
# includes performing global MinMax normalizations
# train_mins_foreground_beam[-1] is the absolute minimum dT_b value in the training set, used for normalization
# train_maxs_foreground_beam[-1] is the absolute maximum dT_b value in the training set, used for normalization
proc_spectra_train = spectra_train.copy()
proc_spectra_train = (spectra_train - train_mins_foreground_beam[-1])/(train_maxs_foreground_beam[-1]-train_mins_foreground_beam[-1])  # global Min-Max normalization
y_train_foreground_beam = proc_spectra_train.copy() # define your normalized training set spectra
proc_spectra_val = spectra_val.copy()
proc_spectra_val = (spectra_val - train_mins_foreground_beam[-1])/(train_maxs_foreground_beam[-1]-train_mins_foreground_beam[-1])
y_val_foreground_beam = torch.from_numpy(proc_spectra_val.copy()) # define your normalized validation set spectra
y_val_foreground_beam = y_val_foreground_beam.to(device)
proc_spectra_train = 0
spectra_train = 0
proc_spectra_val = 0
spectra_val = 0

# Calculate the amplitude of each normalized validation set spectrum, used to compute relative error
amplitude = y_val_foreground_beam[:,0]

# Create normalized training Dataset and DataLoader
train_dataset = NumPyArray2TensorDataset(features_npy=X_train_foreground_beam, targets_npy=y_train_foreground_beam)
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

def error(true_spectrum, emulated_spectrum, relative=True, nu=None, nu_low=None, nu_high=None):
    """
    Compute the relative rms error (Eq. 3 in DJ+25) between the true and emulated spectra

    Parameters
    ----------
    true_spectrum : np.ndarray
        The true spectra created by the model on which the emulator is trained
        An array of brightness temperatures for different redshifts or frequencies
    emulated_spectrum : np.ndarray
        The emulated spectra. Must be same shape as true_spectrum
    relative : bool
        True to compute the rms error in relative (%) units. False for absolute (mK) units. Default : True
    nu : np.ndarray or None
        The array of frequencies corresponding to each spectrum
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
        The computed rms error for each input spectrum

    Raises
    ------
    ValueError : If nu is None and nu_low or nu_high are not None
    """
    if (nu_low or nu_high) and nu is None:
        raise ValueError("Cannot compute error because no frequency array is given.")
    if len(emulated_spectrum.shape) == 1:
        emulated_spectrum = np.expand_dims(emulated_spectrum, axis=0)
        true_spectrum = np.expand_dims(true_spectrum, axis=0)

    if nu_low and nu_high:
        nu_where = np.argwhere((nu >= nu_low) & (nu <= nu_high))[:, 0]
    elif nu_low:
        nu_where = np.argwhere(nu >= nu_low)
    elif nu_high:
        nu_where = np.argwhere(nu <= nu_high)

    if nu_low or nu_high:
        emulated_spectrum = emulated_spectrum[:, nu_where]
        true_spectrum = true_spectrum[:, nu_where]

    err = np.sqrt(np.mean((emulated_spectrum - true_spectrum)**2, axis=1))
    if relative:  # return the rms error as a fraction of the spectrum amplitude in the chosen frequency band
        err /= true_spectrum[:,0]
        err *= 100 # convert to per cent (%)
    return err

class Emulate:
    def __init__(
        self,
        par_train=X_train_foreground_beam,
        par_val=X_val_foreground_beam,
        par_test=X_test_foreground_beam_true,
        spectra_train=y_train_foreground_beam,
        spectra_val=y_val_foreground_beam,
        spectra_test=y_test_foreground_beam_true,
        redshifts=z_list,
        frequencies=None):
        """
        The emulator class for building, training, and using 21cmKAN to emulate spectra in your model
        The default parameters for training/testing 21cmKAN on the 21cmGEM and ARES sets are described in Section 2.2 of DJ+25

        Parameters
        ----------
        par_train : np.ndarray
            Parameters in training set (normalized, used in train() via train_dataloader)
        par_val : np.ndarray
            Parameters in validation set (normalized, used in train())
        par_test : np.ndarray
            Parameters in test set (unnormalized, used in test_error())
        spectra_train : np.ndarray
            spectra in training set (normalized, used in train() via train_dataloader)
        spectra_val : np.ndarray
            spectra in validation set (normalized, used in train())
        spectra_test : np.ndarray
            spectra in test set (unnormalized, used in test_error())
        redshifts : np.ndarray or None
            Array of redshifts corresponding to each spectrum
        frequencies : np.ndarray or None
            Array of frequencies corresponding to each spectrum

        Attributes
        ----------
        par_train : np.ndarray
            Parameters in training set
        par_val : np.ndarray
            Parameters in validation set
        par_test : np.ndarray
            Parameters in test set
        spectra_train : np.ndarray
            spectra in training set
        spectra_val : np.ndarray
            spectra in validation set
        spectra_test : np.ndarray
            spectra in test set
        par_labels : list of str
            Names of the physical parameters
        emulator : tf.keras.Model
            The emulator model
        redshifts : np.ndarray or None
            Array of redshifts corresponding to each spectrum
        frequencies : np.ndarray or None
            Array of frequencies corresponding to each spectrum

        Methods
        -------
        load_model : load an existing instance of 21cmKAN trained on your data
        train : train the emulator on your data
        predict : use the emulator to predict beam-weighted foreground spectra from input physical parameters
        test_error : compute the rms error of the emulator evaluated on your test set
        save : save the model class instance with all attributes
        """
        self.par_train = par_train
        self.par_val = par_val
        self.par_test = par_test
        self.spectra_train = spectra_train
        self.spectra_val = spectra_val
        self.spectra_test = spectra_test

        self.par_labels = [r'$A_1$', r'$\beta_1$', r'$\gamma_1$', r'$A_2$', r'$\beta_2$', r'$\gamma_2$', r'$A_3$', r'$\beta_3$', r'$\gamma_3$',\
		r'$A_4$', r'$\beta_4$', r'$\gamma_4$', r'$A_5$', r'$\beta_5$', r'$\gamma_5$', r'L', r'$\epsilon_{top}$', r'$\epsilon_{bottom}$'] # fill with your parameter labels

        self.emulator = model(layer_nodes, grid_size, spline_order, name="emulator_foreground_beam_meansub")

        self.train_mins = train_mins_foreground_beam
        self.train_maxs = train_maxs_foreground_beam

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
            Default: f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"+"models/emulator_foreground_beam_meansub.pth"

        Raises
        ------
        IOError : if model_path does not point to a valid model instance
        """
        print(f"Loading model from: {model_path}")
        self.emulator = torch.load(model_path, weights_only=False)
        self.emulator.to(device)

    def train(self, num_epochs=num_epochs, batch_size=batch_size, callbacks=[], verbose=2, model_path=model_save_path, shuffle='True'):
        """
        Train an instance of 21cmKAN on your data set

        Parameters
        ----------
        epochs : int
            Number of epochs to train the network for the given batch_size
            Default : 400
        batch_size : int
            Number of spectra whose loss values are averaged at a time to update the network weights during each epoch
            Default : 100
        callbacks : list of tf.keras.callbacks.Callback
            Callbacks to pass to the training loop. Default : []
        verbose : 0, 1, 2
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default : 2
        model_path : str
            The path to the saved model instance
            Default: f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"+"models/emulator_foreground_beam_meansub.pth"

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
                val_loss = torch.mean((val_pred-self.spectra_val)**2)
                val_losses.append(val_loss)
                # code to compute the normalized validation set relative RMSE, if desired to print along with loss
                #RMSE = torch.sqrt(val_loss)
                #rel_error = (RMSE/amplitude)*100
                #mean_rel_error = torch.mean(rel_error)
                #max_rel_error = rel_error.max()
            
            # code to update the learning rate, if desired: scheduler.step() ; scheduler.step(max_error)
            print(f"Epoch: {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

        # save the trained network to model_path
        torch.save(self.emulator, model_path)
        return (train_losses, val_losses)

    def predict(self, params):
        """
        Predict beam-weighted foreground spectra from input physical parameters using trained instance of 21cmKAN

        Parameters
        ----------
        params : np.ndarray
            The values of the physical parameters in the order of par_labels. Input 2D array to predict a set of spectra

        Returns
        -------
        emulated_spectra : np.ndarray
           The predicted beam-weighted foreground spectra
        """
        if len(np.shape(params)) == 1:
            params = np.expand_dims(params, axis=0) # if doing one spectrum at a time

        # include the same parameter preprocessing code performed earlier to transform the input
        # physical parameters to normalized parameters that can be used to train 21cmKAN
        N_proc = np.shape(params)[0] # number of spectra (i.e., parameter sets) to process
        p = np.shape(params)[1] # number of input parameters (# of physical params)
        proc_params = np.zeros((N_proc,p))
        
        for i in range(p):
            x = params[:,i]
            proc_params[:,i] = (x-self.train_mins[i])/(self.train_maxs[i]-self.train_mins[i])

        proc_params_test = torch.from_numpy(proc_params)
        proc_params = 0
        params = 0
        proc_params_test = proc_params_test.to(device)

        self.emulator.eval()
        with torch.no_grad():
            result = self.emulator(proc_params_test) # evaluate trained instance of 21cmKAN with processed parameters

        result = result.cpu().detach().numpy()
        unproc_spectra = result.copy()
        unproc_spectra = (result*(self.train_maxs[-1]-self.train_mins[-1]))+self.train_mins[-1] # denormalize spectra
        #unproc_spectra = unproc_spectra[:,::-1] # flip spectra to be from high-z to low-z
        if unproc_spectra.shape[0] == 1:
            return unproc_spectra[0,:]
        else:
            return unproc_spectra

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
            self.spectra_test,
            self.predict(self.par_test),
            relative=relative,
            nu=self.frequencies,
            nu_low=nu_low,
            nu_high=nu_high)
        return err

