# Welcome to 21cmKAN! 

21cmKAN ([Dorigo Jones et al. 2025](ADS link), referred to as DJ+25) is an emulator of the global 21 cm cosmological signal based on the [Kolmogorov-Arnold Network](https://ui.adsabs.harvard.edu/abs/2024arXiv240419756L/abstract). KANs are a novel type of fully-connected neural network that capture complex relationships by learning data-driven functional transformations, or activation functions, as opposed to using fixed, pre-determined activations. The expressivity of KANs makes them useful for modeling certain structured, lower-dimensional functions or PDEs often found in science, and their transparent architecture makes it easy to interpret and verify their predictions.

**21cmKAN has similar accuracy as the most accurate current emulator of the global 21 cm signal, 21cmLSTM ([Dorigo Jones et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...977...19D/abstract)), while training 75 times faster and predicting each signal in 3.7 milliseconds on average, when utilizing the same typical A100 GPU. 21cmKAN can be trained and used to obtain unbiased physical parameter constraints altogether in under 30 minutes. The speed-accuracy combination of 21cmKAN enables producing many emulator models that can constrain complex feature spaces and covariances across different physical models and parameterizations to fully exploit upcoming observations.**

The tutorial notebooks and commented scripts provided in this GitHub repository make 21cmKAN simple to train, evaluate, employ in Bayesian inference analyses, and apply to different physical models and data sets:
- "21cmKAN_train_test_examples.ipynb" shows how to train and evaluate 21cmKAN on two popular physical models of the global 21 cm signal, [21cmGEM/21cmSPACE](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4845C/abstract) and [ARES](https://github.com/mirochaj/ares). The notebook provides code for reproducing Figures 2, 4, 7, 8, and B1 in DJ+25.
- "21cmKAN_Bayesian_inference_example_21cmGEM.ipynb" shows how to use 21cmKAN to fit a mock global 21 cm signal made by the 21cmGEM model and constrain its physical model parameters. The notebook provides code for reproducing Figures 5, C1, and C2 in DJ+25.
- "21cmKAN_Bayesian_inference_example_ARES.ipynb" shows how to use 21cmKAN to fit a mock global 21 cm signal made by the ARES model and constrain its physical model parameters. The notebook provides code for reproducing Figures 6 and C3 in DJ+25.
- "emulate_21cmGEM.py" and "emulate_ARES.py" are the classes for using 21cmKAN to emulate the 21cmGEM and ARES data sets, respectively. "emulate_yourmodel.py" is an example class that can be edited to use 21cmKAN to emulate a different model or data set of global 21 cm signals
  
Please see the associated paper -- [Dorigo Jones et al. 2025](ADS link) -- for details on the architecture, training, and interpretation of 21cmKAN, as well as high-level and in-depth descriptions of the unique differences and advantages of KANs compared to traditional fully-connected neural networks. We note that all of the data used to train and test 21cmKAN in DJ+25 is publicly-available on Zenodo: [21cmGEM/21cmVAE data set](https://zenodo.org/records/5084114); [ARES data set](https://zenodo.org/records/13840725). Those data, as well as the same trained instances of 21cmKAN used to perform Bayesian inference analyses in DJ+25, are downloaded by this repository upon installation to allow for immediate use of the emulator trained on these two popular physical models of the cosmological 21 cm signal.

### Contact; papers to reference
Please reach out to me at johnny.dorigojones@colorado.edu about any comments, questions, or contributions (can also open an issue or make a pull request). Please reference [Dorigo Jones et al. 2025](
ADS link) and provide a link to this GitHub repository if you utilize this work or emulator in any way.

# Installing 21cmKAN

To ensure that all packages are available, below we provide basic instructions
for setting up an environment that is compatible with 21cmKAN. Although it is not 
necessary, we chose to create a [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) environment. 

1. Create a mamba environment named `21cmkan-env` and activate it
```
module load miniforge
mamba create -n 21cmkan-env python==3.12.8
mamba activate 21cmkan-env
```

2. Grab [efficient-kan](https://github.com/Blealtan/efficient-kan)
```
git clone https://github.com/Blealtan/efficient-kan.git
```
> [!NOTE]
> We would like to extend a HUGE thank you to the individuals who implemented efficient-kan. Without it, creating 21cmKAN may not have been possible. 
3. Depending on your architecture, you may need to install the CUDA version of `torch` and `torchvision` (or else it will default to the CPU version). Here we install `torch` and `torchvision` for CUDA version 12.4:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
4. Install efficient-kan and associated dependencies into environment:
```
cd efficient-kan
pip install .
cd ..
```
5. To perform [Ray Tune](https://docs.ray.io/en/latest/ray-overview/installation.html) Bayesian hyperparameter search, include the following libraries:
```
pip install ray[tune]
pip install optuna
```
6. If you would like to use [Pykan's](https://github.com/KindXiaoming/pykan) LBFGS optimizer, then you will 
need to install it (and some additional dependencies) in your environment: 
```
pip install pykan
pip install scikit-learn
pip install pyyaml
pip install matplotlib
pip install pandas 
```

7. Install 21cmKAN
```
git clone https://github.com/jdorigojones/21cmKAN
cd 21cmKAN
python -m pip install .
cd ..
```

## The following installation steps are for if you wish to use 21cmKAN to perform Bayesian nested sampling parameter inference analyses:

8. Install [PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html)
```
pip install pymultinest
pip install scikit-learn
pip install matplotlib
pip install corner
mamba install conda-forge::cmake=3.16.3 # makes you downgrade Python to 3.12.3 (we are using Python 3.12.8)
mamba install conda-forge::gcc=9.4.0
mamba install conda-forge::gfortran=9.4.0
mamba install conda-forge::libblas=3.9.0
mamba install conda-forge::liblapack=3.9.0
git clone https://github.com/JohannesBuchner/MultiNest
cd MultiNest/build
cmake .. # ignore "Could NOT find MPI..."
make
export LD_LIBRARY_PATH=/path/to/MultiNest/lib:$LD_LIBRARY_PATH # add the compiled libraries to your LD_LIBRARY_PATH. Must do this every time you use the library
cd ..
cd ..
git clone https://github.com/JohannesBuchner/PyMultiNest/
cd PyMultiNest
python setup.py install
```
Test the pymultinest installation by importing the library and running their demo script:
```
python -c 'import pymultinest'
python pymultinest_test.py
cd ..
```

9. Install [distpy](https://github.com/CU-NESS/distpy) and [pylinex](https://github.com/CU-NESS/pylinex/tree/master). Pylinex is used to define the likelihood function evaluated during the nested sampling analyses.
```
git clone https://github.com/CU-NESS/distpy.git
cd distpy
python setup.py develop --user
cd ..
git clone https://github.com/CU-NESS/pylinex.git
cd pylinex
python setup.py develop --user
```

10. Remember to run the following path exports when using the pymultinest, distpy, and pylinex libraries
```
export LD_LIBRARY_PATH=/path/to/MultiNest/lib:$LD_LIBRARY_PATH
export PYLINEX=/path/to/pylinex
export DISTPY=/path/to/distpy
export PYTHONPATH="/path/to/pylinex:$PYTHONPATH"
export PYTHONPATH="/path/to/distpy:$PYTHONPATH"
```

# Contributions
Authors: Johnny Dorigo Jones and Brandon Reyes
