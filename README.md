# Welcome to 21cmKAN! 

21cmKAN ([Dorigo Jones et al. 2025](ADS link), referred to as DJ+25) is an emulator of the global 21 cm cosmological signal based on the [Kolmogorov-Arnold Network](https://ui.adsabs.harvard.edu/abs/2024arXiv240419756L/abstract). KANs are a novel type of fully-connected neural network that learn data-driven functional transformations (i.e., activation functions) to capture complex relationships, as opposed to using fixed, pre-determined activations. 21cmKAN has similar accuracy as the most accurate current emulator of the global 21 cm signal, 21cmLSTM ([Dorigo Jones et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...977...19D/abstract)), while training 75 times faster and predicting each signal in 3.7 milliseconds on average, when utilizing the same typical A100 GPU. The speed-accuracy combination of 21cmKAN enables it to be trained and used to obtain multiple unbiased physical parameter constraints altogether in under 30 minutes. The tutorial notebooks and commented scripts provided in this GitHub repository make 21cmKAN simple to train, evaluate, employ in Bayesian inference analyses, and apply to different physical models and data sets. Please see the associated paper -- [Dorigo Jones et al. 2025](ADS link) -- for details on the architecture, training, and interpretation of 21cmKAN, as well as descriptions of the unique differences and advantages of KANs compared to traditional fully-connected neural networks.

## Installing 21cmKAN

To ensure that all packages are available, below we provide basic instructions
for setting up an environment that is compatible with 21cmKAN. Although it is not 
necessary, we chose to create a [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) environment. 

1. Create a mamba environment named `21cm-kan-env` and activate it
```
mamba create -n 21cm-kan-env python==3.12.8
mamba activate 21cm-kan-env
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
cd ..
```
Test importing the library: python -c 'import pymultinest'
Test the installation by running their demo script: python pymultinest_test.py

9. Install [distpy](https://github.com/CU-NESS/distpy) and [pylinex](https://github.com/CU-NESS/pylinex/tree/master)
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
