# Welcome to 21cmKAN! 

21cmKAN - an emulator of the global 21 cm signal based on the Kolmogorov-Arnold Network

## Setting up an environment 

To ensure that all packages are available, below we provide basic instructions
for setting up an environment that is compatible with 21cmKAN. Although it is not 
necessary, we chose to create a mamba environment. 

1. Create a mamba environment named `21cm-kan-env`
```
mamba create -n 21cm-kan-env python==3.12.8
```
2. Activate the environment:
```
mamba activate 21cm-kan-env
```
3. Grab [efficient-kan](https://github.com/Blealtan/efficient-kan)
```
git clone https://github.com/Blealtan/efficient-kan.git
```
> [!NOTE]
> We would like to extend a HUGE thank you to the individuals who implemented efficient-kan. Without it, creating 21cmKAN may not have been possible. 
4. Depending on your architecture, you may need to install the CUDA version of `torch` and `torchvision` (or else it will default to the CPU version). Here we install `torch` and `torchvision` for CUDA version 12.4:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
5. Install efficient-kan and associated dependencies into environment:
```
cd efficient-kan
pip install .
```
6. To perform Ray Tune Bayesian hyperparameter search, include the following libraries:
```
pip install ray[tune]
pip install optuna
```
7. If you would like to use [Pykan's](https://github.com/KindXiaoming/pykan) LBFGS optimizer, then you will 
need to install it (and some additional dependencies) in your environment: 
```
pip install pykan
pip install scikit-learn
pip install pyyaml
pip install matplotlib
pip install pandas 
```

8. Install 21cmKAN

```
git clone https://github.com/jdorigojones/21cmKAN
cd 21cmKAN
python -m pip install .
```
