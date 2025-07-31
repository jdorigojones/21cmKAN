#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import requests
import shutil
from pathlib import Path
from setuptools import setup, find_packages

def readme(short=False):
    with open('README.md', encoding='utf-8') as f:
        if short:
            return f.readlines()[1].strip()
        else:
            return f.read()

setup(
    name='21cmKAN',
    version='1.0.0',
    description='21cmKAN: A Fast, Accurate, and Transparent Emulator of the Global 21 cm Signal Based on the Kolmogorov-Arnold Network',
    long_description=readme(),
    author='Johnny Dorigo Jones',
    author_email='johnny.dorigojones@colorado.edu',
    url='https://github.com/jdorigojones/21cmKAN',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    license='MIT',
    classifiers=[
               'Intended Audience :: Science/Research',
               'Natural Language :: English',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Astronomy',
               'Topic :: Scientific/Engineering :: Physics',
    ],
)

# the below code creates a hidden folder in $HOME to store needed auxiliary H5 and txt files 
# (i.e., data sets, trained models, training set min and max values) and copies/downloads those files to the folder.
# The location of this folder can be changed here if the user's $HOME has storage limitations by setting the 
# environment variable AUX_DIR to an alternative location. Note: AUX_DIR will need to be set each time the 
# package is used. 
BASE_PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"
MODELS_INSTALL_PATH = os.path.dirname(os.path.realpath(__file__))+'/Global21cmKAN/models/'
emulator_files_list = ['emulator_21cmGEM.pth', 'emulator_ARES.pth', 'train_maxs_21cmGEM.npy',\
                       'train_mins_21cmGEM.npy', 'train_maxs_ARES.npy', 'train_mins_ARES.npy']

if not os.path.exists(BASE_PATH):
   os.makedirs(BASE_PATH)

if not os.path.exists(BASE_PATH+'models/'):
   os.makedirs(BASE_PATH+'models/')

if not Path(BASE_PATH + 'dataset_21cmGEM.h5').exists():
    print('Downloading data set for 21cmGEM (Cohen et al. 2020) used in Dorigo Jones et al. 2025, Dorigo Jones et al. 2024, Bye et al. 2022, and Bevins et al. 2021.')
    r1 = requests.get('https://zenodo.org/record/5084114/files/dataset_21cmVAE.h5?download=1')
    with open(BASE_PATH + 'dataset_21cmGEM.h5', 'wb') as f1:
        f1.write(r1.content)

if not Path(BASE_PATH + 'dataset_ARES.h5').exists():
    print('Downloading data set for ARES used and generated in Dorigo Jones et al. 2024 and also used in Dorigo Jones et al. 2025.')
    r2 = requests.get('https://zenodo.org/record/13840725/files/dataset_ARES.h5?download=1')
    with open(BASE_PATH + 'dataset_ARES.h5', 'wb') as f2:
        f2.write(r2.content)

for item in emulator_files_list:
    shutil.copyfile(MODELS_INSTALL_PATH+item, BASE_PATH+'models/'+item)

