'''
Name: perses/models/KANSignalModel.py
Author: Joshua J. Hibbard & Johnny Dorigo Jones
Date: May 2022
Edited April 2025 to be for 21cmKAN (deleted a lot from original)
Description: A model class which wraps around the 21cmKAN module for various signal or 
			 systematic model evaluations. 
'''
from __future__ import division
import time
import numpy as np
from pylinex import LoadableModel
#from ..util import bool_types, sequence_types, create_hdf5_dataset, get_hdf5_value
import Global21cmKAN as Global21cmKAN
from Global21cmKAN.eval_21cmKAN_21cmGEM import evaluate

try:
	# this runs with no issues in python 2 but raises error in python 3
	basestring
except:
	# this try/except allows for python 2/3 compatible string type checking
	basestring = str

class KANSignalModel(LoadableModel):
    
    def __init__(self, parameters):
        '''
        parameters: list of parameters to accept as input
        '''
        self.parameters = parameters
        
    @property
    def parameters(self):
        """
        Property storing an array of parameters for this model
        """
        return self._parameters
        
    @parameters.setter
    def parameters(self, value):
        """
        Setter for the array of parameters for this model
        value: array of parameters to give to the Global21cmKAN.emulator.Emulate().predict() function
        """
        self._parameters = [element for element in value]
        
    @property
    def neural_network_predictor(self):
        if not hasattr(self, '_neural_network_predictor'):
            self._neural_network_predictor = evaluate()
        return self._neural_network_predictor
        
    def __call__(self, parameters):
        '''
        '''
        signal = self.neural_network_predictor(parameters)
        return signal

