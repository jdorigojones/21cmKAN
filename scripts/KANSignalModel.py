'''
Name: 21cmKAN/scripts/KANSignalModel.py
Author: Johnny Dorigo Jones
Original: May 2022, Edited: August 2025
Description: A model class which wraps around the 21cmKAN module for signal model evaluations
'''
from __future__ import division
import time
import numpy as np
from pylinex import LoadableModel
import Global21cmKAN as Global21cmKAN
from Global21cmKAN.eval_21cmKAN_21cmGEM import evaluate_21cmGEM
from Global21cmKAN.eval_21cmKAN_ARES import evaluate_ARES

try:
	# this runs with no issues in python 2 but raises error in python 3
	basestring
except:
	# this try/except allows for python 2/3 compatible string type checking
	basestring = str

class KANSignalModel_21cmGEM(LoadableModel):
    
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
        value: array of parameters to give to the Global21cmKAN.emulator_21cmKAN_21cmGEM.Emulate().predict() function
        """
        self._parameters = [element for element in value]
        
    @property
    def neural_network_predictor(self):
        if not hasattr(self, '_neural_network_predictor'):
            self._neural_network_predictor = evaluate_21cmGEM()
        return self._neural_network_predictor
        
    def __call__(self, parameters):
        '''
        '''
        signal = self.neural_network_predictor(parameters)
        return signal

class KANSignalModel_ARES(LoadableModel):
    
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
        value: array of parameters to give to the Global21cmKAN.emulator_21cmKAN_ARES.Emulate().predict() function
        """
        self._parameters = [element for element in value]
        
    @property
    def neural_network_predictor(self):
        if not hasattr(self, '_neural_network_predictor'):
            self._neural_network_predictor = evaluate_ARES()
        return self._neural_network_predictor
        
    def __call__(self, parameters):
        '''
        '''
        signal = self.neural_network_predictor(parameters)
        return signal

