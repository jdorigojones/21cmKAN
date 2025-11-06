'''
Name: 21cmKAN/Global21cmKAN/wrapper.py
Author: Johnny Dorigo Jones
Original: May 2022, Edited: August 2025 to include classes for predicting both 21cmGEM and ARES signals
Description: A model class which wraps around the 21cmKAN module for signal model evaluations
'''
from __future__ import division
import time
import os
import numpy as np
from pylinex import LoadableModel
import Global21cmKAN as Global21cmKAN
from Global21cmKAN.evaluate import evaluate_on_21cmGEM, evaluate_on_ARES, evaluate_on_foreground

try:
	# this runs with no issues in python 2 but raises error in python 3
	basestring
except:
	# this try/except allows for python 2/3 compatible string type checking
	basestring = str

PATH = f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"
model_save_path_21cmGEM = PATH+"models/emulator_21cmGEM.pth"
model_save_path_ARES = PATH+"models/emulator_ARES.pth"
model_save_path_foreground = PATH+"models/emulator_foreground_beam_meansub.pth"

class predict_21cmGEM(LoadableModel):
	def __init__(self, parameters, model_path=model_save_path_21cmGEM):
		'''
  		parameters: np.ndarray
    			list of parameters to accept as input
       		model_path : str
	 		The path to the saved 21cmKAN model instance
    			Default: f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"+"models/emulator_21cmGEM.pth"
       		'''
		self.parameters = parameters
		self.model_path = model_path
	
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
    		value: array of parameters to give to the evaluate_on_21cmGEM.__call__() function
      		"""
		self._parameters = [element for element in value]
	
	@property
	def neural_network_predictor(self):
		if not hasattr(self, '_neural_network_predictor'):
			self._neural_network_predictor = evaluate_on_21cmGEM(model_path=self.model_path)
		return self._neural_network_predictor
	
	def __call__(self, parameters):
		'''
  		parameters: np.ndarray
    			list of parameters to accept as input
       		'''
		signal = self.neural_network_predictor(parameters)
		return signal

class predict_ARES(LoadableModel):
	def __init__(self, parameters, model_path=model_save_path_ARES):
		'''
  		parameters: np.ndarray
    			list of parameters to accept as input
       		model_path : str
	 		The path to the saved 21cmKAN model instance
    			Default: f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"+"models/emulator_ARES.pth"
       		'''
		self.parameters = parameters
		self.model_path = model_path
	
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
    		value: array of parameters to give to the evaluate_on_ARES.__call__() function
      		"""
		self._parameters = [element for element in value]
	
	@property
	def neural_network_predictor(self):
		if not hasattr(self, '_neural_network_predictor'):
			self._neural_network_predictor = evaluate_on_ARES(model_path=self.model_path)
		return self._neural_network_predictor
	
	def __call__(self, parameters):
		'''
  		parameters: np.ndarray
    			list of parameters to accept as input
       		'''
		signal = self.neural_network_predictor(parameters)
		return signal

class predict_foreground(LoadableModel):
	def __init__(self, parameters, model_path=model_save_path_foreground):
		'''
  		parameters: np.ndarray
    			list of parameters to accept as input
       		model_path : str
	 		The path to the saved 21cmKAN model instance
    			Default: f"{os.environ.get('AUX_DIR', os.environ.get('HOME'))}/.Global21cmKAN/"+"models/emulator_foreground_beam_meansub.pth"
       		'''
		self.parameters = parameters
		self.model_path = model_path
	
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
    		value: array of parameters to give to the evaluate_on_foreground.__call__() function
      		"""
		self._parameters = [element for element in value]
	
	@property
	def neural_network_predictor(self):
		if not hasattr(self, '_neural_network_predictor'):
			self._neural_network_predictor = evaluate_on_foreground(model_path=self.model_path)
		return self._neural_network_predictor
	
	def __call__(self, parameters):
		'''
  		parameters: np.ndarray
    			list of parameters to accept as input
       		'''
		signal = self.neural_network_predictor(parameters)
		return signal
