import numpy as np
import pandas as pd
from typing import Union, Optional, Sequence


class Data:
	"""Abstract dataset.

	This class implements the basic attributes and methods for a dataset.

	Attributes
	---------
	tasks: list of str
		The names of the tasks.
	n_tasks: int
		The number of tasks.
	n_obs: int or dict of int
		The number of observations.
	n_features: int
		The number of features.
	feature_names: list of str
		The names of the features.
	x_mean: array-like
		An array containing the mean of the features.
	x_std_dev: array-like
		An array containing the std. deviation of the features.
	_x: array-like or dict of array-like
		The features.
	_y: array-like or dict of array-like
		The responses.
	_w : array-like or dict of array-like
		The observation weights.

	Methods
	-------
	x(task)
		Extracts the feature matrix in a given task.
	y(task)
		Extracts the vector of responses in a given task.
	w(task)
		Extracts the vector of weights in a given task.
	n(task)
		Extracts the number of observations in a given task.

	"""

	def __init__(self, **kwargs):
		self._x = None
		self._y = None
		self._w = None
		self.tasks = None
		self.n_tasks = None
		self.n_obs = None
		self.n_features = None
		self.feature_names = None
		self.x_mean = None
		self.x_std_dev = None

	def _summarize(self):
		"""Builds a string summarizing the dataset.

		Returns
		-------
		out: str
			The summary.
		"""
		out = self.__name__ + "\n"
		out += "-" * len(self.__name__) + "\n"
		out += self._summarize_tasks()
		out += "Features (p={}): \n".format(self.n_features)
		for i, feature in enumerate(self.feature_names):
			if i < 10 or i >= self.n_features - 10:
				out += "    {}\n".format(feature)
			elif i == 10:
				out += "    ...\n"
			else:
				pass
		return out

	def _summarize_tasks(self):
		pass

	def __str__(self):
		return self._summarize()

	def x(self, task):
		pass

	def y(self, task):
		pass

	def w(self, task):
		pass

	def n(self, task):
		pass