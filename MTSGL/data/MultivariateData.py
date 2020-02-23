import pandas as pd
import numpy as np
from typing import Union, Sequence, Optional

from MTSGL.data.Data import Data


class MultivariateData(Data):
	"""Multivariate dataset.

	This class implements a Data class in the case of Multi-task data.

	Attributes
	----------
	tasks: list of str
		The names of the tasks.
	n_tasks: int
		The number of tasks.
	n_obs: int
		The number of observations.
	n_features: int
		The number of features.
	feature_names: list of str
		The names of the features.
	x_mean: array-like
		An 1D array containing the mean of the features.
	x_std_dev: array-like
		An 1D array containing the std. deviation of the features.
	_x: array-like
		The features.
	_y: array-like
		The responses.
	_w : array-like
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

	Notes
	-----
	Since we have the same number of observations in each task, we may use arrays to hold the data. The features
	are shared across tasks so we do not copy it.

	"""

	def __init__(
			self,
			df: pd.DataFrame,
			y_cols: Sequence[str],
			w_cols: Union[str, Sequence[str]],
			x_cols: Sequence[str],
			standardize: bool = True,
			intercept: bool = True
	):
		"""Initializes a MultivariateData object.

		Parameters
		----------
		df: pd.DataFrame
			The dataset.
		y_cols: list of str
			The columns of df that contains the responses.
		w_cols: str or list of str
			The column(s) of df that contains the observation weights.
		x_cols: list of str
			The columns of df that contains the features.
		standardize: bool, optional
			True to standardize the features. Defaults to True.
		intercept: bool, optional
			True to include in intercept denoted '(Intercept)'. Note that it will overwrite any column with name
			'(Intercept)' in df. Defaults to True.
		"""
		super().__init__()
		self.__name__ = "MultivariateData"
		self.standardize = standardize
		#  tasks
		self.tasks = y_cols
		self.n_tasks = len(self.tasks)
		#  responses
		self._y = df[y_cols]
		self.n_obs = len(df)
		#  features
		self._x = df[x_cols]
		if intercept:
			self._x.insert(0, "(Intercept)", np.ones((self.n_obs, 1)))
		self.feature_names = self._x.columns
		self.n_features = len(self.feature_names)
		#  weights
		self._w = df[w_cols]
		if self._w.min().min() < 0.0:
			raise ValueError("weights should be non-negative")
		if any([s <= 0 for s in self._w.sum()]):
			raise ValueError("weights should have positive sum")
		self._w = self._w / self._w.sum()
		self._w.columns = self.tasks if self._w.shape[1] > 1 else ["w"]
		#  standardize x and w
		self.x_mean = self._x.mean()
		if intercept:
			self.x_mean["(Intercept)"] = 0.0
		self.x_std_dev = self._x.std()
		self.x_std_dev = self.x_std_dev.where(self.x_std_dev > 1.0e-16, 1.0)
		if self.standardize:
			self._x = (self._x - self.x_mean) / self.x_std_dev

	def _summarize_tasks(self):
		out = "Tasks (n={}): \n".format(self.n_obs)
		for task in self.tasks:
			out += "    {}\n".format(task)
		return out

	def x(self, task):
		return self._x

	def n(self, task: Optional[str]):
		return self.n_obs

	def y(self, task):
		return self._y[task]

	def w(self, task):
		if self._w.shape[1] > 1:
			return self._w[task]
		else:
			return self._w
