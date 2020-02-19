import pandas as pd
import numpy as np
from typing import Union, Sequence, Optional

from MTSGL.data.Data import Data


class MultiTaskData(Data):
	"""Multi-task dataset.

	This class implements a Data class in the case of Multi-task data.

	Attributes
	---------
	tasks: list of str
		The names of the tasks.
	n_tasks: int
		The number of tasks.
	n_obs: dict of int
		The number of observations.
	n_features: int
		The number of features.
	feature_names: list of str
		The names of the features.
	x_mean: np.ndarray
		An 2D array containing the mean of the features.
	x_std_dev: np.ndarray
		An 2D array containing the std. deviation of the features.
	_x: dict of np.ndarray
		The features.
	_y: dict of np.ndarray
		The responses.
	_w : dict of np.ndarray
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
	Since we may have different number of observations in each task, we have to store the data in dictionaries.
	"""

	def __init__(
			self,
			df: pd.DataFrame,
			y_col: str,
			task_col: str,
			w_col: str,
			x_cols: Sequence[str],
			standardize: bool = True,
			intercept: bool = True
	):
		"""Initializes a MultiTaskData object.

		Parameters
		----------
		df: pd.DataFrame
			The dataset.
		y_col: str
			The column of df that contains the responses.
		task_col: str
			The column of df that contains the task identifiers.
		w_col: str
			The column of df that contains the observation weights.
		x_cols: list of str
			The columns of df that contains the features.
		standardize: bool, optional
			True to standardize the features. Defaults to True.
		intercept: bool, optional
			True to include in intercept denoted '(Intercept)'. Note that it will overwrite any column with name
			'(Intercept)' in df. Defaults to True.
		"""
		super().__init__()
		self.__name__ = "MultiTaskData"
		#  tasks
		df[task_col] = df[task_col].astype(str)
		self.tasks = sorted(list(set(df[task_col])))
		self.n_tasks = len(self.tasks)
		#  features
		self.n_features = len(x_cols)
		self.feature_names = x_cols
		#  data into dicts
		self.n_obs = {}
		self._y = {}
		self._w = {}
		self._x = {}
		self.x_mean = pd.DataFrame()
		self.x_std_dev = pd.DataFrame()
		ws_total = 0.0
		for task in self.tasks:
			df_task = df[df[task_col] == task]
			self.n_obs[task] = df_task.shape[0]
			#  responses
			self._y[task] = df_task[y_col]
			#  weights
			if df_task[w_col].min() < 0.0:
				raise ValueError("weights should be non-negative")
			ws = df_task[w_col].sum()
			ws_total += ws
			if ws <= 0.0:
				raise ValueError("weights should have positive sum")
			self._w[task] = df_task[w_col]
			#  features
			self._x[task] = df_task[x_cols]
			if intercept:
				self._x[task].insert(0, "(Intercept)", np.ones((self.n_obs[task], 1)))
			self.x_mean[task] = self._x[task].mean()
			if intercept:
				self.x_mean[task]["(Intercept)"] = 0.0
			st_dev = self._x[task].std()
			st_dev = st_dev.where(st_dev > 1.0e-16, 1.0)
			self.x_std_dev[task] = st_dev
			if standardize:
				self._x[task] = (self._x[task] - self.x_mean[task]) / st_dev
		for task in self.tasks:
			self._w[task] = self._w[task] / ws_total
		#  features
		self.feature_names = self.x(self.tasks[0]).columns
		self.n_features = len(self.feature_names)

	def _summarize_tasks(self):
		out = "Tasks (n={}): \n".format(sum(self.n_obs.values()))
		for task in self.tasks:
			out += "    {} (n={})\n".format(task, self.n_obs[task])
		return out

	def x(self, task):
		return self._x[task]

	def n(self, task):
		return self.n_obs[task]

	def y(self, task):
		return self._y[task]

	def w(self, task):
		return self._w[task]
