import pandas as pd
import numpy as np
from typing import Union, Sequence, Optional

from MTSGL.data.Data import Data


class MultivariateData(Data):

	def __init__(
			self,
			df: pd.DataFrame,
			y_cols: Union[str, Sequence[str]],
			w_cols: Optional[Union[str, Sequence[str]]],
			x_cols: Optional[Sequence[str]],
			standardize: bool = True,
			intercept: bool = True
	):
		super().__init__()
		self.__name__ = "MultivariateData"
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
		if standardize:
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
