import pandas as pd
import numpy as np
from typing import Union, Sequence, Optional

from MTSGL.data.Data import Data


class MultiTaskData(Data):

	def __init__(
			self,
			df: pd.DataFrame,
			y_col: Union[str, Sequence[str]],
			task_col: str,
			w_col: Optional[str] = None,
			x_cols: Optional[Sequence[str]] = None,
			standardize: bool = True
	):
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
			self.x_mean[task] = df_task[x_cols].mean()
			st_dev = df_task[x_cols].std()
			self.x_std_dev[task] = st_dev
			if standardize:
				self._x[task] = (df_task[x_cols] - self.x_mean[task]) / st_dev.where(st_dev > 1.0e-16, 1.0)
			else:
				self._x[task] = df_task[x_cols]
		for task in self.tasks:
			self._w[task] = self._w[task] / ws_total

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
