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
		# --------------------------------------------------------------------------------------------------------------
		print("="*40)
		print(df)
		print(y_col)
		print(task_col)
		print(w_col)
		print(x_cols)
		print(standardize)
		# --------------------------------------------------------------------------------------------------------------

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
		self.y = {}
		self.w = {}
		self.x = {}
		self.x_mean = pd.DataFrame(columns=self.feature_names)
		self.x_std_dev = pd.DataFrame(columns=self.feature_names)
		for task in self.tasks:
			df_task = df[df[task_col] == task]
			self.n_obs[task] = df_task.shape[0]
			#  responses
			self.y[task] = df_task[y_col]
			#  weights
			if df_task[w_col].min() < 0.0:
				raise ValueError("weights should be non-negative")
			ws = df_task[w_col].sum()
			if ws <= 0.0:
				raise ValueError("weights should have positive sum")
			self.w[task] = df_task[w_col] / ws
			#  features
			self.x_mean.loc[task] = df_task[x_cols].mean()
			st_dev = df_task[x_cols].std()
			self.x_std_dev.loc[task] = st_dev
			if standardize:
				self.x[task] = (df_task[x_cols] - self.x_mean.loc[task]) / st_dev.where(st_dev > 1.0e-16, 1.0)
			else:
				self.x[task] = df_task[x_cols]
		# --------------------------------------------------------------------------------------------------------------
		print(self.__name__)
		for attr in dir(self):
			if not attr[0] == "_":
				print(" -  %s = %r" % (attr, getattr(self, attr)))
		# --------------------------------------------------------------------------------------------------------------

	def _check_data(self):
		super()._check_data()

	def _check_features(self):
		super()._check_features()

	def get_x(self, task):
		return self.x[task]

	def get_y(self, task):
		return self.y[task]

	def get_w(self, task):
		return self.w[task]
