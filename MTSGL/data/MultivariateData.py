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
			standardize: bool = True
	):
		#  TODO put in docs that we disregard task_col
		# --------------------------------------------------------------------------------------------------------------
		print("="*40)
		print(df)
		print(y_cols)
		print(w_cols)
		print(x_cols)
		print(standardize)
		# --------------------------------------------------------------------------------------------------------------

		super().__init__()
		self.__name__ = "MultivariateData"
		#  tasks
		self.tasks = y_cols
		self.n_tasks = len(self.tasks)
		#  features
		self.feature_names = x_cols
		self.n_features = len(x_cols)
		self.x = df[x_cols]
		#  responses
		self.y = df[y_cols]
		self.n_obs = len(df)
		#  weights
		self.w = df[w_cols]
		if self.w.min().min() < 0.0:
			raise ValueError("weights should be non-negative")
		if any([s <= 0 for s in self.w.sum()]):
			raise ValueError("weights should have positive sum")
		self.w = self.w / self.w.sum()
		#  standardize x and w
		self.x_mean = self.x.mean()
		self.x_std_dev = self.x.std()
		if standardize:
			self.x = (self.x - self.x_mean) / self.x_std_dev.where(self.x_std_dev > 1.0e-16, 1.0)
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
		pass

	def get_y(self, task):
		pass

	def get_w(self, task):
		pass
