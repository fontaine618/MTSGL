import numpy as np
import pandas as pd
from typing import Union, Optional, Sequence


class Data:

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