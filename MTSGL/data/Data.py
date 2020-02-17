import numpy as np
import pandas as pd
from typing import Union, Optional, Sequence


class Data:

	# TODO getters and setters for x, y, see __getitem__
	# https://rszalski.github.io/magicmethods/

	def __init__(self, **kwargs):
		self.x = None
		self.y = None
		self.w = None
		self.tasks = None
		self.n_tasks = None
		self.n_obs = None
		self.n_features = None
		self.feature_names = None
		self.x_mean = None
		self.x_std_dev = None

	def _check_data(self):
		pass

	def _check_features(self):
		pass

	def _summarize(self):
		out = ""
		return out

	def __str__(self):
		return self._summarize()

	def get_x(self, task):
		pass

	def get_y(self, task):
		pass

	def get_w(self, task):
		pass
