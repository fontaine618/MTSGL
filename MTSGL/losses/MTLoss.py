import numpy as np
import pandas as pd
from MTSGL.losses import Loss
import MTSGL.losses
from MTSGL.data.Data import Data
from MTSGL.data.MultiTaskData import MultiTaskData
from MTSGL.data.MultivariateData import MultivariateData
from typing import Optional


class MTLoss:

	def __init__(self, data: Data, **kwargs):
		self.data = data

	def loss(self, beta: np.ndarray, task: Optional[str]):
		pass

	def gradient(self, beta: np.ndarray, task: Optional[str]):
		pass


class SeparableMTLoss(MTLoss):

	def __init__(self, data: Data, **kwargs):
		super().__init__(data, **kwargs)
		self._losses = dict()

	def __getitem__(self, task):
		return self._losses[task]

	def __repr__(self):
		return repr(self._losses)

	def has_key(self, task):
		return task in self._losses

	def keys(self):
		return self._losses.keys()

	def values(self):
		return self._losses.values()

	def items(self):
		return self._losses.items()

	def __setitem__(self, task: str, loss: Loss):
		self._losses[task] = loss

	def __iter__(self):
		return iter(self._losses)

	def __len__(self):
		return len(self._losses)

	def __contains__(self, task):
		return task in self._losses

	def gradient(self, beta: Optional[np.ndarray] = None, task: Optional[str] = None):
		if task is None:
			if beta is None:
				beta = np.zeros((self.data.n_features, self.data.n_tasks))
			grad = np.column_stack([loss.gradient(beta[:, [k]]) for k, loss in enumerate(self.values())])
			return grad
		else:
			if beta is None:
				beta = np.zeros((self.data.n_features, 1))
			return self[task].gradient(beta)

	def loss(self, beta: Optional[np.ndarray] = None, task: Optional[str] = None):
		if task is None:
			if beta is None:
				beta = np.zeros((self.data.n_features, self.data.n_tasks))
			return sum([loss.loss(beta[:, k]) for k, loss in enumerate(self.values())])
		else:
			if beta is None:
				beta = np.zeros((self.data.n_features, 1))
			return self[task].loss(beta)


class MTWLS(SeparableMTLoss):

	def __init__(self, data: Data):
		super().__init__(data)
		for task in self.data.tasks:
			self[task] = MTSGL.losses.WLS(
				self.data.x(task).to_numpy(),
				self.data.y(task).to_numpy().reshape((-1, 1)),
				self.data.w(task).to_numpy().reshape((-1, 1))
			)