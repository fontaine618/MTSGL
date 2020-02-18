import numpy as np
import pandas as pd
from MTSGL.losses import Loss
import MTSGL.losses
from MTSGL.data.Data import Data
from typing import Optional


class MTLoss:

	def __init__(self, data: Data, **kwargs):
		self.data = data

	def loss(self, beta: np.ndarray, task: Optional[str]):
		pass

	def gradient(self, beta: np.ndarray, task: Optional[str]):
		pass

	def _lin_predictor(self, beta: np.ndarray):
		# if else MVT of MT
		pass


class SeparableMTLoss(MTLoss):

	def __init__(self, data: Data, **kwargs):
		super().__init__(data, **kwargs)
		self._losses = dict()

	def __getitem__(self, task):
		return self._losses[task]

	def __setitem__(self, task: str, loss: Loss):
		self._losses[task] = loss

	def __iter__(self):
		return iter(self._losses)

	def __len__(self):
		return len(self._losses)

	def gradient(self, beta: Optional[np.ndarray] = None, task: Optional[str] = None):
		if task is None:
			if beta is None:
				beta = np.zeros((self.data.n_features, self.data.n_tasks))
			grad = np.zeros((self.data.n_features, self.data.n_tasks))
			for k, task in enumerate(self.data.tasks):
				grad[:, [k]] = self[task].gradient(beta[:, [k]])
			return grad
		else:
			if beta is None:
				beta = np.zeros((self.data.n_features, 1))
			return self[task].gradient(beta)

	def loss(self, beta: Optional[np.ndarray] = None, task: Optional[str] = None):
		if task is None:
			if beta is None:
				beta = np.zeros((self.data.n_features, self.data.n_tasks))
			loss_val = 0.0
			for k, task in enumerate(self.data.tasks):
				loss_val += self[task].loss(beta[:, k])
			return loss_val
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