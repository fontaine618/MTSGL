import numpy as np
import pandas as pd
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
		self._losses = None

	def __getitem__(self, task):
		return self._losses[task]


class MTWLS(SeparableMTLoss):

	def __init__(self, data: Data):
		super().__init__(data)
		self._losses = {
			task: MTSGL.losses.WLS(
				self.data.x(task).to_numpy(),
				self.data.y(task).to_numpy().reshape((-1, 1)),
				self.data.w(task).to_numpy().reshape((-1, 1))
			)
			for task in self.data.tasks
		}

	def gradient(self, beta: Optional[np.ndarray] = None, task: Optional[str] = None):
		if task is None:
			if beta is None:
				beta = np.zeros((self.data.n_features, 1))
			grad = pd.DataFrame(columns=self._losses.keys())
			for task, loss in self._losses.items():
				g = loss.gradient(beta).reshape(-1)
				grad[task] = g
			return grad.to_numpy()
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
