import numpy as np
from MTSGL.losses import Loss, WLS
from MTSGL.data.Data import Data
from typing import Optional


class MTLoss:
	"""Multi-task loss.

	This class implements a loss function with multi-task structure.

	Attributes
	----------
	data: Data
		The data required to defined the loss.

	Methods
	-------
	loss(beta)
		Returns the loss function evaluated at beta.
	gradient(beta)
		Returns the gradient evaluated at beta.

	"""

	def __init__(self, data: Data, **kwargs):
		self.data = data

	def loss(self, beta: np.ndarray, task: Optional[str]):
		pass

	def gradient(self, beta: np.ndarray, task: Optional[str]):
		pass


class SeparableMTLoss(MTLoss):
	"""Multi-task separable loss.

	This class implements a loss function with multi-task structure that is separable across tasks.

	Attributes
	----------
	_losses: dict of Loss
		The individual losses accessible as elements of a dictionary.
	data: Data
		The data required to defined the loss.

	Methods
	-------
	loss(beta)
		Returns the loss function evaluated at beta.
	gradient(beta)
		Returns the gradient evaluated at beta.

	"""

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
	"""Multi-task Weighted Least Squares loss.

	This class implements the multi-task weighted least squares loss.
	"""

	def __init__(self, data: Data):
		super().__init__(data)
		for task in self.data.tasks:
			self[task] = WLS(
				self.data.x(task).to_numpy(),
				self.data.y(task).to_numpy().reshape((-1, 1)),
				self.data.w(task).to_numpy().reshape((-1, 1))
			)