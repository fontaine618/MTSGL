import numpy as np
from .loss import Loss
from .logistic import Logistic
from .wls import WLS
from MTSGL.data import Data
from typing import Optional, Dict


class MTLoss():
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
		self.L = None
		self.L_ls = None
		self.L_saturated = None

	def loss(self, beta: np.ndarray, task: Optional[str]):
		pass

	def loss_from_linear_predictor(self, eta: Dict[str, np.ndarray], task: Optional[str]):
		pass

	def gradient(self, beta: np.ndarray, task: Optional[str]):
		pass

	def gradient_saturated(self, z: Dict[str, np.ndarray]):
		pass

	def hessian_upper_bound(self):
		return self.L

	def hessian_saturated_upper_bound(self):
		return self.L_saturated

	def hessian_ls_upper_bound(self):
		return self.L_ls


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

	def get(self, task, default=None):
		if task in self:
			return self[task]
		else:
			return default

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
			return sum([loss.loss(beta[:, [k]]) for k, loss in enumerate(self.values())])
		else:
			if beta is None:
				beta = np.zeros((self.data.n_features, 1))
			return self[task].loss(beta)

	def loss_from_linear_predictor(self, eta: Optional[Dict[str, np.ndarray]] = None, task: Optional[str] = None):
		if task is None:
			if eta is None:
				eta = {task: np.zeros_like(loss.y) for task, loss in self.items()}
			return sum([loss.loss_from_linear_predictor(eta[task]) for task, loss in self.items()])
		else:
			if eta is None:
				eta = np.zeros_like(self.loss[task].y)
			return self[task].loss_from_linear_predictor(eta)

	def _set_upper_bounds(self):
		self.L = max([loss.hessian_upper_bound() for task, loss in self.items()])
		self.L_ls = max([loss.hessian_ls_upper_bound() for task, loss in self.items()])
		self.L_saturated = max([loss.hessian_saturated_upper_bound() for task, loss in self.items()])

	def gradient_saturated(self, z: Dict[str, np.ndarray]):
		grad = dict()
		for task, loss, in self.items():
			grad[task] = loss.gradient_saturated(z[task])
		return grad


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
		self._set_upper_bounds()


class MTLogReg(SeparableMTLoss):
	"""Multi-task Logistic Regression loss.

	This class implements the multi-label classification loss (y = 0/1 in each task).
	"""

	def __init__(self, data: Data):
		super().__init__(data)
		for task in self.data.tasks:
			self[task] = Logistic(
				self.data.x(task).to_numpy(),
				self.data.y(task).to_numpy().reshape((-1, 1)),
				self.data.w(task).to_numpy().reshape((-1, 1))
			)
		self._set_upper_bounds()