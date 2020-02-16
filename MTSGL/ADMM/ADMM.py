import numpy as np
from MTSGL import Data
from MTSGL.losses import Loss
from MTSGL.regularizations import Regularization


class ADMM:

	def __init__(
			self,
			data: Data,
			loss: Loss,
			reg: Regularization,
			lam: float,
			**kwargs
	):
		if not data.__type == loss.__type:
			raise ValueError(
				"data and loss must correspond to the same type of problem: received {} data and {} loss"
					.format(data.__type, loss.__type)
			)
		self.data = data
		self.loss = loss
		if lam < 0.:
			raise ValueError("lam must be non-negative")
		self.lam = lam
		if "threshold" not in kwargs.keys():
			self.threshold = 1.0e-6
		else:
			self.threshold = float(kwargs["threshold"])
			if self.threshold < 1.0e-16:
				raise ValueError("threshold must be above 1.0e-16")
		if "max_iter" not in kwargs.keys():
			self.max_iter = 1_000
		else:
			self.max_iter = int(kwargs["max_iter"])
			if not (1 <= self.max_iter <= 10_000):
				raise ValueError("max_iter must be between 1 and 10,000")
		if "beta0" not in kwargs.keys():
			self.beta0 = np.zeros(data.n_features, data.n_tasks)
		else:
			if not isinstance(kwargs["beta0"], np.adrray):
				raise TypeError("beta0 should be a numpy array")
			if not kwargs["beta0"].shape == (data.n_features, data.n_tasks):
				raise ValueError(
					"beta0 should be of dimension (p,K): expected ({}, {}), received ({}, {})"
					.format(data.n_features, data.n_tasks, *kwargs["beta0"].shape)
				)
			if not (1 <= self.max_iter <= 10_000):
				raise ValueError("max_iter must be between 1 and 10,000")
			self.beta0 = kwargs["beta0"]
