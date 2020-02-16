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
	) -> None:
		"""
		
		Parameters
		----------
		data : 
		loss : 
		reg : 
		lam :
		beta0 :
		threshold :
		max_iter :

		Returns
		-------
		None
		"""
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
		self.beta0 = None
		self.threshold = None
		self.max_iter = None
		self.set_options(**kwargs)

	def set_beta0(self, beta0):
		if beta0 is None:
			self.beta0 = np.zeros(self.data.n_features, self.data.n_tasks)
		else:
			if not isinstance(beta0, np.adrray):
				raise TypeError("beta0 should be a numpy array")
			if not beta0.shape == (self.data.n_features, self.data.n_tasks):
				raise ValueError(
					"beta0 should be of dimension (p,K): expected ({}, {}), received ({}, {})"
					.format(self.data.n_features, self.data.n_tasks, *beta0.shape)
				)
			self.beta0 = beta0

	def set_options(self, **kwargs):
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
			if not (1 <= self.max_iter <= 100_000):
				raise ValueError("max_iter must be between 1 and 100,000")
		self.set_beta0(kwargs["beta0"])
		self.set_additional_options(**kwargs)

	def set_additional_options(self, **kwargs):
		pass  # if child class does not implement it
