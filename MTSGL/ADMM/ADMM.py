import numpy as np
from typing import Union, List, Dict
from MTSGL.data.Data import Data
from MTSGL.losses import Loss
from MTSGL.regularizations import Regularization


class ADMM:
	# I think this class could be expanded to more general fits than just ADMM.

	def __init__(
			self,
			data: Data, # I think we can drop this and keep it in the fit class to be implemented above that
			losses: Dict[str, Loss],
			reg: Regularization,
			**kwargs
	) -> None:
		self.data = data
		self.losses = losses
		self.reg = reg
		self.beta0 = None
		self.threshold = None
		self.max_iter = None
		self._set_options(**kwargs)
		self._set_lambda(**kwargs)

	def set_beta0(self, beta0):
		if beta0 is None:
			self.beta0 = np.zeros((self.data.n_features, self.data.n_tasks))
		else:
			if not isinstance(beta0, np.adrray):
				raise TypeError("beta0 should be a numpy array")
			if not beta0.shape == (self.data.n_features, self.data.n_tasks):
				raise ValueError(
					"beta0 should be of dimension (p,K): expected ({}, {}), received ({}, {})"
					.format(self.data.n_features, self.data.n_tasks, *beta0.shape)
				)
			self.beta0 = beta0

	def _set_options(self, **kwargs):
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
		if "beta0" in kwargs:
			self.set_beta0(kwargs["beta0"])
		else:
			self.set_beta0(None)
		self._set_additional_options(**kwargs)

	def _set_additional_options(self, **kwargs):
		pass

	def _set_lambda(self, **kwargs):
		if "user_lam" in kwargs:
			#   user-defined sequence (most likely from CV)
			lam = kwargs["user_lam"]
		else:
			#  log decrease, need to first find the largest
			max_lam = self._find_max_lam()
			n_lam = 100 if "n_lam" not in kwargs else kwargs["n_lam"]
			lam_frac = 1.0e-3 if "lam_frac" not in kwargs else kwargs["lam_frac"]
			lam_decrease = np.power(lam_frac, 1.0/(n_lam-1))
			lam = max_lam * np.power(lam_decrease, range(n_lam))
		self.lam = (l for l in lam)

	def _find_max_lam(self):
		return self.reg.max_lam(self.losses)