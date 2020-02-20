import numpy as np
import pandas as pd
from MTSGL.losses import MTLoss
from MTSGL.regularizations import Regularization


class Fit:

	def __init__(
			self,
			loss: MTLoss,
			reg: Regularization,
			**kwargs
	) -> None:
		self.loss = loss
		self.reg = reg
		self.threshold = None
		self.max_iter = None
		self._set_options(**kwargs)
		self._set_lambda(**kwargs)

	def _set_options(self, **kwargs):
		if "threshold" not in kwargs.keys():
			self.threshold = 1.0e-6
		else:
			self.threshold = float(kwargs["threshold"])
			if self.threshold < 1.0e-16:
				raise ValueError("threshold must be above 1.0e-16")
		self.verbose = True if "verbose" not in kwargs.keys() else bool(kwargs["verbose"])
		if "max_iter" not in kwargs.keys():
			self.max_iter = 1_000
		else:
			self.max_iter = int(kwargs["max_iter"])
			if not (1 <= self.max_iter <= 100_000):
				raise ValueError("max_iter must be between 1 and 100,000")
		self._set_additional_options(**kwargs)

	def _set_additional_options(self, **kwargs):
		pass

	def _set_lambda(self, **kwargs):
		self.lam_decrease = None
		self.n_lam = None
		if "user_lam" in kwargs:
			#   user-defined sequence (most likely from CV)
			lam = sorted(kwargs["user_lam"], reverse=True)
			self.n_lam = len(lam)
		else:
			#  log decrease, need to first find the largest
			max_lam = self._find_max_lam()
			n_lam = 100 if "n_lam" not in kwargs else kwargs["n_lam"]
			lam_frac = 1.0e-3 if "lam_frac" not in kwargs else kwargs["lam_frac"]
			lam_decrease = np.power(lam_frac, 1.0 / (n_lam - 1))
			lam = max_lam * np.power(lam_decrease, range(n_lam))
			self.lam_decrease = lam_decrease
			self.n_lam = len(lam)
		self.lam = np.array(lam)

	def _find_max_lam(self):
		return self.reg.max_lam(self.loss)

	def _solution_path(self):
		log = pd.DataFrame(columns=["l", "lambda", "size", "status"])
		beta = np.zeros((self.n_lam, self.loss.data.n_features, self.loss.data.n_tasks))
		for l in range(self.n_lam):
			beta0 = beta[l, :, :]
			lam = self.lam[l]
			p = 0
			try:
				self._solve(beta0, lam)
			except ConvergenceError as error:
				log = log.append(pd.DataFrame({
					"l": [l],
					"lambda": [lam],
					"status": ["error"]
				}), ignore_index=True)
				print("Solution path stopped after {} lambda values :".format(l))
				print(error)
				break
			else:
				log = log.append(pd.DataFrame({
					"l": [l],
					"lambda": [lam],
					"size": [p],
					"status": ["converged"]
				}), ignore_index=True)
			finally:
				if l >= self.n_lam - 1:
					break
				beta[l, :, :] = beta0 + 1.
		if self.verbose:
			print(log)
			print(beta)
		return beta

	def _solve(self, beta0: np.ndarray, lam: float, **kwargs):
		"""

		Parameters
		----------
		beta0 :
		lam :
		kwargs :

		Returns
		-------

		Raises
		------
		ConvergenceError
			If the solver does not reach appropriate convergence.
		NullModelError
			If the solvers return an empty model. Signals that we have to decrease lambda.
		"""
		pass


class ConvergenceError(Exception):
	"""Raised when convergence criterion was not met."""

	def __init__(self, value):
		self.value = "ConvergenceError: " + str(value) + "\nTry increasing the threshold or the number of iterations."

	def __str__(self):
		return str(self.value)
