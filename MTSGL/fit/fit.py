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
		self.max_iter = None
		self._set_options(**kwargs)
		self._set_lambda(**kwargs)
		self.path = self._solution_path()

	def _set_options(self, **kwargs):
		self.verbose = 0 if "verbose" not in kwargs else kwargs["verbose"]
		self.max_iter = 10_000 if "max_iter" not in kwargs else int(kwargs["max_iter"])
		self.max_size = self.loss.data.n_features if "max_size" not in kwargs else int(kwargs["max_size"])
		self.eps_abs = 1.0e-6 if "eps_abs" not in kwargs else float(kwargs["eps_abs"])
		self.eps_rel = 1.0e-3 if "eps_rel" not in kwargs else float(kwargs["eps_rel"])
		self.rho = 1. if "rho" not in kwargs else float(kwargs["rho"])
		if not (1 <= self.max_iter <= 100_000):
			raise ValueError("max_iter must be between 1 and 100,000")
		if not (1 <= self.max_size <= self.loss.data.n_features):
			raise ValueError("max_size must be between 1 and p={}".format(self.loss.data.n_features))
		if self.eps_abs < 1.0e-16:
			raise ValueError("eps_abs must be above 1.0e-16")
		if self.eps_rel < 1.0e-8:
			raise ValueError("eps_rel must be above 1.0e-8")
		if self.rho <= 0.:
			raise ValueError("rho must be positive, received {}".format(self.rho))

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
		self.log = pd.DataFrame(columns=["l", "lambda", "size", "status", "nb. iter", "loss", "obj"])
		self.log_solve = pd.DataFrame(
			columns=["l", "t", "loss", "original obj.", "augmented obj.", "status", "n_grad", "n_prox"]
		)
		beta = np.zeros((self.n_lam, self.loss.data.n_features, self.loss.data.n_tasks))
		beta[:] = np.nan
		b = np.zeros((self.loss.data.n_features, self.loss.data.n_tasks))
		for l, lam in enumerate(self.lam):
			try:
				b, nb_iter, loss, obj = self._solve(b, lam, l)
			except ConvergenceError as error:
				self.log = self.log.append(
					pd.DataFrame({"l": [l], "lambda": [lam], "status": ["error"]}),
					ignore_index=True
				)
				print("Solution path stopped after {} lambda values :".format(l))
				print(error)
				break
			else:
				p = sum(np.apply_along_axis(np.linalg.norm, 1, b, 1) > 0.0)
				self.log = self.log.append(
					pd.DataFrame({
						"l": [l], "lambda": [lam], "size": [p], "status": ["converged"], "nb. iter": [nb_iter],
						"loss": loss, "obj": obj
					}),
					ignore_index=True
				)
				beta[l, :, :] = b
				if p > self.max_size:
					print(
						"Solution path stopped after {} lambda values :\nMaximum model size reached ({}/{})."
						.format(l, p, self.max_size)
					)
					self.log.at[l, "status"] = "max size"
					break
			finally:
				pass
		if self.verbose > 0:
			print(self.log)
		return beta

	def _solve(self, beta0: np.ndarray, lam: float, **kwargs):
		"""

		Parameters
		----------
		beta0 : array-like
			The initial parameter value. (p, K)
		lam : float
			The regularization parameter.

		Returns
		-------
		beta : array-like
			The final estimate. (p, K)

		Raises
		------
		ConvergenceError
			If the solver does not reach appropriate convergence.
		"""
		pass

class ConvergenceError(Exception):
	"""Raised when convergence criteria are not met."""

	def __init__(self, value):
		self.value = "ConvergenceError: " + str(value) + "\nTry increasing the threshold or the number of iterations."

	def __str__(self):
		return str(self.value)
