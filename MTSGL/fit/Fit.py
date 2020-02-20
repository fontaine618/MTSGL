import numpy as np
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
		if self.verbose:
			header = "| {:<5} | {:<10} | {:<10} | {:<10} |".format("l", "lambda", "size", "status")
			print("=" * len(header))
			print("Solution path\n")
			print("-" * len(header))
			print(header)
			print("-" * len(header))
		beta = np.zeros((self.n_lam, self.loss.data.n_features, self.loss.data.n_tasks))
		l = 0
		while True:
			beta0 = beta[l, :, :]
			lam = self.lam[l]
			p = 0
			try:
				self._solve(beta0, lam)
			except ConvergenceError as error:
				if self.verbose:
					print("| {:<5} | {:<10.6f} | {:<10} | {:<10} |".format(l, lam, "-", "error"))
					print("-" * len(header))
				print("Solution path stopped after {} lambda values :".format(l))
				print(error)
				break
			else:
				if self.verbose:
					print("| {:<5} | {:<10.6f} | {:<10} | {:<10} |".format(l, lam, p, "converged"))
			finally:
				pass
			if l >= self.n_lam - 1:
				break
			l += 1
			beta[l, :, :] = beta0 + 1.
		if self.verbose:
			print("=" * len(header))
		pass

	def _solve(self, beta0: np.ndarray, lam: float, **kwargs):
		if lam < 0.005:
			raise ConvergenceError("Good error handling!")


class ConvergenceError(Exception):
	"""Raised when convergence criterion was not met."""

	def __init__(self, value):
		self.value = "ConvergenceError: " + str(value)

	def __str__(self):
		return str(self.value)
