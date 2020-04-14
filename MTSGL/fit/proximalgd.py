from losses import MTLoss
from regularizations import Regularization
from . import Fit, ConvergenceError
from solvers.proxgd import proxgd
import numpy as np
import pandas as pd
from time import time


class ProximalGD(Fit):

	def __init__(
			self,
			loss: MTLoss,
			reg: Regularization,
			**kwargs
	):
		super().__init__(loss, reg, **kwargs)

	def _solve(self, beta0: np.ndarray, lam: float, l: int, **kwargs):
		# initialization
		beta, t = self._initialize(beta0, lam)
		# solve
		while True:
			t0 = time()
			t += 1
			# update
			beta_prev = beta
			beta, n_proxgd = proxgd(
				loss=self.loss,
				reg=self.reg,
				beta0=beta,
				lam=lam,
				rho=1.0,
				ls=False,
				threshold=np.sqrt(self.loss.data.n_features * self.loss.data.n_tasks) * self.eps_abs,
				max_iter=1
			)
			dt = time() - t0
			# logging
			loss, original_obj = self._log(beta, lam, l, t, n_proxgd, n_proxgd, dt)
			# norms for convergence
			norm = np.linalg.norm(beta - beta_prev, 2)
			eps = np.sqrt(self.loss.data.n_features * self.loss.data.n_tasks) * self.eps_abs
			# convergence checks
			if norm < eps:
				break
			if t > self.max_iter:
				print(self.log_solve)
				raise ConvergenceError("maximum number of iteration reached.")
		if self.verbose > 1:
			print(self.log_solve)
		return beta, 1, loss, original_obj

	def _log(self, beta, lam, l, t, n_grad, n_prox, dt):
		loss, augmented_obj, original_obj = self._compute_obj(beta, lam)
		self.log_solve = self.log_solve.append(
			pd.DataFrame({
				"l": l, "t": [t], "loss": loss, "original obj.": [original_obj],
				"augmented obj.": [augmented_obj], "status": ["converged"],
				"n_grad": n_grad, "n_prox": n_prox, "time": dt
			}),
			ignore_index=True
		)
		return loss, original_obj

	def _initialize(self, beta0, lam):
		beta = beta0
		t = 0
		# setup logging
		loss, augmented_obj, original_obj = self._compute_obj(beta, lam)
		self.log_solve = self.log_solve.append(pd.DataFrame({
			"l": [0], "t": [t], "loss": loss, "original obj.": [original_obj], "time": 0.,
			"augmented obj.": [augmented_obj], "status": ["initial"], "n_grad": [0], "n_prox": [0]}
		))
		return beta, t

	def _compute_obj(self, beta, lam):
		loss = self.loss.loss(beta)
		original_obj = loss + lam * self.reg.value(beta)
		augmented_obj = 0.0
		return loss, augmented_obj, original_obj
