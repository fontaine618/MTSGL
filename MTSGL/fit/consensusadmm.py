from losses import SeparableMTLoss
from regularizations import Regularization
from . import Fit, ConvergenceError
import numpy as np
import pandas as pd
from multiprocessing import Pool


class ConsensusADMM(Fit):

	def __init__(
			self,
			loss: SeparableMTLoss,
			reg: Regularization,
			**kwargs
	):
		super().__init__(loss, reg, **kwargs)

	def _solve(self, beta: np.ndarray, lam: float, l: int):
		"""

		Parameters
		----------
		beta : array-like
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
		b, d, t = self._initialize(beta, lam)
		while True:
			t += 1
			# update
			beta, d, r, n_ridge = self._update(b, beta, d, lam)
			# logging
			loss, original_obj = self._log(b, beta, lam, l, r, t, n_ridge)
			# norms for convergence
			eps_dual, eps_primal, r_norm, s_norm = self._compute_convergence_checks(b, beta, d, r)
			# convergence checks
			if r_norm < eps_primal and s_norm < eps_dual:
				break
			if t > self.max_iter:
				print(self.log_solve)
				raise ConvergenceError("maximum number of iteration reached.")
		if self.verbose > 1:
			print(self.log_solve)
		return beta, t, loss, original_obj

	def _update(self, b, beta, d, lam):
		# update B
		n_ridge = np.array([0 for _ in self.loss])
		for k, (task, loss) in enumerate(self.loss.items()):
			b[:, [k]], n_ridge[k] = loss.ridge(
				1. / self.rho,
				beta[:, [k]] - d[:, [k]],
				b[:, [k]],
				threshold=np.sqrt(self.loss.data.n_features * self.loss.data.n_tasks) * self.eps_abs
			)
		# update beta
		beta = self.reg.proximal(b + d, lam / self.rho)
		# update r and d
		r = b - beta
		d += r
		return beta, d, r, n_ridge.mean()

	def _log(self, b, beta, lam, l, r, t, n_ridge):
		loss, augmented_obj, original_obj = self._compute_obj(b, beta, lam, r)
		self.log_solve = self.log_solve.append(
			pd.DataFrame({
				"l": [l], "t": [t], "loss": loss,
				"original obj.": [original_obj], "augmented obj.": [augmented_obj],
				"status": ["ADMM iteration"], "n_grad": n_ridge, "n_prox": 1
			}),
			ignore_index=True
		)
		return loss, original_obj

	def _initialize(self, beta, lam):
		b = beta
		r = b - beta
		d = np.zeros_like(b)
		t = 0
		loss, augmented_obj, original_obj = self._compute_obj(b, beta, lam, r)
		self.log_solve = self.log_solve.append(pd.DataFrame({
			"l": [0], "t": [t], "loss": loss, "original obj.": [original_obj],
			"augmented obj.": [augmented_obj], "status": ["initial"], "n_grad": [0], "n_prox": [0]}
		))
		return b, d, t

	def _compute_obj(self, b, beta, lam, r):
		loss = self.loss.loss(beta)
		original_obj = loss + lam * self.reg.value(beta)
		# TODO check this, I think it is missing a term
		augmented_obj = self.loss.loss(b) + lam * self.reg.value(beta) + self.rho * np.tensordot(r, r) / 2.
		return loss, augmented_obj, original_obj

	def _compute_convergence_checks(self, b, beta, d, r):
		r_norm = np.linalg.norm(r, 'fro')
		s_norm = self.rho * r_norm
		d_norm = np.linalg.norm(d, 'fro')
		b_norm = np.linalg.norm(b, 'fro')
		beta_norm = np.linalg.norm(beta, 'fro')
		eps_primal = np.sqrt(self.loss.data.n_features * self.loss.data.n_tasks) * self.eps_abs
		eps_primal += self.eps_rel * max(b_norm, beta_norm)
		eps_dual = np.sqrt(self.loss.data.n_features * self.loss.data.n_tasks) * self.eps_abs
		eps_dual += self.eps_rel * d_norm * self.rho
		return eps_dual, eps_primal, r_norm, s_norm
