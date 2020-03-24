from MTSGL.losses import SeparableMTLoss
from MTSGL.regularizations import Regularization
from . import Fit, ConvergenceError
import numpy as np
import pandas as pd


class ConsensusADMM(Fit):

	def __init__(
			self,
			loss: SeparableMTLoss,
			reg: Regularization,
			**kwargs
	):
		super().__init__(loss, reg, **kwargs)
		self.path = self._solution_path()

	def _set_additional_options(self, **kwargs):
		if "eps_abs" not in kwargs.keys():
			self.eps_abs = 1.0e-4
		else:
			self.eps_abs = float(kwargs["eps_abs"])
			if self.eps_abs < 1.0e-3:
				raise ValueError("eps_abs must be above 1.0e-16")
		if "eps_rel" not in kwargs.keys():
			self.eps_rel = 1.0e-6
		else:
			self.eps_rel = float(kwargs["eps_rel"])
			if self.eps_rel < 1.0e-8:
				raise ValueError("eps_rel must be above 1.0e-8")
		if "rho" not in kwargs.keys():
			self.rho = 1.
		else:
			self.rho = float(kwargs["rho"])
			if self.rho <= 0.:
				raise ValueError("rho must be positive, received {}".format(self.rho))

	def _solve(self, beta: np.ndarray, lam: float):
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
		b = beta
		r = b - beta
		d = np.zeros_like(b)
		t = 0
		augmented_obj, original_obj = self.compute_obj(b, beta, lam, r)
		self.log_solve = pd.DataFrame({
			"t": [t], "original obj.": [original_obj], "augmented obj.": [augmented_obj], "status": ["initial"]}
		)
		while True:
			t += 1
			# update B
			for k, task in enumerate(self.loss.data.tasks):
				b[:, [k]] = self.loss[task].ridge(
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
			# norms for convergence
			eps_dual, eps_primal, r_norm, s_norm = self.compute_convergence_checks(b, beta, d, r)
			# logging
			augmented_obj, original_obj = self.compute_obj(b, beta, lam, r)
			self.log_solve = self.log_solve.append(
				pd.DataFrame({
					"t": [t], "original obj.": [original_obj], "augmented obj.": [augmented_obj],
					"status": ["converged"]}
				),
				ignore_index=True
			)
			# print(r_norm, eps_primal, s_norm, eps_dual)
			# convergence checks
			if r_norm < eps_primal and s_norm < eps_dual:
				break
			if t > self.max_iter:
				print(self.log_solve)
				raise ConvergenceError("maximum number of iteration reached.")
		if self.verbose > 1:
			print(self.log_solve)
		return beta, t

	def compute_obj(self, b, beta, lam, r):
		original_obj = self.loss.loss(beta) + lam * self.reg.value(beta)
		# TODO check this, I think it is missing a term
		augmented_obj = self.loss.loss(b) + lam * self.reg.value(beta) + self.rho * np.tensordot(r, r) / 2.
		return augmented_obj, original_obj

	def compute_convergence_checks(self, b, beta, d, r):
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
