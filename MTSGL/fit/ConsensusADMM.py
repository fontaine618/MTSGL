from MTSGL.losses import SeparableMTLoss
from MTSGL.regularizations import Regularization
from .Fit import Fit, ConvergenceError
import numpy as np
import pandas as pd

class ConsensusADMM(Fit):

	def __init__(
			self,
			loss: SeparableMTLoss,
			reg: Regularization,
			**kwargs
	):
		self.threshold_ridge_decrease = None
		super().__init__(loss, reg, **kwargs)
		self.path = self._solution_path()

	def _set_additional_options(self, **kwargs):
		if "eps_abs" not in kwargs.keys():
			self.eps_abs = 1.0e-6
		else:
			self.eps_abs = float(kwargs["eps_abs"])
			if self.eps_abs < 1.0e-16:
				raise ValueError("eps_abs must be above 1.0e-16")
		if "eps_rel" not in kwargs.keys():
			self.eps_rel = 1.0e-3
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
		primal_obj = self.loss.loss(beta) + lam * self.reg.value(beta)
		dual_obj = self.loss.loss(b) + lam * self.reg.value(beta) + self.rho * np.tensordot(r, r)
		self.log_solve = pd.DataFrame({
				"t": [t], "primal obj.": [primal_obj], "dual obj.": [dual_obj], "status": ["initial"]}
		)
		while True:
			t += 1
			# update B
			for k, task in enumerate(self.loss.data.tasks):
				b[:, [k]] = self.loss[task].ridge(
					1./self.rho,
					beta[:, [k]] - d[:, [k]],
					b[:, [k]],
					threshold=max(
						np.sqrt(self.loss.data.n_features * self.loss.data.n_tasks) * self.eps_abs,
						1./np.power(2, t)
					)
				)
			# update beta
			beta = self.reg.proximal(b + d, lam / self.rho)
			# update D
			r = b - beta
			d += r
			# norms for convergence
			r_norm = np.linalg.norm(r, 'fro')
			s_norm = self.rho * r_norm
			d_norm = np.linalg.norm(d, 'fro')
			b_norm = np.linalg.norm(b, 'fro')
			beta_norm = np.linalg.norm(beta, 'fro')
			eps_primal = np.sqrt(self.loss.data.n_features * self.loss.data.n_tasks) * self.eps_abs + \
				self.eps_rel * max(b_norm, beta_norm)
			eps_dual = np.sqrt(self.loss.data.n_features * self.loss.data.n_tasks) * self.eps_abs + \
				self.eps_rel * d_norm * self.rho
			# convergence checks
			primal_obj = self.loss.loss(beta) + lam * self.reg.value(beta)
			dual_obj = self.loss.loss(b) + lam * self.reg.value(beta) + self.rho * np.tensordot(r, r) / 2.
			self.log_solve = self.log_solve.append(
				pd.DataFrame({
					"t": [t], "primal obj.": [primal_obj], "dual obj.": [dual_obj], "status": ["converged"]}
				),
				ignore_index=True
			)
			if r_norm < eps_primal and s_norm < eps_dual:
				break
			if t > self.max_iter:
				print(self.log_solve)
				raise ConvergenceError("maximum number of iteration reached.")
		if self.verbose > 1:
			print(self.log_solve)
		return beta, t
