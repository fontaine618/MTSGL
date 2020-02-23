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
		beta_prev = beta
		b = beta
		r = b - beta
		d = np.zeros_like(b)
		t = 0
		primal_obj = self.loss.loss(beta) + lam * self.reg.value(beta)
		primal_obj_prev = -np.inf
		dual_obj = self.loss.loss(b) + lam * self.reg.value(beta) + self.rho * np.tensordot(r, r)
		dual_obj_prev = -np.inf
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
					threshold=1e-3
				)
			# update beta
			beta = self.reg.proximal(b + d, lam / self.rho)
			# update D
			r = b - beta
			d += r
			# convergence checks
			primal_obj_prev = primal_obj
			dual_obj_prev = dual_obj
			primal_obj = self.loss.loss(b) + lam * self.reg.value(beta)
			dual_obj = self.loss.loss(b) + lam * self.reg.value(beta) + self.rho * np.tensordot(r, r) / 2.
			self.log_solve = self.log_solve.append(
				pd.DataFrame({
					"t": [t], "primal obj.": [primal_obj], "dual obj.": [dual_obj], "status": ["converged"]}
				),
				ignore_index=True
			)
			if np.abs(primal_obj_prev - primal_obj) / primal_obj_prev < self.threshold:
				break
			beta_prev = beta
			if t > self.max_iter:
				print(self.log_solve)
				raise ConvergenceError("maximum number of iteration reached.")
		if self.verbose > 1:
			print(self.log_solve)
		return beta, t
