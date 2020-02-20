from MTSGL.losses import SeparableMTLoss
from MTSGL.regularizations import Regularization
from .Fit import Fit, ConvergenceError
import numpy as np


class ConsensusADMM(Fit):

	def __init__(
			self,
			loss: SeparableMTLoss,
			reg: Regularization,
			**kwargs
	):
		self.threshold_ridge_decrease = None
		super().__init__(loss, reg, **kwargs)
		path = self._solution_path()

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
		t = 0
		while True:
			beta = beta + np.random.choice(
				[0, 1],
				(self.loss.data.n_features, self.loss.data.n_tasks), p=[t/(t+1), 1/(t+1)])
			if np.linalg.norm(beta - beta_prev, ord=2) < self.threshold:
				break
			beta_prev = beta
			t += 1
			if t > self.max_iter:
				raise ConvergenceError("maximum number of iteration reached.")
		return beta, t
