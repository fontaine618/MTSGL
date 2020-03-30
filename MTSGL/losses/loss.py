import numpy as np
import MTSGL.solvers
from typing import Optional


class Loss:
	"""Abstract single-task loss.

	Attributes
	----------
	x: array-like
		The features.
	y: array-like
		The responses.

	Methods
	-------
	_lin_predictor(beta)
		Returns the linear predictor evaluated at beta.
	loss(beta)
		Returns the loss evaluated at beta.
	gradient(beta)
		Returns the gradient evaluated at beta.
	hessian_upper_bound
		Returns an upper bound to the Hessian matrix.
	hessian_lower_bound
		Returns a lower bound to the Hessian matrix.
	ridge(tau, v, x0, ...)
		Returns the ridge-regularized minimizer.
	predict(beta)
		Returns the predicted response at beta.
	"""
	def __init__(self, x: np.ndarray, y: np.ndarray):
		self.x = x
		self.y = y

	def lin_predictor(self, beta):
		return np.matmul(self.x, beta)

	def loss(self, beta):
		return self.loss_from_linear_predictor(self.lin_predictor(beta))

	def loss_from_linear_predictor(self, eta):
		pass

	def gradient(self, beta):
		pass

	def hessian_saturated_upper_bound(self, **kwargs):
		pass

	def hessian_upper_bound(self, **kwargs):
		pass

	def hessian_lower_bound(self, **kwargs):
		pass

	def ridge(
			self,
			tau: float,
			v: np.ndarray,
			x0: Optional[np.ndarray] = None,
			**kwargs):
		if x0 is None:
			x0 = np.zeros_like(v)
		return MTSGL.solvers.ridge(
			loss=self,
			x0=x0,
			v=v,
			tau=tau,
			**kwargs
		)

	def predict(self, beta: np.ndarray):
		pass

	def gradient_saturated(self, z: np.ndarray):
		pass
