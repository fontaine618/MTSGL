import numpy as np
from .loss import Loss
from scipy.special import expit


class Logistic(Loss):
	"""Single-task Logistic loss.

	Attributes
	----------
	x: array-like
		The features.
	y: array-like
		The responses (0/1).
	w: array-like
		The observation weights.
	L: float
		Hessian upper bound value.
	mu: float
		Hessian lower bound value.

	Methods
	-------
	lin_predictor(beta)
		Returns the linear predictor evaluated at beta.
	loss(beta)
		Returns the loss evaluated at beta.
	gradient(beta)
		Returns the gradient evaluated at beta.
	hessian_upper_bound
		Returns an upper bound to the Hessian matrix.
	hessian_lower_bound
		Returns a lower bound to the Hessian matrix.
	"""
	def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray):
		super().__init__(x, y)
		self.w = w
		self.n, self.p = x.shape
		eig = np.power(np.linalg.svd(self.x * np.sqrt(self.w), compute_uv=False), 2)
		self.L_ls = np.max(eig)
		self.L = self.L_ls / 4
		self.L_saturated = np.max(self.w)
		self.mu = np.min(eig)

	def loss_from_linear_predictor(self, eta):
		p = expit(eta).clip(1.0e-10, 1. - 1.0e-10)
		return - np.sum(self.w * (self.y * np.log(p) + (1 - self.y) * np.log(1 - p)))

	def gradient(self, beta: np.ndarray):
		return np.matmul(self.x.transpose(), self.gradient_saturated(self.lin_predictor(beta)))

	def predict(self, beta: np.ndarray):
		return expit(self.lin_predictor(beta))

	def hessian_saturated_upper_bound(self):
		return self.L_saturated

	def hessian_ls_upper_bound(self):
		return self.L_ls

	def hessian_upper_bound(self):
		return self.L

	def hessian_lower_bound(self):
		return self.mu

	def gradient_saturated(self, eta: np.ndarray):
		return - self.w * (self.y - expit(eta))
