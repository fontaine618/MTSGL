import numpy as np
from .loss import Loss


class WLS(Loss):
	"""Single-task (Weighted) Least Squares loss.

	Attributes
	----------
	x: array-like
		The features.
	y: array-like
		The responses.
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
		self.L = max(eig)
		self.L_saturated = max(self.w)
		self.mu = min(eig)

	def loss_from_linear_predictor(self, eta):
		return np.sum(self.w * (eta - self.y) ** 2)

	def gradient(self, beta: np.ndarray):
		return np.matmul(self.x.transpose(), self.w * (self.lin_predictor(beta) - self.y))

	def predict(self, beta: np.ndarray):
		return self.lin_predictor(beta)

	def ridge_closed_form(self, tau: float, v: np.ndarray):
		"""Returns the ridge-regularized minimizer using the closed-form solution."""
		mat = np.matmul(self.x.transpose(), self.w * self.x) + np.eye(self.p) / tau
		return np.linalg.solve(mat, np.matmul(self.x.transpose(), self.w * self.y) + v / tau)

	def hessian_saturated_upper_bound(self):
		return self.L_saturated

	def hessian_upper_bound(self):
		return self.L

	def hessian_lower_bound(self):
		return self.mu

	def gradient_saturated(self, eta: np.ndarray):
		return self.w * (eta - self.y)
