import numpy as np
import MTSGL.solvers
from .Loss import Loss


class WLS(Loss):
	def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray):
		super().__init__(x, y)
		self.w = w
		self.__name = "Weighted Least Squares"
		self.n, self.p = x.shape
		eig = np.power(np.linalg.svd(self.x * np.sqrt(self.w), compute_uv=False), 2)
		self.L = max(eig)
		self.mu = min(eig)

	def loss(self, beta: np.ndarray):
		residuals = np.matmul(self.x, beta) - self.y
		return np.matmul(residuals.transpose(), self.w * residuals)[0,0]

	def gradient(self, beta: np.ndarray):
		return np.matmul(self.x.transpose(), self.w * (np.matmul(self.x, beta) - self.y))

	def predict(self, beta: np.ndarray):
		return self.lin_predictor(beta)

	def ridge_closed_form(self, tau: float, v: np.ndarray):
		mat = np.matmul(self.x.transpose(), self.w * self.x) + np.eye(self.p) / tau
		return np.linalg.solve(mat, np.matmul(self.x.transpose(), self.w * self.y)  + v / tau)

	def ridge(self, tau: float, v: np.ndarray, x0: np.ndarray, **kwargs):
		return MTSGL.solvers.ridge(
			loss=self,
			x0=x0,
			v=v,
			tau=tau,
			**kwargs
		)[0]

	def hessian_upper_bound(self):
		return self.L

	def hessian_lower_bound(self):
		return self.mu
