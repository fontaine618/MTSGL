import numpy as np
import MTSGL.solvers


class Loss:
	def __init__(self, x: np.ndarray, y: np.ndarray):
		self.__type = None
		self.x = x
		self.y = y
		self.__type = ""

	def lin_predictor(self, beta):
		return np.matmul(self.x, beta)

	def loss(self, beta):
		pass

	def gradient(self, beta):
		pass

	def hessian_upper_bound(self):
		pass

	def hessian_lower_bound(self):
		pass


# TODO split child classes into different files, I don't know how ...
class LS(Loss):
	def __init__(self, x: np.ndarray, y: np.ndarray):
		super().__init__(x, y)
		self.__name = "Least Squares"
		self.__type = "Regression"
		self.n, self.p = x.shape
		eig = np.power(np.linalg.svd(self.x, compute_uv=False), 2) / self.n
		self.L = max(eig)
		self.mu = min(eig)

	def loss(self, beta: np.ndarray):
		return (np.linalg.norm(np.matmul(self.x, beta) - self.y, 2) ** 2) / (self.n * 2)

	def gradient(self, beta: np.ndarray):
		return np.matmul(self.x.transpose(), np.matmul(self.x, beta) - self.y) / self.n

	def predict(self, beta: np.ndarray):
		return self.lin_predictor(beta)

	def ridge_closed_form(self, tau: float, v: np.ndarray):
		mat = np.matmul(self.x.transpose(), self.x) / self.n + np.eye(self.p) / tau
		return np.linalg.solve(mat, np.matmul(self.x.transpose(), self.y) / self.n + v / tau)

	def ridge(self, tau: float, v: np.ndarray, **kwargs):
		return MTSGL.solvers.ridge(
			loss=self,
			x0=np.zeros((self.p, 1)),
			v=v,
			tau=tau,
			**kwargs
		)[0]

	def hessian_upper_bound(self):
		return self.L

	def hessian_lower_bound(self):
		return self.mu


class WLS(Loss):
	def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray):
		super().__init__(x, y)
		self.w = w
		self.__name = "Weighted Least Squares"
		self.__type = "Regression"

	def loss(self, beta: np.ndarray):
		residuals = np.matmul(self.x, beta) - self.y
		return np.matmul(residuals.transpose(), self.w * residuals)

	def gradient(self, beta: np.ndarray):
		return np.matmul(self.x.transpose(), self.w * (np.matmul(self.x, beta) - self.y))

	def predict(self, beta: np.ndarray):
		return self.lin_predictor(beta)
