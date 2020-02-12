import numpy as np
import MTSGL


class Loss:
	def __init__(self):
		pass

	def loss(self, **kwargs):
		pass

	def gradient(self, **kwargs):
		pass

	def hessian_bounds(selfself, **kwargs):
		pass

	@staticmethod
	def _lin_predictor(beta, x):
		return np.matmul(x, beta)


class LS(Loss):
	def __init__(self, x, y):
		super().__init__()
		self.name = "Least Squares"
		self.n, self.p = x.shape
		self.hessian_bounds(x)

	def hessian_bounds(self, x):
		#eig = np.linalg.eigvalsh(np.matmul(x.transpose(), x) / self.n)
		eig = np.linalg.svd(x, compute_uv=False)**2 / self.n
		self.hessian_upper_bound = max(eig)
		self.hessian_lower_bound = min(eig)

	def loss(self, beta, x, y):
		return (np.linalg.norm(np.matmul(x, beta) - y, 2) ** 2) / (self.n*2)

	def gradient(self, beta, x, y):
		return np.matmul(x.transpose(), np.matmul(x, beta) - y) / self.n

	def predict(self, beta, x):
		return self._lin_predictor(beta, x)

	def ridge(self, x, y, tau, v):
		return self.ridge_closed_form(x, y, tau, v)  # faster n>p, slower n<p

	def ridge_closed_form(self, x, y, tau, v):
		mat = np.matmul(x.transpose(), x) / self.n + np.eye(self.p) / tau
		return np.linalg.solve(mat, np.matmul(x.transpose(), y)/self.n + v / tau)

	def ridge_gd(self, x, y, tau, v):
		return MTSGL.solvers._ridge._ridge_gd(
			loss=self,
			beta0=np.zeros((self.p, 1)),
			v=v,
			tau=tau,
			x=x,
			y=y
		)

class WLS(Loss):
	def __init__(self):
		super().__init__()
		self.name = "Weighted Least Squares"

	def loss(self, beta, x, y, w):
		residuals = np.matmul(x, beta) - y
		return np.matmul(residuals.transpose(), w * residuals)

	def gradient(self, beta, x, y, w):
		return np.matmul(x.transpose(), w * (np.matmul(x, beta) - y))

	def predict(self, x, beta):
		return self._lin_predictor(beta, x)
