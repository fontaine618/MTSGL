import numpy as np


class Loss:
	def __init__(self):
		pass

	def _loss(self, *args, **kwargs):
		pass

	def _gradient(self, *args, **kwargs):
		pass

	@staticmethod
	def _lin_predictor(x, beta):
		return np.matmul(x, beta)


class LS(Loss):
	def __init__(self):
		super().__init__()
		self.name = "Least Squares"

	def _loss(self, x, y, beta):
		return np.linalg.norm(np.matmul(x, beta) - y, 2) ** 2 / y.size

	def _gradient(self, x, y, beta):
		return np.matmul(x.transpose(), np.matmul(x, beta) - y) / y.size

	def _predict(self, x, beta):
		return self._lin_predictor(x, beta)


class WLS(LS):
	def __init__(self):
		super().__init__()
		self.name = "Weighted Least Squares"

	def _loss(self, x, y, w, beta):
		residuals = np.matmul(x, beta) - y
		return np.matmul(residuals.transpose(), w * residuals)

	def _gradient(self, x, y, w, beta):
		return np.matmul(x.transpose(), w * (np.matmul(x, beta) - y))
