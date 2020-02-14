import numpy as np
import MTSGL.solvers
from .Loss import Loss


class WLS(Loss):
	def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray):
		super().__init__(x, y)
		self.w = w
		self.__name = "Weighted Least Squares"

	def loss(self, beta: np.ndarray):
		residuals = np.matmul(self.x, beta) - self.y
		return np.matmul(residuals.transpose(), self.w * residuals)

	def gradient(self, beta: np.ndarray):
		return np.matmul(self.x.transpose(), self.w * (np.matmul(self.x, beta) - self.y))

	def predict(self, beta: np.ndarray):
		return self.lin_predictor(beta)
