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

