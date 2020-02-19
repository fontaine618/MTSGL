import numpy as np
from MTSGL.losses.Loss import Loss


class Regularization:

	def __init__(self, **kwargs):
		self.name = "Regularization"

	def _str_parm(self):
		return None

	@property
	def __str__(self):
		str_parm = self._str_parm()
		return self.name + ("({})".format(str_parm) if str_parm is not None else "")

	def proximal(self, x: np.ndarray, tau: float) -> np.ndarray:
		pass

	def max_lam(self, loss: Loss) -> float:
		pass
