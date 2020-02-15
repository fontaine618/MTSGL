import numpy as np


class Regularization:

	def __init__(self, **kwargs):
		self.name = "Regularization"

	def _str_parm(self):
		return None

	@property
	def __str__(self):
		str_parm = self._str_parm()
		return self.name + ("({})".format(str_parm) if str_parm is not None else "")

	def proximal(self, beta_mat: np.ndarray, **kwargs) -> np.ndarray:
		pass
