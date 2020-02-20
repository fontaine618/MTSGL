import numpy as np
from MTSGL.losses.MTLoss import MTLoss


class Regularization:
	"""Regularization for Multi-task problems.

	Attributes
	----------

	Methods
	-------
	proximal(x, tau)
		Returns the proximal operator evaluated at x with multiplier tau.
	max_lam(loss)
		Returns the maximum regularization value such that all features are excluded for a given loss.
	"""

	def __init__(self, **kwargs):
		pass

	def _str_parm(self):
		return None

	def __str__(self):
		str_parm = self._str_parm()
		return self.name + ("({})".format(str_parm) if str_parm is not None else "")

	def __repr__(self):
		str_parm = self._str_parm()
		return self.name + ("({})".format(str_parm) if str_parm is not None else "")

	def proximal(self, x: np.ndarray, tau: float) -> np.ndarray:
		pass

	def max_lam(self, loss: MTLoss) -> float:
		pass
