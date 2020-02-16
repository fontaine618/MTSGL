from .Regularization import Regularization
from typing import Union
import numpy as np
import MTSGL.proximal


class SparseGroupLasso(Regularization):

	def __init__(self, q: Union[str, int] = 2, alpha: float = 0.5) -> None:
		super().__init__()
		if q not in ["inf", 2]:
			raise NotImplementedError("q = {} not implemented yet".format(q))
		self.q = q
		if not 0. <= alpha <= 1.:
			raise ValueError("alpha must be between 0 and 1")
		self.alpha = alpha
		self.name = "SparseGroupLasso"

	def _str_parm(self) -> str:
		return "q = {}, alpha = {}".format(self.q, self.alpha)

	def proximal(self, x: np.ndarray, tau: float) -> np.ndarray:
		"""

		Parameters
		----------
		x: ndarray
			The matrix at which to evaluate the proximal (p, K)
		tau: float
			The multiplicative factor for the penalty term.

		Returns
		-------
		prox : ndarray
			The proximal value (p, K).

		"""
		return np.apply_along_axis(
			lambda x_col: MTSGL.proximal.proximal_sgl(x_col, tau, self.q, self.alpha),
			0,
			x
		)


class GroupLasso(SparseGroupLasso):

	def __init__(
			self,
			q: Union[str, int] = 2
	):
		super().__init__(q, 0.)
		self.name = "GroupLasso"

	def _str_parm(self):
		return "q = {}".format(self.q)


class Lasso(SparseGroupLasso):

	def __init__(
			self
	):
		super().__init__(2, 1.)
		self.name = "Lasso"

	def _str_parm(self):
		return None
