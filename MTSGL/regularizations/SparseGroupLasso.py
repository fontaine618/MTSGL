from .Regularization import Regularization
from typing import Union, Optional, List, Dict
import numpy as np
import pandas as pd
import MTSGL.proximal
from MTSGL.losses import Loss, MTLoss


class SparseGroupLasso(Regularization):

	def __init__(
			self,
			q: Union[str, int] = 2,
			alpha: float = 0.5,
			weights: Optional[List] = None
	) -> None:
		super().__init__()
		if q not in ["inf", 2]:
			raise NotImplementedError("q = {} not implemented yet".format(q))
		self.q = q
		if self.q == 1:
			self.q_dual = "inf"
		elif self.q == 'inf':
			self.q_dual = 1
		else:
			self.q_dual = self.q / (self.q - 1)
		if not 0. <= alpha <= 1.:
			raise ValueError("alpha must be between 0 and 1")
		self.alpha = alpha
		if weights is not None:
			self.weights = np.array(weights)
		else:
			self.weights = np.ones(1)
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
		# TODO use weights
		return np.apply_along_axis(
			lambda x_col: MTSGL.proximal.proximal_sgl(x_col, tau, self.q, self.alpha),
			0,
			x
		)

	def max_lam(self, loss: MTLoss) -> float:
		# TODO check calculations, this seems to be only for GroupLasso(q)
		# see "A note on the group lasso and a sparse group lasso", section 3.
		grad0 = loss.gradient()
		norms = np.apply_along_axis(np.linalg.norm, 1, grad0, self.q_dual)
		return max(norms / self.weights)


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
