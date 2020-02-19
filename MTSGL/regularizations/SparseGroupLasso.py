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
			self.weights = np.array(weights).reshape(-1)
		else:
			self.weights = None
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

		Notes
		-----
		The proximal operator is performed row-wise (fixed feature j=1..p)


		"""
		p, K = x.shape
		if self.weights is None:
			w = np.ones(p)
		else:
			w = self.weights
		return np.row_stack([
			MTSGL.proximal.proximal_sgl(x[j, :], tau * w[j], self.q, self.alpha)
			for j in range(p)
		])

	def max_lam(self, loss: MTLoss) -> float:
		grad0 = loss.gradient()
		p, K = grad0.shape
		if self.weights is None:
			self.weights = np.ones(p)
		norms = np.apply_along_axis(np.linalg.norm, 1, grad0, self.q_dual)
		denum = self.alpha * np.power(loss.data.n_tasks, 1.0/self.q_dual) + (1-self.alpha)
		return max(norms / self.weights) / denum


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
