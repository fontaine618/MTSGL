from . import Regularization
from typing import Union, Optional, List
import numpy as np
import MTSGL.proximal
from MTSGL.losses import MTLoss


class SparseGroupLasso(Regularization):
	"""Sparse group Lasso regularization for Multi-task problems.

	Attributes
	----------
	q: int or flaot or str
		The group norm. Currently, only q=2 ou q='inf' are implemented.
	alpha: float
		The mixing of the L1 penalty and the Lq penalty: alpha=0 defines pure Lq penalty, alpha=1 defines Lasso penalty.

	Methods
	-------
	proximal(x, tau)
		Returns the proximal operator evaluated at x with multiplier tau.
	max_lam(loss)
		Returns the maximum regularization value such that all features are excluded for a given loss.
	"""

	def __init__(
			self,
			q: Union[str, int, float] = 2,
			alpha: float = 0.5,
			weights: Optional[List] = None
	) -> None:
		"""Initilization of a SparseGroupLasso object.

		Produces an instance of Sparse group Lasso regularization.

		Parameters
		----------
		q : int or float or str
			The group norm. Currently, only q=2 ou q='inf' are implemented.
		alpha : float
			The mixing of the L1 penalty and the Lq penalty: alpha=0 defines pure Lq penalty, alpha=1 defines Lasso penalty.
		weights : array-like
			The weight for each feature.
		"""
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
		"""Proximal operator of a Sparse group Lasso penalty.

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
		"""Returns the maximum regularization value such that all features are excluded for a given loss."""
		grad0 = loss.gradient()
		p, K = grad0.shape
		if self.weights is None:
			self.weights = np.ones(p)
		norms = np.apply_along_axis(np.linalg.norm, 1, grad0, self.q_dual)
		denum = self.alpha * np.power(loss.data.n_tasks, 1.0/self.q_dual) + (1-self.alpha)
		vec = np.divide(norms, self.weights, where=self.weights > 0.0)
		return max(vec) / denum

	def value(self, beta: np.ndarray) -> float:
		"""Returns the value of the penalty."""
		p, K = beta.shape
		if self.weights is None:
			self.weights = np.ones(p)
		P_1 = np.apply_along_axis(lambda x: np.linalg.norm(x, 1), 1, beta)
		if self.q == "inf":
			P_q = np.apply_along_axis(lambda x: max(np.abs(x)), 1, beta)
		else:
			P_q = np.apply_along_axis(lambda x: np.power(sum(np.power(np.abs(x), self.q)), 1/self.q), 1, beta)
		return sum(self.weights * (self.alpha * P_1 + (1. - self.alpha) * P_q))


class GroupLasso(SparseGroupLasso):
	"""Group Lasso regularization for Multi-task problems.

	Attributes
	----------
	q: int or flaot or str
		The group norm. Currently, only q=2 ou q='inf' are implemented.
	"""

	def __init__(
			self,
			q: Union[str, int] = 2,
			**kwargs
	):
		super().__init__(q, 0., **kwargs)
		self.name = "GroupLasso"

	def _str_parm(self):
		return "q = {}".format(self.q)


class Lasso(SparseGroupLasso):
	"""Lasso regularization for Multi-task problems."""

	def __init__(
			self,
			**kwargs
	):
		super().__init__(2, 1., **kwargs)
		self.name = "Lasso"

	def _str_parm(self):
		return None
