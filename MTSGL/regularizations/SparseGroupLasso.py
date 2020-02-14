from .Regularization import Regularization
from typing import Union


class SparseGroupLasso(Regularization):

	def __init__(
			self,
			q: Union[str, int] = 2,
			alpha: float = 0.5
	):
		super().__init__()
		if q not in ["inf", 2]:
			raise NotImplementedError("q = {} not implemented yet".format(q))
		self.q = q
		if not 0. <= alpha <= 1.:
			raise ValueError("alpha must be between 0 and 1")
		self.alpha = alpha
		self.name = "SparseGroupLasso"

	def _str_parm(self):
		return "q = {}, alpha = {}".format(self.q, self.alpha)


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
