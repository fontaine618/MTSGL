from MTSGL import Data
from MTSGL.losses import Loss
from MTSGL.regularizations import Regularization
from .ADMM import ADMM


class ConsensusADMM(ADMM):

	def __init__(
			self,
			data: Data,
			loss: Loss,
			reg: Regularization,
			lam: float,
			**kwargs
	):
		super().__init__(data, loss, reg, lam, **kwargs)
		if "threshold_decrease_ridge" not in kwargs.keys():
			self.threshold_decrease_ridge = 1.0e-1
		else:
			self.threshold_decrease_ridge = float(kwargs["threshold_decrease_ridge"])
			if not 1.0e-3 <= self.threshold_decrease_ridge < 1.:
				raise ValueError("threshold_decrease_ridge must be between 1.0e-1 and 1")
