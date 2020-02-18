from MTSGL.data.Data import Data
from MTSGL.losses import SeparableMTLoss
from MTSGL.regularizations import Regularization
from .ADMM import ADMM


class ConsensusADMM(ADMM):

	def __init__(
			self,
			data: Data,
			loss: SeparableMTLoss,
			reg: Regularization,
			**kwargs
	):
		self.threshold_ridge_decrease = None
		super().__init__(data, loss, reg, **kwargs)

	def _set_additional_options(self, **kwargs):
		if "threshold_ridge_decrease" not in kwargs.keys():
			self.threshold_ridge_decrease = 1.0e-1
		else:
			self.threshold_ridge_decrease = float(kwargs["threshold_ridge_decrease"])
			if self.threshold_ridge_decrease < 1.0e-3:
				raise ValueError("threshold_ridge_decrease must be above 1.0e-3")
