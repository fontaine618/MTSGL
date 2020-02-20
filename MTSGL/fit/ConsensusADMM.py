from MTSGL.losses import SeparableMTLoss
from MTSGL.regularizations import Regularization
from .Fit import Fit


class ConsensusADMM(Fit):

	def __init__(
			self,
			loss: SeparableMTLoss,
			reg: Regularization,
			**kwargs
	):
		self.threshold_ridge_decrease = None
		super().__init__(loss, reg, **kwargs)
		self._solution_path()

	def _set_additional_options(self, **kwargs):
		pass
