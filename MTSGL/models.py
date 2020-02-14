from typing import Union, Sequence, Optional
import MTSGL.data
import pandas as pd

REGRESSION_LOSSES = ["ls"]
CLASSIFICATION_LOSSES = []


class Model:

	def __init__(
			self,
			df: pd.DataFrame,
			y_cols: Union[str, Sequence[str]],
			task_col: Optional[str] = None,
			w_col: Optional[str] = None,
			x_cols: Optional[Sequence[str]] = None
	):
		self.data_raw = MTSGL.data.longdf_to_dict(
			df,
			y_cols,
			task_col,
			w_col,
			x_cols
		)

	def _drop_raw(self):
		delattr(self, 'data_raw')


class LS(Model):

	def __init__(
			self,
			df: pd.DataFrame,
			y_cols: Union[str, Sequence[str]],
			task_col: Optional[str] = None,
			w_col: Optional[str] = None,
			x_cols: Optional[Sequence[str]] = None,
			standardize: Optional[bool] = True,
			keep_df: Optional[bool] = False
		):
		super().__init__(df, y_cols, task_col, w_col, x_cols)
		self.data = MTSGL.data.RegressionData(**self.data_raw, standardize=standardize)
		if not keep_df:
			self._drop_raw()