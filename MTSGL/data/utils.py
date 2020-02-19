import pandas as pd
from typing import Union, List, Optional

from MTSGL.data.Data import Data
from MTSGL.data.MultivariateData import MultivariateData
from MTSGL.data.MultiTaskData import MultiTaskData


def df_to_data(
		df: pd.DataFrame,
		y_cols: Union[str, List[str]],
		task_col: Optional[str] = None,
		w_cols: Optional[Union[str, List[str]]] = None,
		x_cols: Optional[List[str]] = None,
		standardize: bool = True,
		**kwargs
) -> Data:
	if isinstance(y_cols, str):
		y_cols = [y_cols]
	try:
		n_y = len(y_cols)
	except TypeError:
		raise TypeError("y_cols must be a str or a non-empty list")
	if n_y > 1:
		#  TODO put in docs that we disregard task_col
		#  MultivariateData case
		if w_cols is not None:
			if isinstance(w_cols, str):
				#  common w
				w_cols = [w_cols]
			elif isinstance(w_cols, list):
				#  one w per task
				if not len(w_cols) == n_y:
					raise ValueError("not enough w_cols to match y_cols")
				pass
			else:
				raise TypeError("w_cols must be either a str or a list")
		else:
			# no w provided: set to 1/n
			w_cols = "w"
			while w_cols in df.columns:
				w_cols += "_"
			df[w_cols] = 1.0
			w_cols = [w_cols]
		#  prepare x_cols
		df_cols = set(df.columns)
		if x_cols is None:
			x_cols = sorted(list(df_cols - set(y_cols) - set(w_cols)))
		cols = set(x_cols + y_cols + w_cols)
		if len(cols) < len(x_cols) + len(y_cols) + len(w_cols):
			raise ValueError("Some column indices were repeated.")
		if len(cols - df_cols) > 0:
			raise ValueError("Could not match columns {} to the columns of df".format(cols - df_cols))
		#  instantiate
		return MultivariateData(df, y_cols, w_cols, x_cols, standardize)
	elif n_y == 1:
		#  MultiTaskData case
		y_col = y_cols[0]
		#  prepare task_col
		if task_col is None:
			raise ValueError("task_col must be specified if only one response is provided")
		else:
			if not isinstance(task_col, str):
				raise TypeError("task_col must be a str")
		#  prepare w_col
		if w_cols is not None:
			if not isinstance(task_col, str):
				raise TypeError("w_cols must be a str when y_cols specifies a single column")
			w_col = w_cols
		else:
			# no w provided: set to 1/n
			w_col = "w"
			while w_col in df.columns:
				w_col += "_"
			df[w_col] = 1.0
		#  prepare x_cols
		df_cols = set(df.columns)
		if x_cols is None:
			x_cols = sorted(list(df_cols - {y_col, w_col, task_col}))
		cols = set(x_cols)
		cols.update([y_col, w_col, task_col])
		if len(cols) < len(x_cols + [y_col, w_col, task_col]):
			raise ValueError("Some column indices were repeated.")
		if len(cols - df_cols) > 0:
			raise ValueError("Could not match columns {} to the columns of df".format(cols - df_cols))
		#  instantiate
		return MultiTaskData(df, y_col, task_col, w_col, x_cols, standardize)

	else:
		raise TypeError("y_cols must be a str or a non-empty list")
