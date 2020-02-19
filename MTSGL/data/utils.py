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
		**kwargs
) -> Data:
	"""Transforms a data frame into a Data object.

	Given a data frame and column indices for responses, tasks, weights and features, this function returns a Data
	object of the appropriate type (MultiTaskData or MultivariateData).

	Parameters
	----------
	df: DataFrame
		The data frame.
	y_cols: str or list(str)
		The name of the columns(s) containing the response(s).
	task_col: str or None
		The name of the column identifying the tasks. If None, we assume the tasks are given by y_cols.
	w_cols: str or list(str) or None
		The name of the column(s) containing the observation weights. If None, we assume equal weights
		over all observations (1/n, n=sum(n_k)). Is list(str), it must match the length of y_cols.
	x_cols: list(str) or None:
		The name of the columns containing the features. If none, all remaining columns are selected.
	kwargs
		Further arguments to be passed to Data instantiation.

	Returns
	-------
	data: Data
		The Data object.

	Notes
	-----
	If y_cols is a list of length above 1, then we are in the MultivariateData case and task_col is disregarded.
	If y_cols is a string or a list of length 1, then we are in the MultiTaskData case and task_col defines the
	tasks: if no task_col is provided then a single task is assumed.

	"""
	if isinstance(y_cols, str):
		y_cols = [y_cols]
	try:
		n_y = len(y_cols)
	except TypeError:
		raise TypeError("y_cols must be a str or a non-empty list")
	if n_y > 1:
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
		return MultivariateData(df, y_cols, w_cols, x_cols, **kwargs)
	elif n_y == 1:
		#  MultiTaskData case
		y_col = y_cols[0]
		#  prepare task_col
		if task_col is None: #  single task
			task_col = "task"
			while task_col in df.columns:
				task_col += "_"
			df[task_col] = "0"
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
		return MultiTaskData(df, y_col, task_col, w_col, x_cols, **kwargs)

	else:
		raise TypeError("y_cols must be a str or a non-empty list")
