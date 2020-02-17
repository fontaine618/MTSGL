import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Sequence

from MTSGL.data import Data, MultivariateData, MultiTaskData


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
		print(cols)
		print(y_cols)
		print(x_cols)
		print(w_cols)
		if len(cols) < len(x_cols) + len(y_cols) + len(w_cols):
			raise ValueError("Some column indeces were repeated.")
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
			raise ValueError("Some column indeces were repeated.")
		if len(cols - df_cols) > 0:
			raise ValueError("Could not match columns {} to the columns of df".format(cols - df_cols))
		#  instantiate
		return MultiTaskData(df, y_col, task_col, w_col, x_cols, standardize)

	else:
		raise TypeError("y_cols must be a str or a non-empty list")


"""
def _longdf_to_dict(
		df: pd.DataFrame,
		y_cols: Union[str, Sequence[str]],
		task_col: Optional[str] = None,
		w_col: Optional[str] = None,
		x_cols: Optional[Sequence[str]] = None
) -> Dict[str, Union[bool, Dict, np.ndarray]]:
	
	Transforms a data frame into the appropriate dict structure for MTSGL Datasets.

	Parameters
	----------
	df : pandas.DataFrame
		The data frame.
	y_cols : int(s)
		The index of the response(s).
	task_col : int
		The index of the task identifier. Overlooked if len(y_col)>1. Can be None to imply multiple tasks.
	w_col : int
		The index of the observation weights. If None, equal weights are used.
	x_cols
		The index of the features. If None, then all remaining columns are selected.

	Notes
	-----
	If y_col has more than one entry, we use those to define the tasks and set x_same=True.

	Returns
	-------
	x : dict of ndarray, ndarray
		Either a dictionary of ndarrays containing the features in each task
		or a ndarray containing the common features is x_same is True.
	y : dict of ndarray
		The responses.
	w : dict of ndarray
		The weights.
	x_same : bool
		Whether the features are shared or not.
	
	colnames = []
	if not isinstance(df, pd.DataFrame):
		raise TypeError("df should be a pandas Dataframe")
	if not isinstance(y_cols, (str, list)):
		raise TypeError("y_cols should be a str or a list")
	shared = False
	if isinstance(y_cols, list):
		if not all([isinstance(y, str) for y in y_cols]):
			raise TypeError("if y_cols is a list, all entries should be str")
		shared = True
	if task_col is None:
		shared = True
	else:
		if not isinstance(task_col, str):
			raise TypeError("task_col should be a string")
		colnames.append(task_col)
	if shared and not isinstance(y_cols, list):
		# case of single task
		y_cols = [y_cols]
	if shared:
		colnames.extend(y_cols)
	else:
		colnames.append(y_cols)
	if w_col is not None:
		if not isinstance(w_col, str):
			raise TypeError("w_col should be a string")
		colnames.append(w_col)
	if x_cols is not None:
		if not isinstance(x_cols, list):
			raise TypeError("x_cols should be a list")
		if not all([isinstance(x, str) for x in x_cols]):
			raise TypeError("all entries of x_cols should be str")
		colnames.extend(x_cols)
	diff = set(colnames) - set(df.columns)
	if len(diff) > 0:
		raise ValueError("could not match {} from the columns of df".format(diff))
	if len(colnames) > len(set(colnames)):
		raise ValueError("repeated column names provided")
	if x_cols is None:
		x_cols = list(set(df.columns) - set(colnames))
	# construct data
	x = {}
	y = {}
	w = None
	if w_col is not None:
		w = {}
	if shared:
		x = df[x_cols].to_records(index=False)
		tasks = set(y_cols)
		for task in tasks:
			y[task] = df[task].to_numpy().reshape(-1, 1)
			if w_col is not None:
				w[task] = df[w_col].to_numpy().reshape(-1, 1)
	else:
		tasks = set(df[task_col])
		for task in tasks:
			df_task = df.loc[df[task_col] == task]
			y[task] = df_task[y_cols].to_numpy().reshape(-1, 1)
			x[task] = df_task[x_cols].to_records(index=False)
			if w_col is not None:
				w[task] = df_task[w_col].to_numpy().reshape(-1, 1)
	data = {"x": x, "y": y}
	if w_col is not None:
		data["w"] = w
	data["x_same"] = shared
	return data
"""