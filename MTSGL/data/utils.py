import pandas as pd
from typing import Union, Sequence, Optional


def longdf_to_dict(
		df: pd.DataFrame,
		y_cols: Union[str, Sequence[str]],
		task_col: Optional[str] = None,
		w_col: Optional[str] = None,
		x_cols: Optional[Sequence[str]] = None
):
	"""
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
	"""
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
