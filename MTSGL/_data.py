import numpy as np
import pandas as pd
from typing import Union, Sequence, Optional, Any, Dict


class Data:
	"""
	A dataset.

	Attributes
	----------
	x : dict of ndarray, ndarray
		Either a dictionary of ndarrays containing the features in each task
		or a ndarray containing the common features is x_same is True.
	y : dict of ndarray
		The responses.
	w : dict of ndarray
		The weights.
	x_same : bool
		Whether the features are shared or not.
	n_obs : dict of int
		The number of observations per task.
	n_features : int
		The number of features.
	features : list of str
		The feature name ordered.
	n_tasks : int
		The number of tasks.
	tasks : list
		The name of the tasks.
	x_mean : ndarray(n_tasks, n_features)
		The mean of each feature per task.
	x_stdev : ndarray(n_tasks, n_features)
		The standard deviation of each feature per task.
	name : str
		The type of dataset (Regression or Classification).
	"""

	def __init__(
			self,
			x: Union[Dict[Any, np.ndarray], np.ndarray],
			y: Dict[Any, np.ndarray],
			w: Optional[Dict[Any, np.ndarray]] = None,
			x_same: bool = False,
			standardize: bool = True
		):
		"""
		Initialize the dataset.

		Parameters
		----------
		x : dict of ndarray, ndarray
			Either a dictionary of ndarrays containing the features in each task
			or a ndarray containing the common features is x_same is True.
		y : dict of ndarray
			The responses.
		w : dict of ndarray
			The weights.
		x_same : bool
			Whether the features are shared or not.
		standardize : bool
			Whether to standardize the features or not.
		"""
		self.name = "Loss"
		self.x = x
		self.y = y
		self.w = w
		self.x_same = x_same
		self._check_data()
		self._check_features(standardize)

	def _check_data(self):
		"""
		Performs type, value and dimensions checks during initialization.

		Notes
		-----
		We do note check that the tasks have the same number of features since we fill those afterwards in
		_prepare_features.

		"""
		tasks = []
		n_obs = {}
		# Check y
		for task, yk in self.y.items():
			tasks.append(task)
			yksize = yk.shape
			if not len(yksize) == 2:
				raise ValueError(
					"each element of y should be a 2D ndarray: error for task {} with size {}".format(task, yksize)
				)
			if not yksize[1] == 1:
				raise ValueError(
					"each element of y should be have only 1 column: error for task {} with size {}"
					.format(task, yksize)
				)
			n_obs[task] = yksize[0]
		# Check x
		features = set()
		if not self.x_same:
			for task, xk in self.x.items():
				if task not in tasks:
					raise ValueError("tasks in y should match those in x: {} not found in {}".format(task, tasks))
				xksize = xk.shape
				if not len(xksize) == 2:
					raise ValueError(
						"each element of x should be a 1D ndarray: error for task {} with size {}".format(task, xksize)
					)
				if not n_obs[task] == xksize[0]:
					raise ValueError(
						"y should have the same number of observations as x in task {}: received {} but expected {}"
							.format(task, xksize[0], n_obs[task])
					)
				if xk.dtype.names is None:
					xk = np.core.records.fromarrays(
						xk.transpose(),
						names=["X" + str(i) for i in range(xksize[1])],
						formats=[np.float32 for _ in range(xksize[1])]
					)
					self.x[task] = xk
				features.update(xk.dtype.names)
		else:
			xk = self.x
			xksize = xk.shape
			if not len(xksize) == 2:
				raise ValueError(
					"x should be a 1D ndarray:received size {}".format(xksize)
				)
			if not all([n == xksize[0] for n in n_obs.values()]):
				raise ValueError(
					"y should have the same number of observations as x : received {} but expected {}".format(
						xksize[0], n_obs.values()
					))
			if xk.dtype.names is None:
				self.x = np.core.records.fromarrays(
					xk.transpose(),
					names=["X" + str(i) for i in range(xksize[1])],
					formats=[np.float32 for _ in range(xksize[1])]
				)
			features.update(self.x.dtype.names)
		# Check w
		if self.w is not None:
			for task, wk in self.w.items():
				if task not in tasks:
					raise ValueError("tasks in w should match those in x: {} not found in {}".format(task, tasks))
				wksize = wk.shape
				if not len(wksize) == 2:
					raise ValueError(
						"each element of w should be a 2D ndarray: error for task {} with size {}".format(task, wksize)
					)
				if not wksize[1] == 1:
					raise ValueError(
						"each element of w should be have only 1 column: error for task {} with size {}"
						.format(task, wksize)
					)
				if not n_obs[task] == wksize[0]:
					raise ValueError(
						"w should have the same number of observations as x in task {}: received {} but expected {}".format(
							task, wksize[0], n_obs[task]
						))
				if not np.all(wk >= 0.):
					raise ValueError("w should contains only non-negative values : error in task {}".format(task))
				if sum(wk) <= 0.:
					raise ValueError("w should have positive sum : error in task {}".format(task))
				self.w[task] = wk / sum(wk)
		else:
			self.w = {task: np.ones_like(self.y[task]) / n_obs[task] for task in tasks}
		# store dimensions
		self.n_obs = n_obs
		self.n_tasks = len(tasks)
		self.tasks = tasks
		self.features = sorted(features)
		self.n_features = len(features)

	def summarize(self):
		out = ""
		out += "MTSGL " + self.name + " dataset\n"
		out += "Tasks (Nb. Observations):\n".format(self.n_tasks)
		for task, nk in self.n_obs.items():
			out += "    {} ({})\n".format(task, nk)
		out += "Features ({}):\n".format(self.n_features)
		features_str = ", ".join(self.features)
		out += "    " + features_str
		return out

	def __str__(self):
		return self.summarize()

	def _check_features(self, standardize: bool = True):
		"""
		Performs some preparation of the features.

		Parameters
		----------
		standardize : bool
			Whether to standardize the features or not.

		Notes
		-----
		If a task is missing a feature, we fill it with 0s. The ordering of features is changed to follow that in
		features and the names are dropped from here on.

		"""
		x_mean = np.zeros((self.n_tasks, self.n_features))
		x_stdev = np.zeros((self.n_tasks, self.n_features))

		if not self.x_same:
			for k, task in enumerate(self.tasks):
				# fill in missing features
				# re-order following features and drop names
				xk = self._get_x(task)
				xtmp = np.zeros((self.n_obs[task], self.n_features))
				for j, feat in enumerate(self.features):
					if feat in xk.dtype.names:
						xtmp[:, j] = xk[[feat]]
				# store mean and stdev
				xm = np.nanmean(xtmp, axis=0)
				xsd = np.maximum(np.nanstd(xtmp, axis=0, ddof=0), 1e-16)
				x_mean[k, :] = xm
				x_stdev[k, :] = xsd
				if standardize:
					xtmp = xtmp - xm
					xtmp = xtmp / xsd
				# replace nan by 0
				xtmp = np.nan_to_num(xtmp, nan=0.)
				self.x[task] = xtmp
		else:
			xk = self.x
			xtmp = np.zeros((self.n_obs[self.tasks[0]], self.n_features))
			for j, feat in enumerate(self.features):
				if feat in xk.dtype.names:
					xtmp[:, j] = xk[[feat]]
			# store mean and stdev
			xm = np.nanmean(xtmp, axis=0)
			xsd = np.maximum(np.nanstd(xtmp, axis=0, ddof=0), 1e-16)
			for k in range(len(self.tasks)):
				x_mean[k, :] = xm
				x_stdev[k, :] = xsd
			if standardize:
				xtmp = xtmp - xm
				xtmp = xtmp / xsd
			# replace nan by 0
			xtmp = np.nan_to_num(xtmp, nan=0.)
			self.x = xtmp
		self.x_mean = x_mean
		self.x_stdev = x_stdev

	def _get_x(self, task):
		if self.x_same:
			return self.x
		else:
			return self.x[task]


class RegressionData(Data):
	"""
	A Regression dataset.
	"""

	def __init__(
			self,
			x: Union[Dict[Any, np.ndarray], np.ndarray],
			y: Dict[Any, np.ndarray],
			w: Dict[Any, np.ndarray] = None,
			x_same: bool = False,
			standardize: bool = True
		):
		super().__init__(x, y, w, x_same, standardize)
		self.name = "Regression"


class ClassificationData(Data):
	"""
	A (Binary) Classification dataset.

	Notes
	-----
	Class membership is encoded as 0/1.
	"""

	def __init__(
			self,
			x: Union[Dict[Any, np.ndarray], np.ndarray],
			y: Dict[Any, np.ndarray],
			w: Dict[Any, np.ndarray] = None,
			x_same: bool = False,
			standardize: bool = True
		):
		super().__init__(x, y, w, x_same, standardize)
		self.name = "Classification"
# TODO check that y is encoded as 0/1.


def _longdf_to_dict(
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
