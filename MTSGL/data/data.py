import numpy as np
from typing import Union, Optional, Any, Dict


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
	__name : str
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
		self.__type = None
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
				if not len(xksize) <= 2:
					raise ValueError(
						"each element of x should be a 1D or 2D ndarray: error for task {} with size {}".format(task, xksize)
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
			if not len(xksize) <= 2:
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
		out += "MTSGL " + self.__name + " dataset\n"
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
		self.__type = "Regression"


class ClassificationData(Data):
	"""
	A (Binary) Classification dataset.

	Attributes
	----------
	labels: dict[tuple[Any]]
		The encoding of the two classes ordered by (0, 1)

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
		self.__type = "Classification"
		labels = {}
		for task in self.tasks:
			classes = sorted(np.unique(self.y[task]))
			self.y[task] = np.vectorize(lambda y: 0 if y == classes[0] else 1)(self.y[task])
			labels[task] = tuple(classes)
		self.labels = labels
