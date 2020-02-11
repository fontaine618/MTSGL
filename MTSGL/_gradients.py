import numpy as np

#Deprecated

def _ls(X, y, beta):
	"""
	Computes the gradient of the scaled least squares loss.

	Parameters
	----------
	X : ndarray(n, p)
		The features.
	y : ndarray(n, 1)
		The working responses.
	beta : ndarray(p, 1)
		The coefficeints at which to evaluate the gradient.

	Returns
	-------
	The gradient given by
	.. math::

		X'(X\beta - y) / n
	"""
	return np.matmul(X.transpose(), np.matmul(X, beta) - y) / y.size

def _wls(X, y, w, beta):
	"""
	Computes the gradient of the weighted least squares loss.

	Parameters
	----------
	X : ndarray(n, p)
		The features.
	y : ndarray(n, 1)
		The working responses.
	w : ndarray(n, 1)
		The weights
	beta : ndarray(p, 1)
		The coefficeints at which to evaluate the gradient.

	Returns
	-------
	The gradient given by
	.. math::

		X'W(X\beta - y)
	"""
	return np.matmul(X.transpose(), w*(np.matmul(X, beta) - y))