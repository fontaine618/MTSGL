import MTSGL
import numpy as np
import pandas as pd

def _ridge_gd(loss, beta0, v, tau, options={}, **kwargs):
	"""
	Optimizes a loss function with ridge regularization.

	Parameters
	----------
	loss : Loss
		The loss function, which must implement _loss and _gradient.
	beta0 : ndarray
		The initial value.
	v : ndarray
		The vector in the regularizer.
	tau : float
		The regularization multiplier.
	options : dict
		Furhter options to control the optimization.
	**kwargs :
		Further arguments to pass to loss.

	Notes
	-----
	The objective function is as follows:

	.. math::

		L(\beta) + \frac{1}{2\tau}\Vert\beta - V\Vert_2^2

	where :math:`\beta, V\in\mathbb{R}^d` and where the gradient of L w.r.t. :math:`\beta`
	is readily available. We proceed using Gradient descent.

	Returns
	-------
	beta : ndarray
		The solution.

	"""
	if not isinstance(loss, MTSGL._losses.Loss):
		raise TypeError("loss must be a MTSGL Loss")
	if "loss" not in dir(loss):
		raise AttributeError("loss does not implement loss")
	if "gradient" not in dir(loss):
		raise AttributeError("loss does not implement gradient")
	if "hessian_upper_bound" not in dir(loss):
		raise AttributeError("loss does not have attribute hessian_upper_bound")
	if not isinstance(beta0, np.ndarray):
		raise TypeError("beta0 must be a numpy array")
	d = beta0.size
	if not isinstance(v, np.ndarray):
		raise TypeError("v must be a numpy array")
	if not v.size == d:
		raise ValueError("the dimensions of beta0 and v must agree")
	if not isinstance(tau, float):
		raise TypeError("tau must be a float")
	if tau < 0.:
		raise ValueError("tau must be non-negative")
	try:
		loss.loss(beta0, **kwargs)
		loss.gradient(beta0, **kwargs)
	except TypeError:
		raise TypeError("could not evaluate loss or gradient using beta0 and {}".format(kwargs.keys))
	# options
	if not isinstance(options, dict):
		raise TypeError("options must be a dict")
	if "threshold" not in options:
		options["threshold"] = 1.0e-6
	else:
		if not isinstance(options["threshold"], float):
			raise TypeError("option threshold must be a float")
		if options["threshold"] < 1.0e-16:
			raise ValueError("option threshold must be above 1e-16")
	if "max_iter" not in options:
		options["max_iter"] = 1000
	else:
		if not isinstance(options["max_iter"], int):
			raise TypeError("option max_iter must be an int")
		if options["max_iter"] > 10000 or options["max_iter"] < 1:
			raise ValueError("option max_iter must be between 1 and 1e5")
	if "verbose" not in options:
		options["verbose"] = True
	else:
		if not isinstance(options["verbose"], bool):
			raise TypeError("option verbose must be a bool")

	if options["verbose"]:
		print("="*(13+4*12))
		print("| {:<12} | {:>12} | {:>12} | {:>12} |".format("Iteration", "Obj. value", "Loss", "Step norm"))
		print("="*(13+4*12))
	# first iteration
	t = 0
	beta = beta0
	lossval = loss.loss(beta, **kwargs)
	penalty = (np.linalg.norm(beta - v, 2)**2) / (2.*tau)
	obj = lossval + penalty
	step = np.linalg.norm(beta - beta0, 2)
	if options["verbose"]:
		print("| {:<12} | {:>12.6f} | {:>12.6f} | {:>12} |".format(t, obj, lossval, ""))
	while True:
		t+=1
		grad = loss.gradient(beta, **kwargs) + (beta - v) / tau
		beta_prev = beta
		beta = beta - grad * 1./(loss.hessian_upper_bound + 1./tau)
		lossval = loss.loss(beta, **kwargs)
		penalty = (np.linalg.norm(beta - v, 2)**2) / (2.*tau)
		obj = lossval + penalty
		step = np.linalg.norm(beta - beta_prev, 2)
		if options["verbose"]:
			print("| {:<12} | {:>12.6f} | {:>12.6f} | {:>12.6f} |".format(t, obj, lossval, step))
		if t >= options["max_iter"]:
			raise RuntimeError("ridge_gd did not converge in {} iterations".format(t))
		if step < options["threshold"]:
			break
	if options["verbose"]:
		print("="*(13+4*12))
	return beta






