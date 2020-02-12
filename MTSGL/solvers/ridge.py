import MTSGL.losses
import numpy as np


def ridge_gd(loss: MTSGL.losses.Loss, beta0: np.ndarray, v: np.ndarray, tau: float, **kwargs):
	"""
	Optimizes a loss function with ridge regularization.

	Parameters
	----------
	loss : Loss
		The loss function, which must implement loss and gradient and hessian_upper_bound.
	beta0 : ndarray
		The initial value.
	v : ndarray
		The vector in the regularizer.
	tau : float
		The regularization multiplier.
	kwargs : dict
		Further options to control the optimization.

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
	if not v.size == beta0.size:
		raise ValueError("the dimensions of beta0 and v must agree")
	if tau < 0.:
		raise ValueError("tau must be non-negative")
	try:
		loss.loss(beta0)
		loss.gradient(beta0)
	except TypeError:
		raise TypeError("could not evaluate loss or gradient using beta0")
	# options
	threshold = 1.0e-6 if "threshold" not in kwargs else kwargs["threshold"]
	max_iter = 1000 if "max_iter" not in kwargs else kwargs["max_iter"]
	# first iteration
	t = 0
	beta = beta0
	while True:
		t += 1
		grad = loss.gradient(beta) + (beta - v) / tau
		beta_prev = beta
		beta = beta - grad * 1./(loss.hessian_upper_bound() + 1./tau)
		step = np.linalg.norm(beta - beta_prev, 2)
		if t >= max_iter:
			raise RuntimeError("ridge_gd did not converge in {} iterations".format(t))
		if step < threshold:
			break
	return beta, t
