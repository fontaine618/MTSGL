from typing import Tuple
from MTSGL.losses import Loss
import numpy as np


def ridge(
		loss: Loss,
		x0: np.ndarray,
		v: np.ndarray,
		tau: float,
		**kwargs
) -> Tuple[np.ndarray, int]:
	"""Optimizes a loss function with ridge regularizations.

	Parameters
	----------
	loss : Loss
		The loss function, which must implement loss and gradient and hessian_upper_bound.
	x0 : ndarray
		The initial value.
	v : ndarray
		The vector in the regularizer.
	tau : float
		The regularizations multiplier.
	kwargs : dict
		Further options to control the optimization.

	Notes
	-----
	The objective function is as follows:

	.. math::

		L(x) + \frac{1}{2\tau}\Vert x - v\Vert_2^2

	where :math:`x, V\in\mathbb{R}^d` and where the gradient of L w.r.t. :math:`x`
	is readily available.

	In the LS case: Nesterov is almost always the fastest; not much than gd. It is particularly better when p>n.

	Returns
	-------
	xt : ndarray
		The solution.
	t : int
		The number of iterations.

	"""
	if not v.size == x0.size:
		raise ValueError("the dimensions of beta0 and v must agree")
	if tau < 0.:
		raise ValueError("tau must be non-negative")
	try:
		loss.gradient(x0)
	except TypeError:
		raise TypeError("could not evaluate gradient using x0")
	# options
	threshold = 1.0e-6 if "threshold" not in kwargs else kwargs["threshold"]
	max_iter = 1000 if "max_iter" not in kwargs else kwargs["max_iter"]
	# initialize step size to hessian upper bound
	step_size = 1. / (loss.hessian_upper_bound() + 1. / tau)
	# first iteration
	t = 0
	xt = x0
	xtm1 = x0
	while True:
		t += 1
		x_prev = xt
		yt = xt + (xt - xtm1) * (t - 1) / (t + 2)
		grad = loss.gradient(yt) + (yt - v) / tau
		if np.matmul(grad.transpose(), xt - xtm1) > 0.:
			# do a regular gd step
			grad = loss.gradient(xt) + (xt - v) / tau
			xt = xt - grad * step_size
		else:
			# use momentum
			xt = yt - grad * step_size
		xtm1 = x_prev
		step = np.linalg.norm(xt - x_prev, 2)
		if t >= max_iter:
			raise RuntimeError("ridge did not converge in {} iterations".format(t))
		if step < threshold:
			break
	return xt, t
