from MTSGL.regularizations import Regularization
from losses import MTLoss
import numpy as np
from itertools import chain
from typing import Any, Dict, Tuple


def dict_zip(
		*dicts: Tuple[Dict[Any, Any]],
		default: Any = None
) -> Dict[Any, Tuple[Any]]:
	return {key: tuple(d.get(key, default) for d in dicts) for key in set(chain(*dicts))}


def proxgd(
		loss: MTLoss,
		reg: Regularization,
		beta0: np.ndarray,
		v: Dict[str, np.ndarray],
		lam: float,
		rho: float,
		**kwargs
) -> Tuple[np.ndarray, int]:
	"""Optimizes a LS loss for some regularization.

	Parameters
	----------
	loss : MTLoss
		The loss function, only needs access to x, w and cov_upper_bound.
	reg : Regularization
		The regularization used, which must implement proximal.
	beta0 : ndarray
		The initial value (p, K).
	v : dict of ndarray
		The dictionary of working responses in the squared loss (task: n_obs[task]).
	lam : float
		The regularization multiplier.
	rho : float
		The loss multiplier.
	kwargs : dict
		Further options to control the optimization.

	Notes
	-----
	The objective function is as follows:

	.. math::

		\frac{\rho}{2} \sum_{k=1}^{L} \left\Vert X^{(k)} \beta^{(k)} - V^{(k)}\right\Vert + \lambda P_(\beta)


	Returns
	-------
	betat : ndarray
		The solution.
	t : int
		The number of iterations.

	"""
	if lam < 0.:
		raise ValueError("lam must be non-negative")
	if rho < 0.:
		raise ValueError("rho must be non-negative")
	# options
	threshold = 1.0e-6 if "threshold" not in kwargs else kwargs["threshold"]
	max_iter = 1000 if "max_iter" not in kwargs else kwargs["max_iter"]
	# initialize step size to hessian upper bound
	step_size = 1. / (loss.cov_upper_bound() * rho)
	# first iteration
	t = 0
	betat = beta0
	while True:
		t += 1
		beta_prev = betat
		# compute gradient
		grad = np.zeros_like(beta0)
		for k, (task, l) in enumerate(loss.items()):
			grad[:, [k]] = rho * np.matmul(
				(l.x * l.w).T, np.matmul(l.x, betat[:, [k]]) - v[task]
			)
		# gradient step
		betat = betat - step_size * grad
		# proximal
		betat = reg.proximal(betat, lam * step_size)
		# check convergence
		step = np.linalg.norm(betat - beta_prev, 2)
		if t >= max_iter:
			raise RuntimeError("ridge did not converge in {} iterations".format(t))
		if step < threshold:
			break
	return betat, t


def ridge_saturated(
		loss: MTLoss,
		z0: Dict[str, np.ndarray],
		a: Dict[str, np.ndarray],
		tau: float,
		**kwargs
):
	# options
	threshold = 1.0e-6 if "threshold" not in kwargs else kwargs["threshold"]
	max_iter = 1000 if "max_iter" not in kwargs else kwargs["max_iter"]
	# initialize step size to hessian upper bound
	step_size = 1. / (loss.hessian_saturated_upper_bound() + 1. / tau)
	# first iteration
	t = 0
	zt = z0
	ztm1 = z0
	while True:
		t += 1
		z_prev = zt
		yt = {task: zt[task] + (zt[task] - ztm1[task]) * (t - 1) / (t + 2) for task in loss}
		grad = loss.gradient_saturated(yt)
		grad = {task: g + (yt[task] - a[task]) / tau for task, g in grad.items()}
		products = {task: np.matmul(grad[task].transpose(), zt[task] - ztm1[task]) for task in loss}
		if sum(products.values()) > 0.:
			# do a regular gd step
			grad = loss.gradient_saturated(zt)
			grad = {task: g + (zt[task] - a[task]) / tau for task, g in grad.items()}
			zt = {task: zt[task] - grad[task] * step_size for task in loss}
		else:
			# use momentum
			zt = {task: yt[task] - grad[task] * step_size for task in loss}
		ztm1 = z_prev
		step = np.sqrt(sum([np.linalg.norm(zt[task] - z_prev[task], 2) ** 2 for task in loss]))
		if t >= max_iter:
			raise RuntimeError("ridge did not converge in {} iterations".format(t))
		if step < threshold:
			break
	return zt, t
