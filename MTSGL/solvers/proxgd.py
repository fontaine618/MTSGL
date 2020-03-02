from typing import Tuple, Dict
from MTSGL.losses import Loss
from MTSGL.regularizations import Regularization
import numpy as np


def proxgd(
		loss: Loss,
		reg: Regularization,
		beta0: np.ndarray,
		w: Dict[np.ndarray],
		lam: float,
		rho: float,
		**kwargs
) -> Tuple[np.ndarray, int]:
	"""Optimizes a LS loss for some regularization.

	Parameters
	----------
	loss : Loss
		The loss function, only needs access to X and cov_upper_bound.
	reg : Regularization
		The regularization used, which must implement proximal.
	beta0 : ndarray
		The initial value (p, K).
	w : dict of ndarray
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

		\frac{\rho}{2} \sum_{k=1}^{L} \left\Vert X^{(k)} \beta^{(k)} - V^{(k)}\right\Vert
		+ \lambda P_{q,\alpha}(\beta)

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
	# TODO change this to cov_upper_bound and implement it is loss
	step_size = 1. / (loss.hessian_upper_bound() + lam / rho)
	# first iteration
	t = 0
	betat = beta0
	betatm1 = beta0
	while True:
		t += 1
		beta_prev = betat
		# compute gradient
		grad = np.zeros_like(beta0)
		for k, task in enumerate(loss.data.tasks):
			grad[:, [k]] = rho * np.matmul(
				(loss.data.x(task).to_numpy() * loss.data.w(task).to_numpy().reshape((-1, 1))).T,
				np.matmul(loss.data.x(task).to_numpy(), betat[:, [k]]) - w[task].to_numpy().reshape((-1, 1))
			)
		# gradient step
		betat = betat - step_size * grad
		# proximal
		betat = reg.proximal(betat, lam)
		# check convergence
		betatm1 = beta_prev
		step = np.linalg.norm(betat - beta_prev, 2)
		if t >= max_iter:
			raise RuntimeError("ridge did not converge in {} iterations".format(t))
		if step < threshold:
			break
	return betat, t
