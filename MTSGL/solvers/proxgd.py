from typing import Dict, Tuple, Optional
import numpy as np
from losses import MTLoss
from regularizations import Regularization


def proxgd(
		loss: MTLoss,
		reg: Regularization,
		beta0: np.ndarray,
		lam: float,
		rho: float,
		ls: Optional[bool] = False,
		v: Optional[Dict[str, np.ndarray]] = None,
		**kwargs
) -> Tuple[np.ndarray, int]:
	"""Optimizes a loss for some regularization.

	Parameters
	----------
	ls :
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
	if ls and v is None:
		raise ValueError("when ls=True, v must be specified")
	# options
	threshold = 1.0e-6 if "threshold" not in kwargs else kwargs["threshold"]
	max_iter = 1000 if "max_iter" not in kwargs else kwargs["max_iter"]
	# initialize step size to hessian upper bound
	# TODO this should branch on ls=T/F, for now this is fine for LS
	step_size = 1. / (loss.hessian_upper_bound() * rho)
	# first iteration
	t = 0
	betat = beta0
	bt = betat
	thetat = 1.
	# loop until convergence
	while True:
		t += 1
		beta_prev = betat
		# compute gradient
		if ls:
			grad = np.zeros_like(beta0)
			for k, (task, l) in enumerate(loss.items()):
				grad[:, [k]] = rho * np.matmul(l.x.T, l.w * (l.lin_predictor(bt[:, [k]]) - v[task]))
		else:
			# TODO add regular gradient
			pass
		# TODO Adaptive restarts
		# gradient step
		betat = bt - step_size * grad
		# proximal
		betat = reg.proximal(betat, lam * step_size)
		# compute momentum step
		thetatm1 = thetat
		thetat = 0.5 * (1. + np.sqrt(1. + 4. * thetat ** 2))
		bt = betat + (thetatm1 - 1.) * (betat - beta_prev) / thetat
		# check convergence
		step = np.linalg.norm(betat - beta_prev, 2)
		if t >= max_iter:
			raise RuntimeError("proxgd did not converge in {} iterations".format(t))
		if step < threshold:
			break
	return betat, t