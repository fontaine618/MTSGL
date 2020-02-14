import numpy as np
from typing import Union
import MTSGL.proximal._projections


def proximal_sgl(
		x: np.ndarray,
		tau: float,
		q: Union[str, int],
		alpha: float
):
	"""
	Proximal operator on the Lq norm.

	Parameters
	----------
	x : ndarray
		The vector at which the proximal operator is evaluated.
	tau : float
		The multiplicative factor for the penalty term.
	q : int, str
		The type of norm used ('inf': L_infty, 2: L_2)
	alpha : float
		The mixing of penalties (0 = group Lasso, 1 = Lasso)

	Returns
	-------
	prox : ndarray
		The proximal value.

	Notes
	-----
	Computes
	.. math::
		prox_{\tau \Vert\cdot\Vert_q}(v) = \text{argmin}_{x} \tau P_{q,\alpha}(x)
		+ \frac{1}{2}\Vert x-v\Vert_2^2
	where
	.. math::
		P_{q,\alpha}(x) = (1-\alpha)\Vert x\Vert_q + \alpha \Vert x\Vert_1
	"""
	if tau < 0:
		raise ValueError("tau must be non-negative")
	if not (0. <= alpha <= 1.):
		raise ValueError("alpha must be in [0,1]")
	if q in ['inf', 2]:
		if alpha == 0.:  # group Lasso, skip soft-thresholding
			return _proximal_lq(x, tau, q)
		elif alpha == 1.:  # Lasso, skip prox_Lq
			return _proximal_lq(x, tau, 1)
		else:  # SGL
			return _proximal_lq(_proximal_lq(x, alpha * tau, 1), (1 - alpha) * tau, q)
	else:
		raise ValueError("q must be in ['inf', 2]")


def _proximal_lq(
		x: np.ndarray,
		tau: float,
		q: Union[str, int]
):
	"""
	Proximal operator on the Lq norm.

	Parameters
	----------
	x : ndarray
		The vector at which the proximal operator is evaluated.
	tau : float
		The multiplicative factor for the penalty term.
	q : int
		The type of norm used ('inf': L_infty, 1: L_1, 2: L_2)

	Returns
	-------
	prox : ndarray
		The proximal value.

	Notes
	-----
	Computes
	.. math::
		prox_{\tau \Vert\cdot\Vert_q}(v) = \text{argmin}_{x} \tau \Vert x\Vert_q
		+ \frac{1}{2}\Vert x-v\Vert_2^2
	"""
	if q == 1:
		return np.sign(x) * np.maximum(np.abs(x) - tau, 0.)
	elif q == 2:
		norm = np.linalg.norm(x, 2)
		return np.maximum(norm - tau, 0.) * x / norm if norm > 0. else x*0.
	elif q == "inf":
		return x - MTSGL.proximal._projections._l1_projection(x, tau)
	else:
		raise ValueError("q must be in ('inf': L_infty, 1: L_1, 2: L_2)")

