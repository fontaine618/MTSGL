from losses import MTLoss
import numpy as np
from itertools import chain
from typing import Any, Dict, Tuple


def dict_zip(
		*dicts: Dict[Any, Any],
		default: Any = None
) -> Dict[Any, Tuple[Any]]:
	return {key: tuple(d.get(key, default) for d in dicts) for key in set(chain(*dicts))}


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
	print(step_size)
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
