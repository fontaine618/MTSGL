import numpy as np

def _proximal_sgl(x, tau, q, alpha):
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
	x : ndarray
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
	if not isinstance(x, np.ndarray):
		raise TypeError("v must be a numpy array.")
	if not isinstance(tau, float):
		raise TypeError("tau must be a float")
	if tau < 0:
		raise ValueError("tau must be non-negative")
	if not isinstance(alpha, float):
		raise TypeError("alpha must be a float")
	if not (alpha >= 0. and alpha <=1.):
		raise ValueError("alpha must be in [0,1]")
	if q in ['inf', 2]:
		return _proximal_lq(_proximal_lq(x, alpha*tau, 1), (1-alpha)*tau, q)
	else:
		raise ValueError("q must be in ['inf', 2]")


def _proximal_lq(x, tau, q):
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
	x : ndarray
		The proximal value.

	Notes
	-----
	Computes
	.. math::
		prox_{\tau \Vert\cdot\Vert_q}(v) = \text{argmin}_{x} \tau \Vert x\Vert_q
		+ \frac{1}{2}\Vert x-v\Vert_2^2

	References
	----------

	Examples
	--------


	"""
	# if not isinstance(x, np.ndarray):
	# 	raise TypeError("v must be a numpy array.")
	# if not isinstance(tau, float):
	# 	raise TypeError("tau must be a float")
	# if tau < 0:
	# 	raise ValueError("tau must be non-negative")
	# if not isinstance(q, int):
	# 	raise TypeError("q must be an int")
	if q == 1:
		return np.sign(x) * np.maximum(np.abs(x) - tau, 0.)
	elif q == 2:
		norm = np.linalg.norm(x, 2)
		return np.maximum(norm - tau, 0.) * x / norm if norm > 0. else x*0.
	elif q == "inf":
		return x - _l1_projection(x, tau)
	else:
		raise ValueError("q must be in ('inf': L_infty, 1: L_1, 2: L_2)")




def _l1_projection(y, r, alg = "Sort"):
	"""
	Projection onto the L1 ball.

	Parameters
	----------
	y : nparray
		The vector to project.
	r : float
		The radius of the L1 ball on which to project.
	alg : str
		The algorithm used to project onto the simplex.

	Returns
	-------
	x : ndarray
		The projection.
	"""
	# if not isinstance(y, np.ndarray):
	# 	raise TypeError("y must be a numpy array.")
	# if not isinstance(r, (int, float)):
	# 	raise TypeError("r must be numerical (int or float)")
	# if r < 0:
	# 	raise ValueError("r must be non-negative")
	if r == 0:
		return y*0
	p = y.size
	if p == 0:
		return y
	# map to non-negative orthant
	yp = np.abs(y)
	# check is already in L1 ball
	if sum(yp) <= r:
		return y
	# retrieve signs
	sgn = np.sign(y)
	if alg == "Condat":
		xp = _simplex_projection_condat(yp, r)
	elif alg == "Sort":
		xp = _simplex_projection_sort(yp, r)
	else:
		raise ValueError("alg must be in ['Condat', 'Sort']")
	# simplex to L1 ball
	x = xp * sgn
	return x

def _simplex_projection_condat(yp, r):
	"""
	Projection onto the simplex using Condat's algorithm.

	Parameters
	----------
	yp : nparray
		The vector to project, containing non-negative entries.
	r : float
		The radius of the simplex on which to project.

	Returns
	-------
	xp : ndarray
		The projection.

	Notes
	-----
	The projection is performed using Condats's algorithm described in [1]_ with O(n)
	observed runtime and O(n^2) worst-case runtime. The current implementation is very slow
	compared to the sorting algorithm.


	References
	----------
	.. [1] Condat, L. "Fast Projection onto the Simplex and the l1 Ball," Math. Program. 158 (2016), no. 1-2, Ser. A,
	575--585.
	"""
	p = yp.size
	# step 1
	v = [yp[0]]
	vt = []
	rho = v[0] - r
	vl = 1
	# step 2
	for i in range(1,p):
		ytmp = yp[i]
		rho += (ytmp - rho) / (vl + 1)
		if rho > yp[i] - r:
			v.append(ytmp)
			vl += 1
		else:
			vt.extend(v)
			v = [ytmp]
			vl = 1
			rho = ytmp - r
	# step 3
	for ytmp in vt:
		if ytmp > rho:
			v.append(ytmp)
			vl += 1
			rho += (ytmp - rho) / vl
	# step 4
	change = True
	while change:
		change = False
		for ytmp in v:
			if ytmp <= rho:
				v.remove(ytmp)
				vl -= 1
				rho += (rho - ytmp) / vl
				change = True
	# step 5
	xp = np.array([max(ytmp - rho, 0) for ytmp in yp])
	return xp

def _simplex_projection_sort(yp, r):
	"""
	Projection onto the simplex using sorting.

	Parameters
	----------
	yp : nparray
		The vector to project, containing non-negative entries.
	r : float
		The radius of the simplex on which to project.

	Returns
	-------
	xp : ndarray
		The projection.

	Notes
	-----
	Taken from https://github.com/mblondel/projection_simplex.py

	References
	----------
	.. [1] Blondel, M., Fujino, A. and Ueda, N. (2014) "Large-scale Multiclass Support Vector Machine Training via Euclidean
	Projection onto the Simplex." ICPR 2014. Url: http://www.mblondel.org/publications/mblondel-icpr2014.pdf.
	"""
	p = yp.size
	u = np.sort(yp)[::-1]
	cssv = np.cumsum(u) - r
	ind = np.arange(p) + 1
	cond = u - cssv / ind > 0
	rho = ind[cond][-1]
	theta = cssv[cond][-1] / float(rho)
	xp = np.maximum(yp - theta, 0)
	return xp
