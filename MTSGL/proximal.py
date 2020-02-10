import numpy as np

def l1_projection(y, r, alg = "Condat"):
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

	Notes
	-----

	References
	----------

	Examples
	--------


	"""
	if not isinstance(y, np.ndarray):
		raise TypeError("v must be a numpy array.")
	if not isinstance(r, (int, float)):
		raise TypeError("r must be numerical (int or float)")
	if r < 0:
		raise ValueError("r must be non-negative")
	elif r == 0:
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
		xp = simplex_projection_condat(yp, r)
	elif alg == "Sort":
		xp = simplex_projection_sort(yp, r)
	# simplex to L1 ball
	x = xp * sgn
	return x

def simplex_projection_condat(yp, r):
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
	observed runtime and O(n^2) worst-case runtime.


	References
	----------
	.. [1] Condat, L. "Fast Projection onto the Simplex and the l1 Ball," Math. Program. 158 (2016), no. 1-2, Ser. A,
	575--585.

	Examples
	--------


	"""
	if not isinstance(yp, np.ndarray):
		raise TypeError("v must be a numpy array.")
	if not isinstance(r, (int, float)):
		raise TypeError("r must be numerical (int or float)")
	if r < 0:
		raise ValueError("r must be non-negative")
	elif r == 0:
		return yp*0
	p = yp.size
	if p == 0:
		return yp
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

def simplex_projection_sort(yp, r):
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

	Examples
	--------


	"""
	if not isinstance(yp, np.ndarray):
		raise TypeError("v must be a numpy array.")
	if not isinstance(r, (int, float)):
		raise TypeError("r must be numerical (int or float)")
	if r < 0:
		raise ValueError("r must be non-negative")
	elif r == 0:
		return yp*0
	p = yp.size
	if p == 0:
		return yp
	u = np.sort(yp)[::-1]
	cssv = np.cumsum(u) - r
	ind = np.arange(p) + 1
	cond = u - cssv / ind > 0
	rho = ind[cond][-1]
	theta = cssv[cond][-1] / float(rho)
	xp = np.maximum(yp - theta, 0)
	return xp
