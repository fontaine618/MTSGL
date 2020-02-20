import unittest
import MTSGL.solvers
import numpy as np


class TestSolvers(unittest.TestCase):

	def test_ridge_wls(self):
		n = 20
		p = 3
		x = np.random.normal(0, 1, (n, p))
		beta = np.random.normal(0, 1, (p, 1))
		y = np.matmul(x, beta) + np.random.normal(0, 1, (n, 1))
		w = np.random.uniform(0., 1., (n, 1))
		v = np.random.normal(0, 0.1, (p, 1))
		tau = 1.
		threshold = 1.0e-8
		loss = MTSGL.losses.WLS(x, y, w)
		beta_gd = loss.ridge(tau, v, method="GD", threshold=threshold)
		beta_nesterov = loss.ridge(tau, v, method="Nesterov", threshold=threshold)
		beta_closed_form = loss.ridge_closed_form(tau, v)
		try:
			np.testing.assert_array_almost_equal(beta_gd, beta_nesterov)
			np.testing.assert_array_almost_equal(beta_closed_form, beta_nesterov)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "closed_form and ridge do not agree")


if __name__ == '__main__':
	unittest.main()
