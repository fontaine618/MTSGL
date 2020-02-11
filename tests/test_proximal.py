import unittest
import numpy as np
import MTSGL.proximal

class test_proximal(unittest.TestCase):

	def test_proximal_sgl(self):
		"""

		Notes
		-----
		Matlab code to get solutions:
		cvx_setup
		v = [1,2,-6,-8,0]'
		p,k = size(v)
		alpha = 0.4
		tau = 1.5
		q=inf
		cvx_begin
			variable x(p)
			minimize tau * ((alpha) * norm(x,1) + (1-alpha) * norm(x,q)) + sum(square(x-v)) / 2
		cvx_end
		x
		"""
		try:
			np.testing.assert_array_almost_equal(
				MTSGL.proximal._proximal_sgl(np.array([1, 2, -6, -8, 0]), 1.5, 'inf', 0.4),
				np.array([0.4, 1.4, -5.4, -6.5, 0.])
			)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "did not yield the correct proximal (q='inf')")
		try:
			np.testing.assert_array_almost_equal(
				MTSGL.proximal._proximal_sgl(np.array([1, 2, -6, -8, 0]), 1.5, 2, 0.4),
				np.array([0.36118923, 1.26416229, -4.87605456, -6.68200069, 0.])
			)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "did not yield the correct proximal (q=2)")

if __name__ == '__main__':
    unittest.main()