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
		self.assertEqual(
			MTSGL.proximal._proximal_sgl(v=np.array([1, 2, -6, -8, 0]), tau=1.5, q=0, alpha=0.4),
			np.array([ 1.,  2., -6., -6.5,  0.]),
			"does not yield the correct proximal"
		)

if __name__ == '__main__':
    unittest.main()