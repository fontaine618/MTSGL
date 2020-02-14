import unittest
import numpy as np
import MTSGL.proximal


class TestProximalSGL(unittest.TestCase):

	def test_proximal_sgl_inf(self):
		# see proximal_sgl.m for MATLAB code to get solutions
		try:
			np.testing.assert_array_almost_equal(
				MTSGL.proximal.proximal_sgl(np.array([1, 2, -6, -8, 0]), 1.5, 'inf', 0.4),
				np.array([0.4, 1.4, -5.4, -6.5, 0.])
			)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "did not yield the correct proximal (q='inf')")

	def test_proximal_sgl_2(self):
		try:
			np.testing.assert_array_almost_equal(
				MTSGL.proximal.proximal_sgl(np.array([1, 2, -6, -8, 0]), 1.5, 2, 0.4),
				np.array([0.36118923, 1.26416229, -4.87605456, -6.68200069, 0.])
			)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "did not yield the correct proximal (q=2)")

	def test_proximal_sgl_gl(self):
		try:
			np.testing.assert_array_almost_equal(
				MTSGL.proximal.proximal_sgl(np.array([1, 2, -6, -8, 0]), 1.5, 'inf', 0.),
				np.array([1., 2., -6., -6.5, 0.])
			)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "did not yield the correct proximal (q=2, group Lasso)")

	def test_proximal_sgl_lasso(self):
		try:
			np.testing.assert_array_almost_equal(
				MTSGL.proximal.proximal_sgl(np.array([1, 2, -6, -8, 0]), 1.5, 2, 1.),
				np.array([0., 0.5, -4.5, -6.5, 0.])
			)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "did not yield the correct proximal (Lasso)")

	def test_proximal_sgl_bad_q(self):
		self.assertRaises(
			ValueError,
			MTSGL.proximal.proximal_sgl,
			np.array([1]), 1.5, 3, 0.4
		)

	def test_proximal_sgl_bad_tau(self):
		self.assertRaises(
			ValueError,
			MTSGL.proximal.proximal_sgl,
			np.array([1]), -1.5, 3, 0.4
		)

	def test_proximal_sgl_bad_alpha(self):
		self.assertRaises(
			ValueError,
			MTSGL.proximal.proximal_sgl,
			np.array([1]), 1.5, 3, 1.4
		)


if __name__ == '__main__':
	unittest.main()
