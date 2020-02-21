import unittest
import MTSGL.regularizations
import numpy as np
import MTSGL.proximal


class TestSparseGroupLasso(unittest.TestCase):

	def test_sparse_group_lasso(self):
		reg = MTSGL.regularizations.SparseGroupLasso(2, 0.5)
		self.assertEqual(
			str(reg),
			"SparseGroupLasso(q = 2, alpha = 0.5)",
			"Failed to create name for SparseGroupLasso"
		)

	def test_group_lasso(self):
		reg = MTSGL.regularizations.GroupLasso('inf')
		self.assertEqual(
			str(reg),
			"GroupLasso(q = inf)",
			"Failed to create name for GroupLasso"
		)

	def test_lasso(self):
		reg = MTSGL.regularizations.Lasso()
		self.assertEqual(
			str(reg),
			"Lasso",
			"Failed to create name for Lasso"
		)

	def test_proximal(self):
		reg = MTSGL.regularizations.SparseGroupLasso('inf', 0.5, [1., 2., 3.])
		x = np.arange(-2, 4).reshape(3, 2)
		prox = reg.proximal(x, 0.2)
		sol = np.array([[-1.8, -0.9], [0., 0.6], [1.7, 2.4]])
		try:
			np.testing.assert_array_almost_equal(prox, sol)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "Failed to produce the correct proximal.")

	def test_proximal_no_weights(self):
		reg = MTSGL.regularizations.SparseGroupLasso('inf', 0.5)
		x = np.arange(-2, 4).reshape(3, 2)
		prox = reg.proximal(x, 0.2)
		sol = np.array([[-1.8, -0.9], [0., 0.8], [1.9, 2.8]])
		try:
			np.testing.assert_array_almost_equal(prox, sol)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "Failed to produce the correct proximal.")

	def test_sgl_value(self):
		reg = MTSGL.regularizations.SparseGroupLasso('inf', 0.5)
		beta = np.array([0, 1, 2, 0, 1, 2]).reshape((3, 2))
		value = reg.value(beta)
		self.assertEqual(value, 5.5)


if __name__ == '__main__':
	unittest.main()
