import unittest
import MTSGL.regularizations
import numpy as np
import MTSGL.proximal


class TestSparseGroupLasso(unittest.TestCase):

	def test_sparse_group_lasso(self):
		reg = MTSGL.regularizations.SparseGroupLasso(5, 2, 0.5)
		self.assertEqual(
			reg.__str__,
			"SparseGroupLasso(q = 2, alpha = 0.5)",
			"Failed to create name for SparseGroupLasso"
		)

	def test_bad_weights(self):
		self.assertRaises(
			ValueError,
			MTSGL.regularizations.SparseGroupLasso,
			5, 2, 0.5, [1, 2, 3, 4]
		)

	def test_group_lasso(self):
		reg = MTSGL.regularizations.GroupLasso(5, 'inf')
		self.assertEqual(
			reg.__str__,
			"GroupLasso(q = inf)",
			"Failed to create name for GroupLasso"
		)

	def test_lasso(self):
		reg = MTSGL.regularizations.Lasso(5)
		self.assertEqual(
			reg.__str__,
			"Lasso",
			"Failed to create name for Lasso"
		)

	def test_proximal(self):
		reg = MTSGL.regularizations.SparseGroupLasso(5, 'inf', 0.5)
		x = np.arange(-2, 4).reshape(3, 2)
		prox = reg.proximal(x, 1.)
		sol = np.array([[-1.25, -0.5], [0., 0.5], [1.25, 2.]])
		try:
			np.testing.assert_array_almost_equal(prox, sol)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "Failed to produce the correct proximal.")


if __name__ == '__main__':
	unittest.main()
