import unittest
import MTSGL.regularizations


class TestSparseGroupLasso(unittest.TestCase):

	def test_sparse_group_lasso(self):
		reg = MTSGL.regularizations.SparseGroupLasso(2, 0.5)
		self.assertEqual(
			reg.__str__(),
			"SparseGroupLasso(q = 2, alpha = 0.5)",
			"Failed to create name for SparseGroupLasso"
		)

	def test_group_lasso(self):
		reg = MTSGL.regularizations.GroupLasso('inf')
		self.assertEqual(
			reg.__str__(),
			"GroupLasso(q = inf)",
			"Failed to create name for GroupLasso"
		)

	def test_lasso(self):
		reg = MTSGL.regularizations.Lasso()
		self.assertEqual(
			reg.__str__(),
			"Lasso",
			"Failed to create name for Lasso"
		)


if __name__ == '__main__':
	unittest.main()
