import unittest
import numpy as np
import MTSGL

class test_proximal(unittest.TestCase):

	def test_l1_projection(self):
		self.assertRaises(
			MTSGL.l1_projection([0,0], 1),
			TypeError,
			"does not raise a type exception for v"
		)
		self.assertRaises(
			MTSGL.l1_projection(np.array([1]), "a"),
			TypeError,
			"does not raise a type exception for r"
		)
		self.assertRaises(
			MTSGL.l1_projection(np.array([1]), -1),
			ValueError,
			"does not raise a value exception for r<0")
		self.assertEqual(
			MTSGL.l1_projection(np.array([]), 1),
			np.array([]),
			"does not deal with empty ndarray")
		self.assertEqual(
			MTSGL.l1_projection(np.array([-2]), 1),
			-1.,
			"does not deal with 1D case"
		)

if __name__ == '__main__':
    unittest.main()