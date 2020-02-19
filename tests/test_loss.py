import unittest
import numpy as np
import pandas as pd
import MTSGL.losses


class TestLoss(unittest.TestCase):

	def test_wls(self):
		n, p = 100, 3
		x = np.random.normal(0, 1, (n, p))
		y = np.random.normal(0, 1, (n, 1))
		w = np.random.uniform(0, 1, (n, 1))
		w = w / sum(w)
		beta = np.random.normal(0, 1, (p, 1))
		WLS = MTSGL.losses.WLS(x, y, w)
		WLS.ridge_closed_form(1.0, beta)
		WLS.ridge(1.0, beta, np.zeros((p, 1)))


if __name__ == '__main__':
	unittest.main()
