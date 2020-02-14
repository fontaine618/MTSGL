import unittest
import numpy as np
import pandas as pd
import MTSGL.models


class TestLS(unittest.TestCase):

	def test_single_y_col(self):
		try:
			n = 100
			p = 5
			df = pd.DataFrame(data={
				"y": np.random.normal(0, 1, n),
				"w": np.random.uniform(0, 1, n),
				"task": np.random.choice([0, 1, 2], n)
			})
			for i in range(p):
				df["var" + str(i + 1)] = np.random.normal(0, 1, n)
			data = MTSGL.models.LS(df, y_cols="y", task_col="task", w_col="w").data
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "failed to construct LS model with multiple x")

	def test_multi_y_col(self):
		try:
			n = 10
			p = 5
			df = pd.DataFrame(data={
				"y1": np.random.normal(0, 1, n),
				"w": np.random.uniform(0, 1, n),
				"y2": np.random.normal(0, 1, n)
			})
			for i in range(p):
				df["var" + str(i + 1)] = np.random.normal(0, 1, n)
			data = MTSGL.models.LS(df, y_cols=["y1", "y2"], w_col="w").data
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "failed to construct LS model with common x")


if __name__ == '__main__':
	unittest.main()
