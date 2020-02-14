import unittest
import numpy as np
import pandas as pd
import MTSGL.data


class TestLongDfToDict(unittest.TestCase):

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
			data_raw = MTSGL.data.utils.longdf_to_dict(df, y_cols="y", w_col="w", task_col="task")
			self.assertEqual(len(data_raw), 4)
			self.assertFalse(data_raw["x_same"])
			self.assertEqual(len(data_raw["x"]), 3)
			self.assertEqual(len(data_raw["y"]), 3)
			self.assertEqual(len(data_raw["w"]), 3)
			for x, y, w in zip(data_raw["x"].values(), data_raw["y"].values(), data_raw["w"].values()):
				self.assertEqual(x.shape[0], y.shape[0])
				self.assertEqual(y.shape, w.shape)
				self.assertEqual(len(x[0]), p)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "did not produce the correct dict with single column of y")

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
			data_raw = MTSGL.data.utils.longdf_to_dict(df, y_cols=["y1", "y2"], w_col="w")
			self.assertEqual(len(data_raw), 4)
			self.assertTrue(data_raw["x_same"])
			self.assertEqual(len(data_raw["y"]), 2)
			self.assertEqual(len(data_raw["w"]), 2)
			for y, w in zip(data_raw["y"].values(), data_raw["w"].values()):
				self.assertEqual(data_raw["x"].shape[0], y.shape[0])
				self.assertEqual(y.shape, w.shape)
				self.assertEqual(len(data_raw["x"][0]), p)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "did not produce the correct dict with multiple columns of y")


class TestData(unittest.TestCase):

	def test_regression_data_multi_x(self):
		try:
			n = 10
			p = 3
			K = 2
			x = {i: np.random.normal(0, i + 1, (n, p)) + i for i in range(K)}
			y = {i: np.random.normal(0, 1, (n, 1)) for i in range(K)}
			w = {i: np.random.uniform(0, 1, (n, 1)) for i in range(K)}
			data = MTSGL.data.RegressionData(x, y, w, x_same=False)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "failed to construct RegressionData with multiple x")

	def test_regression_data_same_x(self):
		try:
			n = 10
			p = 3
			K = 2
			x = np.random.normal(0, 1, (n, p))
			y = {i: np.random.normal(0, 1, (n, 1)) for i in range(K)}
			w = {i: np.random.uniform(0, 1, (n, 1)) for i in range(K)}
			data = MTSGL.data.RegressionData(x, y, w, x_same=True)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "failed to construct RegressionData with common x")


if __name__ == '__main__':
	unittest.main()
