import unittest
import numpy as np
import pandas as pd
import MTSGL.data


class TestDfToData(unittest.TestCase):

	def test_bad_y(self):
		n = 10
		p = 5
		df = pd.DataFrame(data={
			"y": np.random.normal(0, 1, n),
			"w": np.random.uniform(0, 1, n),
			"task": np.random.choice([0, 1, 2], n)
		})
		for i in range(p):
			df["var" + str(i + 1)] = np.random.normal(0, 1, n)
		self.assertRaises(
			TypeError,
			MTSGL.data.utils.df_to_data,
			df=df,
			y_cols=2,
			task_col="task",
			w_cols="w",
			x_cols=["var1", "var2", "var3"]
		)

	def test_bad_col_name(self):
		n = 10
		p = 5
		df = pd.DataFrame(data={
			"y": np.random.normal(0, 1, n),
			"w": np.random.uniform(0, 1, n),
			"task": np.random.choice([0, 1, 2], n)
		})
		for i in range(p):
			df["var" + str(i + 1)] = np.random.normal(0, 1, n)
		self.assertRaises(
			TypeError,
			MTSGL.data.utils.df_to_data,
			df=df,
			y_cols=2,
			task_col="task",
			w_cols="w2",
			x_cols=["var1", "var2", "var3"]
		)

	def test_one_y_no_task(self):
		n = 10
		p = 5
		df = pd.DataFrame(data={
			"y": np.random.normal(0, 1, n),
			"w": np.random.uniform(0, 1, n),
			"task": np.random.choice([0, 1, 2], n)
		})
		for i in range(p):
			df["var" + str(i + 1)] = np.random.normal(0, 1, n)
		data = MTSGL.data.utils.df_to_data(
			df=df,
			y_cols=["y"],
			w_cols="w",
			x_cols=["var1", "var2", "var3"]
		)
		self.assertEqual(data.y("0").shape, (10, ))

	def test_Multivariate_single_w(self):
		n = 10
		p = 5
		df = pd.DataFrame(data={
			"y1": np.random.normal(0, 1, n),
			"w": np.random.uniform(0, 1, n),
			"y2": np.random.choice([0, 1, 2], n),
			"task": np.random.choice([0, 1, 2], n)
		})
		for i in range(p):
			df["var" + str(i + 1)] = np.random.normal(0, 1, n)
		data = MTSGL.data.utils.df_to_data(
			df=df,
			y_cols=["y1", "y2"],
			task_col="task",
			x_cols=["var1", "var2", "var3"]
		)
		data.x("y")
		self.assertEqual(data._w.shape, (10, 1))

	def test_Multivariate_multi_w(self):
		n = 10
		p = 5
		df = pd.DataFrame(data={
			"y1": np.random.normal(0, 1, n),
			"w1": np.random.uniform(0, 1, n),
			"w2": np.random.uniform(0, 1, n),
			"y2": np.random.choice([0, 1, 2], n),
			"task": np.random.choice([0, 1, 2], n)
		})
		for i in range(p):
			df["var" + str(i + 1)] = np.random.normal(i, i, n)
		data = MTSGL.data.utils.df_to_data(
			df=df,
			y_cols=["y1", "y2"],
			w_cols=["w1", "w2"],
			x_cols=["var1", "var2", "var3"]
		)
		self.assertEqual(data._w.shape, (10, 2))

	def test_Multivariate_no_x(self):
		n = 10
		p = 5
		df = pd.DataFrame(data={
			"y1": np.random.normal(0, 1, n),
			"w": np.random.uniform(0, 1, n),
			"y2": np.random.choice([0, 1, 2], n),
			"task": np.random.choice([0, 1, 2], n)
		})
		for i in range(p):
			df["var" + str(i + 1)] = np.random.normal(0, 1, n)
		data = MTSGL.data.utils.df_to_data(
			df=df,
			y_cols=["y1", "y2"],
			task_col="task",
			w_cols="w",
			intercept=False
		)
		self.assertEqual(data._x.shape, (10, 6))

	def test_MultiTask(self):
		n = 10
		p = 30
		df = pd.DataFrame(data={
			"y": np.random.normal(0, 1, n),
			"w": np.random.uniform(0, 1, n),
			"task": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
		})
		for i in range(p):
			df["var" + str(i + 1)] = np.random.normal(i, i+1, n)
		data = MTSGL.data.utils.df_to_data(
			df=df,
			y_cols="y",
			task_col="task",
			w_cols="w",
			x_cols=["var"+str(i+1) for i in range(p)]
		)
		self.assertEqual(data.n_obs, {"0": 4, "1": 3, "2": 3})

	def test_MultiTask_no_x(self):
		n = 10
		p = 5
		df = pd.DataFrame(data={
			"y": np.random.normal(0, 1, n),
			"w": np.random.uniform(0, 1, n),
			"task": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
		})
		for i in range(p):
			df["var" + str(i + 1)] = np.random.normal(0, 1, n)
		data = MTSGL.data.utils.df_to_data(
			df=df,
			y_cols="y",
			task_col="task",
			w_cols="w",
			intercept=False
		)
		self.assertEqual(
			[data._x[task].shape for task in data.tasks],
			[(4, 5), (3, 5), (3, 5)]
		)


if __name__ == '__main__':
	unittest.main()
