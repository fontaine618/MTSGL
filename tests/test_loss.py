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
		WLS = MTSGL.losses.WLS(x, y, w)
		self.assertEqual(
			WLS.n,
			n
		)

	def test_mtwls(self):
		n = 100
		p = 5
		tasks = ["0", "1", "2"]
		df = pd.DataFrame(data={
			"y": np.random.normal(0, 1, n),
			"w": np.random.uniform(0, 1, n),
			"task": np.random.choice(tasks, n)
		})
		for i in range(p):
			df["var" + str(i + 1)] = np.random.normal(i, i + 1, n)
		data = MTSGL.data.utils.df_to_data(
			df=df,
			y_cols="y",
			task_col="task",
			w_cols="w",
			x_cols=["var" + str(i + 1) for i in range(p)]
		)
		loss = MTSGL.losses.MTWLS(data)
		self.assertEqual(
			[task for task in loss.keys()],
			tasks
		)


if __name__ == '__main__':
	unittest.main()
