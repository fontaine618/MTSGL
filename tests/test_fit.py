import unittest
import numpy as np
import pandas as pd
import MTSGL.fit


class TestDfToData(unittest.TestCase):

	@staticmethod
	def create_model():
		np.random.seed(1)
		n = 100
		p = 5
		x = np.random.normal(0, 1, (n, p))
		beta = np.random.randint(-2, 3, (p, 1))
		tasks = [0, 1, 2]
		task = np.random.choice(tasks, n)
		w = np.random.uniform(0, 1, n)
		y = np.matmul(x, beta) + task.reshape((-1, 1)) + np.random.normal(0, 1, (n, 1))
		df = pd.DataFrame(data={
			"var" + str(i): x[:, i] for i in range(p)
		})
		df["task"] = np.array([str(i) for i in task])
		df["y"] = y
		df["w"] = w / sum(w)
		data = MTSGL.data.utils.df_to_data(
			df=df,
			y_cols="y",
			task_col="task",
			w_cols="w",
			x_cols=["var" + str(i) for i in range(p)],
			standardize=False
		)
		loss = MTSGL.losses.MTWLS(data)
		weights = np.ones(p + 1)
		weights[0] = 0.
		reg = MTSGL.regularizations.SparseGroupLasso(q=2, alpha=0.5, weights=weights)
		return loss, reg

	def test_consensus_admm(self):
		loss, reg = self.create_model()
		model = MTSGL.fit.ConsensusADMM(
			loss, reg, n_lam=10, lam_frac=0.001, rho=1, max_iter=10000, verbose=0
		)
		beta_norm = np.apply_along_axis(lambda x: max(np.abs(x)), 2, model.path)
		try:
			np.testing.assert_array_almost_equal(
				beta_norm[9, ],
				np.array([2.01000651, 2.11337167, 1.09623504, 0.22079414, 1.30309887, 0.30546993])
			)
			res = True
		except AssertionError as err:
			res = False
			print(err)
		self.assertTrue(res, "incorrect solution for fixed problem")


if __name__ == '__main__':
	unittest.main()
