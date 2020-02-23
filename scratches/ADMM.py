import numpy as np
import pandas as pd
import MTSGL

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

n = 300
p = 10

x = np.random.normal(0, 1, (n, p))
beta = np.random.randint(-2, 3, (p, 1))
tasks = [0, 1, 2]
task = np.random.choice(tasks, n)
w = np.random.uniform(0, 1, n)
y = np.matmul(x, beta) + task.reshape((-1, 1)) + np.random.normal(0, 1, (n, 1))

df = pd.DataFrame(data={
	"var"+str(i): x[:, i] for i in range(p)
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

weights = np.ones(p+1)
weights[0] = 0.

reg = MTSGL.regularizations.SparseGroupLasso(q=2, alpha=0.5, weights=weights)

model = MTSGL.fit.ConsensusADMM(
	loss, reg, n_lam=100, lam_frac=0.001, rho=1, max_iter=10000, verbose=1
)

print(np.apply_along_axis(lambda x: max(np.abs(x)), 2, model.path).round(1))
print(3., beta.T)

print(model.path[0, :, :])
print(model.path[99, :, :])

self = model

beta = self.path

beta_d = beta / np.array(self.loss.data.x_std_dev)

beta[99, :, :]
beta_d[99, :, :]