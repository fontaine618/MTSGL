import numpy as np
import pandas as pd
import MTSGL


n = 30
p = 4

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

beta0 = np.zeros((p+1, 3))
w = data._y

lam = 0.1
rho = 1.0

L = max([l.hessian_upper_bound() for task, l in loss.items()])
step_size = 1. / (L + lam / rho)



betat