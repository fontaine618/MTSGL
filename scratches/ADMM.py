import numpy as np
import pandas as pd
import MTSGL

n = 100
p = 5
tasks = ["0", "1", "2"]
df = pd.DataFrame(data={
	"y": np.random.normal(0, 1, n),
	"w": np.random.uniform(0, 1, n),
	"task": np.random.choice(tasks, n)
})
for i in range(p):
	df["var" + str(i + 1)] = np.random.normal(i, i+1, n)

data = MTSGL.data.utils.df_to_data(
	df=df,
	y_cols="y",
	task_col="task",
	w_cols="w",
	x_cols=["var" + str(i + 1) for i in range(p)]
)

loss = MTSGL.losses.MTWLS(data)

reg = MTSGL.regularizations.SparseGroupLasso(q=2, alpha=0.5, weights=np.random.uniform(p))

model = MTSGL.ADMM.ConsensusADMM(data, loss, reg, n_lam=10, lam_frac=0.01)

for l in model.lam:
	print(l)