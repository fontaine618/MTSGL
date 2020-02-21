import numpy as np
import pandas as pd
import MTSGL

n = 300
p = 5

x = np.random.normal(0, 1, (n, p))
beta = np.random.randint(0, 4, (p, 1))
y = np.matmul(x, beta) + 3. + np.random.normal(0, 1, (n,1))

tasks = ["0", "1", "2"]

df = pd.DataFrame(data={
	"var"+str(i): x[:, i] for i in range(p)
})
df["task"] = np.random.choice(tasks, n)
df["y"] = y
df["w"] = np.random.uniform(0, 1, n)

data = MTSGL.data.utils.df_to_data(
	df=df,
	y_cols="y",
	task_col="task",
	w_cols="w",
	x_cols=["var" + str(i) for i in range(p)]
)

loss = MTSGL.losses.MTWLS(data)

reg = MTSGL.regularizations.SparseGroupLasso(q=2, alpha=1.0, weights=np.random.uniform(0,1,p+1))

model = MTSGL.fit.ConsensusADMM(loss, reg, n_lam=100, lam_frac=0.001, rho=1.5, max_iter=10000, verbose=False)

print(np.apply_along_axis(lambda x: max(np.abs(x)), 2, model.path))
print(3., beta.T)
