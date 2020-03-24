import numpy as np
import pandas as pd

import MTSGL
import matplotlib.pyplot as plt
import matplotlib

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
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

model_sharing = MTSGL.fit.SharingADMM(
	loss, reg, n_lam=100, lam_frac=0.001, rho=1., verbose=2
)


model_consensus = MTSGL.fit.ConsensusADMM(
	loss, reg, n_lam=100, lam_frac=0.001, rho=1., verbose=2
)

print(model_consensus.path)
print(model_sharing.path)




N = 256
vals = np.ones((N, 4))
vals[:128, 0] = np.linspace(0, 1, N // 2)
vals[:128, 1] = np.linspace(1, 1, N // 2)
vals[:128, 2] = np.linspace(1, 1, N // 2)
vals[128:, 0] = np.linspace(1, 1, N // 2)
vals[128:, 1] = np.linspace(1, 0, N // 2)
vals[128:, 2] = np.linspace(1, 1, N // 2)
newcmp = matplotlib.colors.ListedColormap(vals)

for model in [model_sharing, model_consensus]:
	beta_norm = np.apply_along_axis(lambda x: max(np.abs(x)), 2, model.path)
	beta_flat = model.path.reshape(100, -1)
	plt.figure(figsize=(10, 10))
	plt.imshow(beta_flat, cmap=newcmp, aspect='auto')
	plt.colorbar()
	plt.show()
	plt.figure(figsize=(10, 10))
	plt.imshow(beta_norm, cmap='inferno', aspect='auto')
	plt.colorbar()
	plt.show()

print(model_sharing.log)
print(model_consensus.log)


w = w.reshape((-1,1))
x * w