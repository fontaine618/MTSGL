import numpy as np
import pandas as pd
import MTSGL

n = 12
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

loss.gradient(task="0")
loss.gradient()

print(loss.loss())
for task in tasks:
	print(loss.loss(task=task))
sum([loss.loss(task=task) for task in tasks])

print(data)

for l in loss.items():
	print(l)

z = {task: l.y + 1.0 for task, l in loss.items()}

loss.gradient_saturated(z)
loss.hessian_saturated_upper_bound()


z0 = {task: l.y + 1.0 for task, l in loss.items()}
a = {task: l.y - 1.0 for task, l in loss.items()}
tau = 1.0

MTSGL.fit.ridge_saturated(loss, z0, a, tau)
