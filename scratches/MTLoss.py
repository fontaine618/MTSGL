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

loss.gradient(task="0")

loss["0"]

print(loss.loss())
for task in tasks:
	print(loss.loss(task=task))
sum([loss.loss(task=task) for task in tasks])