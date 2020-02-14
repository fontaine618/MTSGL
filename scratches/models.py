import numpy as np
import pandas as pd
import MTSGL

n = 1000
p = 3

df = pd.DataFrame(data={
	"y": np.random.normal(0,1,(n)),
	"w": np.random.uniform(0,1,(n)),
	"task": np.random.choice([0,1,2], n)
})

for i in range(p):
	df["var"+str(i+1)] = np.random.normal(0, 1, n)

model = MTSGL.models.LS(df, y_cols="y", task_col="task", w_col="w")

print(model.data)

print(model.data_raw)

