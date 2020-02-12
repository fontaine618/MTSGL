import numpy as np
import pandas as pd
import MTSGL

n = 10
p = 3
K = 2
x = {i:np.random.normal(0,i+1,(n,p))+i for i in range(K)}
y = {i:np.random.normal(0,1,(n,1)) for i in range(K)}
w = {i:np.random.uniform(0,1,(n,1)) for i in range(K)}

data = MTSGL._data.RegressionData(x[0], y, x_same=True)
print(data)

data.x
data.y
data.w

data.tasks
data.x[0].dtype.names
data.features

x[0].mean(axis=0)

xk = data.x[0]

features = xk.dtype.names

xk[["X0"]].view(np.float32).reshape(xk.shape + (-1,))

data.x_mean
data.x_stdev


import numpy as np
import pandas as pd
import MTSGL

n = 10
p = 3
K = 2

df = pd.DataFrame(data={
	"y": np.random.normal(0,1,(n*K)),
	"w": np.random.uniform(0,1,(n*K)),
	"task": np.random.choice([0,1,2], n*K)
})

for i in range(p):
	df["var"+str(i+1)] = np.random.normal(0, 1, (n * K))

data_raw = MTSGL._data._longdf_to_dict(df, y_cols=["y","task"])

data = MTSGL._data.RegressionData(**data_raw)
print(data)