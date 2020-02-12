import numpy as np
import MTSGL

n = 10
p = 3
x = np.random.normal(0,1,(n,p))
beta = np.random.normal(0,1,(p,1))
y = np.matmul(x, beta) + np.random.normal(0,1,(n,1))
w = np.random.uniform(0,1,(n,1))
w = w / sum(w)

b = np.random.normal(0,1,(p,1))

Loss = MTSGL._losses.LS()

print(Loss._gradient(x, y, b))

Loss = MTSGL._losses.WLS()

print(Loss._gradient(x, y, w, b))

print(Loss._predict(x, b))