import numpy as np
import MTSGL
import timeit

n = 1000
p = 1000
x = np.random.normal(0,1,(n,p))
beta = np.random.normal(0,1,(p,1))
y = np.matmul(x, beta) + np.random.normal(0,1,(n,1))
beta0 = np.random.normal(0,1,(p,1))
v = np.random.normal(0,0.1,(p,1))
tau = 1.
threshold = 1.0e-6

loss = MTSGL.losses.LS(x, y)

start_time = timeit.default_timer()
beta_gd = loss.ridge(tau, v, method="GD")
print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
beta_nesterov = loss.ridge(tau, v, method="Nesterov")
print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
beta_nesterov = loss.ridge(tau, v, method="Nesterov", adaptive_restart=False)
print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
beta_closed_form = loss.ridge_closed_form(tau, v)
print(timeit.default_timer() - start_time)

print(np.allclose(
	beta_closed_form,
	beta_gd,
	atol=1.0e-6
))
print(np.allclose(
	beta_nesterov,
	beta_closed_form,
	atol=1.0e-6
))
