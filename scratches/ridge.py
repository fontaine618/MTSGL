import numpy as np
import MTSGL
import timeit

n = 100
p = 1000
x = np.random.normal(0,1,(n,p))
beta = np.random.normal(0,1,(p,1))
y = np.matmul(x, beta) + np.random.normal(0,1,(n,1))
beta0 = np.random.normal(0,1,(p,1))
v = np.random.normal(0,0.1,(p,1))
tau = 1.
threshold = 1.0e-6

loss = MTSGL._losses.LS(x, y)

start_time = timeit.default_timer()
beta_gd = loss.ridge_gd(x, y, tau, v)
print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
beta_closed_form = loss.ridge_closed_form(x, y, tau, v)
print(timeit.default_timer() - start_time)

print(np.allclose(
	beta_gd,
	beta_closed_form,
	atol=1.0e-6
))


loss.hessian_upper_bound
loss.hessian_lower_bound
loss.loss(beta0, x, y)
loss.gradient(beta0, x, y)
loss.gradient(beta, x, y)

beta_opt = MTSGL.solvers._ridge._ridge_gd(
	loss=loss,
	beta0=beta0,
	v=v,
	tau=tau,
	x=x,
	y=y
)

np.linalg.eigvalsh(np.matmul(x.transpose(), x))
np.linalg.svd(x)[1]**2