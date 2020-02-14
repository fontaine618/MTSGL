import MTSGL
import numpy as np

y = np.array([-5,6,3,9,-8,1,0])
r = 1.5

MTSGL.proximal.l1_projection(y, r, "Sort")
MTSGL.proximal.l2_projection(y, r)

MTSGL.proximal.proximal_sgl(y, 5., 2, 0.4)

N = 100
Ys = np.round(np.random.normal(0,3,(N, 3)),3)

Projs = np.apply_along_axis(MTSGL.proximal.l1_projection, 1, Ys, r=1)

import timeit

timeit.timeit(
	stmt="y = np.random.normal(0,3,1000);MTSGL.proximal.l1_projection(y, 2, 'Sort')",
	number=1000,
	globals={"MTSGL":MTSGL, "np":np}
)


N = 1000
Ys = np.round(np.random.normal(0,3,(N, 2)),3)
Projs = np.apply_along_axis(MTSGL.proximal.proximal_sgl, 1, Ys, tau = 3., q=2, alpha=0.5)



MTSGL.proximal.proximal_sgl(np.array([1, 2, -6, -8, 0]), tau = 1.5, q ='inf', alpha = 0.0)