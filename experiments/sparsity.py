import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

np.random.seed(1)
p = 10
K = 5

# sparse
beta = np.ones((p, K))
sparse = np.random.randint(0, 2, (p, K))
group_sparse = np.random.randint(0, 2, (p, 1))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 4))

ax1.grid(False)
ax1.axis("on")
ax1.imshow(beta * sparse)
ax1.set_title("Sparse\n L1")
ax1.set_ylabel("Feature")
ax1.set_xlabel("Task")

ax2.grid(False)
ax2.imshow(beta * group_sparse)
ax2.set_title("Group Sparse\n L1(Lq)")
ax2.set_xlabel("Task")

ax3.grid(False)
ax3.imshow(beta * sparse * group_sparse)
ax3.set_title("Sparse Group Sparse\n L1+L1(Lq)")
ax3.set_xlabel("Task")

plt.tight_layout()
plt.show()