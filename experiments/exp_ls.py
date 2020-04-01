import numpy as np
import pandas as pd
import MTSGL
import matplotlib.pyplot as plt

def gen_data(n=1000, p=100, K=5):
	x = np.random.normal(0, 1, (n, p))
	beta = np.random.randint(-2, 3, (p, K))
	beta[np.random.randint(0, p, p // 2), :] = 0
	tasks = range(K)
	task = np.random.choice(tasks, n)
	w = np.random.uniform(0, 1, n)
	eta = np.choose(task, np.matmul(x, beta).T).reshape((-1, 1))
	y = eta + np.random.normal(0, 1, (n, 1))
	df = pd.DataFrame(data={
		"var" + str(i): x[:, i] for i in range(p)
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
	weights = np.ones(p + 1)
	weights[0] = 0.
	return data, loss, weights, beta

# generate problem
np.random.seed(1)
K = 5
n = 3000
p = 100
data, loss, weights, beta = gen_data(n, p, K)
reg = MTSGL.regularizations.SparseGroupLasso(q=2, alpha=0.5, weights=weights)

# ---------------------------------------------------------
# Solution path
# fit problems
model_proxgd = MTSGL.fit.ProximalGD(
	loss, reg, n_lam=100, lam_frac=0.01, verbose=0
)
model_consensus = MTSGL.fit.ConsensusADMM(
	loss, reg, n_lam=100, lam_frac=0.01, rho=1., verbose=0
)

# compute number of gradients and number of prox
consensus_iter = model_consensus.log_solve[["l", "n_grad", "n_prox"]].groupby("l").agg("sum")
proxgd_iter = model_proxgd.log_solve[["l", "n_grad", "n_prox"]].groupby("l").agg("sum")

# produce plots
fig, ax1 = plt.subplots(figsize=(6, 4))

# plot nb gradients
color = "red"
ax1.set_xlabel("Solution path iteration")
# ax1.set_ylabel("Nb. gradients", color=color)
ax1.plot(consensus_iter.index, consensus_iter["n_grad"].cumsum(),
		 color=color, linestyle="-", label="ConsensusADMM")
ax1.plot(proxgd_iter.index, proxgd_iter["n_grad"].cumsum(),
		 color=color, linestyle="--", label="FISTA")
ax1.tick_params(axis="y", labelcolor=color)
ax1.legend(loc="upper left", title="$\\bf{Nb. gradients}$")

# add number of prox
ax2 = ax1.twinx()
color = "blue"
# ax2.set_ylabel("Nb. prox operators",color=color)
ax2.plot(consensus_iter.index, consensus_iter["n_prox"].cumsum(),
		 color=color, linestyle="-", label="ConsensusADMM")
ax2.plot(proxgd_iter.index, proxgd_iter["n_prox"].cumsum(),
		 color=color, linestyle="--", label="FISTA")
ax2.tick_params(axis="y", labelcolor=color)
ax2.legend(loc="lower right", title="$\\bf{Nb. proximal}$")

fig.tight_layout()
fig.savefig("fig/exp_ls_solution_path.pdf")


# ---------------------------------------------------------
# single iteration (from beta=0)
lam = 0.3

model_proxgd1 = MTSGL.fit.ProximalGD(
	loss, reg, n_lam=100, lam_frac=0.01, verbose=0, user_lam=[lam]
)
model_consensus1 = MTSGL.fit.ConsensusADMM(
	loss, reg, n_lam=100, lam_frac=0.01, rho=1., verbose=0, user_lam=[lam],
	eps_abs=1.0e-6, eps_rel=1.0e-6
)

obj_min = 55.98691

# compute number of gradients and number of prox
consensus_iter1 = model_consensus1.log_solve[["l", "n_grad", "n_prox", "original obj."]]
proxgd_iter1 = model_proxgd1.log_solve[["l", "n_grad", "n_prox", "original obj."]]

# produce plots
fig, ax1 = plt.subplots(figsize=(6, 4))

# plot nb gradients
color = "red"
ax1.set_ylabel("Optimality gap")
ax1.set_yscale('log')
ax1.plot(consensus_iter1["n_grad"].cumsum() / K, consensus_iter1["original obj."] - obj_min,
		 color=color, linestyle="-", label="ConsensusADMM")
ax1.plot(proxgd_iter1["n_grad"].cumsum(), proxgd_iter1["original obj."] - obj_min,
		 color=color, linestyle="--", label="FISTA")
ax1.tick_params(axis="x", labelcolor=color)
ax1.legend(loc="upper right", title="$\\bf{Nb. gradients}$")
ax1.axis('tight')

# add number of prox
ax2 = ax1.twiny()
color = "blue"
ax2.set_yscale('log')
ax2.plot(consensus_iter1["n_prox"].cumsum() / K, consensus_iter1["original obj."] - obj_min,
		 color=color, linestyle="-", label="ConsensusADMM")
ax2.plot(proxgd_iter1["n_prox"].cumsum(), proxgd_iter1["original obj."] - obj_min,
		 color=color, linestyle="--", label="FISTA")
ax2.tick_params(axis="x", labelcolor=color)
ax2.legend(loc="center right", title="$\\bf{Nb. proximal}$")
ax2.axis('tight')

fig.tight_layout()

fig.savefig("fig/exp_ls_single.pdf")