import numpy as np
import pandas as pd
import MTSGL
import matplotlib.pyplot as plt
from scipy.special import expit
plt.style.use("seaborn")

def gen_data(n=1000, p=100, K=5):
	x = np.random.normal(0, 1, (n, p))
	beta = np.random.randint(-2, 3, (p, K))
	beta[np.random.choice(range(p), (4 * p) // 5, replace=False), :] = 0
	tasks = range(K)
	task = np.random.choice(tasks, n)
	w = np.random.uniform(0, 1, n)
	eta = np.matmul(x, beta)
	eta = np.array([[eta[i, task[i]]] for i in range(n)])
	y = np.random.binomial(1, expit(eta))
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
	loss = MTSGL.losses.MTLogReg(data)
	weights = np.ones(p + 1)
	weights[0] = 0.
	return data, loss, weights, beta

# generate problem
np.random.seed(1)
K = 50
n = 100
p = 50
data, loss, weights, beta = gen_data(n, p, K)
reg = MTSGL.regularizations.SparseGroupLasso(q=2, alpha=0.5, weights=weights)

# ---------------------------------------------------------
# Solution path
user_lam = np.logspace(-1.3, -2.5, 50)
# fit problems
model_proxgd = MTSGL.fit.ProximalGD(
	loss, reg, n_lam=100, lam_frac=0.05, verbose=0, user_lam=user_lam,
	eps_abs=1.0e-4, eps_rel=1.0e-3
)
model_consensus = MTSGL.fit.ConsensusADMM(
	loss, reg, n_lam=100, lam_frac=0.05, rho=0.05, verbose=0, user_lam=user_lam,
	eps_abs=1.0e-4, eps_rel=1.0e-3
)

# compute number of gradients and number of prox
consensus = model_consensus.log_solve[["l", "n_grad", "n_prox", "time"]].groupby("l").agg(
	{"n_grad": "sum", "n_prox": "sum", "time": "sum"}
)
proxgd = model_proxgd.log_solve[["l", "n_grad", "n_prox", "time"]].groupby("l").agg(
	{"n_grad": "sum", "n_prox": "sum", "time": "sum"}
)

lam = 0.02

model_proxgd1 = MTSGL.fit.ProximalGD(
	loss, reg, verbose=0, user_lam=[lam],
	eps_abs=1.0e-5, eps_rel=1.0e-4, max_iter=1e5
)
model_consensus1 = MTSGL.fit.ConsensusADMM(
	loss, reg, rho=0.05, verbose=0, user_lam=[lam],
	eps_abs=1.0e-8, eps_rel=1.0e-6
)

# compute number of gradients and number of prox
consensus1 = model_consensus1.log_solve
proxgd1 = model_proxgd1.log_solve

obj_min = min(min(consensus1["original obj."]), min(proxgd1["original obj."])) - 1.0e-4






# produce plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

col_gd, col_px, col_tm = '#4C72B0', '#55A868', '#C44E52'
admm_ls, proxgd_ls = "-", "--"

# solution path plot
ax1.grid(alpha=0.5)
# ax1.set_yscale('log')
ax1.set_xlabel("Solution path iteration")
ln_admm_gd = ax1.plot(consensus.index, consensus["n_grad"].cumsum(),
		 color=col_gd, linestyle=admm_ls, label="ADMM (grad.)")
ln_prox_gd = ax1.plot(proxgd.index, proxgd["n_grad"].cumsum(),
		 color=col_gd, linestyle=proxgd_ls, label="FISTA (grad.)", linewidth=2.2)
ln_admm_px = ax1.plot(consensus.index, consensus["n_prox"].cumsum(),
		 color=col_px, linestyle=admm_ls, label="ADMM (prox.)")
ln_prox_px = ax1.plot(proxgd.index, proxgd["n_prox"].cumsum(),
		 color=col_px, linestyle=proxgd_ls, label="FISTA (prox.)")

ax1.set_ylabel("Nb. gradients/proximals")

ax1t = ax1.twinx()
# ax1t.set_yscale('log')
ax1t.grid(alpha=0.5, linestyle=":")
ln_admm_tm = ax1t.plot(consensus.index, consensus["time"].cumsum(),
		 color=col_tm, linestyle=admm_ls, label="ADMM (time)")
ln_prox_tm = ax1t.plot(proxgd.index, proxgd["time"].cumsum(),
		 color=col_tm, linestyle=proxgd_ls, label="FISTA (time)")
ax1t.tick_params(axis="y", labelcolor=col_tm)
ax1t.set_ylabel("Computing time (s)", color=col_tm)

lns = ln_admm_gd + ln_prox_gd + ln_admm_px + ln_prox_px + ln_admm_tm + ln_prox_tm
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, frameon=True)
ax1.axis('tight')
ax1t.axis('tight')

ax1.set_title("(a) Solution path\n")

# single problem
ax2.grid(alpha=0.5)
ax2.set_ylabel("Optimality gap")
ax2.set_yscale('log')
ln_admm_gd = ax2.plot(consensus1["n_grad"].cumsum(), consensus1["original obj."] - obj_min,
		 color=col_gd, linestyle=admm_ls, label="ADMM (grad.)")
ln_prox_gd = ax2.plot(proxgd1["n_grad"].cumsum(), proxgd1["original obj."] - obj_min,
		 color=col_gd, linestyle=proxgd_ls, label="FISTA (grad.)", linewidth=2.2)
ln_admm_px = ax2.plot(consensus1["n_prox"].cumsum(), consensus1["original obj."] - obj_min,
		 color=col_px, linestyle=admm_ls, label="ADMM (prox.)")
ln_prox_px = ax2.plot(proxgd1["n_prox"].cumsum(), proxgd1["original obj."] - obj_min,
		 color=col_px, linestyle=proxgd_ls, label="FISTA (prox.)")

ax2.set_xlabel("Nb. gradients/proximals")

ax2t = ax2.twiny()
ax2t.set_yscale('log')
ax2t.grid(alpha=0.5, linestyle=":")
ln_admm_tm = ax2t.plot(consensus1["time"].cumsum(), consensus1["original obj."] - obj_min,
		 color=col_tm, linestyle=admm_ls, label="ADMM (time)")
ln_prox_tm = ax2t.plot(proxgd1["time"].cumsum(), proxgd1["original obj."] - obj_min,
		 color=col_tm, linestyle=proxgd_ls, label="FISTA (time)")
ax2t.tick_params(axis="x", labelcolor=col_tm)
ax2t.set_ylabel("Computing time (s)", color=col_tm)

lns = ln_admm_gd + ln_prox_gd + ln_admm_px + ln_prox_px + ln_admm_tm + ln_prox_tm
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, frameon=True)

ax2.axis('tight')
ax2t.axis('tight')
ax2.set_title("(b) Single problem")

fig.tight_layout()
fig.savefig("fig/exp_logreg.pdf")

plt.show()




























#
#
#
#
#
# model_proxgd.log
# model_consensus.log
#
# # compute number of gradients and number of prox
# consensus_iter = model_consensus.log_solve[["l", "n_grad", "n_prox"]].groupby("l").agg("sum")
# proxgd_iter = model_proxgd.log_solve[["l", "n_grad", "n_prox"]].groupby("l").agg("sum")
#
# # produce plots
# fig, ax1 = plt.subplots(figsize=(6, 4))
#
# # plot nb gradients
# color = "red"
# ax1.set_xlabel("Solution path iteration")
# # ax1.set_ylabel("Nb. gradients", color=color)
# ax1.plot(consensus_iter.index, consensus_iter["n_grad"].cumsum(),
# 		 color=color, linestyle="-", label="ConsensusADMM")
# ax1.plot(proxgd_iter.index, proxgd_iter["n_grad"].cumsum(),
# 		 color=color, linestyle="--", label="FISTA")
# ax1.tick_params(axis="y", labelcolor=color)
# ax1.set_yscale('log')
# ax1.legend(loc="upper left", title="$\\bf{Nb. gradients}$")
#
# # add number of prox
# ax2 = ax1.twinx()
# color = "blue"
# # ax2.set_ylabel("Nb. prox operators",color=color)
# ax2.plot(consensus_iter.index, consensus_iter["n_prox"].cumsum(),
# 		 color=color, linestyle="-", label="ConsensusADMM")
# ax2.plot(proxgd_iter.index, proxgd_iter["n_prox"].cumsum(),
# 		 color=color, linestyle="--", label="FISTA")
# ax2.tick_params(axis="y", labelcolor=color)
# ax2.set_yscale('log')
# ax2.legend(loc="lower right", title="$\\bf{Nb. proximal}$")
#
# fig.tight_layout()
# fig.savefig("fig/exp_logreg_solution_path.pdf")
#
#
# # ---------------------------------------------------------
# # single iteration (from beta=0)
# lam = 0.003
#
# model_proxgd1 = MTSGL.fit.ProximalGD(
# 	loss, reg, n_lam=100, lam_frac=0.01, verbose=0, user_lam=[lam],
# 	eps_abs=1.0e-6, eps_rel=1.0e-6
# )
# model_consensus1 = MTSGL.fit.ConsensusADMM(
# 	loss, reg, n_lam=100, lam_frac=0.01, rho=0.01, verbose=0, user_lam=[lam],
# 	eps_abs=1.0e-8, eps_rel=1.0e-6
# )
#
# model_consensus1.log_solve
# model_proxgd1.log_solve
#
# obj_min = 0.487488
#
# # compute number of gradients and number of prox
# consensus_iter1 = model_consensus1.log_solve[["l", "n_grad", "n_prox", "original obj."]]
# proxgd_iter1 = model_proxgd1.log_solve[["l", "n_grad", "n_prox", "original obj."]]
#
# # produce plots
# fig, ax1 = plt.subplots(figsize=(6, 4))
#
# # plot nb gradients
# color = "red"
# ax1.set_ylabel("Optimality gap")
# ax1.set_yscale('log')
# ax1.plot(consensus_iter1["n_grad"].cumsum(), consensus_iter1["original obj."] - obj_min,
# 		 color=color, linestyle="-", label="ConsensusADMM")
# ax1.plot(proxgd_iter1["n_grad"].cumsum(), proxgd_iter1["original obj."] - obj_min,
# 		 color=color, linestyle="--", label="FISTA")
# ax1.tick_params(axis="x", labelcolor=color)
# ax1.legend(loc="upper right", title="$\\bf{Nb. gradients}$")
# ax1.axis('tight')
#
# # add number of prox
# ax2 = ax1.twiny()
# color = "blue"
# ax2.set_yscale('log')
# ax2.plot(consensus_iter1["n_prox"].cumsum(), consensus_iter1["original obj."] - obj_min,
# 		 color=color, linestyle="-", label="ConsensusADMM")
# ax2.plot(proxgd_iter1["n_prox"].cumsum(), proxgd_iter1["original obj."] - obj_min,
# 		 color=color, linestyle="--", label="FISTA")
# ax2.tick_params(axis="x", labelcolor=color)
# ax2.legend(loc="center right", title="$\\bf{Nb. proximal}$")
# ax2.axis('tight')
#
# fig.tight_layout()
#
# fig.savefig("fig/exp_logreg_single.pdf")