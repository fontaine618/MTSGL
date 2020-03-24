from MTSGL.losses import MTLoss
from MTSGL.regularizations import Regularization
from . import Fit, ConvergenceError
from solvers.sharingadmm_utils import dict_zip
from solvers.proxgd import proxgd
import numpy as np
import pandas as pd


class SharingADMM(Fit):

	def __init__(
			self,
			loss: MTLoss,
			reg: Regularization,
			**kwargs
	):
		self.threshold_ridge_decrease = None
		super().__init__(loss, reg, **kwargs)
		self.path = self._solution_path()

	def _set_additional_options(self, **kwargs):
		if "eps_abs" not in kwargs.keys():
			self.eps_abs = 1.0e-4
		else:
			self.eps_abs = float(kwargs["eps_abs"])
			if self.eps_abs < 1.0e-3:
				raise ValueError("eps_abs must be above 1.0e-16")
		if "eps_rel" not in kwargs.keys():
			self.eps_rel = 1.0e-6
		else:
			self.eps_rel = float(kwargs["eps_rel"])
			if self.eps_rel < 1.0e-8:
				raise ValueError("eps_rel must be above 1.0e-8")
		if "rho" not in kwargs.keys():
			self.rho = 1.
		else:
			self.rho = float(kwargs["rho"])
			if self.rho <= 0.:
				raise ValueError("rho must be positive, received {}".format(self.rho))

	def _solve(self, beta0: np.ndarray, lam: float, **kwargs):
		"""

		Parameters
		----------
		beta : array-like
			The initial parameter value. (p, K)
		lam : float
			The regularization parameter.

		Returns
		-------
		beta : array-like
			The final estimate. (p, K)

		Raises
		------
		ConvergenceError
			If the solver does not reach appropriate convergence.
		"""
		# initialization
		beta = beta0
		eta = {task: loss.lin_predictor(beta[:, [k]]) for k, (task, loss) in enumerate(self.loss.items())}
		r = {
			task: etak - loss.lin_predictor(beta[:, [k]])
			for k, (task, (etak, loss))
			in enumerate(dict_zip(eta, self.loss).items())
		}
		z = r
		t = 0
		# setup logging
		augmented_obj, original_obj = self.compute_obj(beta, eta, lam, r, z)
		self.log_solve = pd.DataFrame({
				"t": [t], "original obj.": [original_obj], "augmented obj.": [augmented_obj], "status": ["initial"],
				"n_ridge": 0, "n_proxgd": 0
		})
		while True:
			t += 1
			# update eta
			# eta, n_ridge = ridge_saturated(
			# 	loss=self.loss,
			# 	z0=eta,
			# 	a={task: loss.lin_predictor(beta[:, [k]]) - z[task] for k, (task, loss) in enumerate(self.loss.items())},
			# 	tau=1.0 / self.rho,
			# 	threshold=np.sqrt(sum(self.loss.data.n_obs.values())) * self.eps_abs
			# )
			# for LS, we can do it in closed form:
			for k, (task, loss) in enumerate(self.loss.items()):
				mat = np.diag(loss.w.reshape((-1))) + self.rho * np.eye(loss.n)
				eta[task] = np.linalg.solve(mat, loss.w * loss.y + self.rho * (loss.lin_predictor(beta[:, [k]]) - z[task]))
			n_ridge = 1
			# update beta  using ProximalGD on the LS loss
			beta, n_proxgd = proxgd(
				loss=self.loss,
				reg=self.reg,
				beta0=beta,
				v={task: etak + zk for task, (etak, zk) in dict_zip(eta, z).items()},
				lam=lam,
				rho=self.rho,
				ls=True,
				threshold=np.sqrt(self.loss.data.n_features * self.loss.data.n_tasks) * self.eps_abs
			)
			# print(lam, t, n_ridge, n_proxgd)
			# update z and r
			r = {task: eta[task] - loss.lin_predictor(beta[:, [k]]) for k, (task, loss) in enumerate(self.loss.items())}
			z = {task: zk + rk for task, (zk, rk) in dict_zip(z, r).items()}
			# norms for convergence
			eps_dual, eps_primal, r_norm, s_norm = self.compute_convergence_checks(beta, eta, r, z)
			# logging
			augmented_obj, original_obj = self.compute_obj(beta, eta, lam, r, z)
			self.log_solve = self.log_solve.append(
				pd.DataFrame({
					"t": [t], "original obj.": [original_obj], "augmented obj.": [augmented_obj], "status": ["converged"],
					"n_ridge": n_ridge, "n_proxgd": n_proxgd
				}),
				ignore_index=True
			)
			# print(np.linalg.norm(np.array(r_norm), 2), eps_original,
			# 	  np.linalg.norm(np.array(s_norm), 2), eps_dual)
			# convergence checks
			if np.linalg.norm(np.array(r_norm), 2) < eps_primal and np.linalg.norm(np.array(s_norm), 2) < eps_dual:
				break
			if t > self.max_iter:
				print(self.log_solve)
				raise ConvergenceError("maximum number of iteration reached.")
		if self.verbose > 1:
			print(self.log_solve)
		return beta, t

	def compute_convergence_checks(self, beta, eta, r, z):
		r_norm = [np.linalg.norm(rk, 2) for task, rk in r.items()]
		s_norm = [
			self.rho * np.linalg.norm(np.matmul(loss.x.T, rk), 2)
			for task, (rk, loss) in dict_zip(r, self.loss).items()
		]
		xbeta_norm = np.sqrt(sum([
			np.linalg.norm(loss.lin_predictor(beta[:, [k]]), 2) ** 2
			for k, loss in enumerate(self.loss.values())
		]))
		eta_norm = np.sqrt(sum([
			np.linalg.norm(etak, 2) ** 2
			for etak in eta.values()
		]))
		xtz_norm = self.rho * np.sqrt(sum([
			np.linalg.norm(np.matmul(loss.x.T, zk), 2) ** 2
			for loss, zk in dict_zip(self.loss, z).values()
		]))
		# compute thresholds
		eps_primal = np.sqrt(sum(self.loss.data.n_obs.values())) * self.eps_abs + \
					 self.eps_rel * max(xbeta_norm, eta_norm)
		eps_dual = np.sqrt(self.loss.data.n_features * self.loss.data.n_tasks) * self.eps_abs + \
				   self.eps_rel * xtz_norm
		return eps_dual, eps_primal, r_norm, s_norm

	def compute_obj(self, beta, eta, lam, r, z):
		original_obj = self.loss.loss(beta) + lam * self.reg.value(beta)
		augmented_obj = self.loss.loss_from_linear_predictor(eta)
		augmented_obj += lam * self.reg.value(beta)
		augmented_obj += self.rho * sum([np.matmul(zk.T, rk)[0, 0] for zk, rk in dict_zip(z, r).values()])
		augmented_obj += 0.5 * self.rho * sum([np.linalg.norm(rk, 2) ** 2 for rk in r.values()])
		return augmented_obj, original_obj
