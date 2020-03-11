from MTSGL.losses import MTLoss
from MTSGL.regularizations import Regularization
from . import Fit, ConvergenceError
from .sharingadmm_utils import proxgd, ridge_saturated, dict_zip
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
			self.eps_abs = 1.0e-6
		else:
			self.eps_abs = float(kwargs["eps_abs"])
			if self.eps_abs < 1.0e-16:
				raise ValueError("eps_abs must be above 1.0e-16")
		if "eps_rel" not in kwargs.keys():
			self.eps_rel = 1.0e-3
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

	def _solve(self, beta: np.ndarray, lam: float):
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
		beta = beta
		eta = {task: loss.lin_predictor(beta[:, [k]]) for k, (task, loss) in enumerate(self.loss.items())}
		z = {task: np.zeros_like(eta[task]) for k, (task, loss) in enumerate(self.loss.items())}
		r = {task: etak - loss.lin_predictor(beta) for task, (etak, loss) in dict_zip(eta, self.loss).items()}
		t = 0
		# setup logging
		dual_obj, primal_obj = self.compute_obj(beta, eta, lam, r, z)
		self.log_solve = pd.DataFrame({
				"t": [t], "primal obj.": [primal_obj], "dual obj.": [dual_obj], "status": ["initial"]}
		)
		while True:
			t += 1
			# update eta
			eta, n_ridge = ridge_saturated(
				loss=self.loss,
				z0=eta,
				a={task: loss.lin_predictor(beta[:, [k]]) - z[task] for k, (task, loss) in enumerate(self.loss.items())},
				tau=self.rho
			)
			# update beta
			beta, n_proxgd = proxgd(
				loss=self.loss,
				reg=self.reg,
				beta0=beta,
				v={task: etak + zk for task, (etak, zk) in dict_zip(eta, z).items()},
				lam=lam,
				rho=self.rho
			)
			print(lam, t, n_ridge, n_proxgd)
			# update z and r
			r = {task: eta[task] - loss.lin_predictor(beta[:, [k]]) for k, (task, loss) in enumerate(self.loss.items())}
			z = {task: zk + rk for task, (zk, rk) in dict_zip(z, r).items()}
			# norms for convergence
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
			# convergence checks
			dual_obj, primal_obj = self.compute_obj(beta, eta, lam, r, z)
			self.log_solve = self.log_solve.append(
				pd.DataFrame({
					"t": [t], "primal obj.": [primal_obj], "dual obj.": [dual_obj], "status": ["converged"]}
				),
				ignore_index=True
			)
			if np.linalg.norm(np.array(r_norm), 2) < eps_primal and np.linalg.norm(np.array(s_norm), 2) < eps_dual:
				break
			if t > self.max_iter:
				print(self.log_solve)
				raise ConvergenceError("maximum number of iteration reached.")
		if self.verbose > 1:
			print(self.log_solve)
		return beta, t

	def compute_obj(self, beta, eta, lam, r, z):
		primal_obj = self.loss.loss(beta) + lam * self.reg.value(beta)
		dual_obj = self.loss.loss_from_linear_predictor(eta)
		dual_obj += lam * self.reg.value(beta)
		dual_obj += self.rho * 0.5 * sum([np.linalg.norm(zk + rk, 2) ** 2 for zk, rk in dict_zip(z, r).values()])
		return dual_obj, primal_obj
