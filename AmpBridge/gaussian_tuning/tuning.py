import numpy as np
from .empirical_mse import empirical_tau2
from ..coptimization import vec_bridge_Lq


def grid_search(lam_vec, X, y, q, abs_tol, iter_max):
    """
    lam_seq in ascending order
    """
    k = lam_vec.shape[0]
    Beta_hat = vec_bridge_Lq(X, y, lam_vec, q, abs_tol, iter_max)
    tau2_vec = np.empty(k)
    for i in range(k):
        tau2_vec[i] = empirical_tau2(Beta_hat[:, i], X, y, lam_vec[i], q, abs_tol)
    min_idx = np.argmin(tau2_vec)
    return lam_vec[min_idx], Beta_hat[:, min_idx]

