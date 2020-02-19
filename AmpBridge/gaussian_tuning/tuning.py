import numpy as np
from .empirical_mse import empirical_tau2
from ..coptimization import vec_bridge_Lq


def grid_search(lam_vec, X, y, q, abs_tol, iter_max, details=False, sanity_check_threshold=10.0):
    """
    lam_seq in ascending order
    """
    k = lam_vec.shape[0]
    Beta_hat = vec_bridge_Lq(X, y, lam_vec, q, abs_tol, iter_max)
    tau2_vec = np.empty(k)
    for i in range(k):
        tau2_vec[i] = empirical_tau2(Beta_hat[:, i], X, y, lam_vec[i], q, abs_tol)
    ### need a sanity check below ###
    min_idx = k-1
    min_tau2 = tau2_vec[min_idx]

    for i in range(k-2, -1, -1):
        if tau2_vec[i] < tau2_vec[i+1] / sanity_check_threshold or np.isinf(tau2_vec[i]):
            break
        if tau2_vec[i] < tau2_vec[min_idx]:
            min_tau2 = tau2_vec[i]
            min_idx = i

    if not details:
        return lam_vec[min_idx], Beta_hat[:, min_idx]
    else:
        return lam_vec[min_idx], Beta_hat[:, min_idx], tau2_vec
