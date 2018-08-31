import sys
sys.path.insert(0, "/home/wangsw09/work/Proj-AmpBridge/")

import numpy as np

from AmpBridge import bridge_Lq, vec_bridge_Lq

TOL = 1e-8

def test_bridge_Lq():
    tol = 1e-10
    X = np.array([[1, 1], [1, 3], [2, 1]], dtype=np.float64)
    beta = np.array([1, 0], dtype=np.float64)
    y = np.array([1.5, 0, 3], dtype=np.float64)
    lam = 1.0

    beta_true = np.array([385.0 / 300.0, -0.2], dtype=np.float64)
    beta_hat = bridge_Lq(X, y, lam, 1, abs_tol=tol, iter_max=10000)
    assert np.abs(beta_hat - beta_true).max() < TOL

