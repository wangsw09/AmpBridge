import sys
sys.path.insert(0, "/home/wangsw09/work/Proj-AmpBridge/")

import numpy as np
import numpy.random as npr
from sklearn.linear_model import Lasso, Ridge

from AmpBridge import bridge_Lq, vec_bridge_Lq

TOL = 1e-6

def test_bridge_L1():
    tol = 1e-12
    n = 1000
    p = 200
    X = npr.normal(size=(n, p))
    beta = npr.normal(size=p)
    y = np.dot(X, beta) + 0.5 * npr.normal(size=n)
    lam = 1.0

    las = Lasso(alpha=lam / n, fit_intercept=False, tol=tol, max_iter=50000)
    las.fit(X, y)
    beta_skl = las.coef_
    beta_hat = bridge_Lq(X, y, lam, 1, abs_tol=tol, iter_max=50000)
    assert np.abs(beta_hat - beta_skl).max() < TOL
    
    tol = 1e-10
    n = 500
    p = 700
    X = npr.normal(size=(n, p))
    beta = npr.normal(size=p)
    y = np.dot(X, beta) + 0.5 * npr.normal(size=n)
    lam = 1.0

    las = Lasso(alpha=lam / n, fit_intercept=False, tol=tol, max_iter=100000)
    las.fit(X, y)
    beta_skl = las.coef_
    beta_hat = bridge_Lq(X, y, lam, 1, abs_tol=tol, iter_max=100000)
    assert np.abs(beta_hat - beta_skl).max() < TOL

def test_bridge_L2():
    tol = 1e-10
    n = 1000
    p = 200
    X = npr.normal(size=(n, p))
    beta = npr.normal(size=p)
    y = np.dot(X, beta) + 0.5 * npr.normal(size=n)
    lam = 1.0

    rig = Ridge(alpha=lam * 2, fit_intercept=False, tol=tol, max_iter=10000)
    rig.fit(X, y)
    beta_skl = rig.coef_
    beta_hat = bridge_Lq(X, y, lam, 2, abs_tol=tol, iter_max=10000)
    assert np.abs(beta_hat - beta_skl).max() < TOL
    
    tol = 1e-10
    n = 800
    p = 1000
    X = npr.normal(size=(n, p))
    beta = npr.normal(size=p)
    y = np.dot(X, beta) + 0.5 * npr.normal(size=n)
    lam = 1.0

    rig = Ridge(alpha=lam * 2, fit_intercept=False, tol=tol, max_iter=10000)
    rig.fit(X, y)
    beta_skl = rig.coef_
    beta_hat = bridge_Lq(X, y, lam, 2, abs_tol=tol, iter_max=10000)
    assert np.abs(beta_hat - beta_skl).max() < TOL

