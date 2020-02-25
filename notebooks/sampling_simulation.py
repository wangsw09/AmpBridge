import numpy as np
import numpy.random as npr
import AmpBridge as ab

repeat = 80

delta_vec = [0.8]
eps_vec = [0.2]
sigma_vec = [1.5]
M_vec = [8]
q_vec = [1, 1.2, 2, 4]

p = 5000
n = int(p * delta) 
k = int(p * eps)

abs_tol = 1e-5
max_iter = 10000
lam_vec = 10.0 ** np.arange(-7, 3, 1.0)

for delta in delta_vec:
    for eps in eps_vec:
        for sigma in sigma_vec:
            for M in M_vec:
                for q in q_vec:
                    for i in range(repeat):
                        X = npr.normal(0, 1.0, (n, p)) / sqrt(n)
                        beta = np.ones(p) * M
                        beta[npr.choice(p, p - k, replace=False)] = 0.0
                        w = npr.normal(0, 1, n) * sigma
                        y = np.dot(X, beta) + w
                        
                        lam_opt, beta_opt, tau2_vec = ab.grid_search(lam_vec, X, y, q, abs_tol, max_iter, details=True)


