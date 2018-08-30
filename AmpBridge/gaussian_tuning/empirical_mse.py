import numpy as np

def empirical_mse(beta_hat, X, y, gamma, q, tol):
        """
        empirical mse from each iteration
        u = beta + np.dot(X.T, z)
        theta = gamma * tau_new, gamma is equivalent to alpha
        tau_old = npla.norm(z) / np.sqrt(n)
        """
        n, p = beta_hat.shape
        delta = (1.0 * n) / p
        if q == 1:
            z =  (y - np.dot(X, beta_hat)) / (1.0 - 1 / delta * np.mean(beta_hat))
        else:
            z =  (y - np.dot(X, beta_hat)) / (1.0 - 1.0 / delta * np.mean(np.abs(beta_hat) ** (q - 2.0)))
        XTz = np.dot(X.T, z)
        tau2 = np.mean(z ** 2)
        if q == 1:
            return np.mean(XTz ** 2) - tau2 + 2 * tau2 * np.mean(beta_hat != 0)
        else:
            return np.mean(XTz ** 2) - tau2 + 2 * tau2 * np.mean(1.0 / (1.0 +
                gamma * q * (q - 1.0) * np.abs(beta_hat) ** (q - 2.0)))

def tuning_mapping(lam, beta_hat, q, tol):
    L = lam
    U = lam + 1.0 / delta / q / (q - 1.0) * np.mean(1.0 / np.abs(beta_hat) ** (q - 2.0))

    while U - L > tol:
        mid = (U + L) / 2.0
        if lam / mid > 1.0 - 1 / delta * np.mean(1.0 / (1.0 + mid * q * (q
            - 1.0) * np.abs(beta_hat) ** (q - 2.0))):
            L = mid
        else:
            U = mid
    return (U + L) / 2.0
 
