from math import sqrt
import numpy as np


def empirical_mse(beta_hat, X, y, lam, q, tol):
        """
        empirical mse from each iteration
        """
        tau2 = empirical_tau2(beta_hat, X, y, lam, q, tol)
        if q == 1:
            return np.mean(XTz ** 2) - tau2 + 2 * tau2 * np.mean(beta_hat != 0)
        else:
            gamma = empirical_tuning_mapping(lam, beta_hat, delta, q, tol)
            return np.mean(XTz ** 2) - tau2 + 2 * tau2 * np.mean(1.0 / (1.0 +
                gamma * q * (q - 1.0) * np.abs(beta_hat) ** (q - 2.0)))


def empirical_tau2(beta_hat, X, y, lam, q, tol):
        """
        empirical mse from each iteration
        """
        n, p = X.shape
        delta = (1.0 * n) / p
        if q == 1:
            z =  (y - np.dot(X, beta_hat)) / (1.0 - 1 / delta * np.mean(np.abs(beta_hat) > 1e-7))
        else:
            gamma = empirical_tuning_mapping(lam, beta_hat, delta, q, tol)
            z =  (y - np.dot(X, beta_hat)) / (1.0 - 1.0 / delta * np.mean(1.0 / (1.0 + gamma * q * (q - 1.0) * np.abs(beta_hat) ** (q - 2.0))))
        # XTz = np.dot(X.T, z)
        tau2 = np.mean(z ** 2) / n
        return tau2


def empirical_tuning_mapping(lam, beta_hat, delta, q, tol):
    if q == 1:
        return lam / (1.0 - 1.0 / delta * np.mean(beta_hat != 0))
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
 
