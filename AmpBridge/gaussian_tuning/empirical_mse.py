import numpy as np

def empirical_mse(beta_hat, X, y, gamma, q, tol):
        """
        empirical "mse" tau square of the optimizer
        """
        n, p = X.shape
        delta = (1.0 * n) / p
        if q == 1:
            z =  (y - np.dot(X, beta_hat)) / (1.0 - 1 / delta * np.mean(beta_hat != 0))
        else:
            z =  (y - np.dot(X, beta_hat)) / (1.0 - 1.0 / delta * np.mean(np.abs(beta_hat) ** (q - 2.0)))
        tau2 = np.mean(z ** 2)
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
 
