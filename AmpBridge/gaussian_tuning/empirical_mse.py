import numpy as np

def empirical_mse(beta_hat, X, y, gamma, q, tol):
    """
    Empirical "mse" tau square of the optimizer;
    The quantity is calculated through the AMP mechanism, instead of general out-of-sample estimation methods, such as cross-validation.
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
    """
    This function calculates the value of "gamma" in the paper;
    It is, in some senses, equivalent to the tuning parameter "alpha";
    Only for $q > 2$ does this function make sense.

    """
    if q == 1:
        raise ValueError("value of <q> has to be greater than 1.")
        # return lam / (1.0 - 1.0 / delta * np.mean(beta_hat != 0))
    L = lam
    U = lam + 1.0 / delta / q / (q - 1.0) * np.mean(1.0 / np.abs(beta_hat) ** (q - 2.0))

    while U - L > tol:
        mid = (U + L) / 2.0
        if lam / mid > 1.0 - 1 / delta * np.mean(1.0 / (1.0 + mid * q * (q - 1.0) * np.abs(beta_hat) ** (q - 2.0))):
            L = mid
        else:
            U = mid
    return (U + L) / 2.0
 
