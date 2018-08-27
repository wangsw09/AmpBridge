from .cscalar import mse_Lq, tau_of_alpha, optimal_alpha, alpha_of_lambda_Lq, lambda_of_alpha_Lq

def mseLq(alpha, M, epsilon, delta, sigma, q, tol=1e-9):
    tau = tau_of_alpha(alpha, M, q, epsilon, delta, sigma, tol=tol)
    return mse_Lq(alpha, tau, M, epsilon, q, tol=tol)

def optimal_tuning(M, q, epsilon, delta, sigma, tol=1e-9):
    return optimal_alpha(M, q, epsilon, delta, sigma, tol=tol)

def optimal_mseLq(M, epsilon, delta, sigma, q, tol=1e-9):
    opta = optimal_alpha(M, q, epsilon, delta, sigma, tol=tol)
    return mseLq(opta, M, epsilon, delta, sigma, q, tol=tol)

def tuning_transform(alpha, M, epsilon, delta, sigma, q, tol=1e-9):
    return lambda_of_alpha_Lq(alpha, M, epsilon, delta, sigma, q, tol)

def tuning_transform_inv(lam, M, epsilon, delta, sigma, q, tol=1e-9):
    return alpha_of_lambda_Lq(lam, M, epsilon, delta, sigma, q, tol)

