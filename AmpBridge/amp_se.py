from .cscalar import mse_Lq, tau_of_alpha, optimal_alpha, alpha_of_lambda_Lq, lambda_of_alpha_Lq

def mseLq(M, alpha, epsilon, delta, sigma, q, tol=1e-9):
    tau = tau_of_alpha(alpha, M, epsilon, delta, sigma, q, tol=tol)
    return mse_Lq(M, alpha, tau, epsilon, q, tol=tol)

def optimal_tuning(M, epsilon, delta, sigma, q, tol=1e-9):
    return optimal_alpha(M, epsilon, delta, sigma, q, tol=tol)

def optimal_mseLq(M, epsilon, delta, sigma, q, tol=1e-9):
    opta = optimal_alpha(M, epsilon, delta, sigma, q, tol=tol)
    return mseLq(M, opta, epsilon, delta, sigma, q, tol=tol)

def tuning_transform(alpha, M, epsilon, delta, sigma, q, tol=1e-9):
    return lambda_of_alpha_Lq(alpha, M, epsilon, delta, sigma, q, tol)

def tuning_transform_inv(lam, M, epsilon, delta, sigma, q, tol=1e-9):
    return alpha_of_lambda_Lq(lam, M, epsilon, delta, sigma, q, tol)

