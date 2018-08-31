from amp_mse cimport cmse_Lq, cmse_Lq_dalpha, ctau_of_alpha, coptimal_alpha, clambda_of_alpha_Lq, calpha_of_lambda_Lq
from proximal cimport cproxLq, cproxLq_dx, cproxLq_dt, cproxLq_inv
from gaussian cimport cgaussianMoment, cgaussianPdf, cgaussianCdf

def mse_Lq(double M, double alpha, double tau, double epsilon, double q, double tol=1e-9):
    return cmse_Lq(M, alpha, tau, epsilon, q, tol)

def mse_Lq_dalpha(double alpha, double M, double tau, double epsilon, double q, double tol=1e-9):
    return cmse_Lq_dalpha(alpha, M, tau, epsilon, q, tol)

def tau_of_alpha(double alpha, double M, double epsilon, double delta, double sigma, double q, double tol=1e-9):
    return ctau_of_alpha(alpha, M, epsilon, delta, sigma, q, tol)

def optimal_alpha(double M, double epsilon, double delta, double sigma, double q, double tol=1e-9):
    return coptimal_alpha(M, epsilon, delta, sigma, q, tol)

def lambda_of_alpha_Lq(double alpha, double M, double epsilon, double delta, double sigma, double q, double tol=1e-9):
    return clambda_of_alpha_Lq(alpha, M, epsilon, delta, sigma, q, tol)

def alpha_of_lambda_Lq(double lam, double M, double epsilon, double delta, double sigma, double q, double tol=1e-9):
    return calpha_of_lambda_Lq(lam, M, epsilon, delta, sigma, q, tol)

def proxLq(double x, double t, double q, double tol=1e-8):
    return cproxLq(x, t, q, tol)

def proxLq_dx(double x, double t, double q, double tol=1e-8):
    return cproxLq_dx(x, t, q, tol)

def proxLq_dt(double x, double t, double q, double tol=1e-8):
    return cproxLq_dt(x, t, q, tol)

def proxLq_inv(double z, double t, double q):
    return cproxLq_inv(z, t, q)

def gaussianMoment(double q):
    return cgaussianMoment(q)

def gaussianCdf(double x):
    return cgaussianCdf(x)

def gaussianPdf(double x):
    return cgaussianPdf(x)

