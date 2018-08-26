from amp_mse cimport cmse_Lq, cmse_Lq_dalpha, ctau_of_alpha, coptimal_alpha
from proximal cimport cproxLq, cproxLq_dx, cproxLq_dt, cproxLq_inv

def mse_Lq(double alpha, double tau, double M, double epsilon, double q, double tol=1e-9):
    return cmse_Lq(alpha, tau, M, epsilon, q, tol)

def mse_Lq_dalpha(double alpha, double tau, double M, double epsilon, double q, double tol=1e-9):
    return cmse_Lq_dalpha(alpha, tau, M, epsilon, q, tol)

def tau_of_alpha(double alpha, double M, double q, double epsilon, double delta, double sigma, double tol=1e-9):
    return ctau_of_alpha(alpha, M, q, epsilon, delta, sigma, tol)

def optimal_alpha(double M, double q, double epsilon, double delta, double sigma, double tol=1e-9):
    return coptimal_alpha(M, q, epsilon, delta, sigma, tol)

def proxLq(double x, double t, double q, double tol=1e-8):
    return cproxLq(x, t, q, tol)

def proxLq_dx(double x, double t, double q, double tol=1e-8):
    return cproxLq_dx(x, t, q, tol)

def proxLq_dt(double x, double t, double q, double tol=1e-8):
    return cproxLq_dt(x, t, q, tol)

def proxLq_inv(double z, double t, double q):
    return cproxLq_inv(z, t, q)

