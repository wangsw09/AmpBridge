from libc.math cimport sqrt
import numpy as np
cimport numpy as np
from scipy.integrate import quad

from gaussian cimport cgaussianPdf, cgaussianCdf
from proximal cimport _cproxLq, _cproxLq_dt, _cproxLq_dx

# mse_func
cdef double _cmse_L1_singleton(double x, double alpha, double tau):
    return ( tau ** 2 * (1.0 + alpha ** 2) - x ** 2 ) * (cgaussianCdf(x / tau - alpha) + cgaussianCdf( - x / tau - alpha)) - (tau * x + tau ** 2 * alpha) * cgaussianPdf(alpha - x / tau) + (tau * x - tau ** 2 * alpha) * cgaussianPdf(alpha + x / tau) + x ** 2

cdef double _cmse_L1(double M, double alpha, double tau, double epsilon):
    return _cmse_L1_singleton(M, alpha, tau) * epsilon + _cmse_L1_singleton(0.0, alpha, tau) * (1.0 - epsilon)

cdef double _cmse_L2_singleton(double x, double alpha, double tau):
    return (tau ** 2.0 + 4 * alpha ** 2.0 * x ** 2.0) / (1.0 + 2.0 * alpha) ** 2

cdef double _cmse_L2(double M, double alpha, double tau, double epsilon):
    return _cmse_L2_singleton(M, alpha, tau) * epsilon + _cmse_L2_singleton(0.0, alpha, tau) * (1.0 - epsilon)

cdef double _cmse_Lq_integrand(double z, double x, double alpha, double tau, double q, double tol):
    return (_cproxLq(x + tau * z, alpha * tau ** (2.0 - q), q, tol) - x) ** 2 * cgaussianPdf(z)

cdef double _cmse_Lq_singleton(double x, double alpha, double tau, double q, double tol):
    return quad(_cmse_Lq_integrand, -np.inf, np.inf, args=(x, alpha, tau, q, tol))[0]

cdef double _cmse_Lq(double M, double alpha, double tau, double epsilon, double q, double tol):
    return _cmse_Lq_singleton(M, alpha, tau, q, tol) * epsilon + _cmse_Lq_singleton(0, alpha, tau, q, tol) * (1.0 - epsilon)

cdef double cmse_Lq(double M, double alpha, double tau, double epsilon, double q, double tol=1e-9):
    if q == 1:
        return _cmse_L1(M, alpha, tau, epsilon)
    elif q == 2:
        return _cmse_L2(M, alpha, tau, epsilon)
    else:
        return _cmse_Lq(M, alpha, tau, epsilon, q, tol)

# mse_derivative func
cdef double _cmse_L1_dalpha_singleton(double x, double alpha, double tau):
    return 2.0 * tau ** 2 * alpha * (cgaussianCdf(x / tau - alpha) + cgaussianCdf( - x / tau - alpha)) - 2.0 * tau ** 2 * (cgaussianPdf(alpha - x / tau) + cgaussianPdf(alpha + x / tau) )

cdef double _cmse_L1_dalpha(double alpha, double M, double tau, double epsilon):
    return _cmse_L1_dalpha_singleton(M, alpha, tau) * epsilon + _cmse_L1_dalpha_singleton(0, alpha, tau) * (1.0 - epsilon)

cdef double _cmse_L2_dalpha_singleton(double x, double alpha, double tau):
    return (alpha * x ** 2 - 0.5 * tau ** 2) / (alpha + 0.5) ** 3

cdef double _cmse_L2_dalpha(double alpha, double M, double tau, double epsilon):
    return _cmse_L2_dalpha_singleton(M, alpha, tau) * epsilon + _cmse_L2_dalpha_singleton(0, alpha, tau) * (1.0 - epsilon)

cdef double _cmse_Lq_dalpha_integrand(double z, double x, double alpha, double tau, double q, double tol):
    return 2.0 * tau * (_cproxLq(x + z * tau, alpha * tau ** (2.0 - q), q, tol) - x) * _cproxLq_dt(x / tau + z, alpha, q, tol) * cgaussianPdf(z)

cdef double _cmse_Lq_dalpha_singleton(double x, double alpha, double tau, double q, double tol):
    return quad(_cmse_Lq_dalpha_integrand, -np.inf, np.inf, args=(x, alpha, tau, q, tol))[0]

cdef double _cmse_Lq_dalpha(double alpha, double M, double tau, double epsilon, double q, double tol):
    return _cmse_Lq_dalpha_singleton(M, alpha, tau, q, tol) * epsilon + _cmse_Lq_dalpha_singleton(0, alpha, tau, q, tol) * (1.0 - epsilon)

cdef double cmse_Lq_dalpha(double alpha, double M, double tau, double epsilon, double q, double tol=1e-9):
    if q == 1:
        return _cmse_L1_dalpha(alpha, M, tau, epsilon)
    elif q == 2:
        return _cmse_L2_dalpha(alpha, M, tau, epsilon)
    else:
        return _cmse_Lq_dalpha(alpha, M, tau, epsilon, q, tol)


# Solve for tau
cdef double ctau_of_alpha(double alpha, double M, double epsilon, double delta, double sigma, double q, double tol=1e-9):
    # assume sigma != 0
    cdef double L = sigma
    cdef double incre = sigma
    cdef int Lsign = 1
    cdef double U = L + incre

    while (cmse_Lq(M, alpha, U, epsilon, q, tol) / delta + sigma ** 2 > U ** 2) == Lsign:
        incre = incre * 2
        U = U + incre

    L = U - incre
    cdef double mid = 0

    while U - L > tol:
        mid = (U + L) / 2.0
        if (cmse_Lq(M, alpha, mid, epsilon, q, tol) / delta + sigma ** 2 > mid ** 2) == Lsign:
            L = mid
        else:
            U = mid
    return (U + L) / 2.0

cdef double coptimal_alpha(double M, double epsilon, double delta, double sigma, double q, double tol=1e-9):
    cdef double L = calpha_lb(delta, q, tol)
    cdef double incre = 1.0
    cdef double U = L + incre

    while cmse_Lq_dalpha(U, M, ctau_of_alpha(U, M, epsilon, delta, sigma, q, tol), epsilon, q, tol) < 0:
        incre = incre * 2
        U = U + incre
    L = U - incre

    cdef double mid = 0
    while U - L > tol:
        mid = (U + L) / 2.0
        if cmse_Lq_dalpha(mid, M, ctau_of_alpha(mid, M, epsilon, delta, sigma, q, tol), epsilon, q, tol) < 0:
            L = mid
        else:
            U = mid
    return (U + L) / 2.0

cdef double _cmse_Lq_dtau2_asymp_integrand(double z, double alpha, double q, double tol):
    cdef double eta_abs = _cproxLq(abs(z), alpha, q, tol)
    return (abs(z) - (2.0 - q) * q * alpha * eta_abs ** (q - 1.0)) / (1.0 + alpha * q * (q - 1.0) * eta_abs ** (q - 2.0)) * eta_abs * cgaussianPdf(z)

cdef double _cmse_L1_dtau2_asymp(double alpha):
    return 2 * ((1 + alpha ** 2) * cgaussianCdf(- alpha) - alpha * cgaussianPdf(alpha))

cdef double _cmse_L2_dtau2_asymp(double alpha):
    return 1.0 / (1.0 + 2.0 * alpha) ** 2

cdef double _cmse_Lq_dtau2_asymp(double alpha, double q, double tol):
    return quad(_cmse_Lq_dtau2_asymp_integrand, -np.inf, np.inf, args=(alpha, q, tol))[0]

cdef double calpha_lb(double delta, double q, double tol):
    cdef double L = 0
    cdef double U = 0
    cdef double mid = 0
    cdef double incre = 0
 
    if delta >= 1:
        return 0
    elif q == 1:
        L = 0
        U = 8
        mid = 0
        
        while U - L > tol:
            mid = (U + L) / 2.0
            if _cmse_L1_dtau2_asymp(mid) > delta:
                L = mid
            else:
                U = mid
        return (L + U) / 2.0
    elif q == 2:
        return 0.5 / sqrt(delta) - 0.5
    else:
        L = 0
        incre = 1.0 / sqrt(delta)
        U = L + incre
        mid = 0

        while _cmse_Lq_dtau2_asymp(U, q, tol) > delta:
            incre = incre * 2
            U = U + incre
        L = U - incre

        while U - L > tol:
            mid = (U + L) / 2.0
            if _cmse_Lq_dtau2_asymp(mid, q, tol) > delta:
                L = mid
            else:
                U = mid
        return (L + U) / 2.0

cdef double _clambda_of_alpha_L1_helper_singleton(double x, double alpha, double tau):
    return cgaussianCdf(x / tau - alpha) + cgaussianCdf(- x / tau - alpha)

cdef double _clambda_of_alpha_L1(double M, double alpha, double tau, double epsilon, double delta):
    return (1.0 - (_clambda_of_alpha_L1_helper_singleton(M, alpha, tau) * epsilon + _clambda_of_alpha_L1_helper_singleton(0, alpha, tau) * (1.0 - epsilon)) / delta) * alpha * tau

cdef double _clambda_of_alpha_L2(double alpha, double delta):
    return alpha * (1.0 - 1.0 / delta / (1.0 + 2 * alpha))

cdef double _clambda_of_alpha_Lq_helper_integrand(double z, double x, double alpha, double tau, double q, double tol):
    return _cproxLq_dx(x + z * tau, alpha * tau ** (2.0 - q), q, tol) * cgaussianPdf(z)

cdef double _clambda_of_alpha_Lq_helper_singleton(double x, double alpha, double tau, double q, double tol):
    return quad(_clambda_of_alpha_Lq_helper_integrand, -np.inf, np.inf, args=(x, alpha, tau, q, tol))[0]

cdef double _clambda_of_alpha_Lq(double M, double alpha, double tau, double epsilon, double delta, double q, double tol):
    return (1.0 - (_clambda_of_alpha_Lq_helper_singleton(M, alpha, tau, q, tol) * epsilon + _clambda_of_alpha_Lq_helper_singleton(0, alpha, tau, q, tol) * (1.0 - epsilon)) / delta) * alpha * tau ** (2.0 - q)

cdef double clambda_of_alpha_Lq(double alpha, double M, double epsilon, double delta, double sigma, double q, double tol):
    cdef double tau = ctau_of_alpha(alpha, M, epsilon, delta, sigma, q, tol)
    if q == 1:
        return _clambda_of_alpha_L1(M, alpha, tau, epsilon, delta)
    elif q == 2:
        return _clambda_of_alpha_L2(alpha, delta)
    else:
        return _clambda_of_alpha_Lq(M, alpha, tau, epsilon, delta, q, tol)

cdef double calpha_of_lambda_Lq(double lam, double M, double epsilon, double delta, double sigma, double q, double tol):
    cdef double L = 0
    cdef double U = 0
    cdef double mid = 0
    cdef double incre = 0
 
    if q == 2:
        return 0.25 / delta + 0.5 * lam - 0.25 + sqrt((0.25 - 0.25 / delta - 0.5 * lam) ** 2 + 0.5 * lam)
    else:
        L = calpha_lb(delta, q, tol)
        incre = 1.0 / delta
        U = L + incre
        mid = 0

        while clambda_of_alpha_Lq(U, M, epsilon, delta, sigma, q, tol) < lam:
            incre = incre * 2
            U = U + incre
        L = U - incre

        while U - L > tol:
            mid = (U + L) / 2.0
            if clambda_of_alpha_Lq(mid, M, epsilon, delta, sigma, q, tol) < lam:
                L = mid
            else:
                U = mid
        return (L + U) / 2.0

