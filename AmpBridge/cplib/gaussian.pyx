from libc.math cimport sqrt, tgamma, M_E, M_PI
import numpy as np
cimport numpy as np
from scipy.integrate import quad


cdef double gaussianCdf(double x):
    cdef double a1 = 0.254829592
    cdef double a2 = -0.284496736
    cdef double a3 = 1.421413741
    cdef double a4 = -1.453152027
    cdef double a5 = 1.061405429
    cdef double p = 0.3275911
    cdef double e = 2.71828182846
    
    cdef int sign = 1
    if x < 0:
        sign = -1
    x = abs(x) * 0.70710678118

    cdef double t = 1.0 / (1.0 + p * x)
    cdef double y = 1.0 - ((((( a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * e ** (-x * x)
    return 0.5 * (1.0 + sign * y)

cdef double gaussianPdf(double x):
    cdef double c = 0.39894228040
    return c * M_E ** (- x ** 2 * 0.5)

cdef double gaussianMoment(double q):
    return tgamma((q + 1.0) / 2.0) * (2 ** (q / 2.0)) / sqrt(M_PI)

def gaussianExpectation(f):
    def tmp(x):
        return f(x) * gaussianPdf(x)
    return quad(tmp, -np.inf, np.inf)[0]

# cdef double ccgaussianExpectation(integrand f):
#     return quad(multiplyGaussianKernel, -np.inf, np.inf, args=(f))[0]

# def cgaussianExpectation(f):
#     return ccgaussianExpectation(f)

