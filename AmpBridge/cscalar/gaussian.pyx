from libc.math cimport sqrt, tgamma, erf, M_E, M_PI, M_SQRT1_2, M_2_SQRTPI
import numpy as np
cimport numpy as np
from scipy.integrate import quad

cdef double cgaussianCdf(double x):
    return 0.5 + 0.5 * erf(x * M_SQRT1_2)

cdef double cgaussianPdf(double x):
    return 0.5 * M_SQRT1_2 * M_2_SQRTPI * M_E ** (- x ** 2 * 0.5)

cdef double cgaussianMoment(double q):
    return tgamma((q + 1.0) / 2.0) * (2 ** (q / 2.0)) * M_2_SQRTPI * 0.5

def gaussianExpectation(f):
    def tmp(x):
        return f(x) * cgaussianPdf(x)
    return quad(tmp, -np.inf, np.inf)[0]

# cdef double ccgaussianExpectation(integrand f):
#     return quad(multiplyGaussianKernel, -np.inf, np.inf, args=(f))[0]

# def cgaussianExpectation(f):
#     return ccgaussianExpectation(f)

