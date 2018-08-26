from libc.math cimport sqrt, tgamma, erf, M_E, M_PI, M_SQRT1_2
import numpy as np
cimport numpy as np
from scipy.integrate import quad

cdef double gaussianCdf(double x):
    return 0.5 + 0.5 * erf(x * M_SQRT1_2)

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

