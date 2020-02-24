import numpy as np
from scipy.integrate import quad
from .cscalar.wrapper import gaussianMoment, gaussianPdf, gaussianCdf

class MseExpansion(object):
    def __init__(self):
        pass

    @classmethod
    def large_noise0(self, epsilon, M):
        return epsilon * M ** 2

    def Cq(self, q):
        return gaussianMoment((2.0 - q) / (q - 1.0)) ** 2 / (q - 1.0) ** 2 / gaussianMoment(2.0 / (q - 1.0))

    def large_noise1(self, epsilon, M, q, sigma):
        """
        Only applied to q > 1.
        """
        return - epsilon ** 2 * M ** 4 * self.Cq(q) / sigma ** 2

    def extreme_sparse0(self, epsilon, M):
        return epsolon * M ** 2

    def extreme_sparse1(self, epsilon, M, q, sigma):
        numerator_integrand = lambda z: abs(M / sigma + z) ** (1.0 / (q -
            1.0)) * ((M / sigma + z > 0) * 2 - 1) * gaussianPdf(z)
        return - epsilon ** 2 * M ** 2 * quad(numerator_integrand, -np.inf, np.inf, tol=1e-9)[0] / gaussianMoment(2.0 / (q - 1.0))

    def large_sample1(self, epsilon, delta, sigma, q):
        if q == 1:
            return self.M_1(epsilon, tol=1e-9) * sigma ** 2 / delta
        else:
            return sigma ** 2 / delta

    def large_sample2(self, M, epsilon, delta, sigma, q):
        if q < 2:
            return - (sigma ** 2 / delta) ** q * (1 - epsilon) ** 2 / epsilon * gaussianMoment(q) ** 2 / M ** (2 * q - 2)
        elif q == 2:
            return sigma ** 2 / delta ** 2 * (1.0 - sigma ** 2 / epsilon / M ** 2)
        else:
            return sigma ** 2 / delta ** 2 * (1.0 - epsilon * (q - 1.0) ** 2 * sigma ** 2 / M ** 2)

    def M_1(self, epsilon, tol):
        L = 0
        U = 8
        while U - L > tol:
            mid = (U + L) / 2.0
            if 2 * (1 - epsilon) * (mid * gaussianCdf(-mid) - gaussianPdf(mid)) + epsilon * mid < 0:
                L = mid
            else:
                U = mid
        mid = (U + L) / 2.0  # mid is the value of the minimizer
        return (1.0 - epsilon) * (2.0 * (1.0 + mid ** 2) * gaussianCdf(-mid) - 2.0 * mid * gaussianPdf(mid)) + epsilon * (1.0 + mid ** 2)

