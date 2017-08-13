import scipy as sp

from .AMPbasic import *

def mseOptimal(delta, eps, sigma, signl, q, tol=1e-7):
    ath = amp_theory(eps, delta, sigma, signl)
    alpha_opt = ath.alpha_optimal(q)
    tau_opt   = ath.tau_of_alpha(alpha_opt, q)
    return (tau_opt ** 2 - sigma ** 2) * delta

def mseOptApprox1(delta, eps, sigma, signl, q, scenario, tol=1e-7):
    '''
    Compute the first order approximation of optimal tuned MSE
    '''
    if scenario == "large noise":
        return eps * signl.expectation(lambda x: x ** 2)
    elif scenario == "large sample":
        if q == 1:
            return phase_trans2(eps) * sigma ** 2 / delta
        else:
            return sigma ** 2 / delta

def mseOptApprox2(delta, eps, sigma, signl, q, scenario, tol=1e-7):
    '''
    Compute the second order approximation of optimal tuned MSE
    '''
    first_order = mseOptApprox1(delta, eps, sigma, signl, q, scenario, tol)
    if scenario == "large noise":
        c_q = gaussian_moment((2.0 - q) / (q - 1.0)) ** 2 / (q - 1.0) ** 2 / gaussian_moment(2.0 / (q - 1.0))
        return first_order - first_order ** 2 * c_q / sigma ** 2
    elif scenario == "large sample":
        if q < 2:
            return first_order - (1.0 - eps) ** 2 * gaussian_moment(q) ** 2 / eps / signl.expectation(lambda x: np.absolute(x) ** (2 * q - 2.0)) * first_order ** q
        elif q == 2:
            return first_order + first_order * (1.0 - sigma ** 2 / eps / signl.expectation(lambda x: x ** 2)) / delta
        elif q > 2:
            return first_order + first_order * (1.0 - eps * sigma ** 2 * (q - 1.0) ** 2 * signl.expectation(lambda x: np.absolute(x) ** (q - 2.0)) ** 2 / signl.expectation(lambda x: np.absolute(x) ** (2 * q - 2.0))) / delta

def gaussian_moment(q):
    '''
    calculate Gaussian moment E(|Z| ** q), with Z ~ N(0, 1)
    '''
    return 2 ** (q / 2.0) / np.sqrt(np.pi) * sp.special.gamma(q / 2.0 + 0.5)


