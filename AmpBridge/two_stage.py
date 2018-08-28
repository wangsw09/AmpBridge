from math import sqrt
from .cscalar import gaussianMoment, gaussianPdf, gaussianCdf, proxLq_inv, tau_of_alpha, optimal_alpha

def _vl_(alpha, epsilon):
    '''
    Compute the V value for Lasso
    '''
    return 2 * (1 - epsilon) * gaussianCdf(- alpha)
        
def _tl_(alpha, tau, M, epsilon):
    '''
    Compute the T value for Lasso
    '''
    return epsilon * (gaussianCdf(M / tau - alpha) + gaussianCdf(- M / tau - alpha))

def afdp_atpp_lasso(alpha, M, epsilon, delta, sigma, tol=1e-9):
    tau = tau_of_alpha(alpha, M, q, epsilon, delta, sigma, tol=tol)
    vl = _vl_(alpha, epsilon)
    tl = _tl_(alpha, tau, M, epsilon)
    return (vl / (vl + tl), tl / epsilon)

def _vq_(alpha, tau, epsilon, s, q):
    '''
    Compute the V value for bridge estimator
    '''
    if q == 1:
        return 2 * (1 - epsilon) * gaussianCdf( - s / tau - alpha)
    else:
        return 2 * (1 - epsilon) * gaussianCdf( - proxLq_inv(s / tau, alpha, q))
        
def _tq_(alpha, tau, M, epsilon, s, q):
    '''
    Compute the T value for bridge estimator
    '''
    if q == 1:
        return epsilon * (gaussianCdf((M - s) / tau - alpha) + gaussianCdf(- (M + s) / tau - alpha))
    else:
        return epsilon * (gaussianCdf(M / tau - proxLq_inv(s / tau, alpha, q)) + gaussianCdf(- M / tau - proxLq_inv(s / tau, alpha, q)))

def afdp_atpp_2stage(M, epsilon, delta, sigma, s, q, tol=1e-9):
    alpha = optimal_alpha(M, q, epsilon, delta, sigma, tol=tol)
    tau = tau_of_alpha(alpha, M, q, epsilon, delta, sigma, tol=tol)
    vq = _vq_(alpha, tau, epsilon, s, q)
    tq = _tq_(alpha, tau, M, epsilon, s, q)
    return (vq / (vq + tq), tq / epsilon)

def _vd_(s, tau, epsilon):
    '''
    Compute the V value for debiased bridge estimator
    '''
    return 2 * (1 - epsilon) * gaussianCdf( - s / tau)

def _td_(s, tau, M, epsilon):
    '''
    Compute the T value for debiased bridge estimator
    '''
    return epsilon * (gaussianCdf((M - s) / tau) + gaussianCdf((- s - M) / tau))

def afdp_atpp_db(M, epsilon, delta, sigma, s, q, tol=1e-9):
    alpha = optimal_alpha(M, q, epsilon, delta, sigma, tol=tol)
    tau = tau_of_alpha(alpha, M, q, epsilon, delta, sigma, tol=tol)
    vd = _vd_(s, tau, epsilon)
    td = _td_(s, tau, M, epsilon)
    return (vd / (vd + td), td / epsilon)

def afdp_atpp_sis(M, epsilon, delta, sigma, s):
    tau0 = sqrt(sigma ** 2 + M ** 2 / delta)
    vs = _vd_(s, tau0, epsilon)
    ts = _td_(s, tau0, M, epsilon)
    return (vs / (vs + ts), ts / epsilon)

