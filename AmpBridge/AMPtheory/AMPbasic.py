import sys
import numpy as np
import numpy.random as npr
import scipy.stats as sps
import scipy.integrate as spi

from ..lib.prox_func import *
from ..lib.tools import *
from ..cplib import *


class amp_theory:
    '''
    Compute quantities related to AMP theoretically through the state evolution equation.
    '''
    def __init__(self, eps, delta, sigma, nonzero_dist):
        self.epsilon = eps
        self.delta   = delta
        self.sigma   = sigma
        self.nonzero_dist    = nonzero_dist
        self.dist = ddist(support = nonzero_dist.supp() + [0], prob = [p * eps for p in nonzero_dist.prob()] + [1 - eps])

    def mse_func(self, x, alpha, tau, q):
        '''
        return the function inside MSE part in the state evolution equation.
        parameters:
            x: given beta = x
            alpha: tuning alpha, must be larger than an alpha0
            tau: positive value
        return:
            The value of the integrand of the MSE part
        Detail:
            This function calculates the MSE part in the following equation
            tau ** 2 = sigma ** 2 + Expectation(mse_func(x, alpha, tau)) / delta
        '''
        if q == 1:
            return ( tau ** 2 * (1.0 + alpha ** 2) - x ** 2 ) * (ncdf(x / tau - alpha) + ncdf( - x / tau - alpha)) - (tau * x + tau ** 2 * alpha) * npdf(alpha - x / tau) + (tau * x - tau ** 2 * alpha) * npdf(alpha + x / tau) + x ** 2
        elif q == 2:
            return (tau ** 2.0 + 4 * alpha ** 2.0 * x ** 2.0) / (1.0 + 2.0 * alpha) ** 2
        else:
            temp_func = lambda z: (prox_Lq(x + tau * z, t = alpha * tau ** (2.0 - q), q = q) - x) ** 2 * npdf(z)
            return spi.quad(temp_func, - 5, 5)[0]  # the integral limit may need adjustment for accuracy

    def mse(self, alpha, tau, q):
        '''
        return the MSE in the state evolution
        parameters:
            alpha: tuning alpha, must be larger than an alpha0
            tau: positive value
        return:
            The value of the MSE part
        Detail:
            This function calculates the MSE part in the following equation
            tau ** 2 = sigma ** 2 + MSE(alpha, tau) / delta
        '''
        return self.dist.expectation(self.mse_func, alpha=alpha, tau=tau, q = q)

    def mse_derivative_func(self, x, alpha, tau, q):
        '''
        Compute the derivative of mse_func(x, alpha, tau) w.r.t. alpha
        '''
        if q == 1:
            return 2.0 * tau ** 2 * alpha * (ncdf(x / tau - alpha) + ncdf( - x / tau - alpha)) - 2.0 * tau ** 2 * (npdf(alpha - x / tau) + npdf(alpha + x / tau) )
        elif q == 2:
            return (alpha * x ** 2.0 - 0.5 * tau ** 2) / (alpha + 0.5) ** 3
        else:
            temp_func = lambda z: 2.0 * tau * (prox_Lq(x + z * tau, alpha * tau ** (2.0 - q), q) - x) * prox_Lq_drvt_t(x / tau + z, alpha, q) * npdf(z)
            return spi.quad(temp_func, - 5, 5)[0]  # the integral limit may need adjustment for accuracy

    def mse_derivative(self, alpha, tau, q):
        '''
        Compute the derivative of mse(alpha, tau) w.r.t. alpha
        '''
        return self.dist.expectation(self.mse_derivative_func, alpha=alpha, tau=tau, q = q)

    def tau_of_alpha(self, alpha, q, tol=1e-5):  # need to further modify the upper bound for search
        '''
        Compute tau from alpha. Deduced from state evolution equation
        '''
#        if alpha <= self.alpha_lower_bound(category = 'rough', q = q):
#            raise ValueError('alpha is smaller than its lower bound to find tau')
#       we need to consider the case when sigma=0 for q<=1, calculate phase transition
        temp_func = lambda tau: self.mse(alpha, tau, q) / self.delta + self.sigma ** 2 - tau ** 2
        return bisect_search(temp_func, lower=self.sigma, unit=self.sigma, accuracy = tol)

    def alpha_optimal(self, q):
        temp_func = lambda alpha: self.mse_derivative(alpha, self.tau_of_alpha(alpha, q), q)
        opt_val = bisect_search(temp_func, lower = self.alpha_lower_bound('rough', q) + 0.01, unit = self.sigma)  # the lower bound and upper bound here need to be modified
        return opt_val

    def alpha_lower_bound(self, category, q):
        '''
        find two kinds of lower bound for alpha        
        '''
        if category == 'rough': # result correct
            if self.delta >= 1:
                ans = 0
            else:
                if q == 1:
                    temp_func = lambda x: (1 + x ** 2) * ncdf( - x) - x * npdf(x) - self.delta / 2.0
                    ans = bisect_search(temp_func, lower = 0, upper = 6)  # the 6 here is definitely fine
                elif q == 2:
                    return 0.5 / self.delta ** 0.5 - 0.5
                else:
                    temp_func1 = lambda a: lambda z: prox_Lq(z, a, q) ** 2 * npdf(z)
                    temp_func2 = lambda a: spi.quad(temp_func1(a), -np.inf, np.inf)[0] - self.delta
                    ans = bisect_search(temp_func2, lower = 0, unit = 0.5)
            return ans
        
        elif category == 'accurate':
            if q == 2:
                return max(0.5 / self.delta - 0.5, 0.0)
            ans = self.alpha_of_lambda(0, q)
            return ans

    def lambda_of_alpha(self, alpha, q):
        '''
        the tuning parameter of Lasso as a function of alpha for AMP
        '''
        tau = self.tau_of_alpha(alpha, q)
        if q == 1:
            temp_func = lambda x: ncdf(x / tau - alpha) + ncdf( - x / tau - alpha)
            return alpha * tau * (1.0 - self.dist.expectation(temp_func) / self.delta)
        elif q == 2:
            return alpha * (1.0 - 1.0 / self.delta / (1.0 + 2.0 * alpha))
        else:
            temp_func1 = lambda x: lambda z: prox_Lq_drvt(x + tau * z, alpha * tau ** (2 - q), q) * npdf(z)
            temp_func2 = lambda x: spi.quad(temp_func1(x), -np.inf, np.inf)[0]
            return alpha * tau ** (2 - q) * (1 - self.dist.expectation(temp_func2) / self.delta)
    
    def alpha_of_lambda(self, lam, q):
        '''
        the inversion of the method 'lambda_of_alpha'
        '''
        temp_func = lambda x: self.lambda_of_alpha(x, q) - lam
        return bisect_search(temp_func, lower = self.alpha_lower_bound(category='rough', q = q) + 0.01, unit = 0.5, accuracy=1e-3)
    
    def __VL__(self, alpha):
        '''
        Compute the V value for Lasso
        '''
        return 2 * (1 - self.epsilon) * sps.norm.cdf( - alpha)
        
    def __TL__(self, alpha, tau):
        '''
        Compute the T value for Lasso
        '''
        temp_func = lambda x: sps.norm.cdf(x / tau - alpha) + sps.norm.cdf(- x / tau - alpha)
        return self.epsilon * self.nonzero_dist.expectation(temp_func)
    
    def __Vq__(self, alpha, tau, q, s):
        '''
        Compute the V value for bridge estimator
        '''
        if q == 1:
            return 2 * (1 - self.epsilon) * sps.norm.cdf( - s / tau - alpha)
        else:
            return 2 * (1 - self.epsilon) * sps.norm.cdf( - eta_inv(s / tau, alpha, q))
        
    def __Tq__(self, alpha, tau, q, s):
        '''
        Compute the T value for bridge estimator
        '''
        if q == 1:
            temp_func = lambda x: sps.norm.cdf((x - s) / tau - alpha) + sps.norm.cdf(- (x + s) / tau - alpha)
            return self.epsilon * self.nonzero_dist.expectation(temp_func)
        else:
            temp_func = lambda x: sps.norm.cdf(x / tau - eta_inv(s / tau, alpha, q)) + sps.norm.cdf(- x / tau - eta_inv(s / tau, alpha, q))
            return self.epsilon * self.nonzero_dist.expectation(temp_func)
    
    def __Vd__(self, s, tau):
        '''
        Compute the V value for debiased bridge estimator
        '''
        return 2 * (1 - self.epsilon) * sps.norm.cdf( - s / tau)
    
    def __Td__(self, s, tau):
        '''
        Compute the T value for debiased bridge estimator
        '''
        temp_func = lambda x: sps.norm.cdf((x - s) / tau) + sps.norm.cdf((- s - x) / tau)
        return self.epsilon * self.nonzero_dist.expectation(temp_func)
        
    def tpp_fdp(self, method, **kwargs):
        '''
        Compute (atpp, afdp)
        Parameters:
            method: when method = "one stage", use Lasso and change tuning;
                    when method = "two stage", threshold bridge estimator;
                    when method = "two stage-db", threshold debiased bridge estimator;
                    when method = "sis", use SURE independent screening.

            kwargs: when method = "one stage", kwargs is alpha_arr, with type numpy.ndarray
                    when method = "two stage", kwargs is q and s_arr with type numpy.ndarray
                    when method = "two stage-db", kwargs is q and s_arr with type numpy.ndarray
                    when method = "sis", kwargs is s_arr with type numpy.ndarray
        Value:
            ** when passed single value, single pair is returned (to be implemented);
            when passed array, return (ATPP_array, AFDP_array), where both array are of length equal
            to that of alpha_arr or s_arr.
        '''
        if method == 'one stage':
            k = len(kwargs['alpha'])
            v = np.zeros(k)
            t = np.zeros(k)
            for i in xrange(k):
                v[i] = self.__VL__(kwargs['alpha'][i])
                t[i] = self.__TL__(kwargs['alpha'][i], self.tau_of_alpha(kwargs['alpha'][i], 1))
            return (t / self.epsilon, v / (v + t))

        elif method == 'two stage':
            alpha = self.alpha_optimal(kwargs['q'])
            tau = self.tau_of_alpha(alpha, kwargs['q'])
            k = len(kwargs['s'])
            v = np.zeros(k)
            t = np.zeros(k)
            for i in xrange(k):
                v[i] = self.__Vq__(alpha, tau, kwargs['q'], kwargs['s'][i])
                t[i] = self.__Tq__(alpha, tau, kwargs['q'], kwargs['s'][i])
            return (t / self.epsilon, v / (v + t))

        elif method == 'two stage-dbs':
            alpha = self.alpha_optimal(kwargs['q'])
            tau = self.tau_of_alpha(alpha, kwargs['q'])
            k = len(kwargs['s'])
            v = np.zeros(k)
            t = np.zeros(k)
            for i in xrange(k):
                v[i] = self.__Vd__(kwargs['s'][i], tau)
                t[i] = self.__Td__(kwargs['s'][i], tau)
            return (t / self.epsilon, v / (v + t))

        elif method == "sis":
            tau_inf = np.sqrt(self.sigma ** 2 + 1.0 / self.delta * self.dist.expectation(lambda x: x ** 2))
            k = len(kwargs['s'])
            v = np.zeros(k)
            t = np.zeros(k)
            for i in xrange(k):
                v[i] = self.__Vd__(kwargs['s'][i], tau_inf)
                t[i] = self.__Td__(kwargs['s'][i], tau_inf)
            return (t / self.epsilon, v / (v + t))


def phase_trans1(delta, tol = 1e-7):
    '''
    Compute the phase transition of Lasso given delta
    Parameter:
        delta: delta >= 0
    Values:
        epsilon
    '''
    if delta >= 1:
        return 1
    else:
        # use the parametrized representation of (delta, epsilon) in terms of alpha
        tmp = lambda alpha: 2 * sps.norm.pdf(alpha) / (alpha + 2 * (sps.norm.pdf(alpha) - alpha * sps.norm.cdf( - alpha)))
        low = 0
        upp = 10
        while upp - low > tol:
            mid = (upp + low) / 2.0
            if tmp(mid) > delta:
                low = mid
            else:
                upp = mid
        mid = (upp + low) / 2.0
        return 2 * (sps.norm.pdf(mid) - mid * sps.norm.cdf(-mid)) / (mid + 2 * (sps.norm.pdf(mid) - mid * sps.norm.cdf( - mid)))

def phase_trans2(epsilon, tol = 1e-7):
    '''
    Compute the phase transition of Lasso given epsilon
    Parameter:
        epsilon: 1 >= epsilon >= 0
    Values:
        delta
    '''
    tmp = lambda alpha: 2 * (sps.norm.pdf(alpha) - alpha * sps.norm.cdf(-alpha)) / (alpha + 2 * (sps.norm.pdf(alpha) - alpha * sps.norm.cdf( - alpha)))
    low = 0
    upp = 10
    while upp - low > tol:
        mid = (upp + low) / 2.0
        if tmp(mid) > epsilon:
            low = mid
        else:
            upp = mid
    mid = (upp + low) / 2.0
    return 2 * (1 - epsilon) * sps.norm.cdf(-mid) + epsilon

def atpp_afdp(eps, alpha, tau, signal, q, s):
    '''
    return (atpp, afdp) for two-stage biased bridge estiamtor.
    '''
    if eps <= 0 or alpha <= 0 or tau <= 0 or q < 1 or s < 0:
        raise ValueError('Value of arguments does not satisfy condition;')

    if q == 1:
        quant1 = 2 * sps.norm.cdf(-alpha - s / tau)
        tmp_func = lambda G: sps.norm.cdf( - s / tau - alpha + G / tau) + sps.norm.cdf( - s / tau - alpha - G / tau)
        quant2 = signal.expectation(tmp_func)
    else:
        quant1 = 2 * sps.norm.cdf( - eta_inv(s / tau, alpha, q))
        tmp_func = lambda G: sps.norm.cdf( - eta_inv(s / tau, alpha, q) + G / tau) + sps.norm.cdf( - eta_inv(s / tau, alpha, q) - G / tau)
        quant2 = signal.expectation(tmp_func)

    afdp = (1.0 - eps) * quant1 / ((1.0 - eps) * quant1 + eps * quant2)
    return (quant2, afdp)
