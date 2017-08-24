## modules
from __future__ import print_function
import sys
import numpy as np
import numpy.linalg as npla
import scipy.stats as sps
import sklearn.linear_model as skll

from .lib.prox_func import *
from .lib.tools import *
from .cplib import *


# define a class: linear model
# the input of the class: y, X
# public attributes: y, X, delta
# private attributes: linear regression result, lasso result, lars result, beta_true=None, epsilon, sigma, M(?)
# these private attributes, default set to be None. When calling them,
# if they are None, they call the corresponding function and calculate them.
# store the results to these private attributes (not None anymore).
# Whenever we need to use the result, just detect the private. Not None, use it.
# beta_true is set to be a specific vector if provided. If None, then no true model provided.

class linear_model:
    '''
    construct a linear model and execute various estimation and VS algorithms on it.
    '''
    def __init__(self, y, X):
        self.y = y
        self.X = X
        self.shape = X.shape
        self.delta = X.shape[0] / float(X.shape[1])
        self.sigma = None         # private
        self.signal = None        # private, seems unnecessary
        self.epsilon = None       # private
        self.beta_true = None     # private
        self.X_norm2 = npla.norm(X, axis=0) ** 2
        self.Xy = np.dot(self.X.T, self.y) / self.X_norm2
        self.XX = np.dot(self.X.T, self.X) / self.X_norm2

        self.lasso_result = None  # private
        self.beta_prev    = None
        
        self.amp_result    = {}    # private
        self.bridge_result = {}
        self.slope_result  = {}
        self.S2FDR_result  = {}
        
        self.amp_optimal   = {}
        self.optimal_alpha = {}
        self.optimal_tau   = {}

    def amp(self, alpha, q, threshold = 1e-6, iter_max = 500):
        n, p  = self.shape
        
        beta0 = np.zeros(p)
        z     = self.y[ : ]
        tau   = npla.norm(z) / np.sqrt(n)  # default is l2 norm
        beta  = eta(beta0 + np.dot(self.X.T, z), alpha * tau ** (2.0 - q), q)
        
        iter_count = 0
        
        while npla.norm(beta - beta0) / npla.norm(beta) > threshold:
            if q in (1.0, 2.0, 1.5):
                z     = (self.y - np.dot(self.X, beta)
                     + np.mean(eta_derivative(beta0 + np.dot(self.X.T, z), alpha * tau ** (2.0 - q), q)) / self.delta * z)
            else:
                z = (self.y - np.dot(self.X, beta)
                     + np.mean(1.0 / (1.0 + alpha * tau ** (2.0 - q) * q * (q - 1.0) * np.absolute(beta) ** (q - 2))) / self.delta * z)
            tau   = npla.norm(z) / np.sqrt(n)
            beta0 = beta
            beta  = eta(beta0 + np.dot(self.X.T, z), alpha * tau ** (2.0 - q), q)
            
            iter_count += 1
            if iter_count >= iter_max:
                break

        self.amp_result[(q, alpha)] = (beta, z, tau, iter_count)
        return (beta, z, tau, iter_count)
    
    def two_stage_amp(self, q, s, alpha=None, threshold = 1e-6, iter_max = 500):
        '''
         -- if <s> is an array, return a dictionary in the form: {(q, s[i]) : beta_vec}
         -- if <s> is a scalar, return the beta_vec
        '''
        if alpha != None:
            # since we need optimal tuning for two-stage, is this part necessary?
            if (q, alpha) in self.amp_result:
                amp_fit = self.amp_result[(q, alpha)]
            else:
                amp_fit = self.amp(alpha, q, threshold, iter_max)
            temp = amp_fit[0] + np.dot(self.X.T, amp_fit[1])
            return temp * (np.absolute(temp) > s).astype(int)
        else:
            if q not in self.optimal_alpha:
                self.optimal_tuning(q)
            amp_fit = self.amp(self.optimal_alpha[q], q, threshold, iter_max)
            temp = amp_fit[0] + np.dot(self.X.T, amp_fit[1])
            if type(s) != np.ndarray:
                return temp * (np.absolute(temp) > s).astype(int)
            else:
                ret = {}
                for ss in s:
                    ret[(q, ss)] = temp * (np.absolute(temp) > ss).astype(int)
                return ret
    
    def empirical_mse(self, u, theta, tau_old, q):
        '''
        empirical mse from each iteration
        u = beta + np.dot(X.T, z)
        theta = gamma * tau_new, gamma is equivalent to alpha
        tau_old = npla.norm(z) / np.sqrt(n)
        '''
        if q in (1, 1.5, 2):
            return np.mean((eta(u, theta, q) - u) ** 2) - tau_old ** 2 + 2 * tau_old ** 2 * np.mean(eta_derivative(u, theta, q))
        else:
            v = eta(u, theta, q)
            return np.mean((v - u) ** 2) - tau_old ** 2 + 2 * tau_old ** 2 * np.mean(1.0 / (1.0 + theta * q * (q - 1.0) * np.absolute(v) ** (q - 2.0)))
    
    def __otbisect__(self, u, tau_old, Delta, u_abmax, q, max_iter = 20):
        '''
        Optimal Tuning assistent function through Bisection Search
        search for optimal gamma
        '''
        gamma_upper = u_abmax / float(tau_old)
        gamma_lower = 0.0
        for i in range(15):
            gamma = (gamma_upper + gamma_lower) / 2.0
            diff = (self.empirical_mse(u, tau_old ** (2.0 - q) * (gamma + Delta), tau_old, q) - self.empirical_mse(u, tau_old ** (2.0 - q) * gamma, tau_old, q)) / Delta
            if diff > 0:
                gamma_upper = gamma
            elif diff < 0:
                gamma_lower = gamma
            else:
                break
        return (gamma_upper + gamma_lower) / 2.0
    
    def otbisect(self, u, tau_old, Delta, u_abmax, q, max_iter = 20, plot=False):
        if plot:
            ga_list = np.arange(0.0, 5.0, 0.05)
            mmse_list = []
            for gamma in ga_list:
                mmse_list.append(self.empirical_mse(u, tau_old ** (2.0 - q) * gamma, tau_old, q))
#            plt.plot(ga_list, mmse_list)
#            plt.show()
        
        if type(Delta) == np.ndarray:
            gamma_list = []
            mse_list = []
            for Dt in Delta:
                gamma_list.append(self.__otbisect__(u, tau_old, Dt, u_abmax, q, max_iter))
                mse_list.append(self.empirical_mse(u, tau_old ** (2.0 - q) * gamma_list[-1], tau_old, q))
            i = np.argmin(mse_list)
            return (gamma_list[i], Delta[i])
        else:
            return self.__otbisect__(u, tau_old, Delta, u_abmax, q, max_iter)
            

    def otDelta(self, u, tau_old, u_abmax, q, L1 = 5.0, L2 = 7.0):
        D_list = 5.0  ** np.arange(0, L2, 1.0) * 10.0 ** (- L1)
        mse_list = np.repeat(-1.0, len(D_list))
        gamma_list = np.repeat(-1.0, len(D_list))
       
        for i, Delta in enumerate(D_list):
            gamma_list[i] = self.otbisect(u, tau_old, Delta, u_abmax, q)
            mse_list[i] = self.empirical_mse(u, tau_old ** (2.0 - q) * gamma_list[i], tau_old, q)
        index = np.argmin(mse_list)
        if mse_list[index] < 0:
            raise ValueError('MSE can not be negative0')
            return None
        return (D_list[index], gamma_list[i])       

    def optimal_tuning(self, q, tol = 1e-5, iter_max = 100):
        '''
        a second version of optimal tuning
        maybe provide an argument to suppress the print() information
        '''
        if q in self.amp_optimal:
            return self.amp_optimal[q]
        
        n, p = self.shape
        
        tau = []
        gamma = []
        
        beta0 = np.zeros(p)
        z = self.y[ : ]
        tau.append(npla.norm(z) / np.sqrt(n))
        
        U = beta0 + np.dot(self.X.T, z)
        U_abmax = np.amax(np.absolute(U))
        init = self.otDelta(U, tau[-1], U_abmax, q)
        Delta = init[0]
        gamma.append(init[1])
        beta = eta(U, gamma[-1] * tau[-1] ** (2.0 - q), q)
        
        iter_count = 0
        
        print('-- outer loop --', sep='', end='', file=sys.stderr)
        while npla.norm(beta - beta0) / npla.norm(beta) > tol:
            if q in (1.0, 2.0, 1.5):
                z     = (self.y - np.dot(self.X, beta)
                     + np.mean(eta_derivative(U, gamma[-1] * tau[-1] ** (2.0 - q), q)) / self.delta * z)
            else:
                z     = (self.y - np.dot(self.X, beta)
                     + np.mean(1.0 / (1.0 + gamma[-1] * tau[-1] ** (2.0 - q) * q * (q - 1.0) * np.absolute(U) ** (q - 2))) / self.delta * z)
            tau.append(npla.norm(z) / np.sqrt(n))
            
            beta0 = beta
            U = beta0 + np.dot(self.X.T, z)
            U_abmax = np.amax(np.absolute(U))
#            if iter_count % 5 == 0:
#                temp = self.otbisect(U, tau[-1], 5.0  ** np.arange(0, 7.0, 1.0) * 10.0 ** (- 5), U_abmax, q, plot=True)
#                gamma.append(temp[0])
#                Delta = temp[1]
#            else:
            gamma.append(self.otbisect(U, tau[-1], Delta, U_abmax, q))
    
            beta = eta(U, gamma[-1] * tau[-1] ** (2.0 - q), q)
            
            iter_count += 1
            if iter_count % 5 == 0:
                print(' ', iter_count, ' --', sep='', end='', file=sys.stderr)
            if iter_count >= iter_max:
                break
        print('\n', sep='', end='', file=sys.stderr)
            
        tau = np.array(tau)
        gamma = np.array(gamma)
        self.amp_optimal[q] = (beta, z, tau, gamma, iter_count)
        self.optimal_alpha[q] = np.mean(gamma[-10 : ])
        self.optimal_tau[q] = np.mean(tau[-10 : ])
        return (beta, z, tau, gamma, iter_count)  # make these returns simpler
    
    def bridge(self, lam, q, tol=1e-6, iter_max = 1000, initial = None, store=True):
        '''
        run bridge regression by coordinate descent.
        currently deal with any q >= 1
        I really think I should lower the precision to maybe 1e-3
        '''

        n, p  = self.shape

        iter_count = 0

        if initial is None:
            beta0 = np.zeros(p) - 1.0  # initialization
            beta  = np.zeros(p)  # picked to pass 1st loop
            grad = np.copy(self.Xy)
        else:
            beta0 = np.copy(initial) - 1.0
            beta  = np.copy(initial)
            grad  = np.copy(self.Xy) - np.dot(self.XX.T, beta) + beta

        beta_max = 1.0

        while npla.norm(beta - beta0, np.inf) / beta_max > tol:
            # this is not a good rule when < beta = 0 >
            np.copyto(beta0, beta)

            # update beta
            for i in xrange(p):

                # update beta[i]
                beta[i] = eta(grad[i], lam / self.X_norm2[i], q)

                # update grad after beta[i]
                grad -= self.XX[i, :] * (beta[i] - beta0[i])
                grad[i] += beta[i] - beta0[i]

            beta_max = npla.norm(beta, np.inf)

            iter_count += 1
            if iter_count > iter_max:
                print('warning: iter_max break', file=sys.stderr)
                break

        if self.beta_prev is None:
            self.beta_prev = np.copy(beta)
        else:
            np.copyto(self.beta_prev, beta)

        return beta

    def bridge1(self, lam, q, tol=1e-6, iter_max = 1000, initial = None, store=True):
        '''
        run bridge regression by coordinate descent.
        currently deal with any q >= 1
        I really think I should lower the precision to maybe 1e-3
        '''

        n, p  = self.shape

        iter_count = 0

        if initial is None:
            beta0 = np.zeros(p) - 1.0  # initialization
            beta  = np.zeros(p)  # picked to pass 1st loop
            grad = np.copy(self.Xy)
        else:
            beta0 = np.copy(initial) - 1.0
            beta  = np.copy(initial)
            grad  = np.copy(self.Xy) - np.dot(self.XX.T, beta) + beta

        beta_max = 1.0

        while npla.norm(beta - beta0, np.inf) / beta_max > tol:
            # this is not a good rule when < beta = 0 >
            np.copyto(beta0, beta)

            # update beta
            for i in xrange(p):

                # update beta[i]
                beta[i] = prox_Lq(grad[i], lam / self.X_norm2[i], q)

                # update grad after beta[i]
                grad -= self.XX[i, :] * (beta[i] - beta0[i])
                grad[i] += beta[i] - beta0[i]

            beta_max = npla.norm(beta, np.inf)

            iter_count += 1
            if iter_count > iter_max:
                print('warning: iter_max break', file=sys.stderr)
                break

        if self.beta_prev is None:
            self.beta_prev = np.copy(beta)
        else:
            np.copyto(self.beta_prev, beta)

        return beta
    
    def cbridge(self, lam, q, tol=1e-6, iter_max = 1000, initial = None, store=True):
        '''
        run bridge regression by coordinate descent.
        currently deal with any q >= 1
        I really think I should lower the precision to maybe 1e-3
        '''

        n, p  = self.shape

        if initial:
            beta_init = initial
        else:
            beta_init = np.zeros(p, dtype=np.float64)
        
        return cbridge_decay(self.XX, self.Xy, self.X_norm2, lam, q, beta_init, rel_tol=1e-6)

#    def ccbridge(self, lam, q, tol=1e-6, iter_max = 1000, initial = None, store=True):
#        '''
#        run bridge regression by coordinate descent.
#        currently deal with any q >= 1
#        I really think I should lower the precision to maybe 1e-3
#        '''
#
#        n, p  = self.shape
#
#        if initial:
#            beta_init = initial
#        else:
#            beta_init = np.zeros(p, dtype=np.float64)
#        
#        return cccbridge(self.XX, self.Xy, self.X_norm2, lam, q, beta_init, rel_tol=1e-6)
 
    def bridge_mse(self, lam, q, initial = None):
        '''
        estimate MSE of bridge estimator by debiasing.
        This is not really MSE. It is the (tau ** 2).
        It can also be used for optimal tuning.
        '''
        if q < 1:
            raise ValueError('q must be greater than or equal to 1.')

        if q == 1:  ## new add part, separate L1
            if type(lam) is not np.ndarray:
                clf = skll.Lasso(lam / float(len(self.y)), fit_intercept=False, max_iter=6000)
                # clf = skll.LassoLars(lam / float(len(self.y)), fit_intercept=False, max_iter=3000, fit_path=False)
                clf.fit(self.X, self.y)
                beta_hat = clf.coef_
                
                if np.sum(beta_hat != 0) >= len(self.y):
                    return np.inf
            else:
                _, beta_list, _ = skll.lasso_path(self.X, self.y, alphas=lam / float(len(self.y)), fit_intercept = False, iter_max=6000)
                mse_arr = np.repeat(np.inf, len(lam))
                for i in xrange(len(lam)):
                    if np.sum(beta_list[:, i] != 0) < len(self.y):
                        z = (self.y - np.dot(self.X, beta_list[:, i])) / ( 1.0 - np.mean(beta_list[:, i] != 0) / self.delta )
                        mse_arr[i] = np.mean(z ** 2)
                        # print("lambda: {0}, mse: {1}".format(lam[i], mse_arr[i]))
                return mse_arr

        elif q == 2:
            reg = skll.Ridge(lam * 2.0, fit_intercept=False, tol=0.00001)
            reg.fit(self.X, self.y)
            beta_hat = reg.coef_
        else:
            #if (q, lam) not in self.bridge_result:
            #    self.bridge(lam, q, initial=initial, iter_max = 600) ## MARK!!! 600!!
            beta_hat = self.bridge(lam, q, iter_max=2000, initial=initial)

        if q == 1:
            z = (self.y - np.dot(self.X, beta_hat)) / ( 1.0 - np.mean(beta_hat != 0) / self.delta )
        else:
            gamma = db_gamma(beta_hat, lam, q, self.delta)
            z = (self.y - np.dot(self.X, beta_hat)) / ( 1.0 - np.mean(1.0 / (1.0 + gamma * q * (q - 1.0) * np.absolute(beta_hat) ** (q - 2))) / self.delta )
        
        return np.mean(z ** 2)
    
    def bridge_dbs(self, beta_hat, lam, q):
        '''
        Return the debiased estimator of beta_hat.
        Parameters:
            beta_hat: the estimated coefficients to debias
            lam, q  : tuning and penalty of beta_hat.
                      More accurately, beta_hat = bridge(lam, q)
        Value:
            beta_hat_dbs: debiased version of beta_hat

        This function is based on the estimated beta_hat,
        and does not check the validity of beta_hat. Please
        make sure the estiamtor is good before debiasing.
        '''
        if q < 1:
            raise ValueError('q must be greater than or equal to 1.')

        if q == 1:
            z = (self.y - np.dot(self.X, beta_hat)) / ( 1.0 - np.mean(beta_hat != 0) / self.delta )
        else:
            gamma = db_gamma(beta_hat, lam, q, self.delta)
            z = (self.y - np.dot(self.X, beta_hat)) / ( 1.0 - np.mean(1.0 / (1.0 + gamma * q * (q - 1.0) * np.absolute(beta_hat) ** (q - 2))) / self.delta )

        beta_hat_dbs = beta_hat + np.dot(self.X.T, z)
        
        return beta_hat_dbs
    
    def bridge_optimalTune(self, q):
        '''
        search for the optimal tuning among lambda_arr
        arguments:
            q: q >= 1, Lq bridge
            lambda_arr: array of tuning, must be arranged in decreasing order
        value:
            mse_arr: return the mse array correspond to lambda_arr
            lambda_opt: optimal lambda among lambda_arr
            mse_opt: optimal MSE correspond to lambda_opt
            beta_opt: the estimator correspond to lambda_opt
        '''


        if q < 1: ## we can consider remove some of these conditions
            raise ValueError('q must be greater than or equal to 1.')
        
        lam_unit  = npla.norm(np.dot(self.X.T, self.y), np.inf) / 2.0
        lam_low   = 0.1
        lam_upp   = 0.1 # lam_low, lam_opt specified to pass the 1st while & if.
        grid_size = 15

        mse_min0  = 0
        mse_min   = 100000.0
        i_opt     = 0
        lam_opt0  = 0
        lam_opt   = 0

        while i_opt == grid_size - 1 or i_opt == 0:
            if np.absolute(mse_min0 - mse_min) / mse_min < 0.01:
                return lam_opt0
            mse_min0 = mse_min
            lam_opt0 = lam_opt
            
            if i_opt == grid_size - 1:
                lam_upp =  lam_low
                lam_low /= 10.0
            else:
                lam_low =  lam_upp
                lam_upp += lam_unit
 
            grid = np.linspace(lam_upp, lam_low, grid_size) # by default end-pt included
            # the reason to make it decreasing is
            # directly calculating for small value is unstable and slow
            # start from large and warmly initialize the small can stablize it

            if q == 1:
                mse_arr = self.bridge_mse(grid, 1)
                i_opt   = np.argmin(mse_arr)
                mse_min = mse_arr[i_opt]
                lam_opt = grid[i_opt]
            else:
                for j in xrange(grid_size):
                    mse_tmp = self.bridge_mse(grid[j], q, initial=self.beta_prev)
                    # print("{0} ###### {1}".format(grid[j], mse_tmp))
                    if mse_tmp < mse_min:
                        i_opt   = j
                        mse_min = mse_tmp
                lam_opt = grid[i_opt]
                
            # print("low: {0}  upp: {1}  opt: {2}".format(lam_low, lam_upp, lam_opt), file=sys.stderr)

        return lam_opt
    
    def xbridge_optimalTune(self, q, lam0, Delta = 0.05, tol=0.1):
        '''
        search for the optimal tuning among lambda_arr
        arguments:
            q: q >= 1, Lq bridge
            lambda_arr: array of tuning, must be arranged in decreasing order
        value:
            mse_arr: return the mse array correspond to lambda_arr
            lambda_opt: optimal lambda among lambda_arr
            mse_opt: optimal MSE correspond to lambda_opt
            beta_opt: the estimator correspond to lambda_opt
        '''


        if q < 1:
            raise ValueError('q must be greater than or equal to 1.')

#        h = 0.05
#        lam = lam0
#        lam0 = lam0 + 1
#        count = 1.0
#
#        while (np.absolute(lam - lam0) > tol):
#            lam0 = lam
#            deri = (self.bridge_mse(lam0 + h, q) - self.bridge_mse(lam0 - h, q)) / 2.0 / h
#            lam = max(lam0 - deri / count, 2 * h)
#            count += 1.0
#
#        return lam
        

        if self.bridge_mse(lam0 + Delta, q) - self.bridge_mse(lam0, q) > 0:
            lamR = lam0
            lamL = lam0 / 2.0

            while ((self.bridge_mse(lamL + Delta, q) - self.bridge_mse(lamL, q)) > 0):
                lamR = lamL
                lamL = lamL / 2.0
        else:
            lamL = lam0
            lamR = lam0 * 2.0

            while ((self.bridge_mse(lamR + Delta, q) - self.bridge_mse(lamR, q)) < 0):
                lamL = lamR
                lamR = lamR * 2.0

        while (lamR - lamL) > tol:
            lamM = (lamL + lamR) / 2.0
            if (self.bridge_mse(lamM + Delta, q) - self.bridge_mse(lamM, q)) < 0:
                lamL = lamM
            else:
                lamR = lamM

        return (lamL + lamR) / 2.0


    def variable_selection(self, beta_fit=None, beta_true=None, method=None, fdptpp=False, **kwargs):
        # add the parameter *args/**kwargs to specify alpha/lambda for amp/lasso
        '''
        method in ('L1amp', 'two-stage amp', 'Lasso')
         -- if fdptpp == False, will only return selected index, based on either passed <beta_fit> or computed through specified <method>
         -- if fdptpp == True,  one of <beta_true> or <self.beta_true> is needed
         -- if <method> == 'L1amp', <alpha> should be provided in **kwargs
         -- if <method> == 'Lasso', <lam> should be provided in **kwargs
         -- if <method> == 'two-stage amp', <q> & <s> should be provided in **kwargs; <s> can be an array and an array of (fdp, tpp) in form of tuple will be returned
         -- if <method> == 'slope', **kwargs = {<fdr>, <lam_type>} 
         -- if <method> == '2StageFDR', **kwargs = {fdr, q} 
         -- if <method> == 'TBD', ...
         -- --
         * as long as <method> or <beta_fit> is provided, variable selection can be done;
         * to further compute FDP & TPP (controlled by <fdptpp>=True), beta_true is needed through function arguments or object attribute
         -- --
         ** further error handling needed for all cases
        '''
        # Error handling 1
        if method == None and beta_fit == None:
            raise ValueError('one of <method> or <beta_fit> should be provided')
            sys.exit(0) # or return None?
        # Error handling 2
        if self.beta_true == None and beta_true == None and fdptpp == True:
            raise ValueError('<beta_true> missing; to fix, either pass to function argument or pass to object attribute')
            sys.exit(0) # or return None
        
        if method != None:
            if method == 'L1amp':
                # Error handling 3
                if 'alpha' not in kwargs:
                    raise ValueError('<alpha> needed in **kwargs for \'L1amp\'')
                    sys.exit(0)
                else:
                    alpha = kwargs['alpha']
                    if (1, alpha) not in self.amp_result:
                        self.amp(alpha, 1)
                    fit_vec = self.amp_result[(1, alpha)][0]
            
            elif method == 'Lasso':
                # Eror handling 4
                if 'lam' not in kwargs:
                    raise ValueError('<lam> needed in **kwargs for \'Lasso\'')
                    sys.exit(0)
                else:
                    lam = kwargs['lam']
                    if (1, lam) not in self.bridge_result:
                        self.bridge(lam, 1)
                    fit_vec = self.bridge_result[(1, lam)]
            
            elif method == 'two-stage amp':
                # Error handling 5
                if 'q' not in kwargs or 's' not in kwargs:
                    raise ValueError('<q> & <s> needed in **kwargs for \'two-stage amp\'')
                    sys.exit(0)
                else:
                    q = kwargs['q']
                    s = kwargs['s']
                    fit_vec = self.two_stage_amp(q, s)
            
            elif method == 'slope':
                # Error handling 6
                if 'fdr' not in kwargs or 'lam_type' not in kwargs:
                    raise ValueError('<fdr> & <lam_type> needed in **kwargs for \'SLOPE\'')
                    sys.exit(1)
                else:
                    fdr = kwargs['fdr']
                    lam_type = kwargs['lam_type']
                    fit_vec = self.slope(fdr, lam_type)
            
            elif method == '2StageFDR':
                # Error handling 7
                if 'fdr' not in kwargs or 'q' not in kwargs:
                    raise ValueError('<fdr> & <q> needed in **kwargs for \'2StageFDR\'')
                    sys.exit(1)
                else:
                    fdr = kwargs['fdr']
                    q = kwargs['q']
                    fit_vec = self.twoStageFDR(q, fdr)
            
            
        else:
            fit_vec = beta_fit

        if type(fit_vec) != dict:
            vs_select = (fit_vec != 0).astype(float)
        else:
            vs_select = {(q, ss) : (fit_vec[(q, ss)] != 0).astype(float) for (q, ss) in fit_vec}
        
        if fdptpp == False:
            return vs_select
        else:
            if beta_true is None:
                beta_true = self.beta_true
            vs_true = (beta_true != 0).astype(float)
            if type(vs_select) != dict:
                fdp = np.sum(vs_select * (1.0 - vs_true)) / np.sum(vs_select)
                tpp = np.sum(vs_select * vs_true) / np.sum(vs_true)
                return (vs_select, fdp, tpp)
            else:
                for pair in vs_select:
                    fdp = np.sum(vs_select[pair] * (1.0 - vs_true)) / np.sum(vs_select[pair])
                    tpp = np.sum(vs_select[pair] * vs_true) / np.sum(vs_true)
                    vs_select[pair] = (vs_select[pair], fdp, tpp)
                return vs_select
    
    def slope(self, fdr, lam_type, sigma=None, tol=1e-5, report=True):
        '''
         -- this function performs SLOPE (sorted L-one penalized estimation)
            * reference: [BBSSC15]
         -- <fdr>: level of FDR to be controlled
         -- <lam_type>: choose how to specify the tuning
              if <lam_type> = 'BH', we use (1.5)
              if <lam_type> = 'G', we use (3.7), the lambda-G correction (greater than 'BH')
         -- <sigma>: multiplied to tuning
              if <sigma> is numeric, use it
              if <sigma> = None, estimated based on 'Algorithm 5' in the paper
         -- <report>: report some basic information about the result (or not)
         -- tol: tolerance for interation
        '''
        if (fdr, lam_type) in self.slope_result:
            return self.slope_result[(fdr, lam_type)]
        
        n, p = self.shape
        yn = normalize(self.y, 'c')
        Xn = normalize(self.X, 'cs')
        
        if lam_type == 'BH':
            lam = sps.norm.ppf(1 - fdr * np.arange(1, p + 1) / 2.0 / p)
        elif lam_type == 'G':
            lam_BH = sps.norm.ppf(1 - fdr * np.arange(1, p + 1) / 2.0 / p)
            lam = np.zeros(p)
            lam[0] = lam_BH[0]
            for i in xrange(1, p):
                lam[i] = lam_BH[i] * np.sqrt(1 + sum(lam[: (i - 1)] ** 2) / (n - i - 1.0))
                if lam[i] > lam[i - 1]:
                    lam[i : ] = lam[i - 1]
                    break
        
        # proximal gradient descent
        b  = np.zeros(p)
        t = 1.0 / npla.norm(Xn)
        
        loop_count = 0
        
        if sigma is not None:
            b0 = -1.0 * np.ones(p)
            while npla.norm(b0 - b) / npla.norm(b) > tol:
                b0 = b
                b = prox_sortedL1(b0 - t * np.dot(Xn.T, np.dot(Xn, b0) - yn), t * sigma * lam)
                loop_count += 1
        else:
            sigma = 1.0
            S0 = np.zeros(p, dtype=bool)
            
            while True:
                b0 = -1.0 * np.ones(p)
                while npla.norm(b0 - b) / npla.norm(b) > tol:
                    b0 = b
                    b = prox_sortedL1(b0 - t * np.dot(Xn.T, np.dot(Xn, b0) - yn), t * sigma * lam)
                    loop_count += 1
                S = (b != 0.0)
                if np.all(S == S0):
                    break
                S0 = S
                sigma = np.sqrt((np.sum(yn ** 2)
                                 - np.dot(yn.T, Xn[:, S0]).dot(npla.inv(np.dot(Xn[:, S0].T, Xn[:, S0])).dot(np.dot(Xn[:, S0].T, yn))))
                                / (n - np.count_nonzero(S0) - 1))

        if report:
            print("number of iterations: ", loop_count, sep='', file=sys.stderr)
        
        self.slope_result[(fdr, lam_type)] = b
        print(sigma, file=sys.stderr)
            
        return b  # should this be the correct return?
    
    def twoStageFDR(self, q, fdr, method, iter_max = 500):
        '''
         -- this function does two-stage bridge variable selection with FDR control
         Arguments:
         -- <q> is to use Lq penalty in the 1st stage
         -- <fdr> is the specified FDR level
         Returns:
         
        '''
        
        if (q, fdr) in self.S2FDR_result:
            return self.S2FDR_result[(q, fdr)]
        
        n, p = self.shape
        
        if q not in self.optimal_alpha:
            self.optimal_tuning(q)
        
        amp_fit = self.amp(self.optimal_alpha[q], q, iter_max=iter_max)
        temp = amp_fit[0] + np.dot(self.X.T, amp_fit[1])
        
        if method == 'BH':
            pvals = (1.0 - sps.norm.cdf(np.absolute(temp / amp_fit[2]))) * 2.0
            self.S2FDR_result[(q, fdr, method)] = multi_test(pvals, fdr, method = 'BH')
            return self.S2FDR_result[(q, fdr, method)]
        elif method == 'slope':
            lam_BH = sps.norm.ppf(1 - fdr * np.arange(1, p + 1) / 2.0 / p)
            self.S2FDR_result[(q, fdr, method)] = prox_sortedL1(temp, amp_fit[2] * lam_BH)
            return self.S2FDR_result[(q, fdr, method)]
    
    def get_alpha(self):
        return self.amp_result[0]

    def get_true_par(self, beta_true):
        self.beta_true = beta_true


def db_gamma(beta, lam, q, delta):
    '''
    calculate the debiasing paramter gamma based on bridge estimator beta
    argument:
        beta: np.ndarray, Lq bridge estimator
        lam: tuning of the bridge estimator
        q: Lq loss
        n: number of observation
    value
        gamma
    '''
    if q <= 1:
        raise ValueError('q must be larger than 1 for this function')

    p = len(beta)
    tmp_func = lambda gamma: (lam / float(gamma) - 1.0 + 1.0 / delta
            * np.mean(1.0 / (1.0 + gamma * q * (q - 1.0) * np.absolute(beta) ** (q - 2.0))))
    ## the upper bound is estimable by using the min(beta) or max(beta); double check;
    return bisect_search(tmp_func, lower = lam / 1.1, unit = 10)


class bridge(object):
    '''
    construct a linear model and execute various estimation and VS algorithms on it.
    '''
    def __init__(self, lam=None, q=None, beta_true=None, tol=1e-6,
                       iter_max=1000):
        '''
        Constructor.

        Parameters
        ----------

        Returns
        -------

        '''
        self.lam = lam
        self.q = q
        self.beta_true = beta_true
        self.tol = tol
        self.iter_max = iter_max
   
    def fit(self, X, y, beta_init=None):
        '''
        This function fit the bridge regression.

        Parameters
        ----------
        X: np.ndarray, dtype=np.float64, ndim=2
            The 2-dim array of independent variable, of dimension (n, p)
        y: np.ndarray, dtype=np.float64, ndim=1
            The 1-dim array of response variable, of dimension (n,)
        beta_init: np.ndarray, dtype=np.float64, ndim=1
            beta here means the coefficient vector we would like to find.
            beta_init provides initialization for bridge regression. If None,
            beta_init=np.zeros(p, dtype=np.float64)
        tol: float, >0
            The tolerance of convergence criterion. Here we use it as relative
            tolerance.
        iter_max: int, >0
            The maximal iteration allowed. Quit the loop once exceeded this
            bound.

        Returns
        -------
        beta: np.ndarray, dtype=np.float64, ndim=1 or 2
            The fitted coefficient vector or 2-dim array, depending on whether
            lam is a scalar or an array.
            If lam is a scalar, then beta is an 1-dim array of length p;
            If lam is an 1-dim array with length m, then beta is a 2-dim array
            of shape (p, m), with ith column corresponding to the regressor of
            lam[i].

        We run
            0.5 * \|y - X\beta\|_2^2 + \lambda \|beta\|_q^q
        to obtain the bridge regression estimator for q >= 1.
        '''
        p  = X.shape[1]
        X_norm2 = npla.norm(X, axis=0) ** 2
        Xy = np.dot(X.T, y) / X_norm2
        XX = np.dot(X.T, X) / X_norm2

        if beta_init is None:
            beta_init = np.zeros(p, dtype=np.float64)

        if np.isscalar(self.lam) or self.lam.shape == (1,):
            beta = cbridge_decay(XX, Xy, X_norm2, self.lam, self.q,
                                 beta_init=beta_init, rel_tol=self.tol,
                                 iter_max=self.iter_max)
        else:
            beta = cbridge_decay2(XX, Xy, X_norm2, self.lam, self.q,
                                  beta_init=beta_init, rel_tol=self.tol,
                                  iter_max=self.iter_max)
        return beta
 
    def fmse(self, beta, X, y):
        '''
        Estimate fake-MSE of bridge estimator by debiasing. This is not really
        MSE. It is the (tau ** 2), which equals (sigma **2 + MSE / delta) for
        orthogonal design. Since it is up to a scalar and constant of MSE, we
        can also use it for tuning.

        Parameters
        ----------
        beta: np.ndarray, dtype=np.float64, ndim=1 or 2
            The fitted beta array, either 1-dim or 2-dim, depending whether
            lam is a scalar or an array.

        Returns
        -------
        mse: float or np.ndarray[dtype=np.float64, ndim=1]
            The 
        '''
        n, p = X.shape
        delta = n / float(p)

        if beta.ndim == 1:
            if self.q == 1:
                k = np.count_nonzero(beta)
                if k >= n:
                    return np.inf
                z = (y - np.dot(X, beta)) / (1.0 - k / float(n))
            elif self.q > 1:
                gamma = db_gamma(beta, self.lam, self.q, delta)
                z = (y - np.dot(X, beta)) / (1.0 - np.mean(1.0 / (1.0
                    + gamma * self.q * (self.q - 1.0) * np.absolute(beta) **
                    (self.q - 2))) / delta)
            else:
                raise NotImplementedError("q < 1 not implemented.")
            fmse = np.mean(z ** 2)
        else:
            m = beta.shape[1]
            fmse = np.empty(m, dtype=np.float64)
            if self.q == 1:
                for i in xrange(m):
                    k = np.count_nonzero(beta[:, i], axis=0)
                    if k >= n:
                        fmse[i] = np.inf
                    else:
                        z = (y - np.dot(X, beta[:, i])) / (1.0 - k / float(n))
                        fmse[i] = np.mean(z ** 2)
            elif self.q > 1:
                for i in xrange(m):
                    gamma = db_gamma(beta[:, i], self.lam[i], self.q, delta)
                    z = (y - np.dot(X, beta[:, i])) / (1.0 - np.mean(1.0 /
                        (1.0 + gamma * self.q * (self.q - 1.0) *
                            np.absolute(beta[:, i]) ** (self.q - 2))) / delta)
                    fmse[i] = np.mean(z ** 2)
            else:
                raise NotImplementedError("q < 1 not implemented.")

        return fmse
 
#        if q == 1:  ## new add part, separate L1
#            if type(lam) is not np.ndarray:
#                clf = skll.Lasso(lam / float(len(self.y)), fit_intercept=False, max_iter=6000)
#                # clf = skll.LassoLars(lam / float(len(self.y)), fit_intercept=False, max_iter=3000, fit_path=False)
#                clf.fit(self.X, self.y)
#                beta_hat = clf.coef_
#                
#                if np.sum(beta_hat != 0) >= len(self.y):
#                    return np.inf
#            else:
#                _, beta_list, _ = skll.lasso_path(self.X, self.y, alphas=lam / float(len(self.y)), fit_intercept = False, iter_max=6000)
#                mse_arr = np.repeat(np.inf, len(lam))
#                for i in xrange(len(lam)):
#                    if np.sum(beta_list[:, i] != 0) < len(self.y):
#                        z = (self.y - np.dot(self.X, beta_list[:, i])) / ( 1.0 - np.mean(beta_list[:, i] != 0) / self.delta )
#                        mse_arr[i] = np.mean(z ** 2)
#                        # print("lambda: {0}, mse: {1}".format(lam[i], mse_arr[i]))
#                return mse_arr
#
#        elif q == 2:
#            reg = skll.Ridge(lam * 2.0, fit_intercept=False, tol=0.00001)
#            reg.fit(self.X, self.y)
#            beta_hat = reg.coef_
#        else:
#            #if (q, lam) not in self.bridge_result:
#            #    self.bridge(lam, q, initial=initial, iter_max = 600) ## MARK!!! 600!!
#            beta_hat = self.bridge(lam, q, iter_max=2000, initial=initial)
#
#        if q == 1:
#            z = (self.y - np.dot(self.X, beta_hat)) / ( 1.0 - np.mean(beta_hat != 0) / self.delta )
#        else:
#            gamma = db_gamma(beta_hat, lam, q, self.delta)
#            z = (self.y - np.dot(self.X, beta_hat)) / ( 1.0 - np.mean(1.0 / (1.0 + gamma * q * (q - 1.0) * np.absolute(beta_hat) ** (q - 2))) / self.delta )
#        
#        return np.mean(z ** 2)

    def mse(self, beta, beta_true=None):
        '''
        Calculate the MSE of bridge estimator.

        Parameters
        ----------
        beta: np.ndarray, dtype=np.float64, ndim=1 or 2
            The fitted beta array, either 1-dim or 2-dim, depending whether
            lam is a scalar or an array.
        beta_true: np.ndarray, dtype=np.float64, ndim=1
            True coefficient, must be provided if self.beta_true is None.

        Returns
        -------
        mse: float or np.ndarray[dtype=np.float64, ndim=1]
            The true MSE of estimator beta. If beta is 1-dim, will return
            scalar; np.array otherwise.

        Calculate the MSE by
            MSE = np.mean((beta - beta_true) ** 2)
        '''
        if self.beta_true is not None:
            if beta.ndim == 1:
                mse = np.mean((beta - self.beta_true) ** 2)
            else:
                mse = np.mean((beta - self.beta_true[:, np.newaxis]) ** 2,
                        axis = 0)
        elif beta_true is not None:
            if beta.ndim == 1:
                mse = np.mean((beta - beta_true) ** 2)
            else:
                mse = np.mean((beta - beta_true[:, np.newaxis]) ** 2,
                              axis = 0)
        else:
            raise ValueError(("Please provide at least one of self.beta_true "
                              "or beta_true"))
        return mse

    def debias(self, beta, X, y):
        '''
        Calculate the debiased estimator of beta_hat.

        Parameters
        ----------
        beta: np.ndarray, dtype=np.float64, ndim=1
            The bridge estimator.
        X: np.ndarray, dtype=np.float64, ndim=2
            The 2-dim array of independent variable.
        y: np.ndarray, dtype=np.float64, ndim=1
            The 1-dim array of response variable.

        Returns
        -------
        beta_dbs: debiased version of beta

        This function is based on the estimated beta,
        and does not check the validity of beta_hat. Please
        make sure the estiamtor is good before debiasing.
        '''
        n, p = X.shape
        delta = n / float(p)
        if beta.ndim == 1:
            if self.q == 1:
                k = np.count_nonzero(beta)
                if k >= n:
                    raise ValueError("The estimation is inaccurate.")
                z = (y - np.dot(X, beta)) / ( 1.0 - k / float(n) )
            else:
                gamma = db_gamma(beta, self.lam, self.q, delta)
                z = (y - np.dot(X, beta)) / ( 1.0 - np.mean(1.0 / (1.0 + gamma
                    * self.q * (self.q - 1.0) * np.absolute(beta) ** (self.q - 2))) / delta )

            beta_dbs = beta + np.dot(X.T, z)
        else:
            m = beta.shape[1]
            beta_dbs = np.empty((p, m), dtype=np.float64)
            for i in xrange(m):
                if self.q == 1:
                    k = np.count_nonzero(beta[:, i])
                    if k >= n:
                        raise ValueError("The estimation is inaccurate.")
                    z = (y - np.dot(X, beta[:, i])) / ( 1.0 - k / float(n) )
                else:
                    gamma = db_gamma(beta[:, i], self.lam[i], self.q, delta)
                    z = (y - np.dot(X, beta[:, i])) / ( 1.0 - np.mean(1.0 / (1.0 + gamma
                        * self.q * (self.q - 1.0) * np.absolute(beta[:, i]) ** (self.q - 2))) / delta )

                beta_dbs[:, i] = beta[:, i] + np.dot(X.T, z)
            
        return beta_dbs
    
    def bridge_optimalTune(self, q):
        '''
        search for the optimal tuning among lambda_arr
        Parameters
        ----------
            q: q >= 1, Lq bridge
            lambda_arr: array of tuning, must be arranged in decreasing order
        
        Returns
        -------
            mse_arr: return the mse array correspond to lambda_arr
            lambda_opt: optimal lambda among lambda_arr
            mse_opt: optimal MSE correspond to lambda_opt
            beta_opt: the estimator correspond to lambda_opt
        '''


        if q < 1: ## we can consider remove some of these conditions
            raise ValueError('q must be greater than or equal to 1.')
        
        lam_unit  = npla.norm(np.dot(self.X.T, self.y), np.inf) / 2.0
        lam_low   = 0.1
        lam_upp   = 0.1 # lam_low, lam_opt specified to pass the 1st while & if.
        grid_size = 15

        mse_min0  = 0
        mse_min   = 100000.0
        i_opt     = 0
        lam_opt0  = 0
        lam_opt   = 0

        while i_opt == grid_size - 1 or i_opt == 0:
            if np.absolute(mse_min0 - mse_min) / mse_min < 0.01:
                return lam_opt0
            mse_min0 = mse_min
            lam_opt0 = lam_opt
            
            if i_opt == grid_size - 1:
                lam_upp =  lam_low
                lam_low /= 10.0
            else:
                lam_low =  lam_upp
                lam_upp += lam_unit
 
            grid = np.linspace(lam_upp, lam_low, grid_size) # by default end-pt included

            if q == 1:
                mse_arr = self.bridge_mse(grid, 1)
                i_opt   = np.argmin(mse_arr)
                mse_min = mse_arr[i_opt]
                lam_opt = grid[i_opt]
            else:
                for j in xrange(grid_size):
                    mse_tmp = self.bridge_mse(grid[j], q, initial=self.beta_prev)
                    if mse_tmp < mse_min:
                        i_opt   = j
                        mse_min = mse_tmp
                lam_opt = grid[i_opt]
                
        return lam_opt
    
    def xbridge_optimalTune(self, q, lam0, Delta = 0.05, tol=0.1):
        '''
        search for the optimal tuning among lambda_arr
        arguments:
            q: q >= 1, Lq bridge
            lambda_arr: array of tuning, must be arranged in decreasing order
        value:
            mse_arr: return the mse array correspond to lambda_arr
            lambda_opt: optimal lambda among lambda_arr
            mse_opt: optimal MSE correspond to lambda_opt
            beta_opt: the estimator correspond to lambda_opt
        '''


        if q < 1:
            raise ValueError('q must be greater than or equal to 1.')

#        h = 0.05
#        lam = lam0
#        lam0 = lam0 + 1
#        count = 1.0
#
#        while (np.absolute(lam - lam0) > tol):
#            lam0 = lam
#            deri = (self.bridge_mse(lam0 + h, q) - self.bridge_mse(lam0 - h, q)) / 2.0 / h
#            lam = max(lam0 - deri / count, 2 * h)
#            count += 1.0
#
#        return lam
        

        if self.bridge_mse(lam0 + Delta, q) - self.bridge_mse(lam0, q) > 0:
            lamR = lam0
            lamL = lam0 / 2.0

            while ((self.bridge_mse(lamL + Delta, q) - self.bridge_mse(lamL, q)) > 0):
                lamR = lamL
                lamL = lamL / 2.0
        else:
            lamL = lam0
            lamR = lam0 * 2.0

            while ((self.bridge_mse(lamR + Delta, q) - self.bridge_mse(lamR, q)) < 0):
                lamL = lamR
                lamR = lamR * 2.0

        while (lamR - lamL) > tol:
            lamM = (lamL + lamR) / 2.0
            if (self.bridge_mse(lamM + Delta, q) - self.bridge_mse(lamM, q)) < 0:
                lamL = lamM
            else:
                lamR = lamM

        return (lamL + lamR) / 2.0


class amp(object):
    def __init__(self):
        pass

