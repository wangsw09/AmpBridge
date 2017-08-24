#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs, fmax, fmin

def ncdf(double x):
    cdef double a1 = 0.254829592
    cdef double a2 = -0.284496736
    cdef double a3 = 1.421413741
    cdef double a4 = -1.453152027
    cdef double a5 = 1.061405429
    cdef double p = 0.3275911
    cdef double e = 2.7182818284
    
    cdef int sign = 1
    if x < 0:
        sign = -1
    x = abs(x) * 0.70710678118

    cdef double t = 1.0 / (1.0 + p * x)
    cdef double y = 1.0 - ((((( a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * pow(e, -x * x)
    return 0.5 * (1.0 + sign * y)


def npdf(double x):
    cdef double e = 2.7182818284
    cdef double c = 0.3989422804
    return c * pow(e, - x * x * 0.5)


def prox_Lq(double u, double t, double q, double tol = 1e-9):
    '''
    proximal function of lq penalty tuned at t
    the tol will cause problem for u ~ 5e5, e.g., -564789.688027
    '''
    cdef int sign = 1
    if u < 0:
        sign = -1

    cdef double a
    cdef double b
    cdef double x0
    cdef double x
    cdef double u0

    if q == 1.0:
        return max(abs(u) - t, 0.0) * sign
    elif q == 2.0:
        return u / (1.0 + 2.0 * t)
    elif q == 1.5:
        x0 = (0.5625 * t * t + abs(u)) ** 0.5 - 0.75 * t
        return x0 * x0 * sign
    elif q > 2.0:
        a = t * q * (q - 2.0)
        b = t * q * (q - 1.0)
        x0 = 0
        u0 = abs(u)
        x = u0

        while abs(x - x0) > tol:
            x0 = x
            x = (a * x ** (q - 1.0) + u0) / (1.0 + b * x ** (q - 2.0))
        return x * sign

    else:
        a = t * q * (q - 2.0)
        b = t * q * (q - 1.0)
        u0 = abs(u)

        if u0 <= tol:
            return u
        else:
            x0 = u0
            x = (a * u0 ** (q - 1.0) + u0) / (1.0 + b * u0 ** (q - 2.0))
            if x <= 0:
                x = min( (u0 / (1.0 + t * q)) ** (1.0 / (q - 1.0)), u0 / (1.0 + t * q) )
            
            while abs(x - x0) > tol:
                x0 = x
                x = (a * x + u0 * x ** (2.0 - q)) / (x ** (2.0 - q) + b)
            return x * sign


def fprox_Lq(double u, double t, double q, double tol = 1e-9):
    '''
    proximal function of lq penalty tuned at t
    the tol will cause problem for u ~ 5e5, e.g., -564789.688027
    '''
    cdef int sign = 1
    if u < 0:
        sign = -1

    cdef double a
    cdef double b
    cdef double x0
    cdef double x
    cdef double u0

    if q == 1.0:
        return fmax(fabs(u) - t, 0.0) * sign
    elif q == 2.0:
        return u / (1.0 + 2.0 * t)
    elif q == 1.5:
        x0 = (0.5625 * t * t + fabs(u)) ** 0.5 - 0.75 * t
        return x0 * x0 * sign
    elif q > 2.0:
        a = t * q * (q - 2.0)
        b = t * q * (q - 1.0)
        x0 = 0
        u0 = fabs(u)
        x = u0

        while fabs(x - x0) > tol:
            x0 = x
            x = (a * x ** (q - 1.0) + u0) / (1.0 + b * x ** (q - 2.0))
        return x * sign

    else:
        a = t * q * (q - 2.0)
        b = t * q * (q - 1.0)
        u0 = fabs(u)

        if u0 <= tol:
            return u
        else:
            x0 = u0
            x = (a * u0 ** (q - 1.0) + u0) / (1.0 + b * u0 ** (q - 2.0))
            if x <= 0:
                x = fmin( (u0 / (1.0 + t * q)) ** (1.0 / (q - 1.0)), u0 / (1.0 + t * q) )
            
            while fabs(x - x0) > tol:
                x0 = x
                x = (a * x + u0 * x ** (2.0 - q)) / (x ** (2.0 - q) + b)
            return x * sign



cdef double cprox_Lq(double u, double t, double q, double tol = 1e-9):
    '''
    proximal function of lq penalty tuned at t
    the tol will cause problem for u ~ 5e5, e.g., -564789.688027
    '''
    cdef double sign = 1.0
    if u < 0:
        sign = -1.0

    cdef double a
    cdef double b
    cdef double x0
    cdef double x
    cdef double u0

    if q == 1.0:
        return fmax(fabs(u) - t, 0.0) * sign
    elif q == 2.0:
        return u / (1.0 + 2.0 * t)
    elif q == 1.5:
        x0 = (0.5625 * t * t + fabs(u)) ** 0.5 - 0.75 * t
        return x0 * x0 * sign
    elif q > 2.0:
        a = t * q * (q - 2.0)
        b = t * q * (q - 1.0)
        x0 = 0
        u0 = fabs(u)
        x = u0

        while fabs(x - x0) > tol:
            x0 = x
            x = (a * x ** (q - 1.0) + u0) / (1.0 + b * x ** (q - 2.0))
        return x * sign

    else:
        a = t * q * (q - 2.0)
        b = t * q * (q - 1.0)
        u0 = fabs(u)

        if u0 <= tol:
            return u
        else:
            x0 = u0
            x = (a * u0 ** (q - 1.0) + u0) / (1.0 + b * u0 ** (q - 2.0))
            if x <= 0:
                x = fmin( (u0 / (1.0 + t * q)) ** (1.0 / (q - 1.0)), u0 / (1.0 + t * q) )
            
            while fabs(x - x0) > tol:
                x0 = x
                x = (a * x + u0 * x ** (2.0 - q)) / (x ** (2.0 - q) + b)
            return x * sign



def prox_Lq_drvt(double u, double t, double q, double tol=1e-9):
    '''
    derivative of eta() w.r.t. u
    '''
    cdef double tmp

    if q == 1.0:
        if abs(u) > t:
            return 1
        else:
            return 0
    elif q == 2.0:
        return 1.0 / (1.0 + 2.0 * t)
    elif q == 1.5:
        return 1.0 - t / (t * t + 1.777777778 * abs(u)) ** 0.5
    elif q < 2.0:
        tmp = abs(prox_Lq(u, t, q, tol)) ** (2.0 - q)
        return tmp / (tmp + t * q * (q - 1.0))
    elif q > 2.0:
        tmp = abs(prox_Lq(u, t, q, tol)) ** (q - 2.0)
        return 1.0 / (1.0 + t * q * (q - 1.0) * tmp)


def prox_Lq_drvt_t(double u, double t, double q, double tol=1e-9):  # add discussion on q = 1 and q > 1 here
    '''
    derivative of eta() w.r.t. t
    '''
    cdef int sign = 1
    if u < 0:
        sign = -1

    cdef double tmp

    if q == 1.0:
        if abs(u) > t:
            return -sign
        else:
            return 0
    elif q == 2:
        return - 2.0 * u / (1.0 + 2.0 * t) ** 2.0
    elif q == 1.5:
        return (2.25 * t - (1.6875 * t * t + 1.5 * abs(u)) / (0.5625 * t * t + abs(u)) ** 0.5) * sign
    else:
        tmp = abs(prox_Lq(u, t, q, tol))
        if q < 2.0:
            return - q * tmp * sign / (tmp ** (2.0 - q) + t * q * (q - 1.0))
        else:
            return - q * tmp ** (q - 1.0) * sign / (1.0 + t * q * (q - 1.0) * tmp ** (q - 2.0))


def prox_Lq_inv(double v, double t, double q):
    '''
    Compute u in v = eta(u, t, q), that is, the inverse of eta function.
    '''
    if q == 1.0:
        raise ValueError("q must be larger than 1.")
    cdef int sign = 1
    if v < 0:
        sign = -1
    return v + q * t * abs(v) ** (q - 1.0) * sign


def mse_integrand(double z, double x, double alpha, double tau, double q):
    return (prox_Lq(x + tau * z, t = alpha * tau ** (2.0 - q), q = q) - x) ** 2 * npdf(z)


def mse_drvt_integrand(double z, double x, double alpha, double tau, double q):
    return 2.0 * tau * (prox_Lq(x + z * tau, alpha * tau ** (2.0 - q), q) - x) * prox_Lq_drvt_t(x / tau + z, alpha, q) * npdf(z)


def prox_Lq_vec(double u, double t, double q, double tol = 1e-9):
    '''
    proximal function of lq penalty tuned at t
    the tol will cause problem for u ~ 5e5, e.g., -564789.688027
    '''
    cdef int sign = 1
    if u < 0:
        sign = -1

    cdef double a
    cdef double b
    cdef double x0
    cdef double x
    cdef double u0

    if q == 1.0:
        return max(abs(u) - t, 0.0) * sign
    elif q == 2.0:
        return u / (1.0 + 2.0 * t)
    elif q == 1.5:
        x0 = (0.5625 * t * t + abs(u)) ** 0.5 - 0.75 * t
        return x0 * x0 * sign
    elif q > 2.0:
        a = t * q * (q - 2.0)
        b = t * q * (q - 1.0)
        x0 = 0
        u0 = abs(u)
        x = u0

        while abs(x - x0) > tol:
            x0 = x
            x = (a * x ** (q - 1.0) + u0) / (1.0 + b * x ** (q - 2.0))
        return x * sign

    else:
        a = t * q * (q - 2.0)
        b = t * q * (q - 1.0)
        u0 = abs(u)

        if u0 <= tol:
            return u
        else:
            x0 = u0
            x = (a * u0 ** (q - 1.0) + u0) / (1.0 + b * u0 ** (q - 2.0))
            if x <= 0:
                x = min( (u0 / (1.0 + t * q)) ** (1.0 / (q - 1.0)), u0 / (1.0 + t * q) )
            
            while abs(x - x0) > tol:
                x0 = x
                x = (a * x + u0 * x ** (2.0 - q)) / (x ** (2.0 - q) + b)
            return x * sign


def cbridge_decay(np.ndarray[dtype=np.float64_t, ndim=2] XX, np.ndarray[dtype=np.float64_t, ndim=1] Xy, np.ndarray[dtype=np.float64_t, ndim=1] X_norm2, double lam, double q, np.ndarray[dtype=np.float64_t, ndim=1] beta_init, double abs_tol=1e-3, double rel_tol=1e-3, int iter_max=1000):
    '''
    This function does bridge regression with L_q penalty for q >= 1 given X^TX and X^Ty.

    Parameters
    ----------
    XX: np.ndarray, dtype=np.float64, ndim=2
        This is the matrix X^T * X, then divide the ith column by the square norm of X[:, i].
    Xy: np.ndarray, dtype=np.float64, ndim=1
        The array of X^Ty with the ith component divided by the square norm of X[:, i].
    X_norm2: np.ndarray, dtype=np.float64, ndim=1
        The array of the square norm of the columns of the matrix of independent variables X.
    lam: double, lam >= 0
        Tuning parameter. For lam=0, I think it is interesting to implement it. May need special processing.
    q: double, q >= 0
        Choice of penalty. L_q penalty corresponds to L_q norm.
    beta_init: np.ndarray, dtype=np.float64, ndim=2
        The initialization of the iteration of the algorithm.
    abs_tol: double, > 0
        The absolute tolerance of convergence.
    rel_tol: double, > 0
        The relative tolerance of convergence.
    iter_max: int, > 0
        The maximal number of iteration allowed. The iteration should stop when this number got exceeded.

    Returns
    -------
    beta: np.ndarray, dtype=np.float64, ndim=1
        The bridge estimator of the problem.

    Details
    -------
    For this problem, we solve the following problem:
        min_{\beta} 0.5 * \|y - X \beta\|_2^2 + \lam \|\beta\|_q^q
    The strong convexity of the loss function and the convextiry of the penalty term guarantees the existence and uniqueness of the global minimizer, and also the convergence of the coordinate descent algorithm. Yes, we use coordinate descent.
    '''
    cdef int p = XX.shape[0]
    cdef int iter_count = 0
    cdef int i
    cdef int j
    cdef double beta_max = 1.0
    cdef double diff = 1.0
    cdef double tmp
    cdef double beta0_comp

    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta = np.empty(p, dtype=np.float64) # new step
    cdef np.ndarray[dtype=np.float64_t, ndim=1] grad = Xy - np.dot(XX.T, beta_init) + beta_init # gradient
    cdef np.ndarray[dtype=np.float64_t, ndim=1] lam_arr = np.empty(p, dtype=np.float64) # gradient
    
    cdef double[:] beta_init_view = beta_init
    cdef double[:] beta_view = beta
    cdef double[:] grad_view = grad
    cdef double[:] lam_arr_view = lam_arr
    cdef double[:, :] XX_view = XX

    for i in range(p):
        lam_arr_view[i] = lam / X_norm2[i]
        beta_view[i] = beta_init_view[i]

    while diff / beta_max > rel_tol:
        beta_max = 0.0
        diff = 0.0
        for i in range(p):
            beta0_comp = beta_view[i]
            beta_view[i] = cprox_Lq(grad_view[i], lam_arr_view[i], q)

            # update grad after beta[i]
            tmp = beta_view[i] - beta0_comp

            for j in range(p):
                grad_view[j] -= XX_view[i, j] * tmp
            grad_view[i] += tmp

            if abs(tmp) > diff:
                diff = abs(tmp)

            tmp = abs(beta_view[i])
            if tmp > beta_max:
                beta_max = tmp

        iter_count += 1
        if iter_count > iter_max:
            break
    return beta


def cbridge_decay2(np.ndarray[dtype=np.float64_t, ndim=2] XX, np.ndarray[dtype=np.float64_t, ndim=1] Xy, np.ndarray[dtype=np.float64_t, ndim=1] X_norm2, np.ndarray[dtype=np.float64_t, ndim=1] lam, double q, np.ndarray[dtype=np.float64_t, ndim=1] beta_init, double abs_tol=1e-3, double rel_tol=1e-3, int iter_max=1000):
    '''
    This function does bridge regression with L_q penalty for q >= 1 and a sequence of tuning parameters given X^TX and X^Ty.

    Parameters
    ----------
    XX: np.ndarray, dtype=np.float64, ndim=2
        This is the matrix X^T * X, then divide the ith column by the square norm of X[:, i].
    Xy: np.ndarray, dtype=np.float64, ndim=1
        The array of X^Ty with the ith component divided by the square norm of X[:, i].
    X_norm2: np.ndarray, dtype=np.float64, ndim=1
        The array of the square norm of the columns of the matrix of independent variables X.
    lam: np.ndarray, dtype=np.float64, ndim=1
        Tuning parameter array. The numbers in lam must be in a decreasing order to guarantee valid warm-initialization for further speed-up.
    q: double, q >= 0
        Choice of penalty. L_q penalty corresponds to L_q norm.
    beta_init: np.ndarray, dtype=np.float64, ndim=2
        The initialization of the iteration of the algorithm.
    abs_tol: double, > 0
        The absolute tolerance of convergence.
    rel_tol: double, > 0
        The relative tolerance of convergence.
    iter_max: int, > 0
        The maximal number of iteration allowed. The iteration should stop when this number got exceeded.

    Returns
    -------
    beta: np.ndarray, dtype=np.float64, ndim=2
        The bridge estimators of the problem. With the ith column of the beta array corresponding to lam[i].

    Details
    -------
    For this problem, we solve the following problem:
        min_{\beta} 0.5 * \|y - X \beta\|_2^2 + \lam \|\beta\|_q^q
    The strong convexity of the loss function and the convextiry of the penalty term guarantees the existence and uniqueness of the global minimizer, and also the convergence of the coordinate descent algorithm. Yes, we use coordinate descent.
    '''
    cdef int p = XX.shape[0]
    cdef int m = lam.shape[0]
    cdef int iter_count
    cdef int i
    cdef int j
    cdef int k
    cdef double beta_max
    cdef double diff
    cdef double tmp
    cdef double beta0_comp

    cdef np.ndarray[dtype=np.float64_t, ndim=2] beta = np.empty((p, m), dtype=np.float64) # new step
    cdef np.ndarray[dtype=np.float64_t, ndim=1] lam_arr = np.empty(p, dtype=np.float64) # gradient
    cdef np.ndarray[dtype=np.float64_t, ndim=1] grad = np.empty(p, dtype=np.float64) # gradient

    cdef double[:] beta_init_view = beta_init
    cdef double[:, :] beta_view = beta
    cdef double[:] grad_view = grad
    cdef double[:] lam_view = lam
    cdef double[:] lam_arr_view = lam_arr
    cdef double[:, :] XX_view = XX

    for k in range(m):
        np.copyto(grad, Xy - np.dot(XX.T, beta_init) + beta_init) # gradient
        beta_max = 1.0
        diff = 1.0
        iter_count = 0

        for i in range(p):
            lam_arr_view[i] = lam_view[k] / X_norm2[i]
            beta_view[i, k] = beta_init_view[i]

        while diff / beta_max > rel_tol:
            beta_max = 0.0
            diff = 0.0
            for i in range(p):
                beta0_comp = beta_view[i, k]
                beta_view[i, k] = cprox_Lq(grad_view[i], lam_arr_view[i], q)

                # update grad after beta[i]
                tmp = beta_view[i, k] - beta0_comp

                for j in range(p):
                    grad_view[j] -= XX_view[i, j] * tmp
                grad_view[i] += tmp

                if abs(tmp) > diff:
                    diff = abs(tmp)

                tmp = abs(beta_view[i, k])
                if tmp > beta_max:
                    beta_max = tmp

            if iter_count > iter_max:
                break
            iter_count += 1

        for i in range(p):
            beta_init_view[i] = beta_view[i, k]

    return beta

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def cccbridge(np.ndarray[dtype=np.float64_t, ndim=2] XX, np.ndarray[dtype=np.float64_t, ndim=1] Xy, np.ndarray[dtype=np.float64_t, ndim=1] X_norm2, double lam, double q, np.ndarray[dtype=np.float64_t, ndim=1] beta_init, double abs_tol=1e-3, double rel_tol=1e-3):
#    '''
#    This function does bridge regression with L_q penalty for q >= 1.
#
#    Parameters
#    ----------
#    X: np.ndarray, dtype=np.float64, ndim=2
#        The 2-dim numpy array of independent variables;
#    y: np.ndarray, dtype=np.float64, ndim=1
#        The 1-dim numpy array of response variables;
#    lam: double, lam >= 0
#        Tuning parameter. For lam=0, I think it is interesting to implement it. May need special processing.
#    q: double, q >= 0
#        Choice of penalty. L_q penalty corresponds to L_q norm.
#    beta_init: np.ndarray, dtype=np.float64, ndim=2
#        The initialization of the iteration of the algorithm.
#
#    Returns
#    -------
#    beta_hat: np.ndarray, dtype=np.float64, ndim=1
#        The bridge estimator of the problem.
#
#    Details
#    -------
#    For this problem, we solve the following problem:
#        min_{\beta} 0.5 * \|y - X \beta\|_2^2 + \lam \|\beta\|_q^q
#    The strong convexity of the loss function and the convextiry of the penalty term guarantees the existence and uniqueness of the global minimizer, and also the convergence of the coordinate descent algorithm. Yes, we use coordinate descent.
#    '''
#    cdef int p = XX.shape[0]
#    cdef int iter_count = 0
#    cdef int i
#    cdef int j
#    cdef double beta_max = 1.0
#    cdef double diff = 1.0
#    cdef double tmp
#    cdef double beta0_comp
#
#    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta = np.empty(p, dtype=np.float64) # new step
#    cdef np.ndarray[dtype=np.float64_t, ndim=1] grad = Xy - np.dot(XX.T, beta_init) + beta_init # gradient
#    cdef np.ndarray[dtype=np.float64_t, ndim=1] lam_arr = np.empty(p, dtype=np.float64) # gradient
#    
#    cdef double[:] beta_init_view = beta_init
#    cdef double[:] beta_view = beta
#    cdef double[:] grad_view = grad
#    cdef double[:] lam_arr_view = lam_arr
#    cdef double[:, :] XX_view = XX
#
#    for i in range(p):
#        lam_arr_view[i] = lam / X_norm2[i]
#        beta_view[i] = beta_init_view[i]
#
#    while diff / beta_max > rel_tol:
#        beta_max = 0.0
#        diff = 0.0
#        for i in range(p):
#            beta0_comp = beta_view[i]
#            beta_view[i] = prox_Lq(grad_view[i], lam_arr_view[i], q)
#
#            # update grad after beta[i]
#            tmp = beta_view[i] - beta0_comp
#
#            for j in range(p):
#                grad_view[j] -= XX_view[i, j] * tmp
#            grad_view[i] += tmp
#
#            if abs(tmp) > diff:
#                diff = abs(tmp)
#
#            tmp = abs(beta_view[i])
#            if tmp > beta_max:
#                beta_max = tmp
#
#        iter_count += 1
##        if iter_count > iter_max:
##            print('warning: iter_max break', file=sys.stderr)
##            break
#    return beta

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def ccbridge(np.ndarray[dtype=np.float64_t, ndim=2] XX, np.ndarray[dtype=np.float64_t, ndim=1] Xy, np.ndarray[dtype=np.float64_t, ndim=1] X_norm2, double lam, double q, np.ndarray[dtype=np.float64_t, ndim=1] beta_init, double abs_tol=1e-3, double rel_tol=1e-3):
#    '''
#    This function does bridge regression with L_q penalty for q >= 1.
#
#    Parameters
#    ----------
#    X: np.ndarray, dtype=np.float64, ndim=2
#        The 2-dim numpy array of independent variables;
#    y: np.ndarray, dtype=np.float64, ndim=1
#        The 1-dim numpy array of response variables;
#    lam: double, lam >= 0
#        Tuning parameter. For lam=0, I think it is interesting to implement it. May need special processing.
#    q: double, q >= 0
#        Choice of penalty. L_q penalty corresponds to L_q norm.
#    beta_init: np.ndarray, dtype=np.float64, ndim=2
#        The initialization of the iteration of the algorithm.
#
#    Returns
#    -------
#    beta_hat: np.ndarray, dtype=np.float64, ndim=1
#        The bridge estimator of the problem.
#
#    Details
#    -------
#    For this problem, we solve the following problem:
#        min_{\beta} 0.5 * \|y - X \beta\|_2^2 + \lam \|\beta\|_q^q
#    The strong convexity of the loss function and the convextiry of the penalty term guarantees the existence and uniqueness of the global minimizer, and also the convergence of the coordinate descent algorithm. Yes, we use coordinate descent.
#    '''
#    cdef int p = XX.shape[0]
#    cdef int iter_count = 0
#    cdef int i
#    cdef int j
#    cdef double beta_max = 1.0
#    cdef double diff = 1.0
#    cdef double tmp
#
#    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta0 = np.empty(p, dtype=np.float64) # old step
#    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta = np.empty(p, dtype=np.float64) # new step
#    cdef np.ndarray[dtype=np.float64_t, ndim=1] grad = Xy - np.dot(XX.T, beta_init) + beta_init # gradient
#    cdef np.ndarray[dtype=np.float64_t, ndim=1] lam_arr = np.empty(p, dtype=np.float64) # gradient
#    
#    cdef double[:] beta_init_view = beta_init
#    cdef double[:] beta0_view = beta0
#    cdef double[:] beta_view = beta
#    cdef double[:] grad_view = grad
#    cdef double[:, :] XX_view = XX
#
#    for i in range(p):
#        beta0_view[i] = beta_init_view[i] - 1.0
#        # beta0_view[i] = beta_init_view[i] - 1.0
#        lam_arr[i] = lam / X_norm2[i]
#        beta_view[i] = beta_init_view[i]
#
#    # beta_view[:] = beta_init_view
#
#    while diff / beta_max > rel_tol:
#        for i in range(p):
#            beta0_view[i] = beta_view[i]
#        # beta0_view[:] = beta_view
#        beta_max = 0.0
#        diff = 0.0
#
#        # update beta
#        for i in range(p):
#            # update beta[i]
#            beta_view[i] = prox_Lq(grad[i], lam_arr[i], q)
#            # beta_view[i] = prox_Lq(grad[i], lam_arr[i], q)
#
#            # update grad after beta[i]
#
#            tmp = beta_view[i] - beta0_view[i]
#
#            for j in range(p):
#                grad_view[j] -= XX_view[i, j] * tmp
#            # grad -= XX[i, :] * (beta_view[i] - beta0_view[i])
#            grad_view[i] += tmp
#            # grad[i] += beta_view[i] - beta0_view[i]
#
#            if abs(tmp) > diff:
#                diff = abs(tmp)
#
#            tmp = abs(beta_view[i])
#            if tmp > beta_max:
#                beta_max = tmp
#        # beta_max = np.amax(np.absolute(beta))
#
#        iter_count += 1
##        if iter_count > iter_max:
##            print('warning: iter_max break', file=sys.stderr)
##            break
#    return beta
#
