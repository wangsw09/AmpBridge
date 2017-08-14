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


