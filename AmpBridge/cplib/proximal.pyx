from libc.math cimport sqrt

cdef double _cproxL1(double x, double t):
    return max(abs(x) - t, 0) * ((x > 0) * 2.0 - 1.0)

cdef double _cproxL2(double x, double t):
    return x / (1.0 + 2.0 * t)

cdef double _cproxL1p5(double x, double t):
    return (sqrt(abs(x) + 0.5625 * t ** 2.0) - 0.75 * t) ** 2 * ((x > 0) * 2.0 - 1.0)

cdef double _cproxLq_2L(double x, double t, double q, double tol):
    # q < 2
    cdef double absx = abs(x)
    cdef double L = 0
    cdef double U = absx / (1.0 + t * q * absx ** (q - 2))
    cdef double mid = 0

    cdef int Usign = 1

    while U - L > tol:
        mid = (U + L) / 2.0
        if (mid + t * q * mid ** (q - 1) > absx) == Usign:
            U = mid
        else:
            L = mid
    return (U + L) / 2.0 * ((x > 0) * 2 - 1)

cdef double _cproxLq_2R(double x, double t, double q, double tol):
    cdef double absx = abs(x)
    cdef double L = absx / (1.0 + t * q * absx ** (q - 2))
    cdef double U = absx
    cdef double mid = 0

    cdef int Usign = 1

    while U - L > tol:
        mid = (U + L) / 2.0
        if (mid + t * q * mid ** (q - 1) > absx) == Usign:
            U = mid
        else:
            L = mid
    return (U + L) / 2.0 * ((x > 0) * 2 - 1)

cdef double _cproxLq_2R1(double x, double t, double q, double tol):
    # gradient descent
    cdef double absx = abs(x)
    cdef double z = absx
    cdef double s = 1 / (1.0 + t * q * (q - 1.0) * z ** (q - 2.0))

    cdef double z0 = z + 2.0 * tol

    while abs(z - z0) > tol:
        z0 = z
        z = z - s * (z - absx + t * q * z ** (q - 1.0))
    return z * ((x > 0) * 2 - 1)

cdef double _cproxLq_2R2(double x, double t, double q, double tol):
    # Newton
    cdef double absx = abs(x)
    cdef double z = absx
    cdef double s = 0
    cdef double z0 = z + 2.0 * tol

    while abs(z - z0) > tol:
        z0 = z
        s = 1 / (1.0 + t * q * (q - 1.0) * z ** (q - 2.0))
        z = z - s * (z - absx + t * q * z ** (q - 1.0))
    return z * ((x > 0) * 2 - 1)

cdef double _cproxLq(double x, double t, double q, double tol):
    if q < 2:
        return _cproxLq_2L(x, t, q, tol)
    else:
        return _cproxLq_2R2(x, t, q, tol)

cdef double cproxLq(double x, double t, double q, double tol=1e-8):
    if q == 1:
        return _cproxL1(x, t)
    elif q == 2:
        return _cproxL2(x, t)
    elif q == 1.5:
        return _cproxL1p5(x, t)
    else:
        return _cproxLq(x, t, q, tol)


cdef double _cproxL1_dx(double x, double t):
    return (abs(x) > t)

cdef double _cproxL2_dx(double x, double t):
    return 1.0 / (1.0 + 2.0 * t)

cdef double _cproxLq_dx(double x, double t, double q, double tol):
    return 1.0 / (1.0 + t * q * (q - 1.0) * _cproxLq(abs(x), t, q, tol) ** (q - 2.0))

cdef double cproxLq_dx(double x, double t, double q, double tol=1e-8): # this seems faster than proxlLq
    if q == 1:
        return _cproxL1_dx(x, t)
    elif q == 2:
        return _cproxL2_dx(x, t)
    else:
        return _cproxLq_dx(x, t, q, tol)


cdef double _cproxL1_dt(double x, double t):
    return (abs(x) > t) * ((x < 0) * 2 - 1)

cdef double _cproxL2_dt(double x, double t):
    return - 2.0 * x / (1.0 + 2.0 * t) ** 2

cdef double _cproxLq_dt(double x, double t, double q, double tol):
    cdef double z = _cproxLq(abs(x), t, q, tol)
    return q * z ** (q - 1) / (1.0 + t * q * (q - 1.0) * z ** (q - 2.0)) * ((x < 0) * 2 - 1)

cdef double cproxLq_dt(double x, double t, double q, double tol=1e-8): # this seems faster than proxlLq
    if q == 1:
        return _cproxL1_dt(x, t)
    elif q == 2:
        return _cproxL2_dt(x, t)
    else:
        return _cproxLq_dt(x, t, q, tol)


cdef double cproxLq_inv(double z, double t, double q):
    return z + t * q * abs(z) ** (q - 1) * ((z > 0) * 2 - 1)

