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
