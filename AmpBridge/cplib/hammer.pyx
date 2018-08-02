from type cimport RealFunc1

cdef double biSearchLU(RealFunc1 f, double L, double U, double tol):
    cdef double mid = 0
    cdef int Lsign = (f(L) > 0)
    cdef int Usign = 1 - Lsign

    while U - L > tol:
        mid = (U + L) / 2.0
        if (f(mid) > 0) == Usign:
            U = mid
        else:
            L = mid
    return (U + L) / 2.0

cdef double biSearchL(RealFunc1 f, double L, double incre, double tol):
    cdef int Lsign = (f(L) > 0)
    cdef double U = L + incre

    while (f(U) > 0) == Lsign:
        incre = incre * 2
        U = U + incre

    L = U - incre
    cdef double mid = 0
    cdef int Usign = 1 - Lsign

    while U - L > tol:
        mid = (U + L) / 2.0
        if (f(mid) > 0) == Usign:
            U = mid
        else:
            L = mid
    return (U + L) / 2.0

cdef double biSearchU(RealFunc1 f, double U, double decre, double tol):
    cdef int Usign = (f(U) > 0)
    cdef double L = U - decre

    while (f(L) > 0) == Usign:
        decre = decre * 2
        L = L - decre

    U = L + decre
    cdef double mid = 0
    cdef int Lsign = 1 - Usign

    while U - L > tol:
        mid = (U + L) / 2.0
        if (f(mid) > 0) == Usign:
            U = mid
        else:
            L = mid
    return (U + L) / 2.0

