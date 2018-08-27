import numpy as np
cimport numpy as np
cimport cython

from ..cscalar.proximal cimport cproxLq

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype=np.float64_t, ndim=1] _cbridge_Lq(
        np.ndarray[dtype=np.float64_t, ndim=2] XTX, # XTX[i, j] = <Xi, Xj> / <Xi, Xi>
        np.ndarray[dtype=np.float64_t, ndim=1] XTy, # XTy[i] = <Xi, y> / <Xi, Xi>
        np.ndarray[dtype=np.float64_t, ndim=1] X_norm2,
        np.ndarray[dtype=np.float64_t, ndim=1] beta_init,
        double lam, double q, double abs_tol, int iter_max):

    cdef int p = XTX.shape[0]
    cdef int iter_count = 0
    cdef int i = 0
    cdef int j = 0

    cdef double iter_diff = 1.0
    cdef double local_diff = 0.0
    cdef double betai0 = 0  # record the previous value for each coordinate

    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] grad_move = np.zeros(p, dtype=np.float64)

    for i in range(p):
        beta[i] = beta_init[i]
        grad_move[i] = XTy[i] + XTX[i, i] * beta_init[i]
        for j in range(p):
            grad_move[i] -= XTX[j, i] * beta_init[j]

    while iter_diff > abs_tol:
        iter_diff = 0.0

        for i in range(p):
            betai0 = beta[i]
            beta[i] = cproxLq(grad_move[i], lam / X_norm2[i], q)

            local_diff = beta[i] - betai0
            iter_diff += abs(local_diff)

            if local_diff != 0:
                for j in range(p):
                    if j != i:
                        grad_move[j] -= XTX[i, j] * local_diff

        iter_count += 1
        if iter_count > iter_max:
            break
    return beta


@cython.boundscheck(False)
@cython.wraparound(False)
def cbridge_Lq(
        np.ndarray[dtype=np.float64_t, ndim=2] X, # XTX[i, j] = <Xi, Xj> / <Xi, Xi>
        np.ndarray[dtype=np.float64_t, ndim=1] y, # XTy[i] = <Xi, y> / <Xi, Xi>
        double lam, double q, double abs_tol, int iter_max):

    cdef int n = X.shape[0]
    cdef int p = X.shape[1]

    cdef np.ndarray[dtype=np.float64_t, ndim=1] X_norm2 = np.linalg.norm(X, ord=2, axis=0) ** 2
    cdef np.ndarray[dtype=np.float64_t, ndim=2] XTX = np.dot(X.T, X) / X_norm2
    cdef np.ndarray[dtype=np.float64_t, ndim=1] XTy = np.dot(X.T, y) / X_norm2

    return _cbridge_Lq(XTX, XTy, X_norm2, np.zeros(p, dtype=np.float64), lam, q, abs_tol, iter_max)

