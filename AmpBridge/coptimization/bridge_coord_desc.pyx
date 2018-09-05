import numpy as np
cimport numpy as np
cimport cython

from ..cscalar.proximal cimport cproxLq
from ..clinalg.cython_blas_wrapper cimport st_dgemv, sn_dgemv, s_daxpy, mat_T_mat, s_dcopy

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype=np.float64_t, ndim=1] _cbridge_Lq(
        np.ndarray[dtype=np.float64_t, ndim=2] XTX, # XTX[i, j] = <Xi, Xj> / <Xi, Xi>
        np.ndarray[dtype=np.float64_t, ndim=1] XTy, # XTy[i] = <Xi, y> / <Xi, Xi>
        np.ndarray[dtype=np.float64_t, ndim=1] X_norm2_inv,
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

    s_dcopy(p, &beta_init[0], &beta[0])

    s_dcopy(p, &XTy[0], &grad_move[0])
    s_daxpy(p, 1.0, &beta_init[0], &grad_move[0])
    st_dgemv(p, p, -1.0, &XTX[0, 0], &beta_init[0], 1.0, &grad_move[0])

    while iter_diff > abs_tol:
        iter_diff = 0.0

        for i in range(p):
            betai0 = beta[i]
            beta[i] = cproxLq(grad_move[i], lam * X_norm2_inv[i], q)

            local_diff = beta[i] - betai0
            iter_diff += abs(local_diff)

            s_daxpy(p, -local_diff, &XTX[i, 0], &grad_move[0])
            grad_move[i] += local_diff

#             if local_diff != 0:
#                 for j in range(p):
#                     if j != i:
#                         grad_move[j] -= XTX[i, j] * local_diff

        iter_count += 1
        if iter_count > iter_max:
            break
    return beta


@cython.boundscheck(False)
@cython.wraparound(False)
def bridge_Lq(
        np.ndarray[dtype=np.float64_t, ndim=2] X, # XTX[i, j] = <Xi, Xj> / <Xi, Xi>
        np.ndarray[dtype=np.float64_t, ndim=1] y, # XTy[i] = <Xi, y> / <Xi, Xi>
        double lam, double q, double abs_tol, int iter_max):

    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int i = 0

    cdef np.ndarray[dtype=np.float64_t, ndim=1] X_norm2_inv = np.zeros(p, dtype=np.float64)
    # cdef np.ndarray[dtype=np.float64_t, ndim=2] XTX = np.dot(X.T, X) * X_norm2_inv
    cdef np.ndarray[dtype=np.float64_t, ndim=2] XTX = np.zeros((p, p), dtype=np.float64)
    mat_T_mat(n, p, &X[0, 0], &XTX[0, 0])
    for i in range(p):
        X_norm2_inv[i] = 1.0 / XTX[i, i]

    XTX *= X_norm2_inv

    cdef np.ndarray[dtype=np.float64_t, ndim=1] XTy = np.zeros(p, dtype=np.float64)
    st_dgemv(n, p, 1, &X[0, 0], &y[0], 0, &XTy[0])
    XTy *= X_norm2_inv

    return _cbridge_Lq(XTX, XTy, X_norm2_inv, np.zeros(p, dtype=np.float64), lam, q, abs_tol, iter_max)


@cython.boundscheck(False)
@cython.wraparound(False)
def vec_bridge_Lq(
        np.ndarray[dtype=np.float64_t, ndim=2] X, # XTX[i, j] = <Xi, Xj> / <Xi, Xi>
        np.ndarray[dtype=np.float64_t, ndim=1] y, # XTy[i] = <Xi, y> / <Xi, Xi>
        np.ndarray[dtype=np.float64_t, ndim=1] lams, double q, double abs_tol, int iter_max):
    """
    lams must be in the increasing order.
    """
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0

    cdef np.ndarray[dtype=np.float64_t, ndim=1] X_norm2_inv = 1.0 / np.linalg.norm(X, ord=2, axis=0) ** 2
    cdef np.ndarray[dtype=np.float64_t, ndim=2] XTX = np.dot(X.T, X) * X_norm2_inv
    cdef np.ndarray[dtype=np.float64_t, ndim=1] XTy = np.dot(X.T, y) * X_norm2_inv
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Beta = np.zeros((p, k), dtype=np.float64)

    Beta[:, k-1] =  _cbridge_Lq(XTX, XTy, X_norm2_inv, np.zeros(p, dtype=np.float64), lams[k-1], q, abs_tol, iter_max)
    for i in range(k - 2, -1, -1):
        Beta[:, i] = _cbridge_Lq(XTX, XTy, X_norm2_inv, Beta[:, i+1], lams[i], q, abs_tol, iter_max)
    
    return Beta

