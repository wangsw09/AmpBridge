import numpy as np
cimport numpy as np

# from ..cscalar import *

cdef np.ndarray[dtype=np.float64_t, ndim=1] ccBridgeL1(
        np.ndarray[dtype=np.float64_t, ndim=2] XTX,
        np.ndarray[dtype=np.float64_t, ndim=1] XTy,
        np.ndarray[dtype=np.float64_t, ndim=1] X_norm2,
        np.ndarray[dtype=np.float64_t, ndim=1] beta_init,
        double lam, double abs_tol, double rel_tol, int iter_max):
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


cdef np.ndarray[dtype=np.float64_t, ndim=1] ccBridgeL2(np.ndarray[dtype=np.float64_t, ndim=2] XX, np.ndarray[dtype=np.float64_t, ndim=1] Xy, np.ndarray[dtype=np.float64_t, ndim=1] X_norm2, double lam, double q, np.ndarray[dtype=np.float64_t, ndim=1] beta_init, double abs_tol=1e-3, double rel_tol=1e-3, int iter_max=1000):
    pass


cdef np.ndarray[dtype=np.float64_t, ndim=1] ccBridgeLq(np.ndarray[dtype=np.float64_t, ndim=2] XX, np.ndarray[dtype=np.float64_t, ndim=1] Xy, np.ndarray[dtype=np.float64_t, ndim=1] X_norm2, double lam, double q, np.ndarray[dtype=np.float64_t, ndim=1] beta_init, double abs_tol=1e-3, double rel_tol=1e-3, int iter_max=1000):
    pass
