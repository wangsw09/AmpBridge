import numpy as np
cimport numpy as np

from scipy.linalg.cython_blas cimport dgemv

cdef char TRANS = 116
cdef char NOTRANS = 110
cdef int INCX = 1
cdef int INCY = 1

cdef void dgemv_np(char trans, int m, int n,
        double alpha, double *A_pt, double *x_pt, double beta, double *y):
    if trans == 116:
        dgemv(&NOTRANS, &m, &n, &alpha, A_pt, &m, x_pt, &INCX, &beta, y, &INCY)
    else:
        dgemv(&TRANS, &m, &n, &alpha, A_pt, &m, x_pt, &INCX, &beta, y, &INCY)


