cdef void st_dgemv(int m, int n, double alpha, double *A_pt, double *x_pt, double beta, double *y_pt)
cdef void sn_dgemv(int m, int n, double alpha, double *A_pt, double *x_pt, double beta, double *y_pt)
cdef void s_daxpy(int n, double alpha, double *x_pt, double *y_pt)
cdef void st_dsyrk(int m, int n, double alpha, double *A_pt, double beta, double *C_pt)
cdef void sn_dsyrk(int m, int n, double alpha, double *A_pt, double beta, double *C_pt)
cdef void s_dcopy(int n, double *x_pt, double *y_pt)


cdef void mat_T_mat(int m, int n, double *A_pt, double *C_pt)
cdef void mat_mat_T(int m, int n, double *A_pt, double *C_pt)

