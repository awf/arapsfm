# test_quaternion.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from "Geometry/quaternion.h":
    void quat_Unsafe(double * x, double * q)
    void quatMultiply_Unsafe(double * p, double * q, double * r)
    void rotationMatrix_Unsafe(double * q, double * R)
    void quatMultiply_dp_Unsafe(double * q, double * Dp)
    void quatMultiply_dq_Unsafe(double * p, double * Dq)

def quat(np.ndarray[DTYPE_t, ndim=1] x):
    if x.shape[0] != 3:
        raise ValueError('x.shape[0] != 3')

    cdef np.ndarray[DTYPE_t] q = np.empty(4, dtype=DTYPE)
    quat_Unsafe(<double *>x.data, <double *>q.data)

    return q

def rotationMatrix(np.ndarray[DTYPE_t, ndim=1] q):
    if q.shape[0] != 4:
        raise ValueError('q.shape[0] != 4')

    cdef np.ndarray[DTYPE_t, ndim=2] R = np.empty((3, 3), dtype=DTYPE)
    rotationMatrix_Unsafe(<double *>q.data, <double *>R.data)

    return R

def quatMultiply(np.ndarray[DTYPE_t, ndim=1] p,
                 np.ndarray[DTYPE_t, ndim=1] q):
    if p.shape[0] != 4:
        raise ValueError('p.shape[0] != 4')

    if q.shape[0] != 4:
        raise ValueError('q.shape[0] != 4')

    cdef np.ndarray[DTYPE_t, ndim=1] r = np.empty(4, dtype=DTYPE)

    quatMultiply_Unsafe(<double *>p.data, <double *>q.data, <double *>r.data)

    return r

def quatMultiply_dp_dq(np.ndarray[DTYPE_t, ndim=1] p,
                       np.ndarray[DTYPE_t, ndim=1] q):

    if p.shape[0] != 4:
        raise ValueError('p.shape[0] != 4')

    if q.shape[0] != 4:
        raise ValueError('q.shape[0] != 4')

    cdef np.ndarray[DTYPE_t, ndim=2] Dp = np.empty((4, 4), dtype=DTYPE)
    quatMultiply_dp_Unsafe(<double *>q.data, <double *>Dp.data)

    cdef np.ndarray[DTYPE_t, ndim=2] Dq = np.empty((4, 4), dtype=DTYPE)
    quatMultiply_dq_Unsafe(<double *>p.data, <double *>Dq.data)

    return Dp, Dq

