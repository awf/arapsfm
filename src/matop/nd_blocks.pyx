# nd_blocks.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

from libc.string cimport memcpy, memset

# nd_blocks
def nd_blocks(np.ndarray A, object I, object dim, object cval = 0):
    cdef Py_ssize_t itemsize = A.dtype.itemsize 
    cdef np.uint8_t * A_ = <np.uint8_t *>A.data

    cdef np.ndarray[np.int32_t, ndim=2] J = np.require(
        np.atleast_2d(I), dtype=np.intc)
    cdef Py_ssize_t N = J.shape[0]

    cdef np.ndarray a_cval = np.require(cval, dtype=A.dtype)
    cdef np.uint8_t * a_cval_ = <np.uint8_t *>a_cval.data

    cdef np.ndarray[np.int32_t, ndim=1] d = np.require(dim, dtype=np.intc)

    cdef Py_ssize_t i, D = 1
    for i in range(d.shape[0]):
        D *= d[i]

    cdef np.ndarray B = np.empty((N, D), dtype=A.dtype, order='C')
    cdef np.uint8_t * B_ = <np.uint8_t *>B.data

    cdef Py_ssize_t j, k, l, m, offset, base_offset
    cdef bint out_of_bounds
    cdef np.uint8_t * src

    for i in range(N):
        for k in range(D):
            # set `base_offset`
            base_offset = 0
            out_of_bounds = False
            l = k

            for j in range(A.ndim):
                m = A.ndim - 1 - j
                offset = (J[i, m] + (l % d[m]) - d[m] / 2)
                if offset < 0 or offset >= A.shape[m]:
                    out_of_bounds = True
                    break

                base_offset += A.strides[m] * offset
                l /= d[m]

            if not out_of_bounds:
                src = A_ + base_offset
            else:
                src = a_cval_

            memcpy(B_ + (D * i + k) * itemsize, src, itemsize)

    return B
                
