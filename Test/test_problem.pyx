# test_mesh.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "test_problem.h":
    void test_problem_c "test_problem" ()

# test_problem
def test_problem():
    test_problem_c()
    
