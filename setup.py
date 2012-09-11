# setup.py
# coding=utf-8

import sys, os
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

from numpy import get_include

# Don't have exceptions in MSVC
REMOVE_EXCEPTION_MACROS = [
    ('_HAS_EXCEPTIONS', 0), 
    # ('_STATIC_CPPLIB', 1) # DEPRECATED in MSVC 2010
]

# Include directories and library paths

# sslm 
# from "http://www.inf.ethz.ch/personal/chzach/opensource.html"
SSLM_BASE_SRC = [
    #'Base/v3d_exception.h',
    #'Base/v3d_serialization.h',
]

SSLM_MATH_SRC = [
    #'Math/v3d_linear.h',
    #'Math/v3d_linearbase.h',
    #'Math/v3d_mathutilities.h',
    #'Math/v3d_nonlinlsq.h',
    'Math/v3d_nonlinlsq.cpp',
    #'Math/v3d_optimization.h',
    'Math/v3d_optimization.cpp',
    'Math/v3d_optimization_lm.cpp',
]

SSLM_ROOT = 'SSLM'

def sslm_full_path(path):
    return os.path.join(SSLM_ROOT, path)

SSLM_BASE_SRC = map(sslm_full_path, SSLM_BASE_SRC)
SSLM_MATH_SRC = map(sslm_full_path, SSLM_MATH_SRC)
COLAMD_INC = 'C:/dev/include'
COLAMD_LIB = 'C:/dev/lib'

# Numpy from http://www.scipy.org/Cookbook/SWIG_NumPy_examples
import numpy as np

try:
    NUMPY_INC = np.get_include()
except AttributeError:
    NUMPY_INC = np.get_numpy_include()

include_dirs = [NUMPY_INC, '.', 'shared/']

# Extensions
setup(
    ext_modules=[
        
        Extension('Test.test_shortest_path',
                  ['Test/test_shortest_path.pyx',
                   'Silhouette/shortest_path.cpp'],
                  include_dirs=include_dirs + [SSLM_ROOT],
                  define_macros=REMOVE_EXCEPTION_MACROS,
                  language='c++'),

        # Extension('Test.test_mesh',
        #           ['Test/test_mesh.pyx'],
        #           include_dirs=include_dirs + [SSLM_ROOT],
        #           define_macros=REMOVE_EXCEPTION_MACROS,
        #           language='c++'),

        # Extension('Test.test_problem',
        #           ['Test/test_problem.pyx',
        #            'Solve/problem.cpp',
        #            'Geometry/mesh_walker.cpp'] +
        #           SSLM_BASE_SRC + 
        #           SSLM_MATH_SRC,
        #           include_dirs=include_dirs + [SSLM_ROOT, COLAMD_INC],
        #           library_dirs=[COLAMD_LIB],
        #           libraries=['colamd'],
        #           # library_dirs=[COLAMD_LIB],
        #           # libraries=['colamd'],
        #           define_macros=REMOVE_EXCEPTION_MACROS,
        #           language='c++'),

        # Extension('Test.test_mesh_walker',
        #           ['Test/test_mesh_walker.pyx',
        #            'Geometry/mesh_walker.cpp'],
        #           include_dirs=include_dirs + [SSLM_ROOT],
        #           define_macros=REMOVE_EXCEPTION_MACROS,
        #           language='c++'),
        ],

    cmdclass = {'build_ext' : build_ext},
)

