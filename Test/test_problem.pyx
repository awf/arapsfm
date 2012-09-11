# test_mesh.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "Solve/optimiser_options.h":
    struct OptimiserOptions:
        int maxIterations
        int minIterations
        double tau
        double lambda_ "lambda"
        double gradientThreshold
        double updateThreshold
        double improvementThreshold
        bint useAsymmetricLambda
        int verbosenessLevel

cdef extern from "test_problem.h":
    int test_problem_c "test_problem" (
        np.ndarray V,
        np.ndarray T,
        np.ndarray X,
        np.ndarray V1,
        np.ndarray C,
        np.ndarray P,
        np.ndarray lambdas,
        OptimiserOptions * opt)

    int test_problem2_c "test_problem2" (
        np.ndarray V,
        np.ndarray T,
        np.ndarray C,
        np.ndarray P,
        np.ndarray lambdas,
        OptimiserOptions * opt)

# additional_optimiser_options
DEFAULT_OPTIMISER_OPTIONS = {
    'maxIterations' : 50,
    'minIterations' : 10,
    'tau' : 1e-3,
    'lambda' : 1e-3,
    'gradientThreshold' : 1e-8,
    'updateThreshold' : 1e-8,
    'improvementThreshold' : 1e-8,
    'useAsymmetricLambda' : True,
    'verbosenessLevel' : 1
}

cdef int additional_optimiser_options(OptimiserOptions * options, dict kwargs) except -1:
    for key in kwargs:
        if key not in DEFAULT_OPTIMISER_OPTIONS:
            raise ValueError("'%s' not an optimiser option" % key)

    additional_options = DEFAULT_OPTIMISER_OPTIONS.copy()
    additional_options.update(kwargs)

    options.maxIterations = additional_options['maxIterations']
    options.minIterations = additional_options['minIterations']
    options.tau = additional_options['tau']
    options.lambda_ = additional_options['lambda']
    options.gradientThreshold = additional_options['gradientThreshold']
    options.updateThreshold = additional_options['updateThreshold']
    options.improvementThreshold = additional_options['improvementThreshold']
    options.useAsymmetricLambda = additional_options['useAsymmetricLambda']
    options.verbosenessLevel = additional_options['verbosenessLevel']

    return 0

STATUS_CODES = ['OPTIMIZER_TIMEOUT',
                'OPTIMIZER_SMALL_UPDATE',
                'OPTIMIZER_SMALL_IMPROVEMENT',
                'OPTIMIZER_CONVERGED']

# test_problem
def test_problem(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                 np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                 np.ndarray[np.float64_t, ndim=2, mode='c'] X,
                 np.ndarray[np.float64_t, ndim=2, mode='c'] V1,
                 np.ndarray[np.int32_t, ndim=1] C,
                 np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                 np.ndarray[np.float64_t, ndim=1] lambdas,
                 **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 2:
        raise ValueError('lambdas.shape[0] != 2')

    cdef int status = test_problem_c(V, T, X, V1, C, P, lambdas, &options)

    return STATUS_CODES[status]
    
# test_problem2
def test_problem2(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                  np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                  np.ndarray[np.int32_t, ndim=1] C,
                  np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                  np.ndarray[np.float64_t, ndim=1] lambdas,
                  **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 2:
        raise ValueError('lambdas.shape[0] != 2')

    cdef int status = test_problem2_c(V, T, C, P, lambdas, &options)

    return STATUS_CODES[status]
