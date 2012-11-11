# lm_alt_solvers.pyx
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

cdef extern from "lm_alt_solvers.h":
    int solve_instance_c "solve_instance" (np.ndarray npy_T,
                                           np.ndarray npy_V,
                                           np.ndarray npy_Xg,
                                           np.ndarray npy_s,
                                           np.ndarray npy_X,
                                           np.ndarray npy_V1,
                                           np.ndarray npy_U,
                                           np.ndarray npy_L,
                                           # np.ndarray npy_C,
                                           # np.ndarray npy_P,
                                           np.ndarray npy_S,
                                           np.ndarray npy_SN,
                                           np.ndarray npy_Rx,
                                           np.ndarray npy_Ry,
                                           np.ndarray npy_lambdas,
                                           np.ndarray npy_preconditioners,
                                           np.ndarray npy_piecewisePolynomial,
                                           int narrowBand,
                                           bint uniformWeights,
                                           bint fixedScale,
                                           OptimiserOptions * options)

    int solve_core_c "solve_core" (np.ndarray npy_T,
                                   np.ndarray npy_V,
                                   list list_Xg,
                                   list list_s,
                                   list list_X,
                                   list list_V1,
                                   np.ndarray npy_lambdas,
                                   np.ndarray npy_preconditioners,
                                   int narrowBand,
                                   bint uniformWeights,
                                   OptimiserOptions * options)

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
                'OPTIMIZER_CONVERGED',
                'OPTIMIZER_CANT_BEGIN_ITERATION']

def solve_instance(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] V, 
                   np.ndarray[np.float64_t, ndim=2, mode='c'] Xg, 
                   np.ndarray[np.float64_t, ndim=2, mode='c'] s, 
                   np.ndarray[np.float64_t, ndim=2, mode='c'] X, 
                   np.ndarray[np.float64_t, ndim=2, mode='c'] V1, 
                   np.ndarray[np.float64_t, ndim=2, mode='c'] U,  
                   np.ndarray[np.int32_t, ndim=1] L,
                   # np.ndarray[np.int32_t, ndim=1] C,
                   # np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] S,  
                   np.ndarray[np.float64_t, ndim=2, mode='c'] SN,  
                   np.ndarray[np.float64_t, ndim=2, mode='c'] Rx,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] Ry,
                   np.ndarray[np.float64_t, ndim=1] lambdas,
                   np.ndarray[np.float64_t, ndim=1] preconditioners,
                   np.ndarray[np.float64_t, ndim=1] piecewisePolynomial,
                   int narrowBand,
                   bint uniformWeights,
                   bint fixedScale,
                   **kwargs):

    if lambdas.shape[0] != 4:
        raise ValueError('lambdas.shape[0] != 4')

    if preconditioners.shape[0] != 5:
        raise ValueError('preconditioners.shape[0] != 5')

    if piecewisePolynomial.shape[0] != 2:
        raise ValueError('piecewisePolynomial.shape[0] != 2')

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    cdef int status = solve_instance_c(T, V, Xg, s, X, V1, U, L,
                                       # C, P, 
                                       S, SN, Rx, Ry,
                                       lambdas,
                                       preconditioners,
                                       piecewisePolynomial,
                                       narrowBand,
                                       uniformWeights,
                                       fixedScale,
                                       &options)

    return status, STATUS_CODES[status]

def solve_core(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
               np.ndarray[np.float64_t, ndim=2, mode='c'] V, 
               list Xg,
               list s,
               list X,
               list V1,
               np.ndarray[np.float64_t, ndim=1] lambdas,
               np.ndarray[np.float64_t, ndim=1] preconditioners,
               int narrowBand,
               bint uniformWeights,
               **kwargs):

    if lambdas.shape[0] != 2:
        raise ValueError('lambdas.shape[0] != 2')

    if preconditioners.shape[0] != 4:
        raise ValueError('preconditioners.shape[0] != 4')

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    cdef int status = solve_core_c(T, V, Xg, s, X, V1, 
                                   lambdas, 
                                   preconditioners,
                                   narrowBand,
                                   uniformWeights,
                                   &options)

    return status, STATUS_CODES[status]

