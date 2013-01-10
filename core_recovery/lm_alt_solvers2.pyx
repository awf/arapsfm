# lm_alt_solvers2.pyx
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

cdef extern from "lm_alt_solvers2.h":
    int solve_core_c 'solve_core' (
               np.ndarray npy_T,
               np.ndarray npy_V,
               list list_s,
               np.ndarray npy_kg,
               list list_Xgb,
               list list_yg,
               list list_Xg,
               np.ndarray npy_k,
               np.ndarray npy_Xb,
               list list_y,
               list list_X,
               list list_V1,
               np.ndarray npy_lambdas,
               np.ndarray npy_preconditioners,
               bint uniformWeights,
               bint fixedXgb,
               OptimiserOptions * options,
               object callback)

    int solve_instance_c 'solve_instance' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        np.ndarray npy_s,
        np.ndarray npy_kg,
        list list_Xgb,
        list list_yg,
        list list_Xg,
        np.ndarray npy_k,
        np.ndarray npy_Xb,
        np.ndarray npy_y,
        np.ndarray npy_X, 
        list list_y0,
        list list_X0,
        list list_s0,
        np.ndarray npy_V1, 
        np.ndarray npy_U, 
        np.ndarray npy_L, 
        np.ndarray npy_S, 
        np.ndarray npy_SN, 
        np.ndarray npy_C,
        np.ndarray npy_P,
        np.ndarray npy_lambdas,
        np.ndarray npy_preconditioners,
        np.ndarray npy_piecewisePolynomial,
        int narrowBand,
        bint uniformWeights,
        bint fixedScale,
        bint fixedGlobalRotation,
        bint noSilhouetteUpdate,
        OptimiserOptions * options,
        object callback)

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

# solve_core
def solve_core(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
               np.ndarray[np.float64_t, ndim=2, mode='c'] V, 
               list s,
               np.ndarray[np.int32_t, ndim=1] kg,
               list Xgb,
               list yg,
               list Xg,
               np.ndarray[np.int32_t, ndim=1] k,
               np.ndarray[np.float64_t, ndim=2, mode='c'] Xb, 
               list y,
               list X,
               list V1,
               np.ndarray[np.float64_t, ndim=1] lambdas, 
               np.ndarray[np.float64_t, ndim=1] preconditioners, 
               bint uniformWeights,
               bint fixedXgb,
               **kwargs):

    assert lambdas.shape[0] == 8
    assert preconditioners.shape[0] == 4

    callback = kwargs.pop('callback', None)

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    cdef int status = solve_core_c(
        T, V, s, 
        kg, Xgb, yg, Xg,
        k, Xb, y, X,
        V1, 
        lambdas, preconditioners, uniformWeights, fixedXgb,
        &options, callback)

    return status, STATUS_CODES[status]

# solve_instance
def solve_instance(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] s,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] kg,
                   list Xgb,
                   list yg,
                   list Xg,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] k,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] Xb,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] y,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] X,
                   list y0,
                   list X0,
                   list s0,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] V1,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] U,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] L,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] S,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] SN,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] C,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                   np.ndarray[np.float64_t, ndim=1, mode='c'] lambdas,
                   np.ndarray[np.float64_t, ndim=1, mode='c'] preconditioners,
                   np.ndarray[np.float64_t, ndim=1, mode='c'] piecewisePolynomial,
                   np.int32_t narrowBand,
                   bint uniformWeights,
                   bint fixedScale,
                   bint fixedGlobalRotation,
                   bint noSilhouetteUpdate,
                   **kwargs):

    assert lambdas.shape[0] == 10
    assert preconditioners.shape[0] == 5
    assert piecewisePolynomial.shape[0] == 2

    callback = kwargs.pop('callback', None)

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    cdef int status = solve_instance_c(T, V, s,
        kg,  Xgb,  yg,  Xg,  
        k,  Xb,  y,  X,  y0, X0, s0, 
        V1,  U,  L,  S,  SN,  
        C, P, 
        lambdas,  preconditioners,  piecewisePolynomial, 
        narrowBand, uniformWeights, fixedScale, fixedGlobalRotation,
        noSilhouetteUpdate,
        &options, callback)

    return status, STATUS_CODES[status]

