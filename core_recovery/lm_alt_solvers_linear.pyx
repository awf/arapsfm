# lm_alt_solvers_linear.pyx
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

cdef extern from "lm_alt_solvers_linear.h":
    int solve_instance_c 'solve_instance' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        np.ndarray npy_s,
        np.ndarray npy_kg,
        list list_Xgb,
        list list_yg,
        list list_Xg,
        np.ndarray npy_dg,
        np.ndarray npy_V1,
        np.ndarray npy_U, 
        np.ndarray npy_L, 
        np.ndarray npy_S, 
        np.ndarray npy_SN, 
        np.ndarray npy_C,
        np.ndarray npy_P,
        np.ndarray npy_lambdas,
        np.ndarray npy_preconditioners,
        np.int32_t narrowBand,
        bint uniformWeights,
        bint fixedScale,
        bint fixedGlobalRotation,
        bint fixedTranslation,
        bint noSilhouetteUpdate,
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
                   np.ndarray[np.float64_t, ndim=2, mode='c'] s,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] kg,
                   list Xgb,
                   list yg,
                   list Xg,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] dg,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] V1,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] U,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] L,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] S,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] SN,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] C,
                   np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                   np.ndarray[np.float64_t, ndim=1, mode='c'] lambdas,
                   np.ndarray[np.float64_t, ndim=1, mode='c'] preconditioners,
                   np.int32_t narrowBand,
                   bint uniformWeights,
                   bint fixedScale,
                   bint fixedGlobalRotation,
                   bint fixedTranslation,
                   bint noSilhouetteUpdate,
                   **kwargs):

    assert lambdas.shape[0] == 4
    assert preconditioners.shape[0] == 5

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    cdef int status = solve_instance_c(T, V, s,
        kg, Xgb, yg, Xg, dg,
        V1,  U,  L,  S,  SN,
        C, P, 
        lambdas,  preconditioners,  
        narrowBand, uniformWeights, fixedScale, fixedGlobalRotation,
        fixedTranslation,
        noSilhouetteUpdate,
        &options)

    return status, STATUS_CODES[status]

