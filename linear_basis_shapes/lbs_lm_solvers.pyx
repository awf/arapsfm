# lbs_lm_solvers.pyx
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

cdef extern from "lbs_lm_solvers.h":
    int solve_single_projection_c 'solve_single_projection' (
        np.ndarray[np.int32_t, ndim=2] npy_T,
        list list_Vb,
        np.ndarray npy_s,
        np.ndarray npy_Xg,
        np.ndarray npy_Vd,
        np.ndarray npy_y,
        np.ndarray npy_C,
        np.ndarray npy_P,
        np.ndarray[np.float64_t, ndim=1] npy_lambdas,
        np.ndarray[np.float64_t, ndim=1] npy_preconditioners,
        bint debug,
        OptimiserOptions * options)

    int solve_single_silhouette_c 'solve_single_silhouette' (
        np.ndarray[np.int32_t, ndim=2] npy_T,
        list list_Vb,
        np.ndarray npy_s,
        np.ndarray npy_Xg,
        np.ndarray npy_Vd,
        np.ndarray npy_y,
        np.ndarray npy_U,
        np.ndarray npy_L,
        np.ndarray npy_C,
        np.ndarray npy_P,
        np.ndarray npy_S,
        np.ndarray npy_SN,
        np.ndarray npy_lambdas,
        np.ndarray npy_preconditioners,
        np.int32_t narrowBand,
        bint debug,
        OptimiserOptions * options)

    int solve_multiple_c 'solve_multiple' (
        np.ndarray npy_T,
        list list_Vb,
        list list_s,
        list list_Xg,
        list list_Vd,
        list list_y,
        list list_U,
        list list_L,
        list list_C,
        list list_P,
        list list_S,
        list list_SN,
        np.ndarray npy_lambdas,
        np.ndarray npy_preconditioners,
        np.int32_t narrowBand,
        bint debug,
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

# solve_single_projection
def solve_single_projection(np.ndarray[np.int32_t, ndim=2] npy_T,
                            list list_Vb,
                            np.ndarray[np.float64_t, ndim=2] npy_s,
                            np.ndarray[np.float64_t, ndim=2] npy_Xg,
                            np.ndarray[np.float64_t, ndim=2] npy_Vd,
                            np.ndarray[np.float64_t, ndim=2] npy_y,
                            np.ndarray[np.int32_t, ndim=1] npy_C,
                            np.ndarray[np.float64_t, ndim=2] npy_P,
                            np.ndarray[np.float64_t, ndim=1] npy_lambdas,
                            np.ndarray[np.float64_t, ndim=1] npy_preconditioners,
                            **kwargs):

    cdef bint debug = kwargs.pop('debug', False)

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if npy_lambdas.shape[0] != 2:
        raise ValueError('npy_lambdas.shape[0] != 2')

    if npy_preconditioners.shape[0] != 4:
        raise ValueError('npy_preconditioners.shape[0] != 4')

    cdef int status = solve_single_projection_c(
        npy_T, list_Vb, npy_s, npy_Xg, npy_Vd, npy_y, npy_C, npy_P, npy_lambdas, 
        npy_preconditioners, debug, &options)

    return status, STATUS_CODES[status]

# solve_single_silhouette
def solve_single_silhouette(np.ndarray[np.int32_t, ndim=2] npy_T,
                            list list_Vb,
                            np.ndarray[np.float64_t, ndim=2] npy_s,
                            np.ndarray[np.float64_t, ndim=2] npy_Xg,
                            np.ndarray[np.float64_t, ndim=2] npy_Vd,
                            np.ndarray[np.float64_t, ndim=2] npy_y,
                            np.ndarray[np.float64_t, ndim=2] npy_U,
                            np.ndarray[np.int32_t, ndim=1] npy_L,
                            np.ndarray[np.int32_t, ndim=1] npy_C,
                            np.ndarray[np.float64_t, ndim=2] npy_P,
                            np.ndarray[np.float64_t, ndim=2] npy_S,
                            np.ndarray[np.float64_t, ndim=2] npy_SN,
                            np.ndarray[np.float64_t, ndim=1] npy_lambdas,
                            np.ndarray[np.float64_t, ndim=1] npy_preconditioners,
                            np.int32_t narrowBand,
                            **kwargs):

    cdef bint debug = kwargs.pop('debug', False)

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if npy_lambdas.shape[0] != 5:
        raise ValueError('npy_lambdas.shape[0] != 5')

    if npy_preconditioners.shape[0] != 5:
        raise ValueError('npy_preconditioners.shape[0] != 5')

    cdef int status = solve_single_silhouette_c(
        npy_T, list_Vb, npy_s, npy_Xg, npy_Vd, npy_y, npy_U, npy_L, 
        npy_C, npy_P, npy_S, npy_SN, npy_lambdas, 
        npy_preconditioners, narrowBand, debug, &options)

    return status, STATUS_CODES[status]

# solve_multiple
def solve_multiple (np.ndarray npy_T,
                    list list_Vb,
                    list list_s,
                    list list_Xg,
                    list list_Vd,
                    list list_y,
                    list list_U,
                    list list_L,
                    list list_C,
                    list list_P,
                    list list_S,
                    list list_SN,
                    np.ndarray npy_lambdas,
                    np.ndarray npy_preconditioners,
                    np.int32_t narrowBand,
                    **kwargs):

    cdef bint debug = kwargs.pop('debug', False)
    callback = kwargs.pop('callback', None)

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if npy_lambdas.shape[0] != 5:
        raise ValueError('npy_lambdas.shape[0] != 5')

    if npy_preconditioners.shape[0] != 5:
        raise ValueError('npy_preconditioners.shape[0] != 5')


    cdef int status = solve_multiple_c(npy_T, 
        list_Vb, list_s, list_Xg, list_Vd, list_y, 
        list_U, list_L, 
        list_C, list_P, 
        list_S, list_SN, 
        npy_lambdas, npy_preconditioners, narrowBand, debug, &options,
        callback)

    return status, STATUS_CODES[status]

