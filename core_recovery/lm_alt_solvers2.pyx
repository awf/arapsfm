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
    int solve_instance_c 'solve_instance' (
        np.ndarray T, 
        np.ndarray V, 
        np.ndarray s, 
        int kg, 
        np.ndarray Xgb, 
        np.ndarray yg, 
        np.ndarray Xg, 
        np.ndarray k, 
        np.ndarray Xb, 
        np.ndarray y, 
        np.ndarray X, 
        np.ndarray Vp,
        np.ndarray sp,
        np.ndarray Xgp,
        np.ndarray Xp,
        np.ndarray V1, 
        np.ndarray U, 
        np.ndarray L, 
        np.ndarray S, 
        np.ndarray SN, 
        np.ndarray Rx, 
        np.ndarray Ry,
        np.ndarray C,
        np.ndarray P,
        np.ndarray lambdas,
        np.ndarray preconditioners,
        np.ndarray piecewisePolynomial,
        int narrowBand,
        bint uniformWeights,
        bint fixedScale,
        OptimiserOptions * options)

    int solve_initialisation_first_instance_c 'solve_initialisation_first_instance' (
        np.ndarray T, 
        np.ndarray V, 
        np.ndarray s, 
        np.ndarray Xg, 
        np.ndarray X, 
        np.ndarray V1, 
        np.ndarray C,
        np.ndarray P,
        np.ndarray lambdas,
        np.ndarray preconditioners,
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
               int narrowBand,
               bint uniformWeights,
               OptimiserOptions * options)

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
               int narrowBand,
               bint uniformWeights,
               **kwargs):

    assert lambdas.shape[0] == 2
    assert preconditioners.shape[0] == 4

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    cdef int status = solve_core_c(
        T, V, s, 
        kg, Xgb, yg, Xg,
        k, Xb, y, X,
        V1, 
        lambdas, preconditioners, narrowBand, uniformWeights, &options)

    return status, STATUS_CODES[status]

