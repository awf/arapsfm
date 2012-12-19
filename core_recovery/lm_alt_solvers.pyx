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

    int solve_forward_sectioned_arap_proj_c 'solve_forward_sectioned_arap_proj' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        np.ndarray npy_Xg,
        np.ndarray npy_s,
        np.ndarray npy_Xb,
        np.ndarray npy_y,
        np.ndarray npy_X,
        np.ndarray npy_V1,
        np.ndarray npy_K,
        np.ndarray npy_C,
        np.ndarray npy_P,
        np.ndarray npy_lambdas,
        np.ndarray npy_preconditioners,
        bint isProjection,
        bint uniformWeights,
        bint fixedScale,
        OptimiserOptions * options)

    int solve_instance_sectioned_arap_c 'solve_instance_sectioned_arap' (np.ndarray npy_T,
                                                                         np.ndarray npy_V,
                                                                         np.ndarray npy_Xg,
                                                                         np.ndarray npy_s,
                                                                         np.ndarray npy_K,
                                                                         np.ndarray npy_Xb,
                                                                         np.ndarray npy_y,
                                                                         np.ndarray npy_X,
                                                                         np.ndarray npy_V1,
                                                                         np.ndarray npy_U,
                                                                         np.ndarray npy_L,
                                                                         np.ndarray npy_S,
                                                                         np.ndarray npy_SN,
                                                                         np.ndarray npy_Rx,
                                                                         np.ndarray npy_Ry,
                                                                         np.ndarray npy_C,
                                                                         np.ndarray npy_P,
                                                                         np.ndarray npy_lambdas,
                                                                         np.ndarray npy_preconditioners,
                                                                         np.ndarray npy_piecewisePolynomial,
                                                                         int narrowBand,
                                                                         bint uniformWeights,
                                                                         bint fixedScale,
                                                                         OptimiserOptions * options)

    int solve_core_sectioned_arap_c 'solve_core_sectioned_arap' (np.ndarray npy_T,
                                                                 np.ndarray npy_V,
                                                                 list list_Xg,
                                                                 list list_s,
                                                                 np.ndarray npy_K,
                                                                 np.ndarray Xb,
                                                                 list list_y,
                                                                 list list_X,
                                                                 list list_V1,
                                                                 ## list list_C,
                                                                 ## list list_P,
                                                                 np.ndarray npy_lambdas,
                                                                 np.ndarray npy_preconditioners,
                                                                 int narrowBand,
                                                                 bint uniformWeights,
                                                                 OptimiserOptions * options)

    int solve_instance_sectioned_arap_temporal_c 'solve_instance_sectioned_arap_temporal' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        np.ndarray npy_Xg,
        np.ndarray npy_s,
        np.ndarray npy_K,
        np.ndarray npy_Xb,
        np.ndarray npy_y,
        np.ndarray npy_X,
        np.ndarray npy_V1,
        np.ndarray npy_U,
        np.ndarray npy_L,
        np.ndarray npy_S,
        np.ndarray npy_SN,
        np.ndarray npy_Rx,
        np.ndarray npy_Ry,
        np.ndarray npy_C,
        np.ndarray npy_P,
        list list_Vn,
        np.ndarray npy_omegas,
        np.ndarray npy_lambdas,
        np.ndarray npy_preconditioners,
        np.ndarray npy_piecewisePolynomial,
        int narrowBand,
        bint uniformWeights,
        bint fixedScale,
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

def check_K(K, V, X, Xb, y):
    # check K is specified for all V
    assert K.shape[0] == V.shape[0]

    # check left-entries of K are valid
    assert np.all(K >= -1)

    # check free rotation indices are available in `X`
    X_i = K[K[:,0] == -1, 1]
    assert np.all((X_i >= 0) & (X_i < X.shape[0]))

    # check all rotations in `X` are used
    assert np.unique(X_i).shape[0] == X.shape[0]

    # check basis rotation indices are available in `Xb`
    i = K[:, 0] > 0
    Xb_i = K[i, 1]
    assert np.all((Xb_i >= 0) & (Xb_i < Xb.shape[0]))

    # check all rotations in `Xb` are used
    assert np.unique(Xb_i).shape[0] == Xb.shape[0]

    # check basis coefficients are availabe in `y`
    y_i = K[i, 0] - 1
    assert np.all((y_i >= 0) & (y_i < y.shape[0]))

    # check all coefficients in `y` are used
    assert np.unique(y_i).shape[0] == y.shape[0]

def solve_forward_sectioned_arap_proj(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                      np.ndarray[np.float64_t, ndim=2, mode='c'] V, 
                                      np.ndarray[np.float64_t, ndim=2, mode='c'] Xg, 
                                      np.ndarray[np.float64_t, ndim=2, mode='c'] s, 
                                      np.ndarray[np.float64_t, ndim=2, mode='c'] Xb, 
                                      np.ndarray[np.float64_t, ndim=2, mode='c'] y, 
                                      np.ndarray[np.float64_t, ndim=2, mode='c'] X, 
                                      np.ndarray[np.float64_t, ndim=2, mode='c'] V1, 
                                      np.ndarray[np.int32_t, ndim=2, mode='c'] K, 
                                      np.ndarray[np.int32_t, ndim=1] C,
                                      np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                                      np.ndarray[np.float64_t, ndim=1] lambdas,
                                      np.ndarray[np.float64_t, ndim=1] preconditioners,
                                      bint isProjection,
                                      bint uniformWeights,
                                      bint fixedScale,
                                      **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 2:
        raise ValueError('lambdas.shape[0] != 2')

    if preconditioners.shape[0] != 5:
        raise ValueError('preconditioners.shape[0] != 5')

    check_K(K, V, X, Xb, y)

    cdef int status = solve_forward_sectioned_arap_proj_c(
        T, V, 
        Xg, 
        s,
        Xb,
        y,
        X,
        V1,
        K, C, P, lambdas, preconditioners,
        isProjection,
        uniformWeights,
        fixedScale,
        &options)

    return status, STATUS_CODES[status]

def solve_instance_sectioned_arap(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] V, 
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] Xg, 
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] s, 
                                  np.ndarray[np.int32_t, ndim=2, mode='c'] K, 
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] Xb, 
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] y, 
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] X, 
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] V1, 
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] U,  
                                  np.ndarray[np.int32_t, ndim=1] L,
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] S,  
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] SN,  
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] Rx,
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] Ry,
                                  np.ndarray[np.int32_t, ndim=1, mode='c'] C,
                                  np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                                  np.ndarray[np.float64_t, ndim=1] lambdas,
                                  np.ndarray[np.float64_t, ndim=1] preconditioners,
                                  np.ndarray[np.float64_t, ndim=1] piecewisePolynomial,
                                  int narrowBand,
                                  bint uniformWeights,
                                  bint fixedScale,
                                  **kwargs):

    assert lambdas.shape[0] == 5
    assert preconditioners.shape[0] == 6
    assert piecewisePolynomial.shape[0] == 2

    check_K(K, V, X, Xb, y)

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    cdef int status = solve_instance_sectioned_arap_c(T, V, 
        Xg, s, 
        K, Xb, y, X, 
        V1, 
        U, L, S, SN, 
        Rx, Ry, 
        C, P,
        lambdas, 
        preconditioners, 
        piecewisePolynomial, 
        narrowBand, 
        uniformWeights,
        fixedScale,
        &options)

    return status, STATUS_CODES[status]

def solve_core_sectioned_arap(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                              np.ndarray[np.float64_t, ndim=2, mode='c'] V, 
                              list Xg, 
                              list s, 
                              np.ndarray[np.int32_t, ndim=2, mode='c'] K, 
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
    assert preconditioners.shape[0] == 5

    cdef Py_ssize_t i
    
    for i in range(len(X)):
        check_K(K, V, X[i], Xb, y[i])

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    cdef int status = solve_core_sectioned_arap_c(T, V,
        Xg, s,
        K, Xb, y, X, 
        V1,
        lambdas,
        preconditioners, 
        narrowBand,
        uniformWeights,
        &options)

    return status, STATUS_CODES[status]

def solve_instance_sectioned_arap_temporal(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] V, 
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] Xg, 
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] s, 
                                           np.ndarray[np.int32_t, ndim=2, mode='c'] K, 
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] Xb, 
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] y, 
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] X, 
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] V1, 
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] U,  
                                           np.ndarray[np.int32_t, ndim=1] L,
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] S,  
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] SN,  
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] Rx,
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] Ry,
                                           np.ndarray[np.int32_t, ndim=1, mode='c'] C,
                                           np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                                           list Vn,
                                           np.ndarray[np.float64_t, ndim=1] omegas,
                                           np.ndarray[np.float64_t, ndim=1] lambdas,
                                           np.ndarray[np.float64_t, ndim=1] preconditioners,
                                           np.ndarray[np.float64_t, ndim=1] piecewisePolynomial,
                                           int narrowBand,
                                           bint uniformWeights,
                                           bint fixedScale,
                                           **kwargs):

    assert lambdas.shape[0] == 5
    assert preconditioners.shape[0] == 6
    assert piecewisePolynomial.shape[0] == 2
    assert len(Vn) == omegas.shape[0]

    check_K(K, V, X, Xb, y)

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    cdef int status = solve_instance_sectioned_arap_temporal_c(T, V, 
        Xg, s, 
        K, Xb, y, X, 
        V1, 
        U, L, S, SN, 
        Rx, Ry, 
        C, P,
        Vn, omegas,
        lambdas, 
        preconditioners, 
        piecewisePolynomial, 
        narrowBand, 
        uniformWeights,
        fixedScale,
        &options)

    return status, STATUS_CODES[status]
