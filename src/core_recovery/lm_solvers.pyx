# lm_solvers.pyx
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

cdef extern from "lm_solvers.h":
    int solve_single_arap_proj_c "solve_single_arap_proj" (
        np.ndarray V,
        np.ndarray T,
        np.ndarray X,
        np.ndarray V1,
        np.ndarray C,
        np.ndarray P,
        np.ndarray lambdas,
        OptimiserOptions * opt)

    int solve_single_arap_abs_c "solve_single_arap_abs" (
        np.ndarray V,
        np.ndarray T,
        np.ndarray X,
        np.ndarray V1,
        np.ndarray C,
        np.ndarray P,
        np.ndarray lambdas,
        OptimiserOptions * opt)

    int solve_single_rigid_arap_proj_c "solve_single_rigid_arap_proj" (
        np.ndarray V,
        np.ndarray T,
        np.ndarray X,
        np.ndarray Xg,
        np.ndarray s,
        np.ndarray V1,
        np.ndarray C,
        np.ndarray P,
        np.ndarray lambdas,
        bint uniformWeights,
        OptimiserOptions * options)

    int solve_single_lap_proj_c "solve_single_lap_proj" (
        np.ndarray V,
        np.ndarray T,
        np.ndarray C,
        np.ndarray P,
        np.ndarray lambdas,
        OptimiserOptions * opt)

    int solve_single_lap_proj_silhouette_c "solve_single_lap_proj_silhouette" (
                  np.ndarray T,
                  np.ndarray V,
                  np.ndarray U,
                  np.ndarray L,
                  np.ndarray C,
                  np.ndarray P,
                  np.ndarray S,
                  np.ndarray SN,
                  np.ndarray lambdas,
                  np.ndarray preconditioners,
                  int narrowBand,
                  OptimiserOptions * options)

    int solve_single_lap_proj_sil_spil_c "solve_single_lap_proj_sil_spil" (
                  np.ndarray T,
                  np.ndarray V,
                  np.ndarray U,
                  np.ndarray L,
                  np.ndarray C,
                  np.ndarray P,
                  np.ndarray S,
                  np.ndarray SN,
                  np.ndarray Rx,
                  np.ndarray Ry,
                  np.ndarray lambdas,
                  np.ndarray preconditioners,
                  np.ndarray piecewisePolynomial,
                  int narrowBand,
                  OptimiserOptions * options)

    int solve_single_lap_silhouette_c "solve_single_lap_silhouette" (np.ndarray V,
                  np.ndarray T,
                  np.ndarray U,
                  np.ndarray L,
                  np.ndarray S,
                  np.ndarray SN,
                  np.ndarray Rx,
                  np.ndarray Ry,
                  np.ndarray lambdas,
                  np.ndarray preconditioners,
                  int narrowBand,
                  OptimiserOptions * options)

    int solve_multiview_arap_silhouette_c "solve_multiview_arap_silhouette" (
        np.ndarray T,
        np.ndarray V,
        list multiX,
        list multiV,
        list multiU,
        list multiL,
        list multiS,
        list multiSN,
        np.ndarray lambdas,
        np.ndarray preconditioners,
        int narrowBand,
        OptimiserOptions * options)

    int solve_multiview_lap_silhouette_c "solve_multiview_lap_silhouette" (
        np.ndarray T,
        np.ndarray V,
        list multiX,
        list multiXg,
        list multis,
        list multiV,
        list multiU,
        list multiL,
        list multiS,
        list multiSN,
        list multiRx,
        list multiRy,
        np.ndarray lambdas,
        np.ndarray preconditioners,
        int narrowBand,
        bint uniformWeights,
        OptimiserOptions * options)

    object solve_single_lap_silhouette_with_Jte_c "solve_single_lap_silhouette_with_Jte" (
                      np.ndarray  npy_V,
                      np.ndarray  npy_T,
                      np.ndarray  npy_U,
                      np.ndarray  npy_L,
                      np.ndarray  npy_S,
                      np.ndarray  npy_SN,
                      np.ndarray  npy_lambdas,
                      np.ndarray  npy_preconditioners,
                      int narrowBand,
                      int maxJteStore,
                      OptimiserOptions * options)

    object solve_single_lap_sil_len_adj_with_Jte_c "solve_single_lap_sil_len_adj_with_Jte" (
                      np.ndarray  npy_V,
                      np.ndarray  npy_T,
                      np.ndarray  npy_U,
                      np.ndarray  npy_L,
                      np.ndarray  npy_S,
                      np.ndarray  npy_SN,
                      np.ndarray  npy_lambdas,
                      np.ndarray  npy_preconditioners,
                      int narrowBand,
                      int maxJteStore,
                      OptimiserOptions * options)

    int solve_single_spillage_c "solve_single_spillage" (
        np.ndarray npy_V,
        np.ndarray npy_Rx,
        np.ndarray npy_Ry,
        OptimiserOptions * options)

    int solve_multiview_nonlinear_basis_c "solve_multiview_nonlinear_basis"(
        np.ndarray npy_T,
        np.ndarray npy_V,
        list list_multiXg,
        list list_multis,
        list list_multiX,
        list list_multiy,
        list list_multiV,
        list list_multiU,
        list list_multiL,
        list list_multiS,
        list list_multiSN,
        list list_multiRx,
        list list_multiRy,
        np.ndarray npy_lambdas,
        np.ndarray npy_preconditioners,
        np.ndarray npy_piecewisePolynomial,
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

# solve_single_arap_proj
def solve_single_arap_proj(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
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

    cdef int status = solve_single_arap_proj_c(V, T, X, V1, C, P, lambdas, &options)

    return status, STATUS_CODES[status]

# solve_single_arap_abs
def solve_single_arap_abs(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
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

    cdef int status = solve_single_arap_abs_c(V, T, X, V1, C, P, lambdas, &options)

    return status, STATUS_CODES[status]

# solve_single_rigid_arap_proj
def solve_single_rigid_arap_proj(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                                 np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                 np.ndarray[np.float64_t, ndim=2, mode='c'] X,
                                 np.ndarray[np.float64_t, ndim=2, mode='c'] Xg,
                                 np.ndarray[np.float64_t, ndim=2, mode='c'] s,
                                 np.ndarray[np.float64_t, ndim=2, mode='c'] V1,
                                 np.ndarray[np.int32_t, ndim=1] C,
                                 np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                                 np.ndarray[np.float64_t, ndim=1] lambdas,
                                 bint uniformWeights=False,
                                 **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 2:
        raise ValueError('lambdas.shape[0] != 2')

    cdef int status = solve_single_rigid_arap_proj_c(
        V, T, X, Xg, s, V1, C, P, lambdas, uniformWeights, &options)

    return status, STATUS_CODES[status]
    
# solve_single_lap_proj
def solve_single_lap_proj(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                  np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                  np.ndarray[np.int32_t, ndim=1] C,
                  np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                  np.ndarray[np.float64_t, ndim=1] lambdas,
                  **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 2:
        raise ValueError('lambdas.shape[0] != 2')

    cdef int status = solve_single_lap_proj_c(V, T, C, P, lambdas, &options)

    return status, STATUS_CODES[status]

# solve_single_lap_silhouette
def solve_single_lap_silhouette(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                                np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                np.ndarray[np.float64_t, ndim=2, mode='c'] U,  
                                np.ndarray[np.int32_t, ndim=1] L,
                                np.ndarray[np.float64_t, ndim=2, mode='c'] S,  
                                np.ndarray[np.float64_t, ndim=2, mode='c'] SN,  
                                np.ndarray[np.float64_t, ndim=2, mode='c'] Rx,  
                                np.ndarray[np.float64_t, ndim=2, mode='c'] Ry,  
                                np.ndarray[np.float64_t, ndim=1] lambdas,
                                np.ndarray[np.float64_t, ndim=1] preconditioners,
                                int narrowBand,
                                **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 4:
        raise ValueError('lambdas.shape[0] != 4')

    cdef int status = solve_single_lap_silhouette_c(V, T, U, L, S, SN, Rx, Ry, lambdas, 
        preconditioners, narrowBand, &options)

    return status, STATUS_CODES[status]

# solve_single_lap_proj_silhouette
def solve_single_lap_proj_silhouette(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                                     np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                     np.ndarray[np.float64_t, ndim=2, mode='c'] U,  
                                     np.ndarray[np.int32_t, ndim=1] L,
                                     np.ndarray[np.int32_t, ndim=1] C,
                                     np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                                     np.ndarray[np.float64_t, ndim=2, mode='c'] S,  
                                     np.ndarray[np.float64_t, ndim=2, mode='c'] SN,  
                                     np.ndarray[np.float64_t, ndim=1] lambdas,
                                     np.ndarray[np.float64_t, ndim=1] preconditioners,
                                     int narrowBand,
                                     **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 4:
        raise ValueError('lambdas.shape[0] != 4')

    cdef int status = solve_single_lap_proj_silhouette_c(V, T, U, L, C, P, S, SN,
        lambdas, preconditioners, narrowBand, &options)

    return status, STATUS_CODES[status]

# solve_multiview_arap_silhouette
def solve_multiview_arap_silhouette(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                   np.ndarray[np.float64_t, ndim=2, mode='c'] V, 
                                   list multiX,
                                   list multiV,
                                   list multiU,
                                   list multiL,
                                   list multiS,
                                   list multiSN,
                                   np.ndarray[np.float64_t, ndim=1] lambdas,
                                   np.ndarray[np.float64_t, ndim=1] preconditioners,
                                   int narrowBand,
                                   **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 3:
        raise ValueError('lambdas.shape[0] != 3')

    if preconditioners.shape[0] != 3:
        raise ValueError('preconditioners.shape[0] != 3')

    cdef int status = solve_multiview_arap_silhouette_c(T, V, multiX, multiV, multiU, multiL,
        multiS, multiSN, lambdas, preconditioners, narrowBand, &options)

    return status, STATUS_CODES[status]

# solve_multiview_lap_silhouette
def solve_multiview_lap_silhouette(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                   np.ndarray[np.float64_t, ndim=2, mode='c'] V, 
                                   list multiX,
                                   list multiXg,
                                   list multis,
                                   list multiV,
                                   list multiU,
                                   list multiL,
                                   list multiS,
                                   list multiSN,
                                   list multiRx,
                                   list multiRy,
                                   np.ndarray[np.float64_t, ndim=1] lambdas,
                                   np.ndarray[np.float64_t, ndim=1] preconditioners,
                                   int narrowBand,
                                   bint uniformWeights,
                                   **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 5:
        raise ValueError('lambdas.shape[0] != 5')

    if preconditioners.shape[0] != 4:
        raise ValueError('preconditioners.shape[0] != 4')

    cdef int status = solve_multiview_lap_silhouette_c(T, V, multiX, multiXg, multis, multiV, multiU, multiL,
        multiS, multiSN, multiRx, multiRy,
        lambdas, preconditioners, narrowBand, uniformWeights, &options)

    return status, STATUS_CODES[status]

# solve_single_lap_silhouette_with_Jte
def solve_single_lap_silhouette_with_Jte(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                                         np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                         np.ndarray[np.float64_t, ndim=2, mode='c'] U,  
                                         np.ndarray[np.int32_t, ndim=1] L,
                                         np.ndarray[np.float64_t, ndim=2, mode='c'] S,  
                                         np.ndarray[np.float64_t, ndim=2, mode='c'] SN,  
                                         np.ndarray[np.float64_t, ndim=1] lambdas,
                                         np.ndarray[np.float64_t, ndim=1] preconditioners,
                                         int narrowBand,
                                         int maxJteStore,
                                         **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 3:
        raise ValueError('lambdas.shape[0] != 3')

    status, storedJte = solve_single_lap_silhouette_with_Jte_c(V, T, U, L, S, SN, lambdas, 
        preconditioners, narrowBand, maxJteStore, &options)

    return (status, STATUS_CODES[status]), storedJte

# solve_single_lap_sil_len_adj_with_Jte
def solve_single_lap_sil_len_adj_with_Jte(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                                         np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                         np.ndarray[np.float64_t, ndim=2, mode='c'] U,  
                                         np.ndarray[np.int32_t, ndim=1] L,
                                         np.ndarray[np.float64_t, ndim=2, mode='c'] S,  
                                         np.ndarray[np.float64_t, ndim=2, mode='c'] SN,  
                                         np.ndarray[np.float64_t, ndim=1] lambdas,
                                         np.ndarray[np.float64_t, ndim=1] preconditioners,
                                         int narrowBand,
                                         int maxJteStore,
                                         **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 3:
        raise ValueError('lambdas.shape[0] != 3')

    status, storedJte = solve_single_lap_sil_len_adj_with_Jte_c(V, T, U, L, S, SN, lambdas, 
        preconditioners, narrowBand, maxJteStore, &options)

    return (status, STATUS_CODES[status]), storedJte

# solve_single_spillage
def solve_single_spillage(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                          np.ndarray[np.float64_t, ndim=2, mode='c'] Rx,
                          np.ndarray[np.float64_t, ndim=2, mode='c'] Ry,
                          **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    status = solve_single_spillage_c(V, Rx, Ry, &options)

    return (status, STATUS_CODES[status])

# solve_single_lap_proj_sil_spil
def solve_single_lap_proj_sil_spil(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                                     np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                     np.ndarray[np.float64_t, ndim=2, mode='c'] U,  
                                     np.ndarray[np.int32_t, ndim=1] L,
                                     np.ndarray[np.int32_t, ndim=1] C,
                                     np.ndarray[np.float64_t, ndim=2, mode='c'] P,
                                     np.ndarray[np.float64_t, ndim=2, mode='c'] S,  
                                     np.ndarray[np.float64_t, ndim=2, mode='c'] SN,  
                                     np.ndarray[np.float64_t, ndim=2, mode='c'] Rx,
                                     np.ndarray[np.float64_t, ndim=2, mode='c'] Ry,
                                     np.ndarray[np.float64_t, ndim=1] lambdas,
                                     np.ndarray[np.float64_t, ndim=1] preconditioners,
                                     np.ndarray[np.float64_t, ndim=1] piecewisePolynomial,
                                     int narrowBand,
                                     **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 5:
        raise ValueError('lambdas.shape[0] != 5')

    if piecewisePolynomial.shape[0] < 1:
        raise ValueError('piecewisePolynomial.shape[0] < 1')

    cdef int status = solve_single_lap_proj_sil_spil_c(V, T, U, L, C, P, S, SN,
        Rx, Ry, lambdas, preconditioners, piecewisePolynomial, narrowBand, &options)

    return status, STATUS_CODES[status]

# solve_multiview_nonlinear_basis
def solve_multiview_nonlinear_basis(np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                                   np.ndarray[np.float64_t, ndim=2, mode='c'] V, 
                                   list multiXg,
                                   list multis,
                                   list multiX,
                                   list multiy,
                                   list multiV,
                                   list multiU,
                                   list multiL,
                                   list multiS,
                                   list multiSN,
                                   list multiRx,
                                   list multiRy,
                                   np.ndarray[np.float64_t, ndim=1] lambdas,
                                   np.ndarray[np.float64_t, ndim=1] preconditioners,
                                   np.ndarray[np.float64_t, ndim=1] piecewisePolynomial,
                                   int narrowBand,
                                   bint uniformWeights,
                                   **kwargs):

    cdef OptimiserOptions options
    additional_optimiser_options(&options, kwargs)

    if lambdas.shape[0] != 5:
        raise ValueError('lambdas.shape[0] != 5')

    if preconditioners.shape[0] != 6:
        raise ValueError('preconditioners.shape[0] != 6')

    if piecewisePolynomial.shape[0] != 2:
        raise ValueError('piecewisePolynomial.shape[0] != 2')

    for i, y in enumerate(multiy):
        if y.ndim != 2 or y.shape[0] != len(multiX):
            raise ValueError('y[{i}].ndim != 2 or y[{i}].shape[0] != len(multiX)'
                             .format(i=i))

    cdef int status = solve_multiview_nonlinear_basis_c(
        T, V, multiXg, multis, multiX, multiy, multiV, 
        multiU, multiL, multiS, multiSN, multiRx, multiRy,
        lambdas, preconditioners, piecewisePolynomial,
        narrowBand, uniformWeights, &options)

    return status, STATUS_CODES[status]
