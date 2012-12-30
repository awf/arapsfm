# belief_propagation.pyx

# Imports
import numpy as np
cimport numpy as np
np.import_array()

# Types
ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

# require
def require(arr, ndim, *args, **kwargs):
    if arr.ndim < ndim:
        arr = arr[(np.newaxis,) * (ndim - arr.ndim) + (Ellipsis,)]
    elif arr.ndim > ndim:
        raise ValueError('arr.ndim > %d' % ndim)

    return np.require(arr, *args, **kwargs)

# belief_propagation.h
cdef extern from 'belief_propagation.h':
    object tree_bp_cpp 'tree_bp' (
        object py_unaryBlockList,
        object py_listOfListOfPairwiseEnergyBlockLists,
        object py_listOfColOffsets,
        object py_listOfStates,
        np.ndarray npy_nodeStateIndices,
        np.ndarray npy_graphIndPtr,
        np.ndarray npy_graphIndices,
        np.ndarray npy_graphValues,
        int verbose)

# solve_tree_bp
def solve_tree_bp(list unaryBlocks not None, 
                  list listOfPairwiseEnergies not None,
                  object listOfColOffsets not None,
                  object graph not None,
                  list nodeStates not None,
                  object nodeStateIndices not None,
                  np.int32_t verbose=0):

    unaryBlocks = [require(u, 2, DTYPE, 'C') for u in unaryBlocks]

    if not isinstance(listOfColOffsets, list):
        listOfColOffsets = [listOfColOffsets]
        listOfPairwiseEnergies = [listOfPairwiseEnergies]

    listOflistOfPairwiseEnergyBlockLists = []

    for pairwiseEnergies in listOfPairwiseEnergies:

        listOfPairwiseEnergyBlockLists = []
        for pairwiseBlocks in pairwiseEnergies:
            listOfPairwiseEnergyBlockLists.append(
                [require(a, 2, np.float64, 'C') for a in pairwiseBlocks])

        listOflistOfPairwiseEnergyBlockLists.append(
            listOfPairwiseEnergyBlockLists)

    listOfColOffsets = map(lambda a: require(np.asarray(a), 1, np.int32),
                           listOfColOffsets)

    graphIndPtr = require(graph.indptr, 1, np.int32)
    graphIndices = require(graph.indices, 1, np.int32)
    pairwiseMatrixIndices = require(graph.data, 1, np.int32)

    listOfStates = [require(s, 1, np.int32) for s in nodeStates]
    npy_nodeStateIndices = require(np.asarray(nodeStateIndices), 1, np.int32)

    return tree_bp_cpp(unaryBlocks, 
                       listOflistOfPairwiseEnergyBlockLists,
                       listOfColOffsets,
                       listOfStates,
                       npy_nodeStateIndices,
                       graphIndPtr, 
                       graphIndices,
                       pairwiseMatrixIndices,
                       verbose)

