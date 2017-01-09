# circular.py

# Imports
import heapq
import numpy as np
from scipy import sparse
from belief_propagation import solve_tree_bp
from itertools import count, izip

# solve_csp
def solve_csp(unary, pairwiseEnergies, colOffsets, 
              G, nodeStates, nodeStateIndices, startEnd=None, verbose=False):

    # get start and end of the path
    N = G.shape[0]
    start = np.setdiff1d(np.arange(N), 
                         np.unique(G.indices),
                         assume_unique=True)
    assert start.shape[0] == 1, 'multiple empty columns'

    end = np.argwhere(G.indptr[:-1] == G.indptr[1:]).ravel()
    assert end.shape[0] == 1, 'multiple empty rows'
    end = end[0]

    if verbose:
        print 'start:', start
        print 'end:', end

    # augment the graph with a phantom start node
    G_aug = sparse.lil_matrix((N + 1, N + 1), dtype=np.int32)
    for i, j in izip(*G.nonzero()):
        G_aug[i, j] = 1
    G_aug[end, N] = 1

    G_aug = G_aug.tocsr()

    # augment the unaries with a row of zeros to not double count the unaries
    unaryBlocks = [
        np.require(np.r_['0,2', unary, np.zeros(unary.shape[1])],
                   np.float64, requirements='C')]

    # augment nodeStateIndices with the same available states as the  original
    # node
    nodeStateIndices = np.require(
        np.r_[nodeStateIndices, nodeStateIndices[start]],
        requirements='C')

    # solve initial node states
    E, states = solve_tree_bp(unaryBlocks,
                              pairwiseEnergies,
                              colOffsets,
                              G_aug,
                              nodeStates,
                              nodeStateIndices,
                              verbose)

    if states[-1] == states[start]:
        return E, states[:-1]

    # define `startEnd`
    if startEnd is None:
        startEnd = (0, G.shape[0])

    if verbose:
        print '[%d] E: %.3f : %s : (%d, %d)' % (
            -1, E, startEnd, states[0], states[-1])

    # initial node states don't form a closed path

    # define `make_nodeStates`
    def make_nodeStates(startEnd):
        new_nodeStates = nodeStates[:]
        new_nodeStateIndices = nodeStateIndices.copy()

        new_nodeStateIndices[start] = len(new_nodeStates)
        new_nodeStateIndices[-1] = len(new_nodeStates)

        new_nodeStates.append(
            np.intersect1d(np.arange(startEnd[0], startEnd[1], dtype=np.int32),
                           nodeStates[start]))

        return new_nodeStates, new_nodeStateIndices

    counter = count(0)

    pq = []
    heapq.heappush(pq, (E, next(counter), startEnd, states))

    i = 0
    while pq:
        # get best path with restricted start and end indices
        E, n, startEnd, states = heapq.heappop(pq)
        if verbose:
            print '[%d] E: %.3f : (%d, %d) : (%d, %d)' % (
                i, E, startEnd[0], startEnd[1], states[0], states[-1])
        i += 1

        # terminate if circular
        if states[-1] == states[start]:
            return E, states[:-1]

        # split on first and last indices
        split = (states[start] + states[-1] + 1) / 2
        new_startEnd = ((startEnd[0], split), (split, startEnd[1]))

        for startEnd in new_startEnd:
            if startEnd[0] >= startEnd[1]:
                continue

            nodeStates, nodeStateIndices = make_nodeStates(startEnd)
            if nodeStates[-1].shape[0] <= 0:
                continue

            E, states = solve_tree_bp(unaryBlocks,
                                      pairwiseEnergies,
                                      colOffsets,
                                      G_aug,
                                      nodeStates,
                                      nodeStateIndices)

            if verbose:
                print '    -> E: %.3f : (%d, %d) : (%d, %d)' % (
                    E, startEnd[0], startEnd[1], states[0], states[-1])

            heapq.heappush(pq, (E, next(counter), startEnd, states))

    raise ValueError('unable to find circular shortest path')

