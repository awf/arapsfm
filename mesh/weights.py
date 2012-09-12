# weights.py

# Imports
import numpy as np
from scipy import sparse
from itertools import groupby

# weights
def weights(V, cells, weights_type='cotan'):
    """
    weights(V, cells, weights_type='cotan')

    Calculate the mesh weights and adjacency matrix for a triangular mesh.

    Parameters
    ----------
    V : ndarray[float64, ndim=2], shape=(N, 3)
        Array of vertices (row-vectors).

    cells : ndarray[int32, ndim=1]
        Cell array which describes the vertices of the mesh:
            [n1, i1_1, i1_2, ..., i1_n1, n2, i2_1, ..., i2_n2, ..., iN_nN]

        where n1 is the number of vertices on face 1, and i1_1 is vertex 1 on
        face 1, i1_2 is vertex 2 on face 1 etc.

    weights_type : str optional
        Options for the weights:
            'cotan'   : cotangent weights
            'uniform' : uniform weights

    Returns
    -------
    adj : ndarray
        Array of arrays describing the adjacency of the input mesh.

    W : sparse.lil_matrix
        Sparse weights matrix describing the weights of adjacent vertices.
    """
    # key arguments
    if weights_type == 'cotan':
        cotan_weights = True
    elif weights_type == 'uniform':
        cotan_weights = False
    else:
        raise ValueError('unknown `weights`: %s' % weights)
        
    # construct weights and adjacency lookup
    N = V.shape[0]
    W = sparse.lil_matrix((N,N), dtype=np.float64)

    adj = np.empty(N, dtype=object)
    for i in xrange(N):
        adj[i] = set([])

    i = 0
    while i < cells.shape[0]:
        # get points in cell
        n = cells[i]
        cell = cells[i+1:i+1+n]
        P = V[cell]

        # square lengths and lengths between points in cell
        if cotan_weights:
            n = P.shape[0]
            sq_l = np.empty(n, dtype=np.float32)
            for j in range(n):
                sq_l[j] = np.sum((P[(j+1) % n] - P[j])**2)
            l = np.sqrt(sq_l)

        # get weight contributions and add to weights matrix
        for j in range(n):
            # fill adjacency matrix
            v1, v2 = cell[j-1], cell[j]
            adj[v1].add(v2)
            adj[v2].add(v1)

            # add weights
            if not cotan_weights:
                W[v1,v2] = 1.
                W[v2,v1] = 1.
            else:
                # get angle
                angle = np.arccos((sq_l[j] + sq_l[j-1] - sq_l[(j+1) %n]) / 
                                  (2*l[j]*l[j-1]))

                # negative weights if obtuse triangle
                w = 0.5 * (1. / np.tan(angle))
                if w < 1e-6:
                    continue

                # add to weights
                v1, v2 = cell[(j + 1) % n], cell[j-1]
                W[v1, v2] += w
                W[v2, v1] += w

        i += n + 1

    if not cotan_weights:
        # normalise along rows
        divide_by_row_sums(W)

    # convert sets to arrays
    for i in xrange(N):
        adj[i] = np.array(list(adj[i]))

    return adj, W

# to_vectors
def to_vectors(adj, W):
    # re-build adjacency matrix and weights matrix each as single vectors
    lengths = map(len, adj)
    adj_length = np.sum(lengths) + len(adj)

    adj_vector = np.empty(adj_length, dtype=np.int32)
    W_vector = np.empty(adj_length, dtype=np.float64)

    k = 0
    for i, s in enumerate(adj):
        adj_vector[k] = len(s)
        W_vector[k] = 0.0
        for l, j in enumerate(s):
            adj_vector[k + l + 1] = j
            W_vector[k + l + 1] = W[i,j]
        k += len(s) + 1

    return adj_vector, W_vector
    
# to_pairs
def to_pairs(adj, W):
    N_pairs = np.sum(map(len, adj))

    adj_pairs = np.empty((N_pairs, 2), dtype=np.int32)
    W_pairs = np.empty(N_pairs, dtype=np.float64)

    k = 0
    for i, s in enumerate(adj):
        for j in s:
            adj_pairs[k, 0] = i
            adj_pairs[k, 1] = j

            W_pairs[k] = W[i,j]

            k += 1

    return adj_pairs, W_pairs

# Sparse matrix utilities

# traverse_rows
def traverse_rows(sparse_A):
    rows, cols = sparse_A.nonzero()
    groups = groupby(rows)

    i = [0]
    for row, entries_with_row in groups:
        def traverser():
            for row_ in entries_with_row:
                yield row, cols[i[0]]
                i[0] += 1

        yield traverser()

# divide_by_row_sums
def divide_by_row_sums(sparse_A):
    row_sums = np.empty(sparse_A.shape[0], dtype=np.float64)
    for i, coordinates_on_row in enumerate(traverse_rows(sparse_A)):
        sum_ = 0.
        for r, c in coordinates_on_row:
            sum_ += sparse_A[r, c]

        row_sums[i] = sum_

    for i, coordinates_on_row in enumerate(traverse_rows(sparse_A)):
        for r, c in coordinates_on_row:
            sparse_A[r, c] /= row_sums[i]

