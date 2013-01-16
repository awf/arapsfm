# silhouette.py

# Imports
import numpy as np
from operator import add
from scipy import sparse
from scipy.spatial.distance import cdist
from belief_propagation.circular import solve_csp
from mesh.faces import vertices_to_faces
from mesh.geometry import face_normal
from misc.numpy_ import normalise
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import norm

# solve_silhouette
def solve_silhouette(V, T, S, SN, 
                     SilCandDistances, 
                     SilEdgeCands, 
                     SilEdgeCandParam, 
                     SilCandAssignedFaces,
                     SilCandU,
                     lambdas,
                     radius=None,
                     verbose=False):

    # update candidates
    num_candidates = SilCandDistances.shape[0]
    num_edge_candidates = SilEdgeCandParam.shape[0]

    # vertices
    Q = np.empty((num_candidates, 3), dtype=np.float64)
    Q[:V.shape[0], :] = V

    # vertex normals
    faces_adjacent_to_vertex = vertices_to_faces(V.shape[0], T)
    face_normals = map(lambda i: face_normal(V, T, i, normalise=False),
                       xrange(T.shape[0]))

    # `W` is the weights which are determined by the area of the triangle
    W = np.empty(num_candidates, dtype=np.float64)

    QN = np.empty((num_candidates, 3), dtype=np.float64)
    def vertex_normal_inplace(i):
        adj_normals = map(lambda j: face_normals[j], 
                          faces_adjacent_to_vertex[i])
        W[i] = 0.5 * reduce(add, map(norm, adj_normals))
        QN[i] = normalise(reduce(add, adj_normals))

    map(vertex_normal_inplace, xrange(V.shape[0]))

    # edge candidates
    if num_edge_candidates > 0:
        for l, (i, j) in enumerate(SilEdgeCands):
            t = SilEdgeCandParam[l]

            W[V.shape[0] + l] = (1. - t) * W[i] + t * W[j]
            Q[V.shape[0] + l] = (1. - t) * V[i] + t * V[j]
            QN[V.shape[0] + l] = normalise((1. - t) * QN[i] + t * QN[j])

    n = S.shape[0]
    Q_2 = Q[:,:2]
    SN_3 = np.c_[SN, n * [0.]]
    W = W * W

    unary = np.require(
        lambdas[1] * cdist(S, Q_2, 'sqeuclidean') +
        lambdas[2] * cdist(SN_3, QN, 'sqeuclidean') * W[ np.newaxis, :],
        np.float64, requirements='C')

    pairwiseEnergies = [[np.require(lambdas[0] * SilCandDistances, 
                                    np.float64, requirements='C')]]

    colOffsets = np.array([0], dtype=np.int32)

    G = sparse.lil_matrix((n, n), dtype=np.uint8)
    for i in xrange(n - 1):
        G[i, i + 1] = 1

    G = G.tocsr()

    if radius is None:
        nodeStates = [np.arange(unary.shape[1], dtype=np.int32)]
        nodeStateIndices = np.zeros(n, dtype=np.int32)
    else:
        neighbours = NearestNeighbors(radius=radius)
        neighbours.fit(Q_2)

        nearest = neighbours.radius_neighbors(S, return_distance=False)

        def_node_states = np.arange(unary.shape[1], dtype=np.int32)

        nodeStates = []
        for i, in_radius in enumerate(nearest):
            if in_radius.shape[0] == 0:
                nodeStates.append(def_node_states)
            else:
                nodeStates.append(in_radius)

        nodeStateIndices = np.arange(n, dtype=np.int32)

    E, states = solve_csp(unary, pairwiseEnergies, colOffsets, G, 
                          nodeStates, nodeStateIndices, verbose=verbose)

    U = SilCandU[states]
    l = SilCandAssignedFaces[states]

    return U, l
                        
