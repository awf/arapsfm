# geometry.py

# Imports
import numpy as np

# laplacians
def laplacians(V1, adj_pairs, W_pairs):
    L = np.zeros_like(V1)

    W = {}

    for l, (i, j) in enumerate(adj_pairs):
        w = W_pairs[l]
        L[i] -= w * V1[j]
        W.setdefault(i, []).append(w)

    for i, l in W.iteritems():
        L[i] /= np.sum(l)
        L[i] += V1[i]

    return L

# Barycentric conversion

# make_bary
def make_bary(u):
    u = u[:2]
    return np.r_[u, 1.0 - np.sum(u)]

# bary2pos
def bary2pos(V, u):
    return np.dot(u, V)

# path2pos
def path2pos(V, T, L, U):
    Q = np.empty((U.shape[0], 3), dtype=np.float64)
    for i, face_index in enumerate(L):
        Q[i] = bary2pos(V[T[face_index]], make_bary(U[i]))

    return Q

