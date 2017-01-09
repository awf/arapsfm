# arap.py

# Imports
import numpy as np
import scipy.sparse.linalg
from scipy import sparse

# ARAPVertexSolver
class ARAPVertexSolver(object):
    def __init__(self, adj, W, V):
        # Default constraints on the last index
        C = np.r_[V.shape[0] - 1].astype(int)
        U = V[C]

        # W: weights (csr)
        W = W.tocsr()

        # N: number of vertices including the active constraint set
        # N_F: number of vertices to solve for (free vertices)
        N = adj.shape[0]
        N_F = N - len(C)

        # build the system matrix and solution matrix
        b = np.zeros((N_F, U.shape[1]), dtype=np.float64)
        L = sparse.lil_matrix((N_F, N_F), dtype=np.float64)

        for i in xrange(N_F):
            L[i,i] = W[i].sum() 

            for j in adj[i]:
                if j >= N_F:
                    # active constraint neighbour
                    b[i] += W[i, j] * U[j - N_F]
                else:
                    # free neighbour
                    L[i,j] = -W[i,j]

        # factorise the system matrix
        self.L_solver = sparse.linalg.factorized(L.tocsc())

        # save solution matrix and the reverse mappings
        self.b = b
        self.adj = adj
        self.W = W

        # save original vertices and updated copy
        self.V = V

    # __call__
    def __call__(self, R):
        # restore state
        b = np.copy(self.b)
        N_F = b.shape[0]

        # update b
        for i in xrange(N_F):
            for j in self.adj[i]:
                b[i, :] += (0.5 * self.W[i, j] * 
                            np.dot(self.V[i] - self.V[j], 
                            np.transpose(R[i] + R[j])))

        V1 = self.V.copy()

        # new vertex positions
        for l in xrange(b.shape[1]):
            V1[:N_F, l] = self.L_solver(b[:, l])

        return V1

