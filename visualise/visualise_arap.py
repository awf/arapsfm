# visualise_arap.py

# Imports
import argparse
from visualise import *
from mesh import weights
from mesh.faces import faces_to_cell_array
from itertools import groupby
from operator import itemgetter
from geometry import quaternion as quat

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

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-c', dest='camera_actions', type=str, default=[],
                        action='append', help='camera actions')

    args = parser.parse_args()
    z = np.load(args.input)

    vis = VisualiseMesh()

    # V0 (purple)
    V0 = z['V0']
    vis.add_mesh(V0, z['T'], actor_name='V0')
    lut = vis.actors['V0'].GetMapper().GetLookupTable()
    lut.SetTableValue(0, *int2dbl(255, 0, 255))

    # V0 w/ X (green)
    T_ = faces_to_cell_array(z['T'])
    adj, W = weights.weights(V0, T_, weights_type='cotan')
    solveV0_X = ARAPVertexSolver(adj, W, V0)

    V0_X = solveV0_X([quat.rotationMatrix(quat.quat(x)) for x in z['X']])
    vis.add_mesh(V0_X, z['T'], actor_name='V0_X')
    lut = vis.actors['V0_X'].GetMapper().GetLookupTable()
    lut.SetTableValue(0, *int2dbl(0, 255, 0))

    # V0 w/ Xg (yellow)
    qg = quat.quat(z['Xg'][0])
    Q = [quat.quatMultiply(quat.quat(x), qg) for x in z['X']]
    V0_Xg = solveV0_X([quat.rotationMatrix(q) for q in Q])
    vis.add_mesh(V0_Xg, z['T'], actor_name='V0_Xg')
    lut = vis.actors['V0_Xg'].GetMapper().GetLookupTable()
    lut.SetTableValue(0, *int2dbl(255, 255, 0))

    # V (blue)
    vis.add_mesh(z['V'], z['T'], actor_name='V')

    # input frame
    vis.add_image(z['input_frame'])

    # projection constraints
    vis.add_projection(z['C'], z['P'])

    # apply camera actions sequentially
    for action in args.camera_actions:
        method, tup, save_after = parse_camera_action(action)
        print '%s(*%s), save_after=%s' % (method, tup, save_after)
        vis.camera_actions((method, tup))

    vis.execute()
    
if __name__ == '__main__':
    main()
