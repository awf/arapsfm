# visualise_angle_interpolation.py

# Imports
import numpy as np
from core_recovery.lm_solvers import solve_single_rigid_arap_proj
from mesh.box_model import box_model
from visualise import *
from solvers.arap import ARAPVertexSolver
from mesh import weights, faces
from geometry import quaternion as quat, axis_angle
from itertools import izip, product
from scipy.linalg import block_diag
from matplotlib import cm

# right_multiply_affine_transform
def right_multiply_affine_transform(V0, V):
    t0 = np.mean(V0, axis=0)
    t = np.mean(V, axis=0)

    V0 = V0 - t0
    A = block_diag(V0, V0, V0)
    A_T = np.transpose(A)
    A_pinv = np.dot(np.linalg.inv(np.dot(A_T, A)), A_T)
    x = np.dot(A_pinv, np.ravel(np.transpose(V - t)))

    T = np.transpose(x.reshape(3, 3))
    d = t - np.dot(t0, T)

    return T, d

# main
def main():
    # Input
    lambdas = np.r_[1e0, 1e0].astype(np.float64)
    V, T = box_model(3, 20, 1.0, 1.0)

    # Setup `solve_V_X`
    adj, W = weights.weights(V, faces.faces_to_cell_array(T),
                             weights_type='cotan')
    solve_V_X = ARAPVertexSolver(adj, W, V)

    # Setup `solve_single`
    def solve_single(C, P):
        X = np.zeros_like(V)
        Xg = np.zeros((1, 3), dtype=np.float64)
        s = np.ones((1, 1), dtype=np.float64)
        V1 = V.copy()

        status = solve_single_rigid_arap_proj(
            V, T, X, Xg, s, V1, C, P, lambdas,
            uniformWeights=False,
            maxIterations=100,
            updateThreshold=1e-6,
            gradientThreshold=1e-6,
            improvementThreshold=1e-6)

        return X, V1 

    # Setup `C`, `P1` and `P2`
    heights = np.unique(V[:,2])

    C1 = np.argwhere(V[:,2] >= heights[-3]).flatten()
    L1 = C1.shape[0]
    C2 = np.argwhere(V[:,2] <= heights[2]).flatten()
    L2 = C2.shape[0]
    C = np.r_[C1, C2]

    P1 = V[C, :2].copy()
    P1[:L1, 1] += 10.0

    P2 = V[C, :2].copy()
    P2[:L1, 0] += 10.0

    # Solve for `X1` and `X2`
    X1, V1 = solve_single(C, P1)
    X2, V2 = solve_single(C, P2)

    # Setup `V_from_X1_X2` which "interpolates" `X1` and `X2`
    def V_from_X1_X2(a, b):
        X12 = map(lambda x: 
                  axis_angle.axMakeInterpolated(a, x[0], b, x[1]),
                  izip(X1, X2))

        V12 = solve_V_X(map(lambda x: quat.rotationMatrix(quat.quat(x)), X12))

        # Register bottom fixed layers
        A, d = right_multiply_affine_transform(V12[C[L1:]], V[C[L1:]])

        return np.dot(V12, A) + d

    # Visualise the interpolation of the distortions
    t = np.linspace(-1., 1., 3, endpoint=True)
    N = t.shape[0] * t.shape[0]
    cmap = cm.jet(np.linspace(0., 1., N, endpoint=True))

    vis = VisualiseMesh()

    for i, (u, v) in enumerate(product(t, t)):
        actor_name = 'V12_%d' % i

        V12 = V_from_X1_X2(u, v)
        vis.add_mesh(V12, T, actor_name=actor_name)

        # Change color of actor
        lut = vis.actors[actor_name].GetMapper().GetLookupTable()
        lut.SetTableValue(0, *cmap[i,:3])

        # Change actor to dense surface (instead of default wireframe)
        vis.actor_properties(actor_name, ('SetRepresentation', (3,)))

    vis.camera_actions(('SetParallelProjection', (True,)))
    vis.execute()

if __name__ == '__main__':
    main()

