# visualise_sectioned_arap.py

# Imports
import numpy as np
from core_recovery.lm_alt_solvers import solve_forward_sectioned_arap_proj
from mesh.box_model import box_model
from visualise import *
from geometry import quaternion as quat
from scipy.linalg import norm

# main
def main():
    # Input
    lambdas = np.r_[1e0, 1e2].astype(np.float64)
    preconditioners = np.r_[1.0, 1.0, 128.0, 128.0, 128.0].astype(np.float64)

    V, T = box_model(3, 50, 1.0, 1.0)
    N = V.shape[0]

    # Setup no rotations (fixed) except through middle of the bar
    heights = np.unique(V[:,2])
    middle = np.mean(heights)
    delta = 6.0
    I = np.argwhere(((middle - delta) <= V[:,2]) &
                    (V[:,2] < (middle + delta))).flatten()

    M = I.shape[0]

    K = np.zeros((N, 2), dtype=np.int32)
    K[I,0] = -1
    K[I,1] = np.arange(M)

    # Setup variables for solver
    Xg = np.zeros((1, 3), dtype=np.float64)
    s = np.ones((1, 1), dtype=np.float64)

    Xb = np.array([], dtype=np.float64).reshape(0, 3)
    y = np.array([], dtype=np.float64).reshape(0, 1)

    X = np.zeros((M, 3), dtype=np.float64)

    V1 = V.copy()

    # Setup `C`, `P1` and `P2`
    C1 = np.argwhere(V[:,2] >= heights[-3]).flatten()
    L1 = C1.shape[0]
    C2 = np.argwhere(V[:,2] <= heights[2]).flatten()
    L2 = C2.shape[0]
    C = np.r_[C1, C2]

    # Rotate `C1` vertices about their mean (by -30 on the x-axis)
    x = np.r_[np.deg2rad(-30), 0., 0.]
    R = quat.rotationMatrix(quat.quat(x))

    Q = V[C1]
    c = np.mean(Q, axis=0)
    Q = np.dot(Q - c, np.transpose(R)) + c

    Q[:, 0] += 20
    Q[:, 1] += 30
    Q[:, 2] -= 20

    P = np.ascontiguousarray(np.r_[Q, V[C2]])

    # Solve for `V1`
    status = solve_forward_sectioned_arap_proj(
        T, V, Xg, s, Xb, y, X, V1, K, C, P, lambdas, preconditioners, 
        isProjection=False,
        uniformWeights=False,
        fixedScale=True,
        maxIterations=100, 
        verbosenessLevel=1)

    # View
    # vis = VisualiseMesh()
    # vis.add_mesh(V1, T)
    # vis.add_points(V1[I], sphere_radius=0.2, actor_name='I', color=(0., 0., 1.))
    # vis.add_points(V1[C], sphere_radius=0.2, actor_name='C', color=(1., 0., 0.))
    # vis.camera_actions(('SetParallelProjection', (True,)))
    # vis.execute(magnification=4)

    # Rotate `C1` vertices about their mean (by -180 on the x-axis)
    V1 = V.copy()

    x = np.r_[np.deg2rad(+180), 0., 0.]
    R = quat.rotationMatrix(quat.quat(x))

    Q = V[C1]
    c = np.mean(Q, axis=0)
    Q = np.dot(Q - c, np.transpose(R)) + c

    Q[:, 1] += 10
    Q[:, 2] -= Q[-1, 2]

    P = np.ascontiguousarray(np.r_[Q, V[C2]])
    print 'P:'
    print np.around(P, decimals=3)

    # Assign a single rotation to the upper variables
    I1 = np.setdiff1d(np.argwhere(V[:,2] >= (middle + delta)).flatten(), 
                      C1, assume_unique=True)

    K[I1, 0] = -1
    K[I1, 1] = M

    Xg = np.zeros((1, 3), dtype=np.float64)
    X = np.zeros((M+1, 3), dtype=np.float64)

    # Solve for `V1`
    status = solve_forward_sectioned_arap_proj(
        T, V, Xg, s, Xb, y, X, V1, K, C, P, lambdas, preconditioners, 
        isProjection=False,
        uniformWeights=False,
        fixedScale=True,
        maxIterations=100, 
        verbosenessLevel=1)

    print 's:', s
    print 'X[M] (%.2f):' % np.rad2deg(norm(X[M])), np.around(X[M], decimals=3)

    # View
    # vis = VisualiseMesh()
    # vis.add_mesh(V1, T)
    # vis.add_points(V1[I], sphere_radius=0.2, actor_name='I', color=(0., 0., 1.))
    # vis.add_points(V1[C], sphere_radius=0.2, actor_name='C', color=(1., 0., 0.))
    # vis.camera_actions(('SetParallelProjection', (True,)))
    # vis.execute(magnification=4)

    # Deflect absolute positions further
    Q[:, 0] += 20

    # Assign a single basis rotation to the middle variables
    K[I, 0] = np.arange(M) + 1
    K[I, 1] = 0

    # Assign a single free rotation to the upper variables
    K[I1, 0] = -1
    K[I1, 1] = 0

    V1 = V.copy()

    Xg = np.zeros((1, 3), dtype=np.float64)
    X = np.zeros((1, 3), dtype=np.float64)

    Xb = np.zeros((1, 3), dtype=np.float64)
    y = np.ones((M, 1), dtype=np.float64)

    # Solve for `V1`
    status = solve_forward_sectioned_arap_proj(
        T, V, Xg, s, Xb, y, X, V1, K, C, P, lambdas, preconditioners, 
        isProjection=False,
        uniformWeights=False,
        fixedScale=True,
        maxIterations=100, 
        verbosenessLevel=1)

    print 'X:', np.around(X, decimals=3)
    print 'Xb:', np.around(Xb, decimals=3)
    print 'y:', np.around(y, decimals=3)

    vis = VisualiseMesh()
    vis.add_mesh(V1, T)
    vis.add_points(V1[I], sphere_radius=0.2, actor_name='I', color=(0., 0., 1.))
    vis.add_points(V1[C], sphere_radius=0.2, actor_name='C', color=(1., 0., 0.))
    vis.camera_actions(('SetParallelProjection', (True,)))
    vis.execute(magnification=4)

if __name__ == '__main__':
    main()
