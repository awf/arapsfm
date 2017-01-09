# visualise_single_arap.py

# Imports
import argparse
from visualise import *
from mesh import weights
from mesh.faces import faces_to_cell_array
from itertools import groupby
from operator import itemgetter

from geometry import quaternion as quat, register
from solvers.arap import ARAPVertexSolver

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

    rotM = lambda x: quat.rotationMatrix(quat.quat(x))
    V0_X = solveV0_X(map(rotM, z['X']))
    V0_X += register.displacement(V0_X, z['V'])

    vis.add_mesh(V0_X, z['T'], actor_name='V0_X')
    lut = vis.actors['V0_X'].GetMapper().GetLookupTable()
    lut.SetTableValue(0, *int2dbl(0, 255, 0))

    # V0 w/ Xg (yellow)
    if 'Xg' in z.keys():
        Xg = z['Xg'][0]
        qg = quat.quat(Xg)
        Q = [quat.quatMultiply(quat.quat(x), qg) for x in z['X']]
        V0_Xg = solveV0_X([quat.rotationMatrix(q) for q in Q])
        V0_Xg += register.displacement(V0_Xg, z['V'])
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
