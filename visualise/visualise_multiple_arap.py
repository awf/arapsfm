# visualise_multiple_arap.py

# Imports
import os
import argparse
from visualise import *
from mesh import weights
from mesh.faces import faces_to_cell_array
from itertools import groupby
from operator import itemgetter
from geometry import quaternion as quat

from solvers.arap import ARAPVertexSolver

from pprint import pprint
from matplotlib import cm

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-c', dest='camera_actions', type=str, default=[],
                        action='append', help='camera actions')
    parser.add_argument('--mean_centre', action='store_true', default=False)

    args = parser.parse_args()
    print 'args.input:', args.input
    files = filter(lambda s: s.endswith('.npz'), os.listdir(args.input))
    def sort_key(filename):
        root, ext = os.path.splitext(filename)
        try:
            return int(root)
        except ValueError:
            return -1

    files = map(lambda f: os.path.join(args.input, f), 
                sorted(files, key=sort_key))

    print 'input:'
    pprint(files)
    
    # `V0` taken from first frame
    print 'V0 <- ', files[0]
    z = np.load(files[0])
    V0 = z['V']
    if args.mean_centre:
        V0 -= np.mean(V0, axis=0)
    T = z['T']

    vis = VisualiseMesh()

    # V0 (purple)
    vis.add_mesh(V0, T, actor_name='V0')
    lut = vis.actors['V0'].GetMapper().GetLookupTable()
    lut.SetTableValue(0, *int2dbl(255, 0, 255))

    # setup solveV0_X
    T_ = faces_to_cell_array(z['T'])
    adj, W = weights.weights(V0, T_, weights_type='cotan')
    solveV0_X = ARAPVertexSolver(adj, W, V0)

    # setup color map
    cmap = cm.jet(np.linspace(0., 1., len(files) -1))

    # function to translate axis-angle to rotation matrices
    rotM = lambda x: quat.rotationMatrix(quat.quat(x))

    # add each actor
    for i, file_ in enumerate(files[1:]):
        # load instance file
        print 'V0_X[%d] <- ' % i, file_
        z = np.load(file_)

        # output scale and global rotation (axis-angle)
        def safe_print(var):
            try:
                v = z[var].squeeze()
            except KeyError:
                return

            print ' `%s`:' % var, np.around(v, decimals=3)

        safe_print('s')
        safe_print('Xg')

        # solve for the new coordinates and mean centre
        V0_X = solveV0_X(map(rotM, z['X']))
        if args.mean_centre:
            V0_X -= np.mean(V0_X, axis=0)

        # add actor and adjust lookup table for visualisation
        actor_name = 'V0_X_%d' % i
        vis.add_mesh(V0_X, z['T'], actor_name=actor_name)

        lut = vis.actors[actor_name].GetMapper().GetLookupTable()
        lut.SetTableValue(0, *cmap[i,:3])

    # apply camera actions sequentially
    for action in args.camera_actions:
        method, tup, save_after = parse_camera_action(action)
        print '%s(*%s), save_after=%s' % (method, tup, save_after)
        vis.camera_actions((method, tup))

    vis.execute()

    return

    # V0 w/ X (green)

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
    
if __name__ == '__main__':
    main()
